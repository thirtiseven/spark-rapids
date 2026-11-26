/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids.vortex

import java.io.File
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}
import java.sql.Date
import java.time.LocalDate
import java.util

import scala.collection.mutable.ArrayBuffer

import ai.rapids.cudf.{ColumnVector, CompressionType, DType, ParquetOptions, ParquetWriterOptions, Scalar, Table => CudfTable}
import com.nvidia.spark.rapids._
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.GpuMetric._
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.jni.{LencNative, VortexNative}

import org.apache.spark.TaskContext
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.catalog.{SupportsRead, Table, TableCapability, TableProvider}
import org.apache.spark.sql.connector.expressions.{Expressions, NamedReference, Transform}
import org.apache.spark.sql.connector.read._
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.sources._
import org.apache.spark.sql.types._
import org.apache.spark.sql.util.CaseInsensitiveStringMap
import org.apache.spark.sql.vectorized.ColumnarBatch

case class VortexZoneEntry(
    minVal: Long,
    maxVal: Long,
    hasMinMax: Boolean,
    hasNull: Boolean) extends Serializable

case class VortexFileGroup(
    partVals: Array[Any],
    files: Array[String],
    fileSizes: Array[Long],
    zone: Option[VortexZoneEntry] = None) extends Serializable

/** Evaluates Spark DSv2 partition filters (static & runtime DPP) against Hive partition values. */
object VortexPartitionPruner {
  private def cmp(a: Any, b: Any, dt: DataType): Int = dt match {
    case IntegerType | LongType | ShortType | ByteType =>
      java.lang.Long.compare(a.asInstanceOf[Number].longValue(), b.asInstanceOf[Number].longValue())
    case FloatType | DoubleType =>
      java.lang.Double.compare(a.asInstanceOf[Number].doubleValue(), b.asInstanceOf[Number].doubleValue())
    case DateType =>
      def toDays(v: Any): Int = v match {
        case d: Date => d.toLocalDate.toEpochDay.toInt
        case ld: LocalDate => ld.toEpochDay.toInt
        case n: Number => n.intValue()
        case s: String => LocalDate.parse(s).toEpochDay.toInt
      }
      Integer.compare(toDays(a), toDays(b))
    case _ => a.toString.compareTo(b.toString)
  }

  private def evalOpt(
      f: Filter,
      idxMap: Map[String, Int],
      types: Array[DataType],
      vals: Array[Any]): Option[Boolean] = f match {
    case EqualTo(a, v) =>
      idxMap.get(a).map { i =>
        val pv = vals(i)
        pv != null && v != null && cmp(pv, v, types(i)) == 0
      }
    case EqualNullSafe(a, v) =>
      idxMap.get(a).map { i =>
        val pv = vals(i)
        if (pv == null && v == null) true
        else if (pv == null || v == null) false
        else cmp(pv, v, types(i)) == 0
      }
    case In(a, vs) =>
      idxMap.get(a).map { i =>
        val pv = vals(i)
        pv != null && vs != null && vs.exists(v => v != null && cmp(pv, v, types(i)) == 0)
      }
    case LessThan(a, v) =>
      idxMap.get(a).map(i => vals(i) != null && v != null && cmp(vals(i), v, types(i)) < 0)
    case LessThanOrEqual(a, v) =>
      idxMap.get(a).map(i => vals(i) != null && v != null && cmp(vals(i), v, types(i)) <= 0)
    case GreaterThan(a, v) =>
      idxMap.get(a).map(i => vals(i) != null && v != null && cmp(vals(i), v, types(i)) > 0)
    case GreaterThanOrEqual(a, v) =>
      idxMap.get(a).map(i => vals(i) != null && v != null && cmp(vals(i), v, types(i)) >= 0)
    case IsNull(a) =>
      idxMap.get(a).map(i => vals(i) == null)
    case IsNotNull(a) =>
      idxMap.get(a).map(i => vals(i) != null)
    case And(l, r) =>
      (evalOpt(l, idxMap, types, vals), evalOpt(r, idxMap, types, vals)) match {
        case (Some(false), _) | (_, Some(false)) => Some(false)
        case (Some(true), Some(true)) => Some(true)
        case _ => None
      }
    case Or(l, r) =>
      (evalOpt(l, idxMap, types, vals), evalOpt(r, idxMap, types, vals)) match {
        case (Some(true), _) | (_, Some(true)) => Some(true)
        case (Some(false), Some(false)) => Some(false)
        case _ => None
      }
    case Not(c) => evalOpt(c, idxMap, types, vals).map(!_)
    case _ => None
  }

  def matches(
      f: Filter,
      partColNames: Array[String],
      partColTypes: Array[DataType],
      partVals: Array[Any]): Boolean = {
    if (partColNames.isEmpty) return true
    val idxMap = partColNames.zipWithIndex.toMap
    evalOpt(f, idxMap, partColTypes, partVals).getOrElse(true)
  }

  def isPartitionOnlyFilter(f: Filter, partColSet: Set[String]): Boolean = {
    def refs(flt: Filter): Option[Set[String]] = flt match {
      case EqualTo(a, _) => Some(Set(a))
      case EqualNullSafe(a, _) => Some(Set(a))
      case In(a, _) => Some(Set(a))
      case LessThan(a, _) => Some(Set(a))
      case LessThanOrEqual(a, _) => Some(Set(a))
      case GreaterThan(a, _) => Some(Set(a))
      case GreaterThanOrEqual(a, _) => Some(Set(a))
      case IsNull(a) => Some(Set(a))
      case IsNotNull(a) => Some(Set(a))
      case And(l, r) => for (lr <- refs(l); rr <- refs(r)) yield lr ++ rr
      case Or(l, r) => for (lr <- refs(l); rr <- refs(r)) yield lr ++ rr
      case Not(c) => refs(c)
      case _ => None
    }
    refs(f).exists(r => r.nonEmpty && r.subsetOf(partColSet))
  }

  /** Conservative file-level [min, max] zone-map pruning (never drops any file that could match). */
  def matchesZone(f: Filter, zoneCol: String, z: VortexZoneEntry): Boolean = {
    def toLongOpt(v: Any): Option[Long] = v match {
      case n: Number => Some(n.longValue())
      case d: Date => Some(d.toLocalDate.toEpochDay)
      case ld: LocalDate => Some(ld.toEpochDay)
      case _ => None
    }
    f match {
      case EqualTo(a, v) if a == zoneCol =>
        if (v == null) false
        else toLongOpt(v).forall(vl => z.hasMinMax && vl >= z.minVal && vl <= z.maxVal)
      case EqualNullSafe(a, v) if a == zoneCol =>
        if (v == null) z.hasNull
        else toLongOpt(v).forall(vl => z.hasMinMax && vl >= z.minVal && vl <= z.maxVal)
      case In(a, vs) if a == zoneCol =>
        if (vs == null || vs.isEmpty) false
        else vs.exists { v =>
          if (v == null) z.hasNull
          else toLongOpt(v).forall(vl => z.hasMinMax && vl >= z.minVal && vl <= z.maxVal)
        }
      case LessThan(a, v) if a == zoneCol =>
        if (v == null) false else toLongOpt(v).forall(vl => z.hasMinMax && z.minVal < vl)
      case LessThanOrEqual(a, v) if a == zoneCol =>
        if (v == null) false else toLongOpt(v).forall(vl => z.hasMinMax && z.minVal <= vl)
      case GreaterThan(a, v) if a == zoneCol =>
        if (v == null) false else toLongOpt(v).forall(vl => z.hasMinMax && z.maxVal > vl)
      case GreaterThanOrEqual(a, v) if a == zoneCol =>
        if (v == null) false else toLongOpt(v).forall(vl => z.hasMinMax && z.maxVal >= vl)
      case IsNull(a) if a == zoneCol => z.hasNull
      case IsNotNull(a) if a == zoneCol => z.hasMinMax
      case And(l, r) => matchesZone(l, zoneCol, z) && matchesZone(r, zoneCol, z)
      case Or(l, r) => matchesZone(l, zoneCol, z) || matchesZone(r, zoneCol, z)
      case _ => true
    }
  }
}

/**
 * Tier 1+ Standard Vortex GPU DataSource V2:
 *   - Official vortex-cuda-ffi -> ArrowDeviceArray -> cudf::from_arrow_device zero-copy GPU import.
 *   - Supports Hive key=value partition discovery, static partition pruning, and Spark 3.5
 *     SupportsRuntimeFiltering Dynamic Partition Pruning (DPP) with zero-I/O partition synthesis!
 *   - Opt V1: Single-pass zero-double-copy multi-file coalescing paced cleanly by GpuSemaphore.
 *   - Opt V2: SupportsPushDownFilters + Adaptive GPU 2-Phase Late Materialization.
 */
object VortexSchemaIO {
  val SCHEMA_FILE = "_vortex_schema.json"
  val ZONEMAP_FILE = "_vortex_zonemap.tsv"
  private val HIVE_DEFAULT_PART = "__HIVE_DEFAULT_PARTITION__"

  def stripFileScheme(path: String): String = {
    if (path.startsWith("file://")) path.stripPrefix("file://")
    else if (path.startsWith("file:")) path.stripPrefix("file:")
    else path
  }

  def readDataSchema(dirPath: String): StructType = {
    val clean = stripFileScheme(dirPath)
    val f = new File(clean, SCHEMA_FILE)
    require(f.exists(), s"Missing $SCHEMA_FILE in $clean")
    val json = new String(Files.readAllBytes(f.toPath), StandardCharsets.UTF_8)
    DataType.fromJson(json).asInstanceOf[StructType]
  }

  private def parseTyped(raw: String, dt: DataType): Any = {
    if (raw == null || raw == HIVE_DEFAULT_PART) null
    else dt match {
      case IntegerType => raw.toInt
      case LongType => raw.toLong
      case _ => raw
    }
  }

  private def inferPartType(rawVals: Seq[String]): DataType = {
    val nonNull = rawVals.filter(v => v != null && v != HIVE_DEFAULT_PART)
    if (nonNull.isEmpty) StringType
    else if (nonNull.forall(s => scala.util.Try(s.toInt).isSuccess)) IntegerType
    else if (nonNull.forall(s => scala.util.Try(s.toLong).isSuccess)) LongType
    else StringType
  }

  /**
   * Discovers dataSchema, fullSchema (data + Hive partition cols), Hive partition col metadata,
   * optional file-level zone-map column name (from _vortex_zonemap.tsv), and VortexFileGroups.
   */
  def discoverTable(dirPath: String): (
      StructType, StructType, Array[String], Array[DataType], Array[String], Array[VortexFileGroup]) = {
    val clean = stripFileScheme(dirPath)
    val root = new File(clean)
    require(root.isDirectory, s"Not a directory: $clean")
    val dataSchema = readDataSchema(clean)

    val topEntries = Option(root.listFiles()).getOrElse(Array.empty)
    val topVortex = topEntries
      .filter(f => f.isFile && f.getName.endsWith(".vortex"))
      .sortBy(_.getAbsolutePath)
    val hasPartSubdirs = topEntries.exists(d => d.isDirectory && d.getName.contains("="))

    if (!hasPartSubdirs) {
      val zmFile = new File(root, ZONEMAP_FILE)
      if (zmFile.exists() && topVortex.nonEmpty) {
        val lines = new String(Files.readAllBytes(zmFile.toPath), StandardCharsets.UTF_8)
          .linesIterator.map(_.trim).filter(_.nonEmpty).toArray
        if (lines.nonEmpty && lines(0).startsWith("#COL=")) {
          val zoneCol = lines(0).stripPrefix("#COL=").split("\t")(0)
          val byName = topVortex.map(f => f.getName -> f).toMap
          val groups = new ArrayBuffer[VortexFileGroup]()
          lines.drop(1).foreach { ln =>
            val parts = ln.split("\t")
            if (parts.length >= 4) {
              byName.get(parts(0)).foreach { f =>
                val hasMM = parts(1) != "NULL" && parts(2) != "NULL"
                val mn = if (hasMM) parts(1).toLong else 0L
                val mx = if (hasMM) parts(2).toLong else 0L
                val hn = parts(3) == "1"
                groups += VortexFileGroup(
                  Array.empty, Array(f.getAbsolutePath), Array(f.length()),
                  Some(VortexZoneEntry(mn, mx, hasMM, hn)))
              }
            }
          }
          if (groups.nonEmpty) {
            return (dataSchema, dataSchema, Array.empty, Array.empty, Array(zoneCol), groups.toArray)
          }
        }
      }
      val grp = if (topVortex.nonEmpty) {
        Array(VortexFileGroup(Array.empty, topVortex.map(_.getAbsolutePath), topVortex.map(_.length())))
      } else {
        Array.empty[VortexFileGroup]
      }
      return (dataSchema, dataSchema, Array.empty, Array.empty, Array.empty, grp)
    }

    // Walk Hive col=val/ tree
    val rawGroups = new ArrayBuffer[(Seq[(String, String)], Array[File])]()
    def walk(dir: File, kvs: Seq[(String, String)]): Unit = {
      val entries = Option(dir.listFiles()).getOrElse(Array.empty)
      val vxs = entries.filter(f => f.isFile && f.getName.endsWith(".vortex")).sortBy(_.getAbsolutePath)
      if (vxs.nonEmpty && kvs.nonEmpty) {
        rawGroups += ((kvs, vxs))
      }
      entries.filter(d => d.isDirectory && d.getName.contains("=")).sortBy(_.getName).foreach { sub =>
        val eq = sub.getName.indexOf('=')
        val k = sub.getName.substring(0, eq)
        val v = sub.getName.substring(eq + 1)
        walk(sub, kvs :+ (k -> v))
      }
    }
    walk(root, Seq.empty)

    if (rawGroups.isEmpty) {
      return (dataSchema, dataSchema, Array.empty, Array.empty, Array.empty, Array.empty)
    }

    val partColNames = rawGroups.head._1.map(_._1).toArray
    val partColTypes = partColNames.indices.map { i =>
      inferPartType(rawGroups.map(_._1(i)._2))
    }.toArray
    val dataNames = dataSchema.fieldNames.toSet
    val extraPartFields = partColNames.zip(partColTypes)
      .filterNot { case (n, _) => dataNames.contains(n) }
      .map { case (n, t) => StructField(n, t, nullable = true) }
    val fullSchema = StructType(dataSchema.fields ++ extraPartFields)

    val groups = rawGroups.map { case (kvs, vxs) =>
      val typedVals = partColTypes.indices.map(i => parseTyped(kvs(i)._2, partColTypes(i))).toArray
      VortexFileGroup(typedVals, vxs.map(_.getAbsolutePath), vxs.map(_.length()))
    }.toArray

    (dataSchema, fullSchema, partColNames, partColTypes, Array.empty, groups)
  }

  def readSchema(dirPath: String): StructType = discoverTable(dirPath)._2
}

/** GPU filter evaluator for Spark DataSource V2 Filter trees on Phase-1 Vortex filter columns. */
object VortexGpuFilterEval {
  def isSupportedType(dt: DataType): Boolean = dt match {
    case IntegerType | LongType | FloatType | DoubleType | DateType | StringType => true
    case _ => false
  }

  def isPushable(f: Filter, schema: StructType): Boolean = f match {
    case EqualTo(a, v) => v != null && hasSupportedCol(a, schema)
    case LessThan(a, v) => v != null && hasSupportedCol(a, schema)
    case LessThanOrEqual(a, v) => v != null && hasSupportedCol(a, schema)
    case GreaterThan(a, v) => v != null && hasSupportedCol(a, schema)
    case GreaterThanOrEqual(a, v) => v != null && hasSupportedCol(a, schema)
    case In(a, vs) => vs != null && vs.nonEmpty && vs.forall(_ != null) && hasSupportedCol(a, schema)
    case IsNotNull(a) => hasSupportedCol(a, schema)
    case StringStartsWith(a, v) => v != null && isStringCol(a, schema)
    case StringEndsWith(a, v) => v != null && isStringCol(a, schema)
    case StringContains(a, v) => v != null && isStringCol(a, schema)
    case And(l, r) => isPushable(l, schema) && isPushable(r, schema)
    case Or(l, r) => isPushable(l, schema) && isPushable(r, schema)
    case Not(c) => isPushable(c, schema)
    case _ => false
  }

  private def hasSupportedCol(name: String, schema: StructType): Boolean =
    schema.fields.find(_.name == name).exists(fld => isSupportedType(fld.dataType))

  private def isStringCol(name: String, schema: StructType): Boolean =
    schema.fields.find(_.name == name).exists(_.dataType == StringType)

  def referencedCols(f: Filter): Set[String] = f match {
    case EqualTo(a, _) => Set(a)
    case LessThan(a, _) => Set(a)
    case LessThanOrEqual(a, _) => Set(a)
    case GreaterThan(a, _) => Set(a)
    case GreaterThanOrEqual(a, _) => Set(a)
    case In(a, _) => Set(a)
    case IsNotNull(a) => Set(a)
    case StringStartsWith(a, _) => Set(a)
    case StringEndsWith(a, _) => Set(a)
    case StringContains(a, _) => Set(a)
    case And(l, r) => referencedCols(l) ++ referencedCols(r)
    case Or(l, r) => referencedCols(l) ++ referencedCols(r)
    case Not(c) => referencedCols(c)
    case _ => Set.empty
  }

  private def makeScalar(v: Any, dt: DataType): Scalar = dt match {
    case IntegerType => Scalar.fromInt(v.asInstanceOf[Number].intValue())
    case LongType => Scalar.fromLong(v.asInstanceOf[Number].longValue())
    case FloatType => Scalar.fromFloat(v.asInstanceOf[Number].floatValue())
    case DoubleType => Scalar.fromDouble(v.asInstanceOf[Number].doubleValue())
    case DateType =>
      val days = v match {
        case d: Date => d.toLocalDate.toEpochDay.toInt
        case ld: LocalDate => ld.toEpochDay.toInt
        case n: Number => n.intValue()
      }
      Scalar.timestampDaysFromInt(days)
    case StringType => Scalar.fromString(v.toString)
    case other => throw new UnsupportedOperationException(s"Unsupported Vortex filter dtype: $other")
  }

  private def withDecodedIfDict(cv: ColumnVector)(fn: ColumnVector => ColumnVector): ColumnVector = {
    if (cv.getType == DType.DICTIONARY32) {
      withResource(LencNative.decodeDictionary(cv))(fn)
    } else {
      fn(cv)
    }
  }

  private def evalFilter(
      f: Filter,
      tbl: CudfTable,
      nameToIdx: Map[String, Int],
      types: Array[DataType]): ColumnVector = f match {
    case EqualTo(a, v) =>
      val idx = nameToIdx(a)
      val cv = tbl.getColumn(idx)
      if (cv.getType == DType.DICTIONARY32 && types(idx) == StringType) {
        withResource(LencNative.remapDictionaryToCanonical(cv, Array(v.toString))) { codes =>
          withResource(Scalar.fromInt(0))(s => codes.equalTo(s))
        }
      } else {
        withResource(makeScalar(v, types(idx)))(s => cv.equalTo(s))
      }
    case LessThan(a, v) =>
      val idx = nameToIdx(a)
      withDecodedIfDict(tbl.getColumn(idx)) { cv =>
        withResource(makeScalar(v, types(idx)))(s => cv.lessThan(s))
      }
    case LessThanOrEqual(a, v) =>
      val idx = nameToIdx(a)
      withDecodedIfDict(tbl.getColumn(idx)) { cv =>
        withResource(makeScalar(v, types(idx)))(s => cv.lessOrEqualTo(s))
      }
    case GreaterThan(a, v) =>
      val idx = nameToIdx(a)
      withDecodedIfDict(tbl.getColumn(idx)) { cv =>
        withResource(makeScalar(v, types(idx)))(s => cv.greaterThan(s))
      }
    case GreaterThanOrEqual(a, v) =>
      val idx = nameToIdx(a)
      withDecodedIfDict(tbl.getColumn(idx)) { cv =>
        withResource(makeScalar(v, types(idx)))(s => cv.greaterOrEqualTo(s))
      }
    case In(a, vs) =>
      val idx = nameToIdx(a)
      val cv = tbl.getColumn(idx)
      if (cv.getType == DType.DICTIONARY32 && types(idx) == StringType) {
        val strs = vs.map(_.toString)
        withResource(LencNative.remapDictionaryToCanonical(cv, strs)) { codes =>
          withResource(Scalar.fromInt(0))(z => codes.greaterOrEqualTo(z))
        }
      } else {
        val dt = types(idx)
        val scalars = vs.map(makeScalar(_, dt))
        withResource(scalars) { scs =>
          var acc: ColumnVector = null
          try {
            scs.foreach { s =>
              val eq = cv.equalTo(s)
              if (acc == null) {
                acc = eq
              } else {
                val merged = withResource(eq)(e => withResource(acc)(a0 => a0.or(e)))
                acc = merged
              }
            }
            val res = acc
            acc = null
            res
          } finally {
            if (acc != null) acc.close()
          }
        }
      }
    case IsNotNull(a) =>
      tbl.getColumn(nameToIdx(a)).isNotNull
    case StringStartsWith(a, v) =>
      val idx = nameToIdx(a)
      withDecodedIfDict(tbl.getColumn(idx)) { cv =>
        withResource(Scalar.fromString(v))(s => cv.startsWith(s))
      }
    case StringEndsWith(a, v) =>
      val idx = nameToIdx(a)
      withDecodedIfDict(tbl.getColumn(idx)) { cv =>
        withResource(Scalar.fromString(v))(s => cv.endsWith(s))
      }
    case StringContains(a, v) =>
      val idx = nameToIdx(a)
      withDecodedIfDict(tbl.getColumn(idx)) { cv =>
        withResource(Scalar.fromString(v))(s => cv.stringContains(s))
      }
    case And(l, r) =>
      withResource(evalFilter(l, tbl, nameToIdx, types)) { lc =>
        withResource(evalFilter(r, tbl, nameToIdx, types)) { rc =>
          lc.and(rc)
        }
      }
    case Or(l, r) =>
      withResource(evalFilter(l, tbl, nameToIdx, types)) { lc =>
        withResource(evalFilter(r, tbl, nameToIdx, types)) { rc =>
          lc.or(rc)
        }
      }
    case Not(c) =>
      withResource(evalFilter(c, tbl, nameToIdx, types))(_.not())
    case other =>
      throw new UnsupportedOperationException(s"Unsupported Vortex GPU filter: $other")
  }

  def evalAllConjunctive(
      filters: Array[Filter],
      tbl: CudfTable,
      nameToIdx: Map[String, Int],
      types: Array[DataType]): ColumnVector = {
    var acc: ColumnVector = null
    try {
      filters.foreach { f =>
        val raw = evalFilter(f, tbl, nameToIdx, types)
        // Treat SQL NULL boolean as false so late-materialization never drops valid rows
        val clean = if (raw.getNullCount > 0) {
          withResource(raw) { r =>
            withResource(Scalar.fromBool(false))(fls => r.replaceNulls(fls))
          }
        } else {
          raw
        }
        if (acc == null) {
          acc = clean
        } else {
          val merged = withResource(clean)(c => withResource(acc)(a0 => a0.and(c)))
          acc = merged
        }
      }
      val res = acc
      acc = null
      res
    } finally {
      if (acc != null) acc.close()
    }
  }
}

class VortexDataSourceImpl extends TableProvider {
  override def supportsExternalMetadata(): Boolean = false

  override def inferSchema(options: CaseInsensitiveStringMap): StructType = {
    val path = options.get("path")
    require(path != null && path.nonEmpty, "vortex-gpu datasource requires option 'path'")
    VortexSchemaIO.readSchema(path)
  }

  override def getTable(
      schema: StructType,
      partitioning: Array[Transform],
      properties: util.Map[String, String]): Table = {
    val path = properties.get("path")
    val (dataSchema, fullSchema, partColNames, partColTypes, zoneColNames, fileGroups) =
      VortexSchemaIO.discoverTable(path)
    new VortexTable(
      path, dataSchema, fullSchema, schema, partColNames, partColTypes, zoneColNames, fileGroups)
  }
}

class VortexTable(
    path: String,
    dataSchema: StructType,
    fullSchema: StructType,
    exposedSchema: StructType,
    partColNames: Array[String],
    partColTypes: Array[DataType],
    zoneColNames: Array[String],
    fileGroups: Array[VortexFileGroup]) extends Table with SupportsRead {

  override def name(): String = s"vortex-gpu($path)"

  override def schema(): StructType = exposedSchema

  override def partitioning(): Array[Transform] =
    (partColNames ++ zoneColNames).distinct.map(Expressions.identity)

  override def capabilities(): util.Set[TableCapability] =
    util.EnumSet.of(TableCapability.BATCH_READ)

  override def newScanBuilder(options: CaseInsensitiveStringMap): ScanBuilder =
    new VortexScanBuilder(
      dataSchema, fullSchema, exposedSchema, partColNames, partColTypes, zoneColNames, fileGroups)
}

class VortexScanBuilder(
    dataSchema: StructType,
    fullSchema: StructType,
    exposedSchema: StructType,
    partColNames: Array[String],
    partColTypes: Array[DataType],
    zoneColNames: Array[String],
    fileGroups: Array[VortexFileGroup])
  extends ScanBuilder with SupportsPushDownRequiredColumns with SupportsPushDownFilters {

  private var requiredSchema: StructType = exposedSchema
  private var collectedFilters: Array[Filter] = Array.empty
  private var pushedPartFilters: Array[Filter] = Array.empty
  private var prunedGroups: Array[VortexFileGroup] = fileGroups

  override def pruneColumns(requiredSchema: StructType): Unit = {
    this.requiredSchema = requiredSchema
  }

  override def pushFilters(filters: Array[Filter]): Array[Filter] = {
    val partColSet = partColNames.toSet
    if (partColSet.nonEmpty) {
      val (partFs, dataFs) = filters.partition(VortexPartitionPruner.isPartitionOnlyFilter(_, partColSet))
      if (partFs.nonEmpty) {
        pushedPartFilters = partFs
        prunedGroups = fileGroups.filter(g =>
          partFs.forall(VortexPartitionPruner.matches(_, partColNames, partColTypes, g.partVals)))
      }
      collectedFilters = dataFs.filter(f => VortexGpuFilterEval.isPushable(f, dataSchema))
      dataFs
    } else {
      if (zoneColNames.nonEmpty && filters.nonEmpty) {
        val zc = zoneColNames(0)
        prunedGroups = fileGroups.filter(g =>
          g.zone.forall(z => filters.forall(VortexPartitionPruner.matchesZone(_, zc, z))))
      }
      collectedFilters = filters.filter(f => VortexGpuFilterEval.isPushable(f, dataSchema))
      // Always keep exact row-level filters above zone-map pruned scans for 100% SQL parity!
      filters
    }
  }

  override def pushedFilters(): Array[Filter] = pushedPartFilters

  override def build(): Scan = {
    val reqDataNames = requiredSchema.fieldNames.filterNot(partColNames.contains).toSet
    val applicable = collectedFilters.filter { f =>
      val refs = VortexGpuFilterEval.referencedCols(f)
      refs.nonEmpty && refs.subsetOf(reqDataNames)
    }
    VortexScan(
      dataSchema, fullSchema, requiredSchema,
      partColNames, partColTypes, zoneColNames, prunedGroups, applicable)
  }
}

case class VortexInputPartition(
    filePaths: Array[String],
    groups: Array[VortexFileGroup] = Array.empty) extends InputPartition

case class VortexScan(
    dataSchema: StructType,
    fullSchema: StructType,
    requiredSchema: StructType,
    partColNames: Array[String],
    partColTypes: Array[DataType],
    zoneColNames: Array[String],
    fileGroups: Array[VortexFileGroup],
    lateMatFilters: Array[Filter] = Array.empty)
  extends Scan with Batch with SupportsReportStatistics with SupportsRuntimeFiltering {

  @volatile private var activeGroups: Array[VortexFileGroup] = fileGroups

  def files: Array[String] = activeGroups.flatMap(_.files)
  def fileSizes: Array[Long] = activeGroups.flatMap(_.fileSizes)

  override def readSchema(): StructType = requiredSchema

  override def description(): String =
    s"VortexScan(parts=${activeGroups.length}, files=${files.length}, cols=${requiredSchema.length}, lateMatFilters=${lateMatFilters.length})"

  override def toBatch: Batch = this

  override def filterAttributes(): Array[NamedReference] = {
    val reqSet = requiredSchema.fieldNames.toSet
    (partColNames ++ zoneColNames).distinct.filter(reqSet.contains).map(Expressions.column)
  }

  override def filter(filters: Array[Filter]): Unit = {
    if (filters != null && filters.nonEmpty) {
      if (partColNames.nonEmpty) {
        activeGroups = activeGroups.filter(g =>
          filters.forall(VortexPartitionPruner.matches(_, partColNames, partColTypes, g.partVals)))
      } else if (zoneColNames.nonEmpty) {
        val zc = zoneColNames(0)
        activeGroups = activeGroups.filter(g =>
          g.zone.forall(z => filters.forall(VortexPartitionPruner.matchesZone(_, zc, z))))
      }
    }
  }

  override def estimateStatistics(): Statistics = new Statistics {
    private val sz = math.max(activeGroups.map(_.fileSizes.sum).sum, 1L)
    override def sizeInBytes(): java.util.OptionalLong = java.util.OptionalLong.of(sz)
    override def numRows(): java.util.OptionalLong = java.util.OptionalLong.empty()
  }

  override def planInputPartitions(): Array[InputPartition] = {
    if (partColNames.isEmpty) {
      files.map(f => VortexInputPartition(Array(f)))
    } else {
      activeGroups.map(g => VortexInputPartition(g.files, Array(g)))
    }
  }

  override def createReaderFactory(): PartitionReaderFactory = new PartitionReaderFactory {
    override def supportColumnarReads(partition: InputPartition): Boolean = true
    override def createReader(partition: InputPartition): PartitionReader[InternalRow] =
      throw new UnsupportedOperationException("VortexScan must execute on GPU via GpuVortexScan")
    override def createColumnarReader(partition: InputPartition): PartitionReader[ColumnarBatch] =
      throw new UnsupportedOperationException("VortexScan must execute on GPU via GpuVortexScan")
  }
}

object VortexPartitionHelper {
  def groupFiles(files: Array[String], fileSizes: Array[Long]): Array[Array[String]] = {
    if (files.isEmpty) {
      return Array.empty
    }
    val totalBytes = if (fileSizes.length == files.length) {
      fileSizes.sum
    } else {
      files.map(new File(_).length()).sum
    }
    if (totalBytes <= 16L * 1024L * 1024L) {
      return Array(files)
    }
    val maxPartBytes = SQLConf.get
      .getConfString("spark.rapids.sql.vortex.maxPartitionBytes", "1073741824").toLong
    if (files.length <= 16 || maxPartBytes <= 0L) {
      return files.map(Array(_))
    }
    val minPartitions = 16
    val effectiveMaxBytes = if (totalBytes > 512L * 1024L * 1024L) {
      math.min(maxPartBytes, math.max(totalBytes / minPartitions, 64L * 1024L * 1024L))
    } else {
      maxPartBytes
    }
    val maxFilesPerPart = math.max(1, math.min(64, (files.length + minPartitions - 1) / minPartitions))
    val groups = new ArrayBuffer[Array[String]]()
    val cur = new ArrayBuffer[String]()
    var curBytes = 0L
    files.indices.foreach { idx =>
      val f = files(idx)
      val fsz = if (idx < fileSizes.length) fileSizes(idx) else new File(f).length()
      if (cur.nonEmpty && (curBytes + fsz > effectiveMaxBytes || cur.size >= maxFilesPerPart)) {
        groups += cur.toArray
        cur.clear()
        curBytes = 0L
      }
      cur += f
      curBytes += fsz
    }
    if (cur.nonEmpty) groups += cur.toArray
    groups.toArray
  }

  /** Packs small Hive partition groups into balanced GPU tasks (eliminates 1,824 tiny tasks!). */
  def groupFileGroups(fileGroups: Array[VortexFileGroup]): Array[Array[VortexFileGroup]] = {
    if (fileGroups.isEmpty) return Array.empty
    val totalBytes = fileGroups.map(_.fileSizes.sum).sum
    if (totalBytes <= 16L * 1024L * 1024L) {
      return Array(fileGroups)
    }
    val targetParts = if (totalBytes > 512L * 1024L * 1024L) 32 else 16
    val maxBytesPerTask = math.max(64L * 1024L * 1024L, totalBytes / targetParts)
    val maxGroupsPerTask = math.max(1, math.min(128, (fileGroups.length + targetParts - 1) / targetParts))
    val out = new ArrayBuffer[Array[VortexFileGroup]]()
    val cur = new ArrayBuffer[VortexFileGroup]()
    var curBytes = 0L
    fileGroups.foreach { g =>
      val gsz = g.fileSizes.sum
      if (cur.nonEmpty && (curBytes + gsz > maxBytesPerTask || cur.size >= maxGroupsPerTask)) {
        out += cur.toArray
        cur.clear()
        curBytes = 0L
      }
      cur += g
      curBytes += gsz
    }
    if (cur.nonEmpty) out += cur.toArray
    out.toArray
  }
}

case class GpuVortexScan(
    dataSchema: StructType,
    fullSchema: StructType,
    requiredSchema: StructType,
    partColNames: Array[String],
    partColTypes: Array[DataType],
    zoneColNames: Array[String],
    fileGroups: Array[VortexFileGroup],
    lateMatFilters: Array[Filter] = Array.empty)
  extends GpuScan with Batch with SupportsReportStatistics with SupportsRuntimeFiltering {

  @volatile private var activeGroups: Array[VortexFileGroup] = fileGroups
  @volatile private var extraDict32Cols: Set[String] = Set.empty

  def files: Array[String] = activeGroups.flatMap(_.files)
  def fileSizes: Array[Long] = activeGroups.flatMap(_.fileSizes)

  def addDict32Cols(cols: Set[String]): Unit = {
    if (cols.nonEmpty) {
      extraDict32Cols = extraDict32Cols ++ cols
    }
  }

  override def readSchema(): StructType = requiredSchema

  override def description(): String =
    s"GpuVortexScan(parts=${activeGroups.length}, files=${files.length}, cols=${requiredSchema.length}, lateMatFilters=${lateMatFilters.length})"

  override def withInputFile(): GpuScan = this

  override def toBatch: Batch = this

  override def filterAttributes(): Array[NamedReference] = {
    val reqSet = requiredSchema.fieldNames.toSet
    (partColNames ++ zoneColNames).distinct.filter(reqSet.contains).map(Expressions.column)
  }

  override def filter(filters: Array[Filter]): Unit = {
    if (filters != null && filters.nonEmpty) {
      if (partColNames.nonEmpty) {
        activeGroups = activeGroups.filter(g =>
          filters.forall(VortexPartitionPruner.matches(_, partColNames, partColTypes, g.partVals)))
      } else if (zoneColNames.nonEmpty) {
        val zc = zoneColNames(0)
        activeGroups = activeGroups.filter(g =>
          g.zone.forall(z => filters.forall(VortexPartitionPruner.matchesZone(_, zc, z))))
      }
    }
  }

  override def estimateStatistics(): Statistics = new Statistics {
    private val sz = math.max(activeGroups.map(_.fileSizes.sum).sum, 1L)
    override def sizeInBytes(): java.util.OptionalLong = java.util.OptionalLong.of(sz)
    override def numRows(): java.util.OptionalLong = java.util.OptionalLong.empty()
  }

  override def planInputPartitions(): Array[InputPartition] = {
    if (partColNames.isEmpty) {
      VortexPartitionHelper.groupFiles(files, fileSizes).map(fs => VortexInputPartition(fs))
    } else {
      VortexPartitionHelper.groupFileGroups(activeGroups)
        .map(gs => VortexInputPartition(gs.flatMap(_.files), gs))
    }
  }

  override def createReaderFactory(): PartitionReaderFactory = {
    val colNames = requiredSchema.fieldNames
    val dict32Set = GpuDictionaryUtils.updateDriverDict32Cols(SQLConf.get) ++ extraDict32Cols
    val dict32Mask: Array[Byte] = if (dict32Set.isEmpty) {
      null
    } else {
      val mask = requiredSchema.fields.map { f =>
        if (f.dataType == StringType && dict32Set.contains(f.name)) 1.toByte else 0.toByte
      }
      if (mask.exists(_ != 0)) mask else null
    }
    val coalesceTargetBytes = SQLConf.get
      .getConfString("spark.rapids.sql.vortex.coalesceTargetBytes", "268435456").toLong
    val batchRows = SQLConf.get
      .getConfString("spark.rapids.sql.vortex.batchRows", "0").toLong
    val lateMatEnabled = SQLConf.get
      .getConfString("spark.rapids.sql.vortex.lateMaterializationEnabled", "true").toBoolean
    val activeFilters = if (lateMatEnabled) lateMatFilters else Array.empty[Filter]
    val fallbackCountCol = if (dataSchema.nonEmpty) dataSchema.fields(0).name else ""
    GpuVortexPartitionReaderFactory(
      requiredSchema, colNames, partColNames, partColTypes, fallbackCountCol,
      dict32Mask, coalesceTargetBytes, batchRows, activeFilters, metrics)
  }
}

case class GpuVortexPartitionReaderFactory(
    readSchema: StructType,
    colNames: Array[String],
    partColNames: Array[String],
    partColTypes: Array[DataType],
    fallbackCountCol: String,
    dict32Mask: Array[Byte],
    coalesceTargetBytes: Long,
    batchRows: Long,
    lateMatFilters: Array[Filter],
    metrics: Map[String, GpuMetric]) extends PartitionReaderFactory {

  override def supportColumnarReads(partition: InputPartition): Boolean = true

  override def createReader(partition: InputPartition): PartitionReader[InternalRow] =
    throw new UnsupportedOperationException("GpuVortexScan supports columnar reads only")

  override def createColumnarReader(partition: InputPartition): PartitionReader[ColumnarBatch] = {
    val vp = partition.asInstanceOf[VortexInputPartition]
    val needsPartCols = partColNames.nonEmpty &&
      readSchema.fieldNames.exists(partColNames.contains)
    if (!needsPartCols) {
      // Fast single-CudaExecutionCtx 16-worker Tokio prefetch across all surviving (DPP-pruned) files!
      new GpuVortexPartitionReader(
        vp.filePaths, readSchema, colNames, dict32Mask, coalesceTargetBytes, batchRows,
        lateMatFilters, metrics)
    } else {
      new GpuVortexHivePartitionedReader(
        vp.groups, readSchema, partColNames, partColTypes, fallbackCountCol,
        dict32Mask, coalesceTargetBytes, batchRows, metrics)
    }
  }
}

/**
 * Partition-aware GPU reader for Hive-partitioned Vortex tables:
 *   - Reads only physical data columns from .vortex files (zero I/O for partition columns!)
 *   - Synthesizes requested partition columns via ColumnVector.fromScalar on GPU
 *   - Coalesces multiple small Hive partition files into clean 64-256 MiB GPU batches!
 */
class GpuVortexHivePartitionedReader(
    groups: Array[VortexFileGroup],
    readSchema: StructType,
    partColNames: Array[String],
    partColTypes: Array[DataType],
    fallbackCountCol: String,
    dict32Mask: Array[Byte],
    coalesceTargetBytes: Long,
    batchRows: Long,
    metrics: Map[String, GpuMetric]) extends PartitionReader[ColumnarBatch] {

  private var currentBatch: ColumnarBatch = _
  private val readTypes: Array[DataType] = readSchema.fields.map(_.dataType)
  private val partIdxByName: Map[String, Int] = partColNames.zipWithIndex.toMap
  // For each output col i: Either Left(dataColIdx) or Right(partColIdx)
  private val dataColNamesBuf = new ArrayBuffer[String]()
  private val dataDictMaskBuf = new ArrayBuffer[Byte]()
  private val outColSource: Array[Either[Int, Int]] = readSchema.fields.zipWithIndex.map { case (f, i) =>
    partIdxByName.get(f.name) match {
      case Some(pIdx) => Right(pIdx)
      case None =>
        val dIdx = dataColNamesBuf.length
        dataColNamesBuf += f.name
        dataDictMaskBuf += (if (dict32Mask != null && i < dict32Mask.length) dict32Mask(i) else 0.toByte)
        Left(dIdx)
    }
  }
  private val dataColNames: Array[String] = dataColNamesBuf.toArray
  private val dataDictMask: Array[Byte] =
    if (dataDictMaskBuf.exists(_ != 0)) dataDictMaskBuf.toArray else null
  private val scanCols: Array[String] =
    if (dataColNames.nonEmpty) dataColNames
    else if (fallbackCountCol.nonEmpty) Array(fallbackCountCol)
    else Array.empty[String]
  private val scanDictMask: Array[Byte] =
    if (dataColNames.nonEmpty) dataDictMask else null

  private val opTime = metrics.getOrElse(OP_TIME_LEGACY, NoopMetric)
  private val gpuDecodeTime = metrics.getOrElse(GPU_DECODE_TIME, NoopMetric)
  private val numOutputBatches = metrics.getOrElse(NUM_OUTPUT_BATCHES, NoopMetric)
  private val targetCoalesceBytes = if (coalesceTargetBytes > 0L) coalesceTargetBytes else 128L * 1024L * 1024L

  private var groupIdx = 0
  private var activeScan: VortexNative.MultiFileScan = _
  private var activeGroup: VortexFileGroup = _

  private def makePartScalar(v: Any, dt: DataType): Scalar = {
    if (v == null) {
      Scalar.fromNull(GpuColumnVector.getNonNestedRapidsType(dt))
    } else dt match {
      case IntegerType => Scalar.fromInt(v.asInstanceOf[Number].intValue())
      case LongType => Scalar.fromLong(v.asInstanceOf[Number].longValue())
      case StringType => Scalar.fromString(v.toString)
      case other => throw new UnsupportedOperationException(s"Unsupported Vortex partition col dtype: $other")
    }
  }

  private def buildBatchTable(dataTbl: CudfTable, partVals: Array[Any]): CudfTable = {
    val nRows = dataTbl.getRowCount.toInt
    val cols = new Array[ColumnVector](outColSource.length)
    var ok = false
    try {
      var i = 0
      while (i < outColSource.length) {
        outColSource(i) match {
          case Left(dIdx) =>
            cols(i) = dataTbl.getColumn(dIdx).incRefCount()
          case Right(pIdx) =>
            withResource(makePartScalar(partVals(pIdx), partColTypes(pIdx))) { s =>
              cols(i) = ColumnVector.fromScalar(s, nRows)
            }
        }
        i += 1
      }
      val t = new CudfTable(cols: _*)
      ok = true
      t
    } finally {
      cols.foreach(c => if (c != null) c.close())
    }
  }

  override def next(): Boolean = {
    if (currentBatch != null) {
      currentBatch.close()
      currentBatch = null
    }
    if (groups.isEmpty || scanCols.isEmpty) return false

    GpuSemaphore.acquireIfNecessary(TaskContext.get())
    opTime.ns {
      val pending = new ArrayBuffer[CudfTable]()
      var accBytes = 0L
      var accRows = 0L
      var emptyColRows = 0L
      val canCoalesce = readTypes.nonEmpty && (dataDictMask == null)
      try {
        while ((pending.isEmpty && emptyColRows == 0L) ||
            (canCoalesce && accBytes < targetCoalesceBytes && accRows < 4000000L)) {
          if (activeScan == null) {
            if (groupIdx >= groups.length) {
              if (pending.nonEmpty || emptyColRows > 0L) {
                return emitPending(pending, emptyColRows)
              } else {
                return false
              }
            }
            activeGroup = groups(groupIdx)
            groupIdx += 1
            activeScan = VortexNative.openMultiScan(activeGroup.files, scanCols, batchRows, 1)
          }
          val dataTbl = gpuDecodeTime.ns {
            activeScan.nextCoalescedBatch(scanDictMask, targetCoalesceBytes)
          }
          if (dataTbl == null) {
            activeScan.safeClose()
            activeScan = null
          } else {
            withResource(dataTbl) { dt =>
              val nr = dt.getRowCount
              if (nr > 0) {
                if (readTypes.isEmpty) {
                  emptyColRows += nr
                } else {
                  val fullT = buildBatchTable(dt, activeGroup.partVals)
                  pending += fullT
                  accBytes += fullT.getDeviceMemorySize
                  accRows += nr
                }
              }
            }
          }
        }
        emitPending(pending, emptyColRows)
      } finally {
        pending.safeClose()
      }
    }
  }

  private def emitPending(pending: ArrayBuffer[CudfTable], emptyColRows: Long): Boolean = {
    if (readTypes.isEmpty) {
      if (emptyColRows <= 0L) return false
      currentBatch = new ColumnarBatch(Array.empty, emptyColRows.toInt)
      numOutputBatches += 1
      return true
    }
    if (pending.isEmpty) return false
    val finalTbl = if (pending.length == 1) {
      pending.remove(0)
    } else {
      val arr = pending.toArray
      pending.clear()
      withResource(arr)(tbls => CudfTable.concatenate(tbls: _*))
    }
    withResource(finalTbl) { t =>
      currentBatch = GpuColumnVector.from(t, readTypes)
      numOutputBatches += 1
    }
    true
  }

  override def get(): ColumnarBatch = {
    val ret = currentBatch
    currentBatch = null
    ret
  }

  override def close(): Unit = {
    if (currentBatch != null) {
      currentBatch.close()
      currentBatch = null
    }
    if (activeScan != null) {
      activeScan.safeClose()
      activeScan = null
    }
  }
}

class GpuVortexPartitionReader(
    filePaths: Array[String],
    readSchema: StructType,
    colNames: Array[String],
    dict32Mask: Array[Byte],
    coalesceTargetBytes: Long,
    batchRows: Long,
    lateMatFilters: Array[Filter],
    metrics: Map[String, GpuMetric]) extends PartitionReader[ColumnarBatch] {

  private var currentBatch: ColumnarBatch = _
  private val readTypes: Array[DataType] = readSchema.fields.map(_.dataType)

  // Determine whether GPU 2-Phase Late Materialization applies:
  // Filter touches a strict subset of projected columns AND excludes non-trivial filters (not just IsNotNull)!
  private val meaningfulFilters: Array[Filter] = lateMatFilters.filter {
    case _: IsNotNull => false
    case _ => true
  }
  private val filterColSet: Set[String] =
    meaningfulFilters.flatMap(VortexGpuFilterEval.referencedCols).toSet
  private val filterColIndices: Array[Int] =
    colNames.indices.filter(i => filterColSet.contains(colNames(i))).toArray
  private val payloadColIndices: Array[Int] =
    colNames.indices.filter(i => !filterColSet.contains(colNames(i))).toArray
  private var useLateMat: Boolean =
    meaningfulFilters.nonEmpty && filterColIndices.nonEmpty && payloadColIndices.nonEmpty &&
      readTypes.nonEmpty && filePaths.nonEmpty

  private val filterColNames: Array[String] = filterColIndices.map(colNames)
  private val filterColTypes: Array[DataType] = filterColIndices.map(readTypes)
  private val filterNameToIdx: Map[String, Int] = filterColNames.zipWithIndex.toMap
  private val filterDictMask: Array[Byte] =
    filterColIndices.map { i =>
      if (readTypes(i) == StringType) 1.toByte else 0.toByte
    }
  private val payloadColNames: Array[String] = payloadColIndices.map(colNames)
  private val payloadHasStrings: Boolean =
    payloadColIndices.exists(i => readTypes(i) == StringType && (dict32Mask == null || dict32Mask(i) == 0))
  private val payloadDictMask: Array[Byte] =
    if (dict32Mask == null) null else payloadColIndices.map(dict32Mask)

  // Only enable 2-phase late materialization when payload actually has heavy STRING columns
  // (or user explicitly enables it for all types) since vortex-cuda decodes numeric CudaFlat blocks
  // fast enough that opening a file twice only pays off when skipping heavy strings or 0-match files!
  private val lateMatForceAll: Boolean = SQLConf.get
    .getConfString("spark.rapids.sql.vortex.lateMatAllTypes", "false").toBoolean
  private val partTotalBytes: Long =
    filePaths.map(p => new java.io.File(p).length()).sum
  private val avgFileBytes: Long =
    if (filePaths.nonEmpty) partTotalBytes / filePaths.length else 0L
  private lazy val useZeroCopyForPart: Boolean =
    zeroCopyEnabled && (filePaths.length <= 2 || avgFileBytes >= (24L << 20))
  useLateMat = useLateMat && (payloadHasStrings || lateMatForceAll) &&
    partTotalBytes >= (64L << 20) && avgFileBytes >= (24L << 20)

  private var fileIdx = 0
  private var multiScan: VortexNative.MultiFileScan =
    if (!useLateMat && readTypes.nonEmpty && filePaths.nonEmpty) {
      VortexNative.openMultiScan(filePaths, colNames, batchRows, 0)
    } else {
      null
    }

  private val opTime = metrics.getOrElse(OP_TIME_LEGACY, NoopMetric)
  private val gpuDecodeTime = metrics.getOrElse(GPU_DECODE_TIME, NoopMetric)
  private val numOutputBatches = metrics.getOrElse(NUM_OUTPUT_BATCHES, NoopMetric)

  private def normalizeFilterColForOutput(cv: ColumnVector, reqColIdx: Int): ColumnVector = {
    val wantDict = dict32Mask != null && dict32Mask(reqColIdx) != 0
    if (cv.getType == DType.DICTIONARY32 && !wantDict) {
      LencNative.decodeDictionary(cv)
    } else {
      cv.incRefCount()
    }
  }

  private def readOneFileLateMat(path: String): Option[CudfTable] = {
    // Phase 1: Decode ONLY filter columns on GPU via vortex-cuda!
    val fTbl = gpuDecodeTime.ns {
      VortexNative.scanFileWithRows(path, filterColNames, null, filterDictMask)
    }
    if (fTbl == null) return None
    withResource(fTbl) { ft =>
      val nRows = ft.getRowCount
      if (nRows == 0L) return None
      withResource(VortexGpuFilterEval.evalAllConjunctive(
          meaningfulFilters, ft, filterNameToIdx, filterColTypes)) { mask =>
        // Check exact pass count via 8-byte GPU sum() scalar BEFORE copying any row indices D2H!
        val nPass: Long = withResource(mask.sum(DType.INT64)) { s =>
          if (s.isValid) s.getLong else 0L
        }
        if (nPass == 0L) {
          // 100% of file skipped — ZERO payload bytes read or decoded!
          return None
        }
        // Tightened adaptive selectivity gate (fixes Q04 3.8% & Q22 50% double-open regressions!):
        // Since vortex-cuda's with_row_indices still decodes CudaFlat chunks before filtering,
        // 2-phase late materialization only pays off when selectivity <= 2% (nPass * 50 <= nRows)
        // or entire files are skipped (nPass == 0, e.g. NF_J0202). Otherwise fall back immediately
        // to 1-pass true zero-copy for all remaining files!
        if (nPass * 50L > nRows) {
          useLateMat = false
          val pTbl = gpuDecodeTime.ns {
            VortexNative.scanFileWithRows(path, payloadColNames, null, payloadDictMask)
          }
          if (pTbl == null) return None
          withResource(pTbl) { pt =>
            val outCols = new Array[ColumnVector](colNames.length)
            try {
              filterColIndices.indices.foreach { fi =>
                val dstIdx = filterColIndices(fi)
                outCols(dstIdx) = normalizeFilterColForOutput(ft.getColumn(fi), dstIdx)
              }
              payloadColIndices.indices.foreach { pi =>
                outCols(payloadColIndices(pi)) = pt.getColumn(pi).incRefCount()
              }
              val res = new CudfTable(outCols: _*)
              return Some(res)
            } finally {
              outCols.foreach(c => if (c != null) c.close())
            }
          }
        }
        // Phase 2: Ultra-low selectivity (<= 2%) -> Vortex takes ONLY surviving rows of heavy payload!
        val rowIndices = VortexNative.maskToRowIndices(mask)
        val pTbl = gpuDecodeTime.ns {
          VortexNative.scanFileWithRows(path, payloadColNames, rowIndices, payloadDictMask)
        }
        if (pTbl == null) return None
        withResource(pTbl) { pt =>
          // If surviving payload is tiny (< 48 bytes/row, e.g. narrow dict strings), disable
          // 2-phase for remaining files so we never double-open for cheap columns!
          if (nPass > 0L && pt.getDeviceMemorySize < nPass * 48L) {
            useLateMat = false
          }
          withResource(ft.filter(mask)) { filtFt =>
            val outCols = new Array[ColumnVector](colNames.length)
            try {
              filterColIndices.indices.foreach { fi =>
                val dstIdx = filterColIndices(fi)
                outCols(dstIdx) = normalizeFilterColForOutput(filtFt.getColumn(fi), dstIdx)
              }
              payloadColIndices.indices.foreach { pi =>
                outCols(payloadColIndices(pi)) = pt.getColumn(pi).incRefCount()
              }
              Some(new CudfTable(outCols: _*))
            } finally {
              outCols.foreach(c => if (c != null) c.close())
            }
          }
        }
      }
    }
  }

  private def hasAnyDict32(t: CudfTable): Boolean = {
    var i = 0
    val nc = t.getNumberOfColumns
    while (i < nc) {
      if (t.getColumn(i).getType == DType.DICTIONARY32) return true
      i += 1
    }
    false
  }

  private val zeroCopyEnabled: Boolean = SQLConf.get
    .getConfString("spark.rapids.sql.vortex.zeroCopyEnabled", "true").toBoolean

  override def next(): Boolean = {
    if (currentBatch != null) {
      currentBatch.close()
      currentBatch = null
    }
    if (readTypes.isEmpty) return false

    // Acquire GpuSemaphore BEFORE opening prefetch scan & opTime.ns:
    // keeps active prefetched VRAM bounded to concurrentGpuTasks(6) x prefetch(1),
    // while the 16-worker Tokio I/O pool overlaps file i+1 open+read with file i GPU decode!
    GpuSemaphore.acquireIfNecessary(TaskContext.get())
    opTime.ns {
      if (!useLateMat && multiScan == null && fileIdx < filePaths.length) {
        val rem = filePaths.slice(fileIdx, filePaths.length)
        fileIdx = filePaths.length
        multiScan = VortexNative.openMultiScan(rem, colNames, batchRows, 1)
      }
      if (useLateMat) {
        val pending = new ArrayBuffer[CudfTable]()
        var accBytes = 0L
        var accRows = 0L
        var canCoalesce = coalesceTargetBytes > 0L
        try {
          while (useLateMat && fileIdx < filePaths.length &&
              (pending.isEmpty || (canCoalesce && accBytes < coalesceTargetBytes && accRows < 80000000L))) {
            val path = filePaths(fileIdx)
            fileIdx += 1
            readOneFileLateMat(path).foreach { t =>
              if (hasAnyDict32(t)) canCoalesce = false
              pending += t
              accBytes += t.getDeviceMemorySize
              accRows += t.getRowCount
            }
          }
          // If adaptive selectivity gate tripped mid-partition, open direct MultiFileScan for remaining files!
          if (!useLateMat && fileIdx < filePaths.length && multiScan == null) {
            val rem = filePaths.slice(fileIdx, filePaths.length)
            fileIdx = filePaths.length
            multiScan = VortexNative.openMultiScan(rem, colNames, batchRows, 1)
          }
          if (pending.nonEmpty) {
            val finalTbl = if (pending.length == 1) {
              pending.remove(0)
            } else {
              val all = pending.toArray
              pending.clear()
              withResource(all)(tbls => CudfTable.concatenate(tbls: _*))
            }
            withResource(finalTbl) { t =>
              currentBatch = GpuColumnVector.from(t, readTypes)
              numOutputBatches += 1
            }
            return true
          }
        } finally {
          pending.safeClose()
        }
      }

      if (multiScan == null) return false
      if (useZeroCopyForPart) {
        val cvs = gpuDecodeTime.ns {
          multiScan.nextZeroCopyBatch(dict32Mask)
        }
        if (cvs == null) {
          multiScan.safeClose()
          multiScan = null
          return false
        }
        val numRows = if (cvs.nonEmpty) cvs(0).getRowCount.toInt else 0
        val gpuCols = new Array[org.apache.spark.sql.vectorized.ColumnVector](cvs.length)
        var ok = false
        try {
          var i = 0
          while (i < cvs.length) {
            gpuCols(i) = GpuColumnVector.from(cvs(i), readTypes(i))
            cvs(i) = null
            i += 1
          }
          currentBatch = new ColumnarBatch(gpuCols, numRows)
          numOutputBatches += 1
          ok = true
        } finally {
          if (!ok) {
            cvs.foreach(c => if (c != null) c.close())
            gpuCols.foreach(c => if (c != null) c.close())
          }
        }
      } else {
        val tbl = gpuDecodeTime.ns {
          multiScan.nextCoalescedBatch(dict32Mask, coalesceTargetBytes)
        }
        if (tbl == null) {
          multiScan.safeClose()
          multiScan = null
          return false
        }
        withResource(tbl) { t =>
          currentBatch = GpuColumnVector.from(t, readTypes)
          numOutputBatches += 1
        }
      }
    }
    true
  }

  override def get(): ColumnarBatch = {
    val ret = currentBatch
    currentBatch = null
    ret
  }

  override def close(): Unit = {
    if (currentBatch != null) {
      currentBatch.close()
      currentBatch = null
    }
    if (multiScan != null) {
      multiScan.safeClose()
      multiScan = null
    }
  }
}

case class VortexScanRuleMeta(
    a: VortexScan,
    override val conf: RapidsConf,
    p: Option[RapidsMeta[_, _, _]],
    r: DataFromReplacementRule) extends ScanMeta[VortexScan](a, conf, p, r) {

  override val supportsRuntimeFilters: Boolean = true

  override def tagSelfForGpu(): Unit = {}

  override def convertToGpu(): GpuScan =
    GpuVortexScan(
      a.dataSchema, a.fullSchema, a.requiredSchema,
      a.partColNames, a.partColTypes, a.zoneColNames, a.fileGroups, a.lateMatFilters)
}

/** Converts a directory of Parquet files (flat OR Hive col=val/ partitioned!) directly to standard .vortex files! */
class VortexConverterImpl extends java.util.function.Function[Array[String], java.lang.Long] {
  private def collectParquetFiles(dir: File): Array[File] = {
    val buf = new ArrayBuffer[File]()
    def walk(f: File): Unit = {
      val list = f.listFiles()
      if (list != null) {
        list.foreach { child =>
          if (child.isDirectory) walk(child)
          else if (child.isFile && child.getName.endsWith(".parquet")) buf += child
        }
      }
    }
    walk(dir)
    buf.toArray.sortBy(_.getAbsolutePath)
  }

  override def apply(args: Array[String]): java.lang.Long = {
    if (args.length >= 9 && args(5) == "COMPACT_SORTED_PAIR") {
      return compactHiveParquetToSortedPair(
        rawHivePqDir = VortexSchemaIO.stripFileScheme(args(0)),
        outVxDir = VortexSchemaIO.stripFileScheme(args(1)),
        fullSchemaJson = args(2),
        numThreads = math.max(1, args(4).toInt),
        outPqDir = VortexSchemaIO.stripFileScheme(args(6)),
        partCol = args(7),
        targetChunkBytes = math.max(64L, args(8).toLong) * 1024L * 1024L)
    }

    val parquetDir = VortexSchemaIO.stripFileScheme(args(0))
    val vortexDir = VortexSchemaIO.stripFileScheme(args(1))
    val schemaJson = args(2)
    val includeCols: Array[String] =
      if (args.length >= 4 && args(3).nonEmpty) args(3).split(",") else Array.empty
    val numThreads: Int =
      if (args.length >= 5 && args(4).nonEmpty) args(4).toInt else 4

    val schema = DataType.fromJson(schemaJson).asInstanceOf[StructType]
    val colNames = if (includeCols.nonEmpty) includeCols else schema.fieldNames

    val srcRoot = new File(parquetDir).getCanonicalFile
    val outDir = new File(vortexDir)
    outDir.mkdirs()
    Files.write(
      Paths.get(vortexDir, VortexSchemaIO.SCHEMA_FILE),
      schemaJson.getBytes(StandardCharsets.UTF_8))

    val pqFiles = collectParquetFiles(srcRoot)
    val optsBuilder = ParquetOptions.builder()
    if (includeCols.nonEmpty) optsBuilder.includeColumn(includeCols: _*)
    val opts = optsBuilder.build()

    // Preserve Hive col=val/ subdirectories automatically (1 clean .vortex file per leaf partition!)
    val hasHiveDirs = pqFiles.exists(f => f.getParentFile.getCanonicalPath != srcRoot.getPath)
    val totalBytes = new java.util.concurrent.atomic.AtomicLong(0L)
    val pool = java.util.concurrent.Executors.newFixedThreadPool(math.max(1, numThreads))
    try {
      if (!hasHiveDirs) {
        val futures = pqFiles.zipWithIndex.map { case (pqFile, i) =>
          pool.submit(new Runnable {
            override def run(): Unit = {
              val outPath = new File(outDir, f"part-$i%05d.vortex")
              if (!outPath.exists() || outPath.length() == 0L) {
                withResource(CudfTable.readParquet(opts, pqFile)) { table =>
                  VortexNative.writeTable(table, colNames, outPath.getAbsolutePath)
                }
              }
              totalBytes.addAndGet(outPath.length())
            }
          })
        }
        futures.foreach(_.get())
      } else {
        val byParent = pqFiles.groupBy(_.getParentFile.getCanonicalFile).toArray.sortBy(_._1.getPath)
        val futures = byParent.map { case (parentDir, filesInPart) =>
          pool.submit(new Runnable {
            override def run(): Unit = {
              val rel = srcRoot.toPath.relativize(parentDir.toPath)
              val partOutDir = outDir.toPath.resolve(rel).toFile
              partOutDir.mkdirs()
              val outPath = new File(partOutDir, "part-00000.vortex")
              if (!outPath.exists() || outPath.length() == 0L) {
                if (filesInPart.length == 1) {
                  withResource(CudfTable.readParquet(opts, filesInPart(0))) { table =>
                    VortexNative.writeTable(table, colNames, outPath.getAbsolutePath)
                  }
                } else {
                  withResource(filesInPart.safeMap(f => CudfTable.readParquet(opts, f))) { tbls =>
                    withResource(CudfTable.concatenate(tbls: _*)) { merged =>
                      VortexNative.writeTable(merged, colNames, outPath.getAbsolutePath)
                    }
                  }
                }
              }
              totalBytes.addAndGet(outPath.length())
            }
          })
        }
        futures.foreach(_.get())
      }
    } finally {
      pool.shutdown()
    }
    java.lang.Long.valueOf(totalBytes.get())
  }

  /**
   * Compacts a Hive col=val/ Parquet directory into 1:1 identical contiguous date-sorted
   * big Parquet files + Standard Vortex files + _vortex_zonemap.tsv for Spark 3.5 DPP!
   */
  private def compactHiveParquetToSortedPair(
      rawHivePqDir: String,
      outVxDir: String,
      fullSchemaJson: String,
      numThreads: Int,
      outPqDir: String,
      partCol: String,
      targetChunkBytes: Long): java.lang.Long = {
    val fullSchema = DataType.fromJson(fullSchemaJson).asInstanceOf[StructType]
    val fullColNames = fullSchema.fieldNames
    val partColIdx = fullSchema.fieldIndex(partCol)
    val partColType = fullSchema.fields(partColIdx).dataType
    val dataColNames = fullSchema.fields.filterNot(_.name == partCol).map(_.name)

    val srcRoot = new File(rawHivePqDir).getCanonicalFile
    val pqOutDir = new File(outPqDir)
    val vxOutDir = new File(outVxDir)
    pqOutDir.mkdirs()
    vxOutDir.mkdirs()
    Files.write(
      Paths.get(outVxDir, VortexSchemaIO.SCHEMA_FILE),
      fullSchemaJson.getBytes(StandardCharsets.UTF_8))

    // Collect leaf Hive partCol=val/ directories separated into sorted numeric & NULL partitions
    val prefix = partCol + "="
    val partDirs = Option(srcRoot.listFiles()).getOrElse(Array.empty)
      .filter(d => d.isDirectory && d.getName.startsWith(prefix))
    val numericParts = new ArrayBuffer[(Long, Array[File], Long)]()
    val nullFiles = new ArrayBuffer[File]()
    partDirs.foreach { d =>
      val rawV = d.getName.stripPrefix(prefix)
      val fs = collectParquetFiles(d)
      if (fs.nonEmpty) {
        if (rawV == "__HIVE_DEFAULT_PARTITION__") {
          nullFiles ++= fs
        } else {
          numericParts += ((rawV.toLong, fs, fs.map(_.length()).sum))
        }
      }
    }
    val sortedNum = numericParts.sortBy(_._1)

    // Pack consecutive date_sk partitions into contiguous ~targetChunkBytes chunks
    case class ChunkSpec(
        chunkIdx: Int,
        days: Array[(Option[Long], Array[File])],
        minVal: Option[Long],
        maxVal: Option[Long],
        hasNull: Boolean)

    val chunks = new ArrayBuffer[ChunkSpec]()
    val curDays = new ArrayBuffer[(Option[Long], Array[File])]()
    var curBytes = 0L
    var curMin = Long.MaxValue
    var curMax = Long.MinValue
    sortedNum.foreach { case (dVal, fs, sz) =>
      if (curDays.nonEmpty && curBytes + sz > targetChunkBytes) {
        chunks += ChunkSpec(chunks.length, curDays.toArray, Some(curMin), Some(curMax), hasNull = false)
        curDays.clear()
        curBytes = 0L
        curMin = Long.MaxValue
        curMax = Long.MinValue
      }
      curDays += ((Some(dVal), fs))
      curBytes += sz
      if (dVal < curMin) curMin = dVal
      if (dVal > curMax) curMax = dVal
    }
    if (curDays.nonEmpty) {
      chunks += ChunkSpec(chunks.length, curDays.toArray, Some(curMin), Some(curMax), hasNull = false)
    }
    if (nullFiles.nonEmpty) {
      // Split NULL partition into <= targetChunkBytes chunks as well
      val curNull = new ArrayBuffer[File]()
      var curNBytes = 0L
      nullFiles.foreach { f =>
        if (curNull.nonEmpty && curNBytes + f.length() > targetChunkBytes) {
          chunks += ChunkSpec(chunks.length, Array((None, curNull.toArray)), None, None, hasNull = true)
          curNull.clear()
          curNBytes = 0L
        }
        curNull += f
        curNBytes += f.length()
      }
      if (curNull.nonEmpty) {
        chunks += ChunkSpec(chunks.length, Array((None, curNull.toArray)), None, None, hasNull = true)
      }
    }

    val readOpts = ParquetOptions.builder().includeColumn(dataColNames: _*).build()
    val writeOpts = SchemaUtils
      .writerOptionsFromSchema(ParquetWriterOptions.builder(), fullSchema, nullable = false)
      .withCompressionType(CompressionType.SNAPPY)
      .withRowGroupSizeBytes(128L * 1024L * 1024L)
      .withStatisticsFrequency(ParquetWriterOptions.StatisticsFrequency.ROWGROUP)
      .build()

    def makePartCol(optVal: Option[Long], nRows: Int): ColumnVector = {
      val s = optVal match {
        case None => Scalar.fromNull(GpuColumnVector.getNonNestedRapidsType(partColType))
        case Some(v) => partColType match {
          case IntegerType => Scalar.fromInt(v.toInt)
          case LongType => Scalar.fromLong(v)
          case _ => Scalar.fromInt(v.toInt)
        }
      }
      withResource(s)(sc => ColumnVector.fromScalar(sc, nRows))
    }

    def attachPartCol(dataTbl: CudfTable, optVal: Option[Long]): CudfTable = {
      val nRows = dataTbl.getRowCount.toInt
      val cols = new Array[ColumnVector](fullColNames.length)
      try {
        var dIdx = 0
        var i = 0
        while (i < fullColNames.length) {
          if (i == partColIdx) {
            cols(i) = makePartCol(optVal, nRows)
          } else {
            cols(i) = dataTbl.getColumn(dIdx).incRefCount()
            dIdx += 1
          }
          i += 1
        }
        new CudfTable(cols: _*)
      } finally {
        cols.foreach(c => if (c != null) c.close())
      }
    }

    val totalVxBytes = new java.util.concurrent.atomic.AtomicLong(0L)
    val pool = java.util.concurrent.Executors.newFixedThreadPool(numThreads)
    try {
      val futures = chunks.map { ch =>
        pool.submit(new Runnable {
          override def run(): Unit = {
            val pqFile = new File(pqOutDir, f"part-${ch.chunkIdx}%05d.parquet")
            val vxFile = new File(vxOutDir, f"part-${ch.chunkIdx}%05d.vortex")
            if (!pqFile.exists() || pqFile.length() == 0L || !vxFile.exists() || vxFile.length() == 0L) {
              val dayTables = new ArrayBuffer[CudfTable]()
              try {
                ch.days.foreach { case (optVal, files) =>
                  files.foreach { f =>
                    withResource(CudfTable.readParquet(readOpts, f)) { dt =>
                      if (dt.getRowCount > 0) {
                        dayTables += attachPartCol(dt, optVal)
                      }
                    }
                  }
                }
                if (dayTables.nonEmpty) {
                  val merged = if (dayTables.length == 1) {
                    dayTables.remove(0)
                  } else {
                    val arr = dayTables.toArray
                    dayTables.clear()
                    withResource(arr)(ts => CudfTable.concatenate(ts: _*))
                  }
                  withResource(merged) { tbl =>
                    if (!pqFile.exists() || pqFile.length() == 0L) {
                      withResource(CudfTable.writeParquetChunked(writeOpts, pqFile)) { w =>
                        val totalRows = tbl.getRowCount
                        val maxChunkRows = 16000000L
                        if (totalRows <= maxChunkRows) {
                          w.write(tbl)
                        } else {
                          val splitPts = (maxChunkRows until totalRows by maxChunkRows).map(_.toInt).toArray
                          withResource(tbl.contiguousSplit(splitPts: _*)) { parts =>
                            parts.foreach(ct => w.write(ct.getTable))
                          }
                        }
                      }
                    }
                    if (!vxFile.exists() || vxFile.length() == 0L) {
                      VortexNative.writeTable(tbl, fullColNames, vxFile.getAbsolutePath)
                    }
                  }
                }
              } finally {
                dayTables.safeClose()
              }
            }
            if (vxFile.exists()) {
              totalVxBytes.addAndGet(vxFile.length())
            }
          }
        })
      }
      futures.foreach(_.get())
    } finally {
      pool.shutdown()
    }

    // Write _vortex_zonemap.tsv manifest for Spark 3.5 SupportsRuntimeFiltering DPP
    val sb = new StringBuilder()
    sb.append(s"#COL=$partCol\t${partColType.simpleString}\n")
    chunks.foreach { ch =>
      val fn = f"part-${ch.chunkIdx}%05d.vortex"
      val mn = ch.minVal.map(_.toString).getOrElse("NULL")
      val mx = ch.maxVal.map(_.toString).getOrElse("NULL")
      val hn = if (ch.hasNull) "1" else "0"
      sb.append(s"$fn\t$mn\t$mx\t$hn\n")
    }
    Files.write(
      Paths.get(outVxDir, VortexSchemaIO.ZONEMAP_FILE),
      sb.toString.getBytes(StandardCharsets.UTF_8))

    java.lang.Long.valueOf(totalVxBytes.get())
  }
}
