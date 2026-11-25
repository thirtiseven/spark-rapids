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

import ai.rapids.cudf.{ColumnVector, DType, ParquetOptions, Scalar, Table => CudfTable}
import com.nvidia.spark.rapids._
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.GpuMetric._
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.jni.{LencNative, VortexNative}

import org.apache.spark.TaskContext
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.catalog.{SupportsRead, Table, TableCapability, TableProvider}
import org.apache.spark.sql.connector.expressions.Transform
import org.apache.spark.sql.connector.read._
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.sources._
import org.apache.spark.sql.types._
import org.apache.spark.sql.util.CaseInsensitiveStringMap
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * Tier 1+ Standard Vortex GPU DataSource V2:
 *   - Official vortex-cuda-ffi -> ArrowDeviceArray -> cudf::from_arrow_device zero-copy GPU import.
 *   - Opt V1: Single-pass zero-double-copy multi-file coalescing paced cleanly by GpuSemaphore.
 *   - Opt V2: SupportsPushDownFilters + Adaptive GPU 2-Phase Late Materialization via
 *     vx_cuda_scan_path_device_batches_with_rows (skips decoding unselected payload rows on
 *     selective queries like Q06/Q10/Q12/Q14/Q19, and auto-bypasses on high-selectivity scans like Q01!).
 */
object VortexSchemaIO {
  val SCHEMA_FILE = "_vortex_schema.json"

  def stripFileScheme(path: String): String = {
    if (path.startsWith("file://")) path.stripPrefix("file://")
    else if (path.startsWith("file:")) path.stripPrefix("file:")
    else path
  }

  def readSchema(dirPath: String): StructType = {
    val clean = stripFileScheme(dirPath)
    val f = new File(clean, SCHEMA_FILE)
    require(f.exists(), s"Missing $SCHEMA_FILE in $clean")
    val json = new String(Files.readAllBytes(f.toPath), StandardCharsets.UTF_8)
    DataType.fromJson(json).asInstanceOf[StructType]
  }

  def listVortexFilesWithSizes(dirPath: String): (Array[String], Array[Long]) = {
    val clean = stripFileScheme(dirPath)
    val dir = new File(clean)
    require(dir.isDirectory, s"Not a directory: $clean")
    val arr = dir.listFiles()
      .filter(f => f.isFile && f.getName.endsWith(".vortex"))
      .sortBy(_.getAbsolutePath)
    (arr.map(_.getAbsolutePath), arr.map(_.length()))
  }
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
    val fullSchema = VortexSchemaIO.readSchema(path)
    val (files, fileSizes) = VortexSchemaIO.listVortexFilesWithSizes(path)
    new VortexTable(path, fullSchema, schema, files, fileSizes)
  }
}

class VortexTable(
    path: String,
    fullSchema: StructType,
    exposedSchema: StructType,
    files: Array[String],
    fileSizes: Array[Long]) extends Table with SupportsRead {

  override def name(): String = s"vortex-gpu($path)"

  override def schema(): StructType = exposedSchema

  override def capabilities(): util.Set[TableCapability] =
    util.EnumSet.of(TableCapability.BATCH_READ)

  override def newScanBuilder(options: CaseInsensitiveStringMap): ScanBuilder =
    new VortexScanBuilder(fullSchema, exposedSchema, files, fileSizes)
}

class VortexScanBuilder(
    fullSchema: StructType,
    exposedSchema: StructType,
    files: Array[String],
    fileSizes: Array[Long])
  extends ScanBuilder with SupportsPushDownRequiredColumns with SupportsPushDownFilters {

  private var requiredSchema: StructType = exposedSchema
  private var collectedFilters: Array[Filter] = Array.empty

  override def pruneColumns(requiredSchema: StructType): Unit = {
    this.requiredSchema = requiredSchema
  }

  override def pushFilters(filters: Array[Filter]): Array[Filter] = {
    // Capture pushable filters for GPU 2-Phase Late Materialization inside GpuVortexScan;
    // return all filters as post-scan residuals so Spark keeps GpuFilterExec/GenericJitReduce
    // above the scan for 100% SQL parity and compatibility!
    collectedFilters = filters.filter(f => VortexGpuFilterEval.isPushable(f, fullSchema))
    filters
  }

  override def pushedFilters(): Array[Filter] = Array.empty

  override def build(): Scan = {
    // Keep only filters whose referenced columns are actually present in requiredSchema
    val reqNames = requiredSchema.fieldNames.toSet
    val applicable = collectedFilters.filter { f =>
      val refs = VortexGpuFilterEval.referencedCols(f)
      refs.nonEmpty && refs.subsetOf(reqNames)
    }
    VortexScan(fullSchema, requiredSchema, files, fileSizes, applicable)
  }
}

case class VortexInputPartition(filePaths: Array[String]) extends InputPartition

case class VortexScan(
    fullSchema: StructType,
    requiredSchema: StructType,
    files: Array[String],
    fileSizes: Array[Long],
    lateMatFilters: Array[Filter] = Array.empty) extends Scan with Batch with SupportsReportStatistics {

  override def readSchema(): StructType = requiredSchema

  override def description(): String =
    s"VortexScan(files=${files.length}, cols=${requiredSchema.length}, lateMatFilters=${lateMatFilters.length})"

  override def toBatch: Batch = this

  override def estimateStatistics(): Statistics = new Statistics {
    private val sz = math.max(fileSizes.sum, 1L)
    override def sizeInBytes(): java.util.OptionalLong = java.util.OptionalLong.of(sz)
    override def numRows(): java.util.OptionalLong = java.util.OptionalLong.empty()
  }

  override def planInputPartitions(): Array[InputPartition] =
    files.map(f => VortexInputPartition(Array(f)))

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
    // Coalesce small tables (<= 16 MiB, e.g. nation 64 KiB / region 32 KiB) into 1 single task
    // so they don't spawn 13 tiny tasks waiting on GpuSemaphore!
    if (totalBytes <= 16L * 1024L * 1024L) {
      return Array(files)
    }
    val maxPartBytes = SQLConf.get
      .getConfString("spark.rapids.sql.vortex.maxPartitionBytes", "1073741824").toLong
    if (files.length <= 16 || maxPartBytes <= 0L) {
      return files.map(Array(_))
    }
    val maxFilesPerPart = 64
    val groups = new ArrayBuffer[Array[String]]()
    val cur = new ArrayBuffer[String]()
    var curBytes = 0L
    files.indices.foreach { idx =>
      val f = files(idx)
      val fsz = if (idx < fileSizes.length) fileSizes(idx) else new File(f).length()
      if (cur.nonEmpty && (curBytes + fsz > maxPartBytes || cur.size >= maxFilesPerPart)) {
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
}

case class GpuVortexScan(
    fullSchema: StructType,
    requiredSchema: StructType,
    files: Array[String],
    fileSizes: Array[Long],
    lateMatFilters: Array[Filter] = Array.empty) extends GpuScan with Batch with SupportsReportStatistics {

  override def readSchema(): StructType = requiredSchema

  override def description(): String =
    s"GpuVortexScan(files=${files.length}, cols=${requiredSchema.length}, lateMatFilters=${lateMatFilters.length})"

  override def withInputFile(): GpuScan = this

  override def toBatch: Batch = this

  override def estimateStatistics(): Statistics = new Statistics {
    private val sz = math.max(fileSizes.sum, 1L)
    override def sizeInBytes(): java.util.OptionalLong = java.util.OptionalLong.of(sz)
    override def numRows(): java.util.OptionalLong = java.util.OptionalLong.empty()
  }

  override def planInputPartitions(): Array[InputPartition] =
    VortexPartitionHelper.groupFiles(files, fileSizes).map(VortexInputPartition)

  override def createReaderFactory(): PartitionReaderFactory = {
    val colNames = requiredSchema.fieldNames
    val dict32Set = GpuDictionaryUtils.updateDriverDict32Cols(SQLConf.get)
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
    GpuVortexPartitionReaderFactory(
      requiredSchema, colNames, dict32Mask, coalesceTargetBytes, batchRows, activeFilters, metrics)
  }
}

case class GpuVortexPartitionReaderFactory(
    readSchema: StructType,
    colNames: Array[String],
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
    new GpuVortexPartitionReader(
      vp.filePaths, readSchema, colNames, dict32Mask, coalesceTargetBytes, batchRows,
      lateMatFilters, metrics)
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
  private val payloadHasStrings: Boolean = payloadColIndices.exists(i => readTypes(i) == StringType)
  private val payloadDictMask: Array[Byte] =
    if (dict32Mask == null) null else payloadColIndices.map(dict32Mask)

  // Only enable 2-phase late materialization when payload actually has heavy STRING columns
  // (or user explicitly enables it for all types) since vortex-cuda decodes numeric CudaFlat blocks
  // fast enough that opening a file twice only pays off when skipping heavy strings or 0-match files!
  private val lateMatForceAll: Boolean = SQLConf.get
    .getConfString("spark.rapids.sql.vortex.lateMatAllTypes", "false").toBoolean
  useLateMat = useLateMat && (payloadHasStrings || lateMatForceAll)

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
        // Adaptive selectivity gate: if > 20% pass, fall back to direct 1-pass multi-file scan
        // for all remaining files (zero wasted rowIndices D2H copy!)
        if (nPass * 5L > nRows) {
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
        // Phase 2: Low selectivity (<= 20%) -> Vortex takes ONLY surviving rows of heavy payload columns!
        val rowIndices = VortexNative.maskToRowIndices(mask)
        val pTbl = gpuDecodeTime.ns {
          VortexNative.scanFileWithRows(path, payloadColNames, rowIndices, payloadDictMask)
        }
        if (pTbl == null) return None
        withResource(pTbl) { pt =>
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

    // Acquire GpuSemaphore BEFORE opTime.ns so opTime reflects actual scan work, not semaphore wait!
    GpuSemaphore.acquireIfNecessary(TaskContext.get())
    opTime.ns {
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
            multiScan = VortexNative.openMultiScan(rem, colNames, batchRows, 0)
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
      if (zeroCopyEnabled) {
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

  override def tagSelfForGpu(): Unit = {}

  override def convertToGpu(): GpuScan =
    GpuVortexScan(a.fullSchema, a.requiredSchema, a.files, a.fileSizes, a.lateMatFilters)
}

/** Converts a directory of Parquet files directly to standard CUDA-optimized .vortex files! */
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
    val parquetDir = VortexSchemaIO.stripFileScheme(args(0))
    val vortexDir = VortexSchemaIO.stripFileScheme(args(1))
    val schemaJson = args(2)
    val includeCols: Array[String] =
      if (args.length >= 4 && args(3).nonEmpty) args(3).split(",") else Array.empty
    val numThreads: Int =
      if (args.length >= 5 && args(4).nonEmpty) args(4).toInt else 4

    val schema = DataType.fromJson(schemaJson).asInstanceOf[StructType]
    val colNames = if (includeCols.nonEmpty) includeCols else schema.fieldNames

    val outDir = new File(vortexDir)
    outDir.mkdirs()
    Files.write(
      Paths.get(vortexDir, VortexSchemaIO.SCHEMA_FILE),
      schemaJson.getBytes(StandardCharsets.UTF_8))

    val pqFiles = collectParquetFiles(new File(parquetDir))
    val optsBuilder = ParquetOptions.builder()
    if (includeCols.nonEmpty) optsBuilder.includeColumn(includeCols: _*)
    val opts = optsBuilder.build()

    val totalBytes = new java.util.concurrent.atomic.AtomicLong(0L)
    val pool = java.util.concurrent.Executors.newFixedThreadPool(math.max(1, numThreads))
    try {
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
    } finally {
      pool.shutdown()
    }
    java.lang.Long.valueOf(totalBytes.get())
  }
}
