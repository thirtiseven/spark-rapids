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
import java.util

import scala.collection.mutable.ArrayBuffer

import ai.rapids.cudf.{ParquetOptions, Table => CudfTable}
import com.nvidia.spark.rapids._
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.GpuMetric._
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.jni.VortexNative

import org.apache.spark.TaskContext
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.catalog.{SupportsRead, Table, TableCapability, TableProvider}
import org.apache.spark.sql.connector.expressions.Transform
import org.apache.spark.sql.connector.read._
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types._
import org.apache.spark.sql.util.CaseInsensitiveStringMap
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * Tier 1 Standard Vortex GPU DataSource V2 implementation.
 * Coexists cleanly alongside Parquet and .lenc (which continues to provide GPU-native Shuffle):
 *   - Uses official vortex-cuda-ffi (vx_cuda_scan_path_device_batches) -> ArrowDeviceArray ->
 *     cudf::from_arrow_device zero-copy GPU import!
 *   - Implements SupportsPushDownRequiredColumns + SupportsReportStatistics (optimal Catalyst joins).
 *   - Preserves DICTIONARY32 for AutoDict32 / Dict-Aware AST-JIT and coalesces small file batches
 *     on GPU up to spark.rapids.sql.vortex.coalesceTargetBytes (Opt D).
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
  extends ScanBuilder with SupportsPushDownRequiredColumns {

  private var requiredSchema: StructType = exposedSchema

  override def pruneColumns(requiredSchema: StructType): Unit = {
    this.requiredSchema = requiredSchema
  }

  override def build(): Scan = {
    VortexScan(fullSchema, requiredSchema, files, fileSizes)
  }
}

case class VortexInputPartition(filePaths: Array[String]) extends InputPartition

case class VortexScan(
    fullSchema: StructType,
    requiredSchema: StructType,
    files: Array[String],
    fileSizes: Array[Long]) extends Scan with Batch with SupportsReportStatistics {

  override def readSchema(): StructType = requiredSchema

  override def description(): String =
    s"VortexScan(files=${files.length}, cols=${requiredSchema.length})"

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
    fileSizes: Array[Long]) extends GpuScan with Batch with SupportsReportStatistics {

  override def readSchema(): StructType = requiredSchema

  override def description(): String =
    s"GpuVortexScan(files=${files.length}, cols=${requiredSchema.length})"

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
    GpuVortexPartitionReaderFactory(
      requiredSchema, colNames, dict32Mask, coalesceTargetBytes, batchRows, metrics)
  }
}

case class GpuVortexPartitionReaderFactory(
    readSchema: StructType,
    colNames: Array[String],
    dict32Mask: Array[Byte],
    coalesceTargetBytes: Long,
    batchRows: Long,
    metrics: Map[String, GpuMetric]) extends PartitionReaderFactory {

  override def supportColumnarReads(partition: InputPartition): Boolean = true

  override def createReader(partition: InputPartition): PartitionReader[InternalRow] =
    throw new UnsupportedOperationException("GpuVortexScan supports columnar reads only")

  override def createColumnarReader(partition: InputPartition): PartitionReader[ColumnarBatch] = {
    val vp = partition.asInstanceOf[VortexInputPartition]
    new GpuVortexPartitionReader(
      vp.filePaths, readSchema, colNames, dict32Mask, coalesceTargetBytes, batchRows, metrics)
  }
}

class GpuVortexPartitionReader(
    filePaths: Array[String],
    readSchema: StructType,
    colNames: Array[String],
    dict32Mask: Array[Byte],
    coalesceTargetBytes: Long,
    batchRows: Long,
    metrics: Map[String, GpuMetric]) extends PartitionReader[ColumnarBatch] {

  private var currentBatch: ColumnarBatch = _
  private val readTypes: Array[DataType] = readSchema.fields.map(_.dataType)
  private var fileIdx = 0
  private var activeScan: VortexNative.DeviceScan = _

  private val opTime = metrics.getOrElse(OP_TIME_LEGACY, NoopMetric)
  private val gpuDecodeTime = metrics.getOrElse(GPU_DECODE_TIME, NoopMetric)
  private val numOutputBatches = metrics.getOrElse(NUM_OUTPUT_BATCHES, NoopMetric)

  private def nextRawTable(): Option[CudfTable] = {
    while (true) {
      if (activeScan != null) {
        val tbl = gpuDecodeTime.ns { activeScan.nextBatch(dict32Mask) }
        if (tbl != null) {
          return Some(tbl)
        }
        activeScan.safeClose()
        activeScan = null
      }
      if (fileIdx >= filePaths.length) {
        return None
      }
      val path = filePaths(fileIdx)
      fileIdx += 1
      activeScan = VortexNative.openScan(path, colNames, batchRows)
    }
    None
  }

  private def hasAnyDict32(t: CudfTable): Boolean = {
    var i = 0
    val nc = t.getNumberOfColumns
    while (i < nc) {
      if (t.getColumn(i).getType == ai.rapids.cudf.DType.DICTIONARY32) return true
      i += 1
    }
    false
  }

  override def next(): Boolean = {
    if (currentBatch != null) {
      currentBatch.close()
      currentBatch = null
    }
    if (readTypes.isEmpty) {
      // Zero-column projection (e.g. COUNT(*)): scan 1st col or count via full projection
      return false
    }
    opTime.ns {
      GpuSemaphore.acquireIfNecessary(TaskContext.get())
      val firstOpt = nextRawTable()
      if (firstOpt.isEmpty) {
        return false
      }
      val pending = new ArrayBuffer[CudfTable]()
      try {
        val first = firstOpt.get
        pending += first
        var accBytes = first.getDeviceMemorySize
        var accRows = first.getRowCount
        // Opt D: Coalesce small decoded Vortex tables on GPU up to coalesceTargetBytes (when no Dict32)
        val canCoalesce = coalesceTargetBytes > 0L && !hasAnyDict32(first)
        while (canCoalesce && accBytes < coalesceTargetBytes && accRows < 100000000L) {
          nextRawTable() match {
            case Some(t) =>
              if (hasAnyDict32(t)) {
                // Keep separate if dictionary appeared
                pending += t
                accBytes = coalesceTargetBytes
              } else {
                pending += t
                accBytes += t.getDeviceMemorySize
                accRows += t.getRowCount
              }
            case None =>
              accBytes = coalesceTargetBytes
          }
        }
        val finalTbl = if (pending.length == 1) {
          pending.remove(0)
        } else {
          // Concatenate tables cleanly; decode any unexpected dict32 first
          val all = pending.toArray
          pending.clear()
          withResource(all) { tbls =>
            CudfTable.concatenate(tbls: _*)
          }
        }
        withResource(finalTbl) { t =>
          currentBatch = GpuColumnVector.from(t, readTypes)
          numOutputBatches += 1
        }
      } finally {
        pending.safeClose()
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
    if (activeScan != null) {
      activeScan.safeClose()
      activeScan = null
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
    GpuVortexScan(a.fullSchema, a.requiredSchema, a.files, a.fileSizes)
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
