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

package com.nvidia.spark.rapids.lenc

import java.io.File
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}
import java.util

import ai.rapids.cudf.{HostMemoryBuffer, ParquetOptions, Table => CudfTable}
import com.nvidia.spark.rapids._
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.GpuMetric._
import com.nvidia.spark.rapids.jni.LencNative

import org.apache.spark.TaskContext
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.catalog.{SupportsRead, Table, TableCapability, TableProvider}
import org.apache.spark.sql.connector.expressions.Transform
import org.apache.spark.sql.connector.read._
import org.apache.spark.sql.types._
import org.apache.spark.sql.util.CaseInsensitiveStringMap
import org.apache.spark.sql.vectorized.ColumnarBatch

object LencSchemaIO {
  val SCHEMA_FILE = "_lenc_schema.json"

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

  def listLencFiles(dirPath: String): Array[String] = {
    val clean = stripFileScheme(dirPath)
    val dir = new File(clean)
    require(dir.isDirectory, s"Not a directory: $clean")
    dir.listFiles()
      .filter(f => f.isFile && f.getName.endsWith(".lenc"))
      .map(_.getAbsolutePath)
      .sorted
  }

  def fusedOutputSchema(fusedExpr: String): Option[StructType] = fusedExpr match {
    case "W1a_1" | "W1b_8" | "W1b_24" | "W1a_1_sum" =>
      Some(StructType(Seq(StructField("o0", LongType, nullable = false))))
    case "W1a_8" =>
      Some(StructType((0 until 8).map(k => StructField(s"o$k", LongType, nullable = false))))
    case _ => None
  }

  def fusedSpec(fusedExpr: String, fullSchema: StructType): (Int, Array[Int]) = {
    val nameToIdx = fullSchema.fieldNames.zipWithIndex.toMap
    fusedExpr match {
      case "W1a_1" =>
        (0, Array("c0", "c1", "c2", "c3").map(nameToIdx))
      case "W1a_8" =>
        (1, Array("c0", "c1", "c2", "c3", "c4").map(nameToIdx))
      case "W1b_8" =>
        (2, Array("c0", "c1", "c2", "c3", "c4", "s1").map(nameToIdx))
      case "W1b_24" =>
        (3, Array("c0", "c1", "c2", "c3", "c4", "s1").map(nameToIdx))
      case "W1a_1_sum" =>
        (4, Array("c0", "c1", "c2", "c3").map(nameToIdx))
      case other =>
        throw new IllegalArgumentException(s"Unsupported fusedExpr: $other")
    }
  }
}

class LencDataSourceImpl extends TableProvider {
  override def supportsExternalMetadata(): Boolean = false

  override def inferSchema(options: CaseInsensitiveStringMap): StructType = {
    val path = options.get("path")
    require(path != null && path.nonEmpty, "lenc datasource requires option 'path'")
    val fusedExpr = Option(options.get("fusedExpr")).filter(_.nonEmpty)
    fusedExpr.flatMap(LencSchemaIO.fusedOutputSchema)
      .getOrElse(LencSchemaIO.readSchema(path))
  }

  override def getTable(
      schema: StructType,
      partitioning: Array[Transform],
      properties: util.Map[String, String]): Table = {
    val path = properties.get("path")
    val fusedExpr = Option(properties.get("fusedExpr")).filter(_.nonEmpty)
    val fullSchema = LencSchemaIO.readSchema(path)
    val files = LencSchemaIO.listLencFiles(path)
    new LencTable(path, fullSchema, schema, files, fusedExpr)
  }
}

class LencTable(
    path: String,
    fullSchema: StructType,
    exposedSchema: StructType,
    files: Array[String],
    fusedExpr: Option[String]) extends Table with SupportsRead {

  override def name(): String = s"lenc($path)"

  override def schema(): StructType = exposedSchema

  override def capabilities(): util.Set[TableCapability] =
    util.EnumSet.of(TableCapability.BATCH_READ)

  override def newScanBuilder(options: CaseInsensitiveStringMap): ScanBuilder =
    new LencScanBuilder(fullSchema, exposedSchema, files, fusedExpr)
}

class LencScanBuilder(
    fullSchema: StructType,
    exposedSchema: StructType,
    files: Array[String],
    fusedExpr: Option[String])
  extends ScanBuilder with SupportsPushDownRequiredColumns {

  private var requiredSchema: StructType = exposedSchema

  override def pruneColumns(requiredSchema: StructType): Unit = {
    this.requiredSchema = requiredSchema
  }

  override def build(): Scan = {
    LencScan(fullSchema, requiredSchema, files, fusedExpr)
  }
}

case class LencScan(
    fullSchema: StructType,
    requiredSchema: StructType,
    files: Array[String],
    fusedExpr: Option[String]) extends Scan with Batch {
  override def readSchema(): StructType = requiredSchema
  override def description(): String =
    s"LencScan(files=${files.length}, fused=${fusedExpr.getOrElse("none")})"
  override def toBatch: Batch = this
  override def planInputPartitions(): Array[InputPartition] =
    files.map(f => LencInputPartition(Array(f)))
  override def createReaderFactory(): PartitionReaderFactory = new PartitionReaderFactory {
    override def createReader(partition: InputPartition): PartitionReader[InternalRow] =
      throw new UnsupportedOperationException("LencScan must run on GPU via GpuLencScan")
  }
}

case class LencInputPartition(filePaths: Array[String]) extends InputPartition

case class GpuLencScan(
    fullSchema: StructType,
    requiredSchema: StructType,
    files: Array[String],
    fusedExpr: Option[String]) extends GpuScan with Batch {

  override def readSchema(): StructType = requiredSchema

  override def description(): String =
    s"GpuLencScan(files=${files.length}, cols=${requiredSchema.length}, fused=${fusedExpr.getOrElse("none")})"

  override def withInputFile(): GpuScan = this

  override def toBatch: Batch = this

  override def planInputPartitions(): Array[InputPartition] = {
    // 1 file per partition (32 files -> 32 partitions, matching Spark Parquet partitions)
    files.map(f => LencInputPartition(Array(f)))
  }

  override def createReaderFactory(): PartitionReaderFactory = {
    val nameToIdx = fullSchema.fieldNames.zipWithIndex.toMap
    val (fusedKind, projectedIndices) = fusedExpr match {
      case Some(expr) =>
        val (kind, idxs) = LencSchemaIO.fusedSpec(expr, fullSchema)
        (kind, idxs)
      case None =>
        (-1, requiredSchema.fieldNames.map(nameToIdx))
    }
    GpuLencPartitionReaderFactory(
      requiredSchema,
      projectedIndices,
      fusedKind,
      metrics)
  }
}

case class GpuLencPartitionReaderFactory(
    readSchema: StructType,
    projectedIndices: Array[Int],
    fusedKind: Int,
    metrics: Map[String, GpuMetric]) extends PartitionReaderFactory {

  override def supportColumnarReads(partition: InputPartition): Boolean = true

  override def createReader(partition: InputPartition): PartitionReader[InternalRow] =
    throw new UnsupportedOperationException("GpuLencScan only supports columnar GPU reads")

  override def createColumnarReader(partition: InputPartition): PartitionReader[ColumnarBatch] = {
    val lp = partition.asInstanceOf[LencInputPartition]
    new GpuLencPartitionReader(lp.filePaths, readSchema, projectedIndices, fusedKind, metrics)
  }
}

class GpuLencPartitionReader(
    filePaths: Array[String],
    readSchema: StructType,
    projectedIndices: Array[Int],
    fusedKind: Int,
    metrics: Map[String, GpuMetric]) extends PartitionReader[ColumnarBatch] {

  private var idx: Int = 0
  private var currentBatch: ColumnarBatch = _
  private val readTypes: Array[DataType] = readSchema.fields.map(_.dataType)

  private val opTime = metrics.getOrElse(OP_TIME_LEGACY, NoopMetric)
  private val bufferTime = metrics.getOrElse(BUFFER_TIME, NoopMetric)
  private val gpuDecodeTime = metrics.getOrElse(GPU_DECODE_TIME, NoopMetric)
  private val numOutputBatches = metrics.getOrElse(NUM_OUTPUT_BATCHES, NoopMetric)

  override def next(): Boolean = {
    if (currentBatch != null) {
      currentBatch.close()
      currentBatch = null
    }
    if (idx >= filePaths.length) {
      return false
    }
    val path = filePaths(idx)
    idx += 1

    opTime.ns {
      withResource(LencNative.planRead(path, projectedIndices)) { plan =>
        val totalBytes = plan.totalPackedBytes()
        // Step 1: Read projected compressed column ranges into Spark's pinned host pool OFF semaphore
        val hostBuf = bufferTime.ns {
          val buf = HostMemoryBuffer.allocate(math.max(totalBytes, 64L))
          closeOnExcept(buf) { b =>
            if (totalBytes > 0L) {
              plan.readToHost(path, b)
            }
          }
          buf
        }
        // Step 2: Acquire GPU semaphore, H2D + GPU decode (or fused decode+eval) -> GpuColumnVector
        withResource(hostBuf) { hb =>
          GpuSemaphore.acquireIfNecessary(TaskContext.get())
          val table = gpuDecodeTime.ns {
            if (fusedKind >= 0) {
              plan.decodeAndEvalFused(hb, fusedKind, 0L)
            } else {
              plan.decodeFromHost(hb, null)
            }
          }
          withResource(table) { t =>
            currentBatch = GpuColumnVector.from(t, readTypes)
            numOutputBatches += 1
          }
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
  }
}

case class LencScanRuleMeta(
    a: LencScan,
    override val conf: RapidsConf,
    p: Option[RapidsMeta[_, _, _]],
    r: DataFromReplacementRule) extends ScanMeta[LencScan](a, conf, p, r) {

  override def tagSelfForGpu(): Unit = {}

  override def convertToGpu(): GpuScan =
    GpuLencScan(a.fullSchema, a.requiredSchema, a.files, a.fusedExpr)
}

/** Converts a directory of Parquet files directly to .lenc files + _lenc_schema.json on GPU. */
class LencConverterImpl extends java.util.function.Function[Array[String], java.lang.Long] {
  override def apply(args: Array[String]): java.lang.Long = {
    val parquetDir = LencSchemaIO.stripFileScheme(args(0))
    val lencDir = LencSchemaIO.stripFileScheme(args(1))
    val schemaJson = args(2)

    val outDir = new File(lencDir)
    outDir.mkdirs()
    Files.write(
      Paths.get(lencDir, LencSchemaIO.SCHEMA_FILE),
      schemaJson.getBytes(StandardCharsets.UTF_8))

    val pqFiles = new File(parquetDir).listFiles()
      .filter(f => f.isFile && f.getName.endsWith(".parquet"))
      .sortBy(_.getName)

    var totalBytes = 0L
    val opts = ParquetOptions.DEFAULT
    pqFiles.zipWithIndex.foreach { case (pqFile, i) =>
      val outPath = new File(outDir, f"part-$i%05d.lenc")
      withResource(CudfTable.readParquet(opts, pqFile)) { table =>
        LencNative.writeTable(table, outPath.getAbsolutePath)
      }
      totalBytes += outPath.length()
    }
    java.lang.Long.valueOf(totalBytes)
  }
}
