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

import java.net.URI

import scala.collection.mutable.ArrayBuffer

import ai.rapids.cudf.{ColumnVector, Scalar}
import com.nvidia.spark.VortexFileFormat
import com.nvidia.spark.rapids._
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.GpuMetric._
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.jni.VortexNative

import org.apache.hadoop.conf.Configuration
import org.apache.spark.TaskContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.read.{InputPartition, PartitionReader, PartitionReaderFactory}
import org.apache.spark.sql.execution.FileSourceScanExec
import org.apache.spark.sql.execution.datasources.{FilePartition, PartitionedFile}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids.GpuFileSourceScanExec
import org.apache.spark.sql.sources.Filter
import org.apache.spark.sql.types._
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.util.SerializableConfiguration

/**
 * GPU V1 FileFormat reader for Standard Vortex (.vortex) files.
 *
 * Uses one single VortexNative.MultiFileScan per Spark FilePartition (1 reused CudaExecutionCtx,
 * 16-worker Tokio off-GpuSemaphore host prefetch, Rust VortexFile footer cache, true zero-copy
 * ArrowDeviceArray -> cuDF import, and O(1) getLastFileIndex() zero-I/O partition synthesis).
 */
class GpuReadVortexFileFormat extends VortexFileFormat with GpuReadFileFormatWithMetrics {
  override def buildReaderWithPartitionValuesAndMetrics(
      sparkSession: SparkSession,
      dataSchema: StructType,
      partitionSchema: StructType,
      requiredSchema: StructType,
      filters: Seq[Filter],
      options: Map[String, String],
      hadoopConf: Configuration,
      metrics: Map[String, GpuMetric]): PartitionedFile => Iterator[InternalRow] = {
    throw new UnsupportedOperationException("GpuReadVortexFileFormat uses multi-file reader only")
  }

  override def isPerFileReadEnabled(conf: RapidsConf): Boolean = false

  override def createMultiFileReaderFactory(
      broadcastedConf: Broadcast[SerializableConfiguration],
      pushedFilters: Array[Filter],
      fileScan: GpuFileSourceScanExec): PartitionReaderFactory = {
    val coalesceTargetBytes = SQLConf.get
      .getConfString("spark.rapids.sql.vortex.coalesceTargetBytes", "268435456").toLong
    val batchRows = SQLConf.get
      .getConfString("spark.rapids.sql.vortex.batchRows", "0").toLong
    val fullSchemaBytes = math.max(
      fileScan.relation.dataSchema.defaultSize + fileScan.relation.partitionSchema.defaultSize, 1)
    val fallbackCountCol =
      if (fileScan.relation.dataSchema.nonEmpty) fileScan.relation.dataSchema.fields(0).name else ""
    GpuVortexMultiFilePartitionReaderFactory(
      fileScan.requiredSchema,
      fileScan.readPartitionSchema,
      fullSchemaBytes,
      fallbackCountCol,
      coalesceTargetBytes,
      batchRows,
      fileScan.allMetrics)
  }
}

object GpuReadVortexFileFormat {
  def tagSupport(meta: SparkPlanMeta[FileSourceScanExec]): Unit = {
    val fsse = meta.wrapped
    val allTypes = (fsse.requiredSchema.fields ++ fsse.relation.partitionSchema.fields).map(_.dataType)
    val unsupported = allTypes.find {
      case BooleanType | ByteType | ShortType | IntegerType | LongType |
           FloatType | DoubleType | DateType | TimestampType | StringType |
           _: DecimalType => false
      case _ => true
    }
    unsupported.foreach(dt => meta.willNotWorkOnGpu(s"Unsupported Vortex column type: $dt"))
  }
}

case class GpuVortexMultiFilePartitionReaderFactory(
    readDataSchema: StructType,
    readPartitionSchema: StructType,
    fullSchemaBytes: Int,
    fallbackCountCol: String,
    coalesceTargetBytes: Long,
    batchRows: Long,
    metrics: Map[String, GpuMetric]) extends PartitionReaderFactory {

  override def supportColumnarReads(partition: InputPartition): Boolean = true

  override def createReader(partition: InputPartition): PartitionReader[InternalRow] =
    throw new UnsupportedOperationException("GpuReadVortexFileFormat supports columnar reads only")

  override def createColumnarReader(partition: InputPartition): PartitionReader[ColumnarBatch] = {
    val fp = partition.asInstanceOf[FilePartition]
    new GpuVortexFilePartitionReader(
      fp.files,
      readDataSchema,
      readPartitionSchema,
      fullSchemaBytes,
      fallbackCountCol,
      coalesceTargetBytes,
      batchRows,
      metrics)
  }
}

class GpuVortexFilePartitionReader(
    partFiles: Array[PartitionedFile],
    readDataSchema: StructType,
    readPartitionSchema: StructType,
    fullSchemaBytes: Int,
    fallbackCountCol: String,
    coalesceTargetBytes: Long,
    batchRows: Long,
    metrics: Map[String, GpuMetric]) extends PartitionReader[ColumnarBatch] {

  private var currentBatch: ColumnarBatch = _
  private val dataColNames: Array[String] = readDataSchema.fieldNames
  private val dataTypes: Array[DataType] = readDataSchema.fields.map(_.dataType)
  private val partTypes: Array[DataType] = readPartitionSchema.fields.map(_.dataType)
  private val outTypes: Array[DataType] = dataTypes ++ partTypes

  private val scanCols: Array[String] =
    if (dataColNames.nonEmpty) dataColNames
    else if (fallbackCountCol.nonEmpty) Array(fallbackCountCol)
    else Array.empty[String]

  private val localPaths: Array[String] = partFiles.map { pf =>
    val s = pf.filePath.toString
    if (s.startsWith("file:")) new URI(s).getPath else s
  }
  private val avgFileBytes: Long =
    if (partFiles.nonEmpty) partFiles.map(_.length).sum / partFiles.length else 0L
  private val avgProjBytes: Long =
    (avgFileBytes.toDouble * math.max(readDataSchema.defaultSize, 1) / math.max(fullSchemaBytes, 1)).toLong
  private val zeroCopyFastPath: Boolean =
    outTypes.nonEmpty && (partFiles.length <= 2 || avgProjBytes >= (24L << 20))
  private val targetCoalesceBytes: Long =
    if (coalesceTargetBytes > 0L) coalesceTargetBytes else 128L * 1024L * 1024L

  private val opTime = metrics.getOrElse(OP_TIME_NEW, metrics.getOrElse(OP_TIME_LEGACY, NoopMetric))
  private val gpuDecodeTime = metrics.getOrElse(GPU_DECODE_TIME, NoopMetric)
  private val numOutputBatches = metrics.getOrElse(NUM_OUTPUT_BATCHES, NoopMetric)
  private val outputBytes = metrics.getOrElse(GPU_OUTPUT_BATCH_BYTES, NoopMetric)

  private var activeScan: VortexNative.MultiFileScan = _
  private var eof = false

  private def makePartScalar(row: InternalRow, idx: Int, dt: DataType): Scalar = {
    if (row == null || row.isNullAt(idx)) {
      Scalar.fromNull(GpuColumnVector.getNonNestedRapidsType(dt))
    } else dt match {
      case IntegerType => Scalar.fromInt(row.getInt(idx))
      case LongType => Scalar.fromLong(row.getLong(idx))
      case ShortType => Scalar.fromShort(row.getShort(idx))
      case ByteType => Scalar.fromByte(row.getByte(idx))
      case BooleanType => Scalar.fromBool(row.getBoolean(idx))
      case FloatType => Scalar.fromFloat(row.getFloat(idx))
      case DoubleType => Scalar.fromDouble(row.getDouble(idx))
      case DateType => Scalar.timestampDaysFromInt(row.getInt(idx))
      case StringType => Scalar.fromString(row.getUTF8String(idx).toString)
      case other => throw new UnsupportedOperationException(s"Unsupported Vortex partition col dtype: $other")
    }
  }

  private def buildBatchFromZeroCopy(cvs: Array[ColumnVector], pRow: InternalRow): ColumnarBatch = {
    val nRows = if (cvs.nonEmpty) cvs(0).getRowCount.toInt else 0
    val nData = dataTypes.length
    val nPart = partTypes.length
    val gpuCols = new Array[org.apache.spark.sql.vectorized.ColumnVector](nData + nPart)
    var ok = false
    try {
      var i = 0
      while (i < nData) {
        gpuCols(i) = GpuColumnVector.from(cvs(i), dataTypes(i))
        cvs(i) = null
        i += 1
      }
      var p = 0
      while (p < nPart) {
        withResource(makePartScalar(pRow, p, partTypes(p))) { s =>
          gpuCols(nData + p) = GpuColumnVector.from(ColumnVector.fromScalar(s, nRows), partTypes(p))
        }
        p += 1
      }
      val batch = new ColumnarBatch(gpuCols, nRows)
      ok = true
      batch
    } finally {
      cvs.foreach(c => if (c != null) c.close())
      if (!ok) {
        gpuCols.foreach(c => if (c != null) c.close())
      }
    }
  }

  override def next(): Boolean = {
    if (currentBatch != null) {
      currentBatch.close()
      currentBatch = null
    }
    if (eof || localPaths.isEmpty || scanCols.isEmpty) return false

    // Start pure-CPU off-GpuSemaphore Tokio host prefetch BEFORE acquiring GpuSemaphore!
    // Negative prefetch requests VX_CUDA_SCAN_FLAG_DECODE_DICT | VX_CUDA_SCAN_FLAG_LAZY_NULL_COUNT
    // (fused GPU dict decode + zero per-file DtoH null_count stalls on batched multi-file decode).
    if (activeScan == null) {
      activeScan = VortexNative.openMultiScan(localPaths, scanCols, batchRows, -3)
    }

    GpuSemaphore.acquireIfNecessary(TaskContext.get())
    opTime.ns {
      if (zeroCopyFastPath) {
        while (true) {
          val cvs = gpuDecodeTime.ns {
            activeScan.nextZeroCopyBatch(null)
          }
          if (cvs == null) {
            eof = true
            activeScan.safeClose()
            activeScan = null
            return false
          }
          val nr = if (cvs.nonEmpty) cvs(0).getRowCount else 0L
          if (nr <= 0L) {
            cvs.safeClose()
          } else {
            val fIdx = math.min(math.max(activeScan.getLastFileIndex(), 0), partFiles.length - 1)
            currentBatch = buildBatchFromZeroCopy(cvs, partFiles(fIdx).partitionValues)
            numOutputBatches += 1
            outputBytes += GpuColumnVector.getTotalDeviceMemoryUsed(currentBatch)
            return true
          }
        }
      }

      // Path A2: Batched Multi-File GPU Decode — pops up to 64 small daily files in ONE JNI call,
      // concatenates data columns in ONE fused GPU kernel (with exact __popc null_count & async
      // zero-sync input release), and synthesizes any requested partition columns from RLE runs!
      val bt = gpuDecodeTime.ns {
        activeScan.nextBatchedMultiFile(64, 32000000L, targetCoalesceBytes)
      }
      if (bt == null) {
        eof = true
        activeScan.safeClose()
        activeScan = null
        return false
      }
      withResource(bt.table) { dataTbl =>
        val totalRows = dataTbl.getRowCount.toInt
        if (totalRows <= 0) return next()
        if (outTypes.isEmpty) {
          currentBatch = new ColumnarBatch(Array.empty, totalRows)
          numOutputBatches += 1
          return true
        }
        if (partTypes.isEmpty) {
          currentBatch = GpuColumnVector.from(dataTbl, dataTypes)
          numOutputBatches += 1
          outputBytes += GpuColumnVector.getTotalDeviceMemoryUsed(currentBatch)
          return true
        }
        val nData = dataTypes.length
        val nPart = partTypes.length
        val gpuCols = new Array[org.apache.spark.sql.vectorized.ColumnVector](nData + nPart)
        var ok = false
        try {
          var i = 0
          while (i < nData) {
            gpuCols(i) = GpuColumnVector.from(dataTbl.getColumn(i).incRefCount(), dataTypes(i))
            i += 1
          }
          var p = 0
          while (p < nPart) {
            val pCol = buildPartitionColFromRuns(p, partTypes(p), bt.fileIndices, bt.rowCounts)
            gpuCols(nData + p) = GpuColumnVector.from(pCol, partTypes(p))
            p += 1
          }
          currentBatch = new ColumnarBatch(gpuCols, totalRows)
          numOutputBatches += 1
          outputBytes += GpuColumnVector.getTotalDeviceMemoryUsed(currentBatch)
          ok = true
          true
        } finally {
          if (!ok) {
            gpuCols.foreach(c => if (c != null) c.close())
          }
        }
      }
    }
  }

  private def buildPartitionColFromRuns(
      partIdx: Int,
      dt: DataType,
      fileIndices: Array[Int],
      rowCounts: Array[Int]): ColumnVector = {
    if (fileIndices.length == 1) {
      val fIdx = math.min(math.max(fileIndices(0), 0), partFiles.length - 1)
      withResource(makePartScalar(partFiles(fIdx).partitionValues, partIdx, dt)) { s =>
        return ColumnVector.fromScalar(s, rowCounts(0))
      }
    }
    val segs = new ArrayBuffer[ColumnVector](fileIndices.length)
    try {
      var i = 0
      while (i < fileIndices.length) {
        val fIdx = math.min(math.max(fileIndices(i), 0), partFiles.length - 1)
        withResource(makePartScalar(partFiles(fIdx).partitionValues, partIdx, dt)) { s =>
          segs += ColumnVector.fromScalar(s, rowCounts(i))
        }
        i += 1
      }
      ColumnVector.concatenate(segs.toArray: _*)
    } finally {
      segs.safeClose()
    }
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
