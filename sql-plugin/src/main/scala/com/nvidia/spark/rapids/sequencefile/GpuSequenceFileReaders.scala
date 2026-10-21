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

package com.nvidia.spark.rapids.sequencefile

import java.io.{DataOutputStream, EOFException, IOException, OutputStream}
import java.net.URI
import java.util
import java.util.concurrent.Future

import scala.collection.mutable

import ai.rapids.cudf.{ColumnVector, DType, HostColumnVector, HostColumnVectorCore,
  HostMemoryBuffer}
import com.nvidia.spark.rapids._
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.GpuMetric._
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.RmmRapidsRetryIterator.withRetryNoSplit
import com.nvidia.spark.rapids.io.async.{AsyncMetrics, AsyncResult, AsyncRunner, FastReleaseResult,
  MemoryBoundedAsyncRunner}
import com.nvidia.spark.rapids.jni.{RmmSpark, TaskPriority}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FSDataInputStream, FSInputStream, Path}
import org.apache.hadoop.io.{DataOutputBuffer, SequenceFile}

import org.apache.spark.TaskContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.connector.read.{PartitionReader, PartitionReaderFactory}
import org.apache.spark.sql.execution.datasources.PartitionedFile
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids.execution.TrampolineUtil
import org.apache.spark.sql.sources.Filter
import org.apache.spark.sql.types.BinaryType
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector => SparkVector}
import org.apache.spark.util.SerializableConfiguration

private[rapids] final class SequenceFileReadAhead[T] {
  private var armed = false
  private var ready = false
  private var hasNextChunk = false
  private var taken = false
  private var submit: () => Option[T] = null
  private var continuation: Option[T] = None

  def armSuccessor(submitSuccessor: () => Option[T]): Unit = synchronized {
    require(!armed, "SequenceFile successor was already prepared")
    armed = true
    if (!ready || hasNextChunk) {
      submit = submitSuccessor
      submitIfReady()
    }
  }

  def markReady(hasMore: Boolean): Unit = synchronized {
    require(!ready, "SequenceFile chunk was already published")
    ready = true
    hasNextChunk = hasMore
    if (hasMore) {
      submitIfReady()
    } else {
      submit = null
    }
  }

  def takeContinuation(): Option[T] = synchronized {
    require(ready, "SequenceFile chunk has not been published")
    require(!taken, "SequenceFile continuation was already consumed")
    taken = true
    val result = continuation
    continuation = None
    result
  }

  private def submitIfReady(): Unit = {
    if (ready && hasNextChunk && submit != null) {
      // Completion and consumer promotion can race; only one may submit the successor.
      val submitSuccessor = submit
      submit = null
      continuation = submitSuccessor()
    }
  }
}

private object SequenceFileReaderLimits {
  val MAX_BATCH_OUTPUT_BYTES: Long = 256L * 1024 * 1024
  val MAX_BATCH_ROWS: Int = 4 * 1024 * 1024
  val MAX_OFFSETS_FRACTION_DENOMINATOR: Int = 4
  val MIN_INITIAL_DATA_BYTES: Long = 64L * 1024
  val MIN_INITIAL_ROWS: Int = 1024
  val ESTIMATED_COMPRESSED_BYTES_PER_ROW: Long = 1024
}

private final class EofTrackingInputStream(input: FSDataInputStream) extends FSInputStream {
  private var eof = false

  def reachedEof: Boolean = eof

  override def read(): Int = {
    val result = input.read()
    if (result < 0) eof = true
    result
  }

  override def read(bytes: Array[Byte], offset: Int, length: Int): Int = {
    val result = input.read(bytes, offset, length)
    if (result < 0) eof = true
    result
  }

  override def seek(position: Long): Unit = input.seek(position)

  override def getPos: Long = input.getPos

  override def seekToNewSource(position: Long): Boolean = input.seekToNewSource(position)

  override def close(): Unit = input.close()
}

private final class BinaryRecordReader(
    reader: SequenceFile.Reader,
    input: EofTrackingInputStream,
    file: PartitionedFile,
    splitEnd: Long) extends AutoCloseable {
  private val key = new DataOutputBuffer
  private val value = reader.createValueBytes()
  private var more = reader.getPosition < splitEnd

  def next(): Boolean = {
    if (!more) {
      return false
    }
    key.reset()
    val position = reader.getPosition
    val hasNext = try {
      reader.nextRaw(key, value) >= 0
    } catch {
      case e: EOFException if input.reachedEof =>
        val truncated = new EOFException(s"Truncated SequenceFile in ${file.filePath}")
        truncated.initCause(e)
        throw truncated
    }
    if (!hasNext && input.reachedEof) {
      throw new EOFException(s"Truncated SequenceFile in ${file.filePath}")
    }
    if (!hasNext || (position >= splitEnd && reader.syncSeen)) {
      more = false
    }
    more
  }

  def writeKey(out: DataOutputStream): Unit = out.write(key.getData, 0, key.getLength)

  def writeValue(out: DataOutputStream): Unit = value.writeUncompressedBytes(out)

  override def close(): Unit = reader.close()
}

private object BinaryRecordReader {
  def open(conf: Configuration, file: PartitionedFile): BinaryRecordReader = {
    val path = new Path(new URI(file.filePath.toString))
    val fileSystem = path.getFileSystem(conf)
    val rawInput = fileSystem.open(path)
    val input = new EofTrackingInputStream(rawInput)
    var reader: SequenceFile.Reader = null
    try {
      val fileSize = if (file.fileSize > 0) file.fileSize
      else fileSystem.getFileStatus(path).getLen
      reader = new SequenceFile.Reader(
        conf,
        SequenceFile.Reader.stream(new FSDataInputStream(input)),
        SequenceFile.Reader.length(fileSize))
      if (file.start > reader.getPosition) {
        reader.sync(file.start)
      }
      val splitEnd = math.min(fileSize, Math.addExact(file.start, file.length))
      new BinaryRecordReader(reader, input, file, splitEnd)
    } catch {
      case t: Throwable =>
        if (reader == null) {
          input.safeClose(t)
        } else {
          reader.safeClose(t)
        }
        throw t
    }
  }
}

private final class BatchCapacityExceeded extends IOException

private[rapids] final class BinaryHostColumnBuilder(
    maxDataCapacity: Long,
    maxRowCapacity: Int,
    initialDataCapacity: Long,
    initialRowCapacity: Int,
    reallocations: GpuMetric = NoopMetric,
    copiedBytes: GpuMetric = NoopMetric) extends AutoCloseable {
  private var data: HostMemoryBuffer = null
  private var offsets: HostMemoryBuffer = null
  private var hostOutput: HostMemoryOutputStream = null
  private var dataOutput: DataOutputStream = null
  private var rows = 0
  private var writeLimit = maxDataCapacity

  private val growableOutput = new OutputStream {
    override def write(value: Int): Unit = {
      growData(Math.addExact(hostOutput.getPos, 1L))
      hostOutput.write(value)
    }

    override def write(bytes: Array[Byte], offset: Int, length: Int): Unit = {
      growData(Math.addExact(hostOutput.getPos, length.toLong))
      hostOutput.write(bytes, offset, length)
    }
  }

  try {
    require(initialDataCapacity > 0 && initialDataCapacity <= maxDataCapacity)
    require(initialRowCapacity > 0 && initialRowCapacity <= maxRowCapacity)
    data = HostAlloc.alloc(initialDataCapacity, preferPinned = true)
    offsets = HostAlloc.alloc(
      (initialRowCapacity.toLong + 1L) * DType.INT32.getSizeInBytes, preferPinned = true)
    hostOutput = new HostMemoryOutputStream(data)
    dataOutput = new DataOutputStream(growableOutput)
    offsets.setInt(0, 0)
  } catch {
    case t: Throwable =>
      dataOutput.safeClose(t)
      hostOutput.safeClose(t)
      data.safeClose(t)
      offsets.safeClose(t)
      throw t
  }

  private def recordGrowth(bytes: Long): Unit = {
    // Several reader threads share the scan's non-atomic SQL metrics.
    if (reallocations ne NoopMetric) {
      reallocations.synchronized { reallocations += 1 }
    }
    if (copiedBytes ne NoopMetric) {
      copiedBytes.synchronized { copiedBytes += bytes }
    }
  }

  private def growData(requiredCapacity: Long): Unit = {
    if (requiredCapacity > writeLimit) {
      throw new BatchCapacityExceeded
    }
    if (requiredCapacity > data.getLength) {
      val newCapacity = math.min(maxDataCapacity,
        math.max(requiredCapacity, Math.multiplyExact(data.getLength, 2L)))
      val oldPosition = hostOutput.getPos
      closeOnExcept(HostAlloc.alloc(newCapacity, preferPinned = true)) { newData =>
        newData.copyFromHostBuffer(0, data, 0, oldPosition)
        recordGrowth(oldPosition)
        val newHostOutput = new HostMemoryOutputStream(newData)
        newHostOutput.seek(oldPosition)
        hostOutput.safeClose(null)
        data.safeClose(null)
        data = newData
        hostOutput = newHostOutput
      }
    }
  }

  private def growOffsets(requiredRows: Int): Unit = {
    val currentRowCapacity = offsets.getLength / DType.INT32.getSizeInBytes - 1L
    if (requiredRows > currentRowCapacity) {
      val newRowCapacity = math.min(maxRowCapacity.toLong,
        math.max(requiredRows.toLong, Math.multiplyExact(currentRowCapacity, 2L)))
      closeOnExcept(HostAlloc.alloc(
          (newRowCapacity + 1L) * DType.INT32.getSizeInBytes, preferPinned = true)) {
        newOffsets =>
          val bytes = (rows.toLong + 1L) * DType.INT32.getSizeInBytes
          newOffsets.copyFromHostBuffer(0, offsets, 0, bytes)
          recordGrowth(bytes)
          offsets.safeClose(null)
          offsets = newOffsets
      }
    }
  }

  def mark(): Long = hostOutput.getPos

  def nextInitialSize: (Long, Int) = {
    // Round actual contents, not allocation slack or a rolled-back partial record.
    // Small row-width changes should not force another full-buffer copy.
    def rounded(size: Long, unit: Long, limit: Long): Long =
      math.min(limit, math.max(1L, ((size + unit - 1L) / unit) * unit))

    (rounded(hostOutput.getPos, SequenceFileReaderLimits.MIN_INITIAL_DATA_BYTES, maxDataCapacity),
      rounded(rows, SequenceFileReaderLimits.MIN_INITIAL_ROWS, maxRowCapacity).toInt)
  }

  def rollback(rowCount: Int, position: Long): Unit = {
    rows = rowCount
    hostOutput.seek(position)
  }

  def append(write: DataOutputStream => Unit, maxBytes: Long = maxDataCapacity): Long = {
    require(rows < maxRowCapacity, "SequenceFile binary exceeds the row limit")
    require(maxBytes >= 0, "SequenceFile binary write budget must be nonnegative")
    val start = hostOutput.getPos
    // A compressed value can write in several calls; reject overflow before growing either buffer.
    writeLimit = start + math.min(maxBytes, maxDataCapacity - start)
    write(dataOutput)
    val written = hostOutput.getPos - start
    growOffsets(rows + 1)
    rows += 1
    offsets.setInt(rows.toLong * DType.INT32.getSizeInBytes,
      Math.toIntExact(hostOutput.getPos))
    written
  }

  def finish(): Array[SpillableHostBuffer] = {
    var dataSpill: SpillableHostBuffer = null
    var offsetsSpill: SpillableHostBuffer = null
    try {
      val dataBuffer = data
      dataSpill = SpillableHostBuffer(
        dataBuffer, hostOutput.getPos, SpillPriorities.ACTIVE_BATCHING_PRIORITY)
      data = null
      val offsetsBuffer = offsets
      offsetsSpill = SpillableHostBuffer(
        offsetsBuffer,
        (rows.toLong + 1L) * DType.INT32.getSizeInBytes,
        SpillPriorities.ACTIVE_BATCHING_PRIORITY)
      offsets = null
      Array(dataSpill, offsetsSpill)
    } catch {
      case t: Throwable =>
        dataSpill.safeClose(t)
        offsetsSpill.safeClose(t)
        throw t
    }
  }

  override def close(): Unit = {
    dataOutput.safeClose(null)
    hostOutput.safeClose(null)
    data.safeClose(null)
    offsets.safeClose(null)
    hostOutput = null
    data = null
    offsets = null
  }
}

private[sequencefile] class GpuSequenceFilePartitionReader(
    conf: Configuration,
    files: Array[PartitionedFile],
    keyFirst: Boolean,
    poolConf: ThreadPoolConf,
    maxNumFileProcessed: Int,
    execMetrics: Map[String, GpuMetric],
    maxReadBatchSizeRows: Int,
    maxReadBatchSizeBytes: Long,
    maxGpuColumnSizeBytes: Long,
    keepReadsInOrder: Boolean,
    combineWaitTime: Int)
  extends MultiFileCloudPartitionReaderBase(
    conf,
    files,
    poolConf,
    maxNumFileProcessed,
    Array.empty,
    execMetrics,
    maxReadBatchSizeRows,
    maxReadBatchSizeBytes,
    keepReadsInOrder = keepReadsInOrder,
    combineConf = CombineConf(Seq(
      SequenceFileReaderLimits.MAX_BATCH_OUTPUT_BYTES,
      maxReadBatchSizeBytes,
      maxGpuColumnSizeBytes).min,
      combineWaitTime)) with MultiFileReaderFunctions {

  private val projectedColumns = 2
  private val batchCapacity = Seq(
    SequenceFileReaderLimits.MAX_BATCH_OUTPUT_BYTES,
    maxReadBatchSizeBytes,
    maxGpuColumnSizeBytes).min
  private val maxRowsForOffsets = batchCapacity /
    (projectedColumns.toLong * DType.INT32.getSizeInBytes *
      SequenceFileReaderLimits.MAX_OFFSETS_FRACTION_DENOMINATOR) - 1L
  private val rowCapacity = math.min(
    math.min(maxReadBatchSizeRows.toLong, SequenceFileReaderLimits.MAX_BATCH_ROWS.toLong),
    maxRowsForOffsets).toInt
  require(rowCapacity > 0, "SequenceFile binary batch limit is too small")
  private val offsetsCapacity = projectedColumns.toLong * (rowCapacity.toLong + 1L) *
    DType.INT32.getSizeInBytes
  private val payloadCapacity = batchCapacity - offsetsCapacity
  require(payloadCapacity > 0, "SequenceFile binary batch limit leaves no room for payloads")
  // This covers both columns plus one builder's grow-and-copy overlap.
  private val requiredHostMemory = Math.multiplyExact(3L, batchCapacity)
  private val keyIndex = if (keyFirst) 0 else 1
  private val valueIndex = 1 - keyIndex
  private val tracker = new ResultTracker
  private val readStates = mutable.ArrayBuffer.empty[ReadState]
  private val outputBatchesMetric = execMetrics.getOrElse(NUM_OUTPUT_BATCHES, NoopMetric)
  private val copyMetric = execMetrics.getOrElse(COPY_BUFFER_TIME, NoopMetric)
  private val reallocationsMetric =
    execMetrics.getOrElse(GpuSequenceFileRDDReader.HOST_BUFFER_REALLOCATIONS, NoopMetric)
  private val copiedBytesMetric =
    execMetrics.getOrElse(GpuSequenceFileRDDReader.HOST_BUFFER_BYTES_COPIED, NoopMetric)

  private sealed trait SequenceFileBuffers extends HostMemoryBuffersWithMetaDataBase {
    def sourceBuffers: Array[SequenceFileHostBuffers]
  }

  private final class SequenceFileHostBuffers(
      val state: ReadState,
      val runner: ReadBatchRunner,
      val atEnd: Boolean,
      override val memBuffersAndSizes: Array[SingleHMBAndMeta],
      override val bytesRead: Long) extends SequenceFileBuffers {
    require(memBuffersAndSizes.length == 1)

    override val partitionedFile: PartitionedFile = state.file

    override lazy val sourceBuffers: Array[SequenceFileHostBuffers] = Array(this)

    override def close(): Unit = synchronized {
      tracker.remove(this)
      super.close()
    }
  }

  private final class CombinedSequenceFileHostBuffers(
      override val sourceBuffers: Array[SequenceFileHostBuffers]) extends SequenceFileBuffers {
    require(sourceBuffers.nonEmpty)

    override val partitionedFile: PartitionedFile = sourceBuffers.head.partitionedFile
    override val memBuffersAndSizes: Array[SingleHMBAndMeta] =
      sourceBuffers.flatMap(_.memBuffersAndSizes)
    override val bytesRead: Long = sourceBuffers.foldLeft(0L) { (total, buffers) =>
      Math.addExact(total, buffers.bytesRead)
    }

    setExecutionTime(
      sourceBuffers.foldLeft(0L)((total, buffers) =>
        Math.addExact(total, buffers.getFilterTime)),
      sourceBuffers.foldLeft(0L)((total, buffers) =>
        Math.addExact(total, buffers.getBufferTime)))
    setScheduleTime(sourceBuffers.foldLeft(0L)((total, buffers) =>
      Math.addExact(total, buffers.getScheduleTime)))

    override def close(): Unit = sourceBuffers.safeClose()
  }

  private final class ResultTracker extends AutoCloseable {
    private val published = mutable.HashSet.empty[SequenceFileHostBuffers]
    private var closed = false

    def publish(result: SequenceFileHostBuffers, release: () => Unit): Boolean = synchronized {
      if (closed) {
        false
      } else {
        result.addReleaseResourceCallback(release)
        published += result
        true
      }
    }

    def remove(result: SequenceFileHostBuffers): Unit = synchronized {
      published -= result
    }

    def claim(results: Array[SequenceFileHostBuffers]): Boolean = synchronized {
      if (closed || !results.forall(published.contains)) {
        false
      } else {
        results.foreach(published.remove)
        true
      }
    }

    override def close(): Unit = {
      val toClose = synchronized {
        closed = true
        val copy = published.toArray
        published.clear()
        copy
      }
      toClose.safeClose()
    }
  }

  private class ReadBatchRunner(state: ReadState)
    extends MemoryBoundedAsyncRunner[BufferInfo] {

    private val readAhead =
      new SequenceFileReadAhead[(ReadBatchRunner, Future[RunnerResult])]

    def prepareNextRunner(): Unit = {
      val next = new ReadBatchRunner(state)
      // Preparing on the consumer thread preserves its task context and metrics.
      val prepared = prepareReadRunner(state.file, next)
      readAhead.armSuccessor(() => state.submitIfOpen(prepared).map(next -> _))
    }

    def readContinuation(): Future[RunnerResult] = {
      readAhead.takeContinuation() match {
        case Some((next, future)) =>
          // The consumed chunk has released its buffers, so another chunk may be prefetched.
          if (state.canRead) next.prepareNextRunner()
          future
        case None =>
          state.checkOpen()
          throw new IllegalStateException("SequenceFile continuation was not submitted")
      }
    }

    // The reader owns cancellation; retaining a task listener per chunk would grow with the split.
    override def sparkTaskContext: Option[TaskContext] = None
    override val priority: Long = state.priority

    override val requiredMemoryBytes: Long = requiredHostMemory

    override protected def buildResult(
        resultData: BufferInfo,
        metrics: AsyncMetrics): AsyncResult[BufferInfo] = {
      val releaseCallback = () => {
        withStateLock { _ =>
          if (isHoldingResource) releaseResourceCallback()
        }
      }
      // Bind the reservation before publishing; task closure can leave the future unconsumed.
      val buffers = resultData.asInstanceOf[SequenceFileHostBuffers]
      if (tracker.publish(buffers, releaseCallback)) {
        if (keepReadsInOrder) readAhead.markReady(!buffers.atEnd)
        val resultMetrics = metrics
        new AsyncResult[BufferInfo] {
          override val data: BufferInfo = resultData
          override val metrics: AsyncMetrics = resultMetrics
        }
      } else {
        resultData.close()
        new FastReleaseResult(resultData, metrics)
      }
    }

    override protected def callImpl(): BufferInfo = {
      state.checkOpen()
      val taskContext = state.taskContext
      TrampolineUtil.setTaskContext(taskContext)
      RmmSpark.poolThreadWorkingOnTask(taskContext.taskAttemptId())
      val start = System.nanoTime()
      try {
        val result = state.readNext(this)
        result.setExecutionTime(0L, System.nanoTime() - start)
        result
      } finally {
        RmmSpark.poolThreadFinishedForTask(taskContext.taskAttemptId())
        TrampolineUtil.unsetTaskContext()
      }
    }
  }

  private def closeBuilders(
      builders: Array[BinaryHostColumnBuilder],
      error: Throwable): Unit = {
    builders.filter(_ != null).safeClose(error)
  }

  private def finishBuilders(
      builders: Array[BinaryHostColumnBuilder],
      rows: Int): SingleHMBAndMeta = {
    val buffers = new Array[SpillableHostBuffer](builders.length * 2)
    try {
      var index = 0
      var bytes = 0L
      builders.foreach { builder =>
        val finished = builder.finish()
        buffers(index) = finished(0)
        buffers(index + 1) = finished(1)
        bytes = Math.addExact(bytes, Math.addExact(finished(0).length, finished(1).length))
        index += 2
      }
      SingleHMBAndMeta(buffers, bytes, rows, Seq.empty)
    } catch {
      case t: Throwable =>
        buffers.safeClose(t)
        throw t
    }
  }

  private def newBuilders(
      file: PartitionedFile,
      previousSizes: Array[(Long, Int)]): Array[BinaryHostColumnBuilder] = {
    val builders = new Array[BinaryHostColumnBuilder](projectedColumns)
    try {
      val cappedFileLength = math.min(file.length, payloadCapacity)
      val initialPayloadCapacity = math.min(payloadCapacity,
        math.max(SequenceFileReaderLimits.MIN_INITIAL_DATA_BYTES,
          Math.multiplyExact(cappedFileLength, 2L)))
      val initialDataCapacity = math.max(1L, initialPayloadCapacity / projectedColumns)
      val initialRowCapacity = math.min(rowCapacity.toLong,
        math.max(SequenceFileReaderLimits.MIN_INITIAL_ROWS.toLong,
          file.length / SequenceFileReaderLimits.ESTIMATED_COMPRESSED_BYTES_PER_ROW)).toInt
      var index = 0
      while (index < builders.length) {
        val (dataCapacity, rows) = if (previousSizes == null) {
          (initialDataCapacity, initialRowCapacity)
        } else {
          previousSizes(index)
        }
        builders(index) = new BinaryHostColumnBuilder(
          payloadCapacity, rowCapacity, dataCapacity, rows, reallocationsMetric, copiedBytesMetric)
        index += 1
      }
      builders
    } catch {
      case t: Throwable =>
        closeBuilders(builders, t)
        throw t
    }
  }

  private def byteLimitExceeded(file: PartitionedFile): UnsupportedOperationException = {
    new UnsupportedOperationException(
      s"SequenceFile binary record exceeds the $payloadCapacity byte decompressed " +
        s"single-batch limit: ${file.filePath}")
  }

  private final class ReadState(
      val taskContext: TaskContext,
      val file: PartitionedFile,
      config: Configuration) extends AutoCloseable {
    val priority: Long = TaskPriority.getTaskPriority(taskContext.taskAttemptId())
    private var reader: BinaryRecordReader = null
    private var pendingRecord = false
    private var atEnd = false
    private var running = false
    private var previousSizes: Array[(Long, Int)] = null
    @volatile private var closed = false

    def canRead: Boolean = !closed && !taskContext.isInterrupted()

    def checkOpen(): Unit = {
      if (!canRead) {
        throw new InterruptedException("SequenceFile reader was closed")
      }
    }

    def submitIfOpen(runner: AsyncRunner[BufferInfo]): Option[Future[RunnerResult]] = synchronized {
      if (canRead) Some(submitReadRunner(runner)) else None
    }

    def readNext(runner: ReadBatchRunner): SequenceFileHostBuffers = {
      synchronized {
        require(!running, "Concurrent reads of a SequenceFile split are not supported")
        running = true
      }
      val startingBytesRead = fileSystemBytesRead()
      var builders: Array[BinaryHostColumnBuilder] = null
      var readFailure: Throwable = null
      try {
        checkOpen()
        if (reader == null) {
          reader = BinaryRecordReader.open(new Configuration(config), file)
        }
        // Reuse the writers so each record does not allocate two closures.
        val writeKey = reader.writeKey _
        val writeValue = reader.writeValue _
        var rows = 0
        var bytes = 0L
        var keepReading = true
        while (keepReading && rows < rowCapacity && !closed) {
          if (!pendingRecord && !reader.next()) {
            atEnd = true
            keepReading = false
          } else {
            pendingRecord = true
            if (builders == null) builders = newBuilders(file, previousSizes)
            val keyMark = builders(keyIndex).mark()
            val valueMark = builders(valueIndex).mark()
            val written = try {
              val remaining = payloadCapacity - bytes
              val keyBytes = builders(keyIndex).append(writeKey, remaining)
              val valueBytes = builders(valueIndex).append(writeValue, remaining - keyBytes)
              Math.addExact(keyBytes, valueBytes)
            } catch {
              case _: BatchCapacityExceeded =>
                // Both columns must roll back, including a partially written compressed value.
                builders(keyIndex).rollback(rows, keyMark)
                builders(valueIndex).rollback(rows, valueMark)
                if (rows == 0) throw byteLimitExceeded(file)
                keepReading = false
                0L
            }
            if (keepReading) {
              bytes = Math.addExact(bytes, written)
              rows += 1
              pendingRecord = false
            }
          }
        }
        checkOpen()
        val bytesRead = Math.max(0L, fileSystemBytesRead() - startingBytesRead)
        val info = if (builders == null) {
          SingleHMBAndMeta.empty()
        } else {
          val nextSizes = if (atEnd) null else builders.map(_.nextInitialSize)
          val result = finishBuilders(builders, rows)
          previousSizes = nextSizes
          result
        }
        closeOnExcept(info) { ownedInfo =>
          if (builders != null) {
            closeBuilders(builders, null)
            builders = null
          }
          if (atEnd) closeReader()
          new SequenceFileHostBuffers(this, runner, atEnd, Array(ownedInfo), bytesRead)
        }
      } catch {
        case t: Throwable =>
          closed = true
          readFailure = t
          throw t
      } finally {
        try {
          if (builders != null) closeBuilders(builders, readFailure)
        } finally {
          synchronized {
            running = false
            if (closed || atEnd) closeReader(readFailure)
          }
        }
      }
    }

    private def closeReader(error: Throwable = null): Unit = {
      val toClose = reader
      reader = null
      toClose.safeClose(error)
    }

    override def close(): Unit = synchronized {
      closed = true
      if (!running) closeReader()
    }
  }

  private def copyToDevice(
      dataSpill: SpillableHostBuffer,
      offsetsSpill: SpillableHostBuffer,
      rows: Int): ColumnVector = {
    withResource(dataSpill.getDataHostBuffer()) { data =>
      withResource(offsetsSpill.getDataHostBuffer()) { offsets =>
        data.incRefCount()
        val child = try {
          new HostColumnVectorCore(
            DType.UINT8,
            dataSpill.length,
            util.Optional.of[java.lang.Long](0L),
            data,
            null,
            null,
            new util.ArrayList[HostColumnVectorCore]())
        } catch {
          case t: Throwable =>
            data.close()
            throw t
        }
        val hostColumn = closeOnExcept(child) { ownedChild =>
          val children = new util.ArrayList[HostColumnVectorCore](1)
          children.add(ownedChild)
          offsets.incRefCount()
          try {
            new HostColumnVector(
              DType.LIST,
              rows,
              util.Optional.of[java.lang.Long](0L),
              null,
              null,
              offsets,
              children)
          } catch {
            case t: Throwable =>
              offsets.close()
              throw t
          }
        }
        withResource(hostColumn)(_.copyToDevice())
      }
    }
  }

  private def copyAndConcatenate(
      infos: Array[SingleHMBAndMeta],
      columnIndex: Int): ColumnVector = {
    val deviceColumns = new Array[ColumnVector](infos.length)
    try {
      var index = 0
      while (index < infos.length) {
        val info = infos(index)
        deviceColumns(index) = copyToDevice(
          info.hmbs(columnIndex * 2), info.hmbs(columnIndex * 2 + 1), info.numRows.toInt)
        index += 1
      }
      if (deviceColumns.length == 1) {
        val result = deviceColumns.head
        deviceColumns(0) = null
        result
      } else {
        ColumnVector.concatenate(deviceColumns: _*)
      }
    } finally {
      deviceColumns.safeClose()
    }
  }

  override def readBatches(
      fileBufsAndMeta: HostMemoryBuffersWithMetaDataBase): Iterator[ColumnarBatch] = {
    val buffers = fileBufsAndMeta.asInstanceOf[SequenceFileBuffers]
    if (!tracker.claim(buffers.sourceBuffers)) {
      buffers.close()
      throw new IllegalStateException("SequenceFile buffers were closed before GPU transfer")
    }
    val batch = withRetryNoSplit(buffers) { current =>
      GpuSemaphore.acquireIfNecessary(TaskContext.get())
      val infos = current.memBuffersAndSizes.filter(_.numRows > 0)
      val rows = Math.toIntExact(infos.foldLeft(0L)((total, info) =>
        Math.addExact(total, info.numRows)))
      val columns = new Array[SparkVector](projectedColumns)
      closeOnExcept(columns) { _ =>
        copyMetric.ns {
          var index = 0
          while (index < columns.length) {
            val device = copyAndConcatenate(infos, index)
            columns(index) = closeOnExcept(device) { ownedDevice =>
              GpuColumnVector.from(ownedDevice, BinaryType)
            }
            index += 1
          }
        }
        new ColumnarBatch(columns, rows)
      }
    }
    outputBatchesMetric += 1
    new SingleGpuColumnarBatchIterator(batch)
  }

  override def canUseCombine: Boolean = true

  override def combineHMBs(
      results: Array[HostMemoryBuffersWithMetaDataBase]): HostMemoryBuffersWithMetaDataBase = {
    require(results.nonEmpty)
    val buffers = results.map(_.asInstanceOf[SequenceFileBuffers])
    var accepted = 0
    var bytes = 0L
    var rows = 0L
    var keepAdding = true
    while (accepted < buffers.length && keepAdding) {
      val next = buffers(accepted)
      val nextBytes = next.memBuffersAndSizes.foldLeft(0L)((total, info) =>
        Math.addExact(total, info.bytes))
      val nextRows = next.memBuffersAndSizes.foldLeft(0L)((total, info) =>
        Math.addExact(total, info.numRows))
      require(nextBytes <= batchCapacity && nextRows <= rowCapacity)
      if (accepted == 0 ||
          (nextBytes <= batchCapacity - bytes && nextRows <= rowCapacity - rows)) {
        bytes += nextBytes
        rows += nextRows
        accepted += 1
      } else {
        keepAdding = false
      }
    }
    if (accepted < results.length) {
      combineLeftOverFiles = Some(results.drop(accepted))
    }
    val sources = buffers.take(accepted).flatMap(_.sourceBuffers)
    if (sources.length == 1) sources.head else new CombinedSequenceFileHostBuffers(sources)
  }

  override protected def getNumFilesInHostBuffers(
      fileInfo: HostMemoryBuffersWithMetaDataBase): Int =
    fileInfo.asInstanceOf[SequenceFileBuffers].sourceBuffers.length

  override protected def getNumCompletedFiles(fileInfo: BufferInfo): Int =
    fileInfo.asInstanceOf[SequenceFileBuffers].sourceBuffers.count(_.atEnd)

  override protected def getReadContinuations(
      fileInfo: BufferInfo): Seq[Future[RunnerResult]] = {
    fileInfo.asInstanceOf[SequenceFileBuffers].sourceBuffers.filterNot(_.atEnd).map { source =>
      if (keepReadsInOrder) {
        source.runner.readContinuation()
      } else {
        submitReadRunner(prepareReadRunner(
          source.partitionedFile, new ReadBatchRunner(source.state)))
      }
    }.toSeq
  }

  override protected def canCombineAfter(fileInfo: BufferInfo): Boolean =
    !keepReadsInOrder || fileInfo.asInstanceOf[SequenceFileBuffers].sourceBuffers.forall(_.atEnd)

  override def getBatchRunner(
      tc: TaskContext,
      file: PartitionedFile,
      config: Configuration,
      filters: Array[Filter]): MemoryBoundedAsyncRunner[BufferInfo] = {
    val state = new ReadState(tc, file, config)
    readStates += state
    val runner = new ReadBatchRunner(state)
    if (keepReadsInOrder) runner.prepareNextRunner()
    runner
  }

  override final def getFileFormatShortName: String = "SequenceFileBinary"

  // Closed states reject pending reads. Let runners finish their pool accounting hooks;
  // cancelling their futures can bypass those hooks after acquiring a reservation.
  override protected def closeAsyncRead(task: Future[RunnerResult]): Option[BufferInfo] = None

  override def close(): Unit = {
    try {
      readStates.safeClose()
    } finally {
      try {
        tracker.close()
      } finally {
        super.close()
      }
    }
  }
}

private[rapids] case class GpuSequenceFilePartitionReaderFactory(
    @transient sqlConf: SQLConf,
    broadcastedConf: Broadcast[SerializableConfiguration],
    keyFirst: Boolean,
    @transient rapidsConf: RapidsConf,
    poolConfBuilder: ThreadPoolConfBuilder,
    metrics: Map[String, GpuMetric])
  extends MultiFilePartitionReaderFactoryBase(sqlConf, broadcastedConf, rapidsConf) {

  override protected val canUseCoalesceFilesReader: Boolean = false

  override protected val canUseMultiThreadReader: Boolean = true

  private val maxNumFileProcessed = rapidsConf.multiThreadReadNumThreads
  private val keepReadsInOrder = rapidsConf.getMultithreadedReaderKeepOrder
  private val combineWaitTime = rapidsConf.getMultithreadedCombineWaitTime

  override protected def buildBaseColumnarReaderForCloud(
      files: Array[PartitionedFile],
      conf: Configuration): PartitionReader[ColumnarBatch] = {
    new GpuSequenceFilePartitionReader(
      conf,
      files,
      keyFirst,
      poolConfBuilder.build(),
      maxNumFileProcessed,
      metrics,
      maxReadBatchSizeRows,
      maxReadBatchSizeBytes,
      maxGpuColumnSizeBytes,
      keepReadsInOrder,
      combineWaitTime)
  }

  override protected def buildBaseColumnarReaderForCoalescing(
      files: Array[PartitionedFile],
      conf: Configuration): PartitionReader[ColumnarBatch] = {
    throw new IllegalStateException("SequenceFile binary does not support coalescing reads")
  }

  override protected def getFileFormatShortName: String = "SequenceFileBinary"
}

object GpuSequenceFileRDDReader {
  val HOST_BUFFER_REALLOCATIONS = "hostBufferReallocations"
  val HOST_BUFFER_BYTES_COPIED = "hostBufferBytesCopied"

  def createReaderFactory(
      sqlConf: SQLConf,
      broadcastedConf: Broadcast[SerializableConfiguration],
      keyFirst: Boolean,
      rapidsConf: RapidsConf,
      metrics: Map[String, GpuMetric]): PartitionReaderFactory = {
    GpuSequenceFilePartitionReaderFactory(
      sqlConf,
      broadcastedConf,
      keyFirst,
      rapidsConf,
      ThreadPoolConfBuilder(rapidsConf),
      metrics)
  }
}
