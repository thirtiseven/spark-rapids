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
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}
import java.util

import ai.rapids.cudf.{HostMemoryBuffer, ParquetOptions, Table => CudfTable}
import com.nvidia.spark.rapids._
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.GpuMetric._
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.jni.LencNative
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path

import org.apache.spark.TaskContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.connector.catalog.{SupportsRead, Table, TableCapability, TableProvider}
import org.apache.spark.sql.connector.expressions.Transform
import org.apache.spark.sql.connector.read._
import org.apache.spark.sql.types._
import org.apache.spark.sql.util.CaseInsensitiveStringMap
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.util.SerializableConfiguration

/**
 * Item 1: Distributed / S3A / HDFS + Local POSIX I/O layer for .lenc.
 * - Local paths (`file://` or bare `/...`) use zero-copy native POSIX pread().
 * - Remote URIs (`s3a://`, `hdfs://`, `gs://`, etc.) use Hadoop FileSystem / FSDataInputStream
 *   with 256 KiB range coalescing directly into HostMemoryBuffer — zero GPU kernel changes!
 */
object LencFileIO {
  val HEADER_SIZE = 64
  val REMOTE_MERGE_GAP = 256L * 1024L // coalesce adjacent S3/HDFS ranges within 256 KiB

  private lazy val remoteIoPool: java.util.concurrent.ExecutorService =
    java.util.concurrent.Executors.newFixedThreadPool(16, new java.util.concurrent.ThreadFactory {
      override def newThread(r: Runnable): Thread = {
        val t = new Thread(r, "lenc-remote-s3-io")
        t.setDaemon(true)
        t
      }
    })

  def isRemoteUri(path: String): Boolean = {
    val idx = path.indexOf("://")
    idx > 0 && !path.startsWith("file://")
  }

  def stripFileScheme(path: String): String = {
    if (path.startsWith("file://")) path.stripPrefix("file://")
    else if (path.startsWith("file:")) path.stripPrefix("file:")
    else path
  }

  def activeHadoopConf(): Configuration = {
    SparkSession.getActiveSession
      .map(_.sessionState.newHadoopConf())
      .getOrElse(new Configuration())
  }

  def loadPlanAndHost(
      path: String,
      colIndices: Array[Int],
      rangePreds: Array[LencRangePred],
      serConf: SerializableConfiguration): Option[LencPrefetchedFile] = {
    if (!isRemoteUri(path)) {
      val clean = stripFileScheme(path)
      val plan = LencNative.planRead(clean, colIndices)
      closeOnExcept(plan) { p =>
        if (rangePreds.nonEmpty && LencRangePruner.canSkipFile(p, rangePreds)) {
          p.close()
          None
        } else {
          val totalBytes = p.totalPackedBytes()
          val buf = HostMemoryBuffer.allocate(math.max(totalBytes, 64L))
          closeOnExcept(buf) { b =>
            if (totalBytes > 0L) {
              p.readToHost(clean, b)
            }
            Some(LencPrefetchedFile(path, p, b))
          }
        }
      }
    } else {
      val conf = if (serConf != null) serConf.value else new Configuration()
      val hpath = new Path(path)
      val fs = hpath.getFileSystem(conf)
      withResource(fs.open(hpath)) { in =>
        val hdr = new Array[Byte](HEADER_SIZE)
        in.readFully(0L, hdr)
        val bb = ByteBuffer.wrap(hdr).order(ByteOrder.LITTLE_ENDIAN)
        // FileHeader: magic[8], num_rows(4), num_cols(4), chunk_size(4), res(4), footer_off(8), footer_sz(4)
        val footerOff = bb.getLong(24)
        val footerSz = bb.getInt(32)
        val ftr = new Array[Byte](math.max(footerSz, 0))
        if (footerSz > 0) {
          in.readFully(footerOff, ftr)
        }
        val plan = LencNative.planReadFromBuffers(hdr, ftr, colIndices)
        closeOnExcept(plan) { p =>
          val totalBytes = p.totalPackedBytes()
          val buf = HostMemoryBuffer.allocate(math.max(totalBytes, 64L))
          closeOnExcept(buf) { b =>
            if (totalBytes > 0L) {
              val rawRanges = p.readRanges()
              // Triples: (fileOff, len, dstOff) sorted by fileOff for S3/HDFS range coalescing
              case class Rng(fileOff: Long, len: Int, dstOff: Long)
              val rngs = (0 until rawRanges.length by 3).map { i =>
                Rng(rawRanges(i), rawRanges(i + 1).toInt, rawRanges(i + 2))
              }.sortBy(_.fileOff)

                // Coalesce into merged byte spans and fetch spans in parallel across S3/HDFS!
                case class Span(startOff: Long, spanLen: Int, items: Seq[Rng])
                val spans = new scala.collection.mutable.ArrayBuffer[Span]()
                var i = 0
                while (i < rngs.length) {
                  val startOff = rngs(i).fileOff
                  var endOff = startOff + rngs(i).len
                  var j = i + 1
                  while (j < rngs.length && rngs(j).fileOff <= endOff + REMOTE_MERGE_GAP &&
                      (rngs(j).fileOff + rngs(j).len - startOff) <= (64L << 20)) {
                    endOff = math.max(endOff, rngs(j).fileOff + rngs(j).len)
                    j += 1
                  }
                  spans += Span(startOff, (endOff - startOff).toInt, rngs.slice(i, j))
                  i = j
                }
                if (spans.size <= 1) {
                  spans.foreach { sp =>
                    val tmp = new Array[Byte](sp.spanLen)
                    in.readFully(sp.startOff, tmp, 0, sp.spanLen)
                    sp.items.foreach { r =>
                      b.setBytes(r.dstOff, tmp, (r.fileOff - sp.startOff).toInt, r.len)
                    }
                  }
                } else {
                  val futs = spans.map { sp =>
                    remoteIoPool.submit(new Runnable {
                      override def run(): Unit = {
                        withResource(fs.open(hpath)) { sin =>
                          val tmp = new Array[Byte](sp.spanLen)
                          sin.readFully(sp.startOff, tmp, 0, sp.spanLen)
                          sp.items.foreach { r =>
                            b.setBytes(r.dstOff, tmp, (r.fileOff - sp.startOff).toInt, r.len)
                          }
                        }
                      }
                    })
                  }
                  futs.foreach(_.get())
                }
            }
            Some(LencPrefetchedFile(path, p, b))
          }
        }
      }
    }
  }
}

object LencSchemaIO {
  val SCHEMA_FILE = "_lenc_schema.json"

  def stripFileScheme(path: String): String = LencFileIO.stripFileScheme(path)

  def readSchema(dirPath: String): StructType = {
    if (!LencFileIO.isRemoteUri(dirPath)) {
      val clean = stripFileScheme(dirPath)
      val f = new File(clean, SCHEMA_FILE)
      require(f.exists(), s"Missing $SCHEMA_FILE in $clean")
      val json = new String(Files.readAllBytes(f.toPath), StandardCharsets.UTF_8)
      DataType.fromJson(json).asInstanceOf[StructType]
    } else {
      val conf = LencFileIO.activeHadoopConf()
      val sp = new Path(new Path(dirPath), SCHEMA_FILE)
      val fs = sp.getFileSystem(conf)
      require(fs.exists(sp), s"Missing $SCHEMA_FILE in $dirPath")
      withResource(fs.open(sp)) { in =>
        val out = new java.io.ByteArrayOutputStream()
        val buf = new Array[Byte](8192)
        var n = in.read(buf)
        while (n >= 0) {
          if (n > 0) out.write(buf, 0, n)
          n = in.read(buf)
        }
        val json = new String(out.toByteArray, StandardCharsets.UTF_8)
        DataType.fromJson(json).asInstanceOf[StructType]
      }
    }
  }

  def listLencFilesWithSizes(dirPath: String): (Array[String], Array[Long]) = {
    if (!LencFileIO.isRemoteUri(dirPath)) {
      val clean = stripFileScheme(dirPath)
      val dir = new File(clean)
      require(dir.isDirectory, s"Not a directory: $clean")
      val arr = dir.listFiles()
        .filter(f => f.isFile && f.getName.endsWith(".lenc"))
        .sortBy(_.getAbsolutePath)
      (arr.map(_.getAbsolutePath), arr.map(_.length()))
    } else {
      val conf = LencFileIO.activeHadoopConf()
      val dp = new Path(dirPath)
      val fs = dp.getFileSystem(conf)
      val arr = fs.listStatus(dp)
        .filter(st => st.isFile && st.getPath.getName.endsWith(".lenc"))
        .sortBy(_.getPath.toString)
      (arr.map(_.getPath.toString), arr.map(_.getLen))
    }
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
      case "W1a_1" => (0, Array("c0", "c1", "c2", "c3").map(nameToIdx))
      case "W1a_8" => (1, Array("c0", "c1", "c2", "c3", "c4").map(nameToIdx))
      case "W1b_8" => (2, Array("c0", "c1", "c2", "c3", "c4", "s1").map(nameToIdx))
      case "W1b_24" => (3, Array("c0", "c1", "c2", "c3", "c4", "s1").map(nameToIdx))
      case "W1a_1_sum" => (4, Array("c0", "c1", "c2", "c3").map(nameToIdx))
      case other => throw new IllegalArgumentException(s"Unsupported fusedExpr: $other")
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
    val (files, fileSizes) = LencSchemaIO.listLencFilesWithSizes(path)
    val serConf = new SerializableConfiguration(LencFileIO.activeHadoopConf())
    new LencTable(path, fullSchema, schema, files, fileSizes, fusedExpr, serConf)
  }
}

class LencTable(
    path: String,
    fullSchema: StructType,
    exposedSchema: StructType,
    files: Array[String],
    fileSizes: Array[Long],
    fusedExpr: Option[String],
    serConf: SerializableConfiguration) extends Table with SupportsRead {

  override def name(): String = s"lenc($path)"

  override def schema(): StructType = exposedSchema

  override def capabilities(): util.Set[TableCapability] =
    util.EnumSet.of(TableCapability.BATCH_READ)

  override def newScanBuilder(options: CaseInsensitiveStringMap): ScanBuilder =
    new LencScanBuilder(fullSchema, exposedSchema, files, fileSizes, fusedExpr, serConf)
}

class LencScanBuilder(
    fullSchema: StructType,
    exposedSchema: StructType,
    files: Array[String],
    fileSizes: Array[Long],
    fusedExpr: Option[String],
    serConf: SerializableConfiguration)
  extends ScanBuilder with SupportsPushDownRequiredColumns {

  private var requiredSchema: StructType = exposedSchema

  override def pruneColumns(requiredSchema: StructType): Unit = {
    this.requiredSchema = requiredSchema
  }

  override def build(): Scan = {
    LencScan(fullSchema, requiredSchema, files, fileSizes, fusedExpr, serConf)
  }
}

case class LencScan(
    fullSchema: StructType,
    requiredSchema: StructType,
    files: Array[String],
    fileSizes: Array[Long],
    fusedExpr: Option[String],
    serConf: SerializableConfiguration) extends Scan with Batch {
  override def readSchema(): StructType = requiredSchema
  override def description(): String =
    s"LencScan(files=${files.length}, cols=${requiredSchema.length}, fused=${fusedExpr.getOrElse("none")})"
  override def toBatch: Batch = this
  override def planInputPartitions(): Array[InputPartition] =
    files.map(f => LencInputPartition(Array(f)))
  override def createReaderFactory(): PartitionReaderFactory = new PartitionReaderFactory {
    override def supportColumnarReads(partition: InputPartition): Boolean = true
    override def createReader(partition: InputPartition): PartitionReader[InternalRow] =
      throw new UnsupportedOperationException("LencScan must execute on GPU via GpuLencScan")
    override def createColumnarReader(partition: InputPartition): PartitionReader[ColumnarBatch] =
      throw new UnsupportedOperationException("LencScan must execute on GPU via GpuLencScan")
  }
}

case class LencInputPartition(filePaths: Array[String]) extends InputPartition

case class GpuLencScan(
    fullSchema: StructType,
    requiredSchema: StructType,
    files: Array[String],
    fileSizes: Array[Long],
    fusedExpr: Option[String],
    serConf: SerializableConfiguration) extends GpuScan with Batch {

  override def readSchema(): StructType = requiredSchema

  override def description(): String =
    s"GpuLencScan(files=${files.length}, cols=${requiredSchema.length}, fused=${fusedExpr.getOrElse("none")})"

  override def withInputFile(): GpuScan = this

  override def toBatch: Batch = this

  override def planInputPartitions(): Array[InputPartition] = {
    LencPartitionHelper.groupFiles(files, fileSizes).map(LencInputPartition)
  }

  override def createReaderFactory(): PartitionReaderFactory = {
    val nameToIdx = fullSchema.fieldNames.zipWithIndex.toMap
    val (fusedKind, projectedIndices) = fusedExpr match {
      case Some(expr) => LencSchemaIO.fusedSpec(expr, fullSchema)
      case None => (-1, requiredSchema.fieldNames.map(nameToIdx))
    }
    val dict32Set = GpuDictionaryUtils.updateDriverDict32Cols(
      org.apache.spark.sql.internal.SQLConf.get)
    val dict32Mask: Array[Byte] = if (fusedKind >= 0 || dict32Set.isEmpty) {
      null
    } else {
      val mask = requiredSchema.fields.map { f =>
        if (f.dataType == StringType && dict32Set.contains(f.name)) 1.toByte else 0.toByte
      }
      if (mask.exists(_ != 0)) mask else null
    }
    GpuLencPartitionReaderFactory(
      requiredSchema, projectedIndices, fusedKind, dict32Mask, serConf, metrics)
  }
}

case class GpuLencPartitionReaderFactory(
    readSchema: StructType,
    projectedIndices: Array[Int],
    fusedKind: Int,
    dict32Mask: Array[Byte],
    serConf: SerializableConfiguration,
    metrics: Map[String, GpuMetric]) extends PartitionReaderFactory {

  override def supportColumnarReads(partition: InputPartition): Boolean = true

  override def createReader(partition: InputPartition): PartitionReader[InternalRow] =
    throw new UnsupportedOperationException("GpuLencScan supports columnar reads only")

  override def createColumnarReader(partition: InputPartition): PartitionReader[ColumnarBatch] = {
    val lp = partition.asInstanceOf[LencInputPartition]
    new GpuLencPartitionReader(
      lp.filePaths, readSchema, projectedIndices, fusedKind, dict32Mask, serConf, metrics)
  }
}

class GpuLencPartitionReader(
    filePaths: Array[String],
    readSchema: StructType,
    projectedIndices: Array[Int],
    fusedKind: Int,
    dict32Mask: Array[Byte],
    serConf: SerializableConfiguration,
    metrics: Map[String, GpuMetric]) extends PartitionReader[ColumnarBatch] {

  private var currentBatch: ColumnarBatch = _
  private val readTypes: Array[DataType] = readSchema.fields.map(_.dataType)

  private val opTime = metrics.getOrElse(OP_TIME_LEGACY, NoopMetric)
  private val bufferTime = metrics.getOrElse(BUFFER_TIME, NoopMetric)
  private val gpuDecodeTime = metrics.getOrElse(GPU_DECODE_TIME, NoopMetric)
  private val numOutputBatches = metrics.getOrElse(NUM_OUTPUT_BATCHES, NoopMetric)
  private val prefetcher = new LencAsyncPrefetcher(filePaths, projectedIndices, Array.empty, serConf)

  override def next(): Boolean = {
    if (currentBatch != null) {
      currentBatch.close()
      currentBatch = null
    }
    val itemOpt = opTime.ns {
      bufferTime.ns {
        prefetcher.next()
      }
    }
    if (itemOpt.isEmpty) {
      return false
    }
    val item = itemOpt.get

    opTime.ns {
      withResource(item.plan) { plan =>
        withResource(item.hostBuf) { hb =>
          GpuSemaphore.acquireIfNecessary(TaskContext.get())
          val table = gpuDecodeTime.ns {
            if (fusedKind >= 0) {
              plan.decodeAndEvalFused(hb, fusedKind, 0L)
            } else {
              plan.decodeFromHost(hb, dict32Mask)
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
    prefetcher.close()
  }
}

object LencPartitionHelper {
  def groupFiles(files: Array[String], fileSizes: Array[Long] = Array.empty): Array[Array[String]] = {
    val maxPartBytes = org.apache.spark.sql.internal.SQLConf.get
      .getConfString("spark.rapids.sql.lenc.maxPartitionBytes", "268435456").toLong
    if (files.length <= 64 || maxPartBytes <= 0L) {
      return files.map(Array(_))
    }
    val maxFilesPerPart = 32
    val groups = new scala.collection.mutable.ArrayBuffer[Array[String]]()
    val cur = new scala.collection.mutable.ArrayBuffer[String]()
    var curBytes = 0L
    files.indices.foreach { idx =>
      val f = files(idx)
      val fsz =
        if (idx < fileSizes.length) fileSizes(idx)
        else if (!LencFileIO.isRemoteUri(f)) new File(LencFileIO.stripFileScheme(f)).length()
        else 64L * 1024L * 1024L
      if (cur.nonEmpty && (curBytes + fsz > maxPartBytes || cur.size >= maxFilesPerPart)) {
        groups += cur.toArray
        cur.clear()
        curBytes = 0L
      }
      cur += f
      curBytes += fsz
    }
    if (cur.nonEmpty) {
      groups += cur.toArray
    }
    groups.toArray
  }
}

final case class LencPrefetchedFile(
    filePath: String,
    plan: LencNative.ReadPlan,
    hostBuf: HostMemoryBuffer)

final class LencAsyncPrefetcher(
    filePaths: Array[String],
    colIndices: Array[Int],
    rangePreds: Array[LencRangePred] = Array.empty,
    serConf: SerializableConfiguration = null) extends AutoCloseable {

  private var nextSubmitIdx = 0
  private val useAsync = filePaths.length > 1
  private val exec: java.util.concurrent.ExecutorService = if (useAsync) {
    java.util.concurrent.Executors.newSingleThreadExecutor(new java.util.concurrent.ThreadFactory {
      override def newThread(r: Runnable): Thread = {
        val t = new Thread(r, "lenc-io-prefetch")
        t.setDaemon(true)
        t
      }
    })
  } else null

  private var pending: java.util.concurrent.Future[Option[LencPrefetchedFile]] = _

  private def loadSync(path: String): Option[LencPrefetchedFile] =
    LencFileIO.loadPlanAndHost(path, colIndices, rangePreds, serConf)

  private def submitNext(): Unit = {
    if (useAsync && nextSubmitIdx < filePaths.length) {
      val path = filePaths(nextSubmitIdx)
      nextSubmitIdx += 1
      pending = exec.submit(new java.util.concurrent.Callable[Option[LencPrefetchedFile]] {
        override def call(): Option[LencPrefetchedFile] = loadSync(path)
      })
    }
  }

  if (useAsync) {
    submitNext()
  }

  def next(): Option[LencPrefetchedFile] = {
    if (!useAsync) {
      while (nextSubmitIdx < filePaths.length) {
        val path = filePaths(nextSubmitIdx)
        nextSubmitIdx += 1
        val opt = loadSync(path)
        if (opt.isDefined) return opt
      }
      None
    } else {
      while (pending != null) {
        val fut = pending
        pending = null
        val opt = fut.get()
        submitNext()
        if (opt.isDefined) return opt
      }
      None
    }
  }

  override def close(): Unit = {
    if (pending != null) {
      try {
        pending.get().foreach { item =>
          item.hostBuf.safeClose()
          item.plan.safeClose()
        }
      } catch {
        case _: Throwable =>
      }
      pending = null
    }
    if (exec != null) {
      exec.shutdownNow()
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
    GpuLencScan(a.fullSchema, a.requiredSchema, a.files, a.fileSizes, a.fusedExpr, a.serConf)
}

/** Converts a directory of Parquet files directly to .lenc files + _lenc_schema.json on GPU. */
class LencConverterImpl extends java.util.function.Function[Array[String], java.lang.Long] {
  private def collectParquetFiles(dir: File): Array[File] = {
    val buf = new scala.collection.mutable.ArrayBuffer[File]()
    def walk(f: File): Unit = {
      val list = f.listFiles()
      if (list != null) {
        list.foreach { child =>
          if (child.isDirectory) {
            walk(child)
          } else if (child.isFile && child.getName.endsWith(".parquet")) {
            buf += child
          }
        }
      }
    }
    walk(dir)
    buf.toArray.sortBy(_.getAbsolutePath)
  }

  override def apply(args: Array[String]): java.lang.Long = {
    val parquetDir = LencSchemaIO.stripFileScheme(args(0))
    val lencDir = LencSchemaIO.stripFileScheme(args(1))
    val schemaJson = args(2)
    val includeCols: Array[String] =
      if (args.length >= 4 && args(3).nonEmpty) args(3).split(",") else Array.empty
    val numThreads: Int =
      if (args.length >= 5 && args(4).nonEmpty) args(4).toInt else 4

    val outDir = new File(lencDir)
    outDir.mkdirs()
    Files.write(
      Paths.get(lencDir, LencSchemaIO.SCHEMA_FILE),
      schemaJson.getBytes(StandardCharsets.UTF_8))

    val pqFiles = collectParquetFiles(new File(parquetDir))
    val optsBuilder = ParquetOptions.builder()
    if (includeCols.nonEmpty) {
      optsBuilder.includeColumn(includeCols: _*)
    }
    val opts = optsBuilder.build()

    val totalBytes = new java.util.concurrent.atomic.AtomicLong(0L)
    val pool = java.util.concurrent.Executors.newFixedThreadPool(math.max(1, numThreads))
    try {
      val futures = pqFiles.zipWithIndex.map { case (pqFile, i) =>
        pool.submit(new Runnable {
          override def run(): Unit = {
            val outPath = new File(outDir, f"part-$i%05d.lenc")
            if (!outPath.exists() || outPath.length() == 0L) {
              withResource(CudfTable.readParquet(opts, pqFile)) { table =>
                LencNative.writeTable(table, outPath.getAbsolutePath)
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
