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

import java.util.concurrent.{ConcurrentHashMap, ConcurrentLinkedQueue}

import scala.collection.mutable

import ai.rapids.cudf.{DType, HostMemoryBuffer}
import ai.rapids.cudf.ast
import com.nvidia.spark.rapids._
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.GpuMetric._
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.jni.LencNative
import com.nvidia.spark.rapids.shims.GpuBatchScanExecBase

import org.apache.spark.{Partition, SparkContext, TaskContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.{Attribute, Expression, ExprId, NamedExpression}
import org.apache.spark.sql.execution.{LeafExecNode, SparkPlan}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types._
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * AST leaf expression representing an inlineable LTO-IR .lenc column decoder UDF call
 * (`cudf::ast::jit::call` to `lenc_decode_*_sK`), aligned with NVIDIA/cudf#24869.
 */
case class GpuLencDecodeLeaf(
    slot: Int,
    symbol: String,
    dataType: DataType,
    cudfDType: DType,
    nullable: Boolean) extends GpuLeafExpression {

  override def selfSupportsAstJit: Boolean = true

  override def convertToAst(numFirstTableColumns: Int): ast.AstExpression = {
    new ast.JitUdfOperation(
      s"${symbol}_s$slot",
      cudfDType,
      Array.empty[DType])
  }

  override def columnarEval(batch: ColumnarBatch): GpuColumnVector =
    throw new UnsupportedOperationException("GpuLencDecodeLeaf is only for LTO AST-JIT")

  override def toString: String = s"${symbol}_s$slot()"
}

/** Executor-level thread-safe pool of compiled LTO Decode-in-Register + AST-JIT programs. */
object GpuLencLtoProgramPool {
  final case class PoolKey(
      exprs: Vector[Expression],
      encodings: Vector[Byte],
      types: Vector[Byte])

  private val MAX_KEYS = 128
  private val MAX_PER_KEY = 8
  private val pools = new ConcurrentHashMap[PoolKey, ConcurrentLinkedQueue[LencNative.LtoProgram]]()

  Runtime.getRuntime.addShutdownHook(new Thread(() => {
    val it = pools.values().iterator()
    while (it.hasNext) {
      val q = it.next()
      var p = q.poll()
      while (p != null) {
        p.safeClose()
        p = q.poll()
      }
    }
    pools.clear()
  }))

  def withProgram[T](
      key: PoolKey,
      build: => LencNative.LtoProgram)(body: LencNative.LtoProgram => T): (T, Boolean) = {
    val q = pools.computeIfAbsent(key, _ => new ConcurrentLinkedQueue[LencNative.LtoProgram]())
    val existing = q.poll()
    val hit = existing != null
    val prog = if (hit) existing else build
    try {
      (body(prog), hit)
    } finally {
      if (pools.size() > MAX_KEYS || q.size() >= MAX_PER_KEY) {
        prog.safeClose()
      } else {
        q.offer(prog)
      }
    }
  }
}

final case class LtoExprSpec(
    fileColIndices: Array[Int],
    compactExprs: Vector[Expression],
    dictLiteralsPerSlot: Array[Array[String]])

object LencLtoFusion {
  val LTO_FUSION_CONF = "spark.rapids.sql.lenc.ltoFusionEnabled"

  // Must match lenc.hpp Encoding & ColType enum values
  private val ENC_FOR_BITPACK: Byte = 0
  private val ENC_DICT_BITPACK: Byte = 1
  private val ENC_RLE_INT: Byte = 2
  private val ENC_RAW_FIXED: Byte = 3
  private val ENC_SCALED_FLOAT_FOR: Byte = 4
  private val ENC_DELTA_FOR_BITPACK: Byte = 6
  private val ENC_NVCOMP_LZ4_FIXED: Byte = 8

  private val TYPE_INT32: Byte = 0
  private val TYPE_INT64: Byte = 1
  private val TYPE_FLOAT32: Byte = 2
  private val TYPE_FLOAT64: Byte = 3
  private val TYPE_DATE32: Byte = 4
  private val TYPE_STRING: Byte = 5

  def decodeSymbolAndType(encoding: Byte, colType: Byte): (String, DataType, DType) = {
    // DELTA_FOR_BITPACK and NVCOMP_LZ4_FIXED are transparently pre-decoded into raw_ptr in lenc.cu
    val effEnc =
      if (encoding == ENC_DELTA_FOR_BITPACK || encoding == ENC_NVCOMP_LZ4_FIXED) ENC_RAW_FIXED
      else encoding
    (effEnc, colType) match {
      case (ENC_FOR_BITPACK, TYPE_INT64) => ("lenc_decode_for_i64", LongType, DType.INT64)
      case (ENC_FOR_BITPACK, TYPE_INT32) => ("lenc_decode_for_i32", IntegerType, DType.INT32)
      case (ENC_FOR_BITPACK, TYPE_DATE32) => ("lenc_decode_for_date32", DateType, DType.TIMESTAMP_DAYS)
      case (ENC_SCALED_FLOAT_FOR, TYPE_FLOAT32) => ("lenc_decode_scaled_f32", FloatType, DType.FLOAT32)
      case (ENC_SCALED_FLOAT_FOR, TYPE_FLOAT64) => ("lenc_decode_scaled_f64", DoubleType, DType.FLOAT64)
      case (ENC_RLE_INT, TYPE_INT64) => ("lenc_decode_rle_i64", LongType, DType.INT64)
      case (ENC_RLE_INT, TYPE_INT32) => ("lenc_decode_rle_i32", IntegerType, DType.INT32)
      case (ENC_DICT_BITPACK, TYPE_STRING) => ("lenc_decode_dict_code", IntegerType, DType.INT32)
      case (ENC_RAW_FIXED, TYPE_INT64) => ("lenc_decode_raw_i64", LongType, DType.INT64)
      case (ENC_RAW_FIXED, TYPE_INT32) => ("lenc_decode_raw_i32", IntegerType, DType.INT32)
      case (ENC_RAW_FIXED, TYPE_DATE32) => ("lenc_decode_raw_date32", DateType, DType.TIMESTAMP_DAYS)
      case (ENC_RAW_FIXED, TYPE_FLOAT32) => ("lenc_decode_raw_f32", FloatType, DType.FLOAT32)
      case (ENC_RAW_FIXED, TYPE_FLOAT64) => ("lenc_decode_raw_f64", DoubleType, DType.FLOAT64)
      case (e, t) =>
        throw new UnsupportedOperationException(s"Unsupported .lenc LTO (encoding=$e, type=$t)")
    }
  }

  def buildLtoProgram(
      compactExprs: Vector[Expression],
      encodings: Array[Byte],
      types: Array[Byte]): LencNative.LtoProgram = {
    val leafInfo = encodings.indices.map { slot =>
      decodeSymbolAndType(encodings(slot), types(slot))
    }
    val ltoExprs = compactExprs.map { expr =>
      expr.transformDown {
        case ref: GpuBoundReference =>
          val (sym, sparkDt, cudfDt) = leafInfo(ref.ordinal)
          GpuLencDecodeLeaf(ref.ordinal, sym, sparkDt, cudfDt, ref.nullable)
      }.asInstanceOf[GpuExpression]
    }
    withResource(ltoExprs.safeMap(_.convertToAst(Int.MaxValue).compileJit())) { compiled =>
      LencNative.compileLtoProgram(compiled: _*)
    }
  }

  private def unwrapAlias(expr: Expression): Expression = expr match {
    case GpuAlias(child, _) => child
    case other => other
  }

  private def isSupportedLeafType(dt: DataType): Boolean = dt match {
    case IntegerType | LongType | FloatType | DoubleType | DateType | StringType => true
    case _ => false
  }

  private def buildSpec(
      unwrapped: Seq[Expression],
      childOutput: Seq[Attribute],
      fullSchema: StructType): Option[LtoExprSpec] = {
    val allEligible = unwrapped.forall {
      case g: GpuExpression =>
        g.supportsAstJit && g.dataType != StringType &&
          GpuAstJitExpression.isSupportedDataType(g.dataType)
      case _ => false
    }
    if (!allEligible) {
      return None
    }

    val refOrds = unwrapped.flatMap(_.collect {
      case ref: GpuBoundReference => ref.ordinal
    }).distinct.sorted
    if (refOrds.isEmpty || refOrds.size > 16) {
      return None
    }
    if (!refOrds.forall(ord => isSupportedLeafType(childOutput(ord).dataType))) {
      return None
    }

    val oldOrdToSlot = refOrds.zipWithIndex.toMap
    val fullNameToIdx = fullSchema.fieldNames.zipWithIndex.toMap
    val fileColIndices = refOrds.map(ord => fullNameToIdx(childOutput(ord).name)).toArray

    var compactExprs: Vector[Expression] = unwrapped.map { e =>
      e.transformDown {
        case ref: GpuBoundReference =>
          GpuBoundReference(oldOrdToSlot(ref.ordinal), ref.dataType, ref.nullable)(
            ExprId(ref.exprId.id), ref.name)
      }
    }.toVector

    val dictLiteralsPerSlot = Array.fill[Array[String]](refOrds.size)(Array.empty[String])
    refOrds.zipWithIndex.foreach { case (oldOrd, slot) =>
      if (childOutput(oldOrd).dataType == StringType) {
        val litSet = mutable.HashSet.empty[String]
        val ok = compactExprs.forall(GpuDictAwareAstJit.collectDictPredicates(_, slot, litSet)) &&
          litSet.nonEmpty
        if (!ok) {
          return None
        }
        val sortedLits = litSet.toArray.sorted
        val litToId = sortedLits.zipWithIndex.toMap
        dictLiteralsPerSlot(slot) = sortedLits
        compactExprs = compactExprs.map { e =>
          GpuDictAwareAstJit.rewriteExprForDict(e, slot, litToId, sortedLits.length)
        }
      }
    }
    Some(LtoExprSpec(fileColIndices, compactExprs, dictLiteralsPerSlot))
  }

  /**
   * Pattern 1: Fuse `GpuFilterExec(gpuCond, GpuBatchScanExec(GpuLencScan))` into
   * two-stage late-materialized `GpuLencLtoFilterScanExec`.
   */
  def tryFuseFilter(gpuCond: Expression, gpuChild: SparkPlan): Option[GpuExec] = {
    val enabled = SQLConf.get.getConfString(LTO_FUSION_CONF, "false").toBoolean
    if (!enabled) {
      return None
    }
    val lencScan = gpuChild match {
      case bs: GpuBatchScanExecBase =>
        bs.scan match {
          case s: GpuLencScan if s.fusedExpr.isEmpty => s
          case _ => return None
        }
      case _ => return None
    }
    val boundCond = GpuBindReferences.bindGpuReference(gpuCond, gpuChild.output, Map.empty)
    val filterSpecOpt = buildSpec(Seq(boundCond), gpuChild.output, lencScan.fullSchema)
    filterSpecOpt.map { filterSpec =>
      val fullNameToIdx = lencScan.fullSchema.fieldNames.zipWithIndex.toMap
      val outFileIndices = gpuChild.output.map(a => fullNameToIdx(a.name)).toArray
      val dict32Set = GpuDictionaryUtils.updateDriverDict32Cols(SQLConf.get)
      val outDict32Mask: Array[Byte] =
        if (dict32Set.isEmpty) Array.empty[Byte]
        else gpuChild.output.map(a =>
          if (a.dataType == StringType && dict32Set.contains(a.name)) 1.toByte else 0.toByte
        ).toArray
      GpuLencLtoFilterScanExec(
        gpuChild.output,
        lencScan.fullSchema,
        lencScan.files,
        filterSpec,
        outFileIndices,
        outDict32Mask)
    }
  }

  /**
   * Pattern 2 & 3:
   * - `GpuProjectExec(gpuExprs, GpuBatchScanExec(GpuLencScan))` -> `GpuLencLtoScanExec`
   * - `GpuProjectExec(gpuExprs, GpuLencLtoFilterScanExec)` -> `GpuLencLtoFilterProjectScanExec`
   */
  def tryFuse(gpuExprs: Seq[NamedExpression], gpuChild: SparkPlan): Option[GpuExec] = {
    val enabled = SQLConf.get.getConfString(LTO_FUSION_CONF, "false").toBoolean
    if (!enabled || gpuExprs.isEmpty) {
      return None
    }
    gpuChild match {
      case bs: GpuBatchScanExecBase =>
        bs.scan match {
          case s: GpuLencScan if s.fusedExpr.isEmpty =>
            val bound = GpuBindReferences.bindGpuReferences(gpuExprs, gpuChild.output, Map.empty)
            val unwrapped = bound.map(unwrapAlias)
            buildSpec(unwrapped, gpuChild.output, s.fullSchema).map { projSpec =>
              GpuLencLtoScanExec(gpuExprs.map(_.toAttribute), s.files, projSpec)
            }
          case _ => None
        }

      case fscan: GpuLencLtoFilterScanExec =>
        val bound = GpuBindReferences.bindGpuReferences(gpuExprs, gpuChild.output, Map.empty)
        val unwrapped = bound.map(unwrapAlias)
        buildSpec(unwrapped, gpuChild.output, fscan.fullSchema).map { projSpec =>
          GpuLencLtoFilterProjectScanExec(
            gpuExprs.map(_.toAttribute),
            fscan.files,
            fscan.filterSpec,
            projSpec)
        }

      case _ => None
    }
  }
}

private case class LencLtoPartition(index: Int, filePaths: Array[String]) extends Partition

/** Pattern 2 RDD: Project(Scan) with async 1-slot prefetch + dynamic LTO Decode-in-Register. */
private class GpuLencLtoRdd(
    sc: SparkContext,
    fileGroups: Array[Array[String]],
    projSpec: LtoExprSpec,
    outTypes: Array[DataType],
    metrics: Map[String, GpuMetric]) extends RDD[ColumnarBatch](sc, Nil) {

  override protected def getPartitions: Array[Partition] =
    fileGroups.zipWithIndex.map { case (fs, i) => LencLtoPartition(i, fs) }

  override def compute(split: Partition, context: TaskContext): Iterator[ColumnarBatch] = {
    val lp = split.asInstanceOf[LencLtoPartition]
    val opTime = metrics.getOrElse(OP_TIME_LEGACY, NoopMetric)
    val bufferTime = metrics.getOrElse(BUFFER_TIME, NoopMetric)
    val gpuDecodeTime = metrics.getOrElse(GPU_DECODE_TIME, NoopMetric)
    val numOutputRows = metrics.getOrElse(NUM_OUTPUT_ROWS, NoopMetric)
    val numOutputBatches = metrics.getOrElse(NUM_OUTPUT_BATCHES, NoopMetric)

    val prefetcher = new LencAsyncPrefetcher(lp.filePaths, projSpec.fileColIndices)
    if (context != null) {
      context.addTaskCompletionListener[Unit](_ => prefetcher.close())
    }

    new Iterator[ColumnarBatch] {
      private var nextBatch: ColumnarBatch = _

      private def advance(): Unit = {
        while (nextBatch == null) {
          val itemOpt = opTime.ns { bufferTime.ns { prefetcher.next() } }
          if (itemOpt.isEmpty) {
            prefetcher.close()
            return
          }
          val item = itemOpt.get
          opTime.ns {
            withResource(item.plan) { plan =>
              val encodings = plan.columnEncodings()
              val types = plan.columnTypes()
              withResource(item.hostBuf) { hb =>
                GpuSemaphore.acquireIfNecessary(TaskContext.get())
                val table = gpuDecodeTime.ns {
                  val poolKey = GpuLencLtoProgramPool.PoolKey(
                    projSpec.compactExprs, encodings.toVector, types.toVector)
                  val (tbl, _) = GpuLencLtoProgramPool.withProgram(
                    poolKey,
                    LencLtoFusion.buildLtoProgram(projSpec.compactExprs, encodings, types)) { prog =>
                    prog.run(plan, hb, projSpec.dictLiteralsPerSlot)
                  }
                  tbl
                }
                withResource(table) { t =>
                  numOutputRows += t.getRowCount
                  numOutputBatches += 1
                  nextBatch = GpuColumnVector.from(t, outTypes)
                }
              }
            }
          }
        }
      }

      override def hasNext: Boolean = {
        if (nextBatch == null) advance()
        nextBatch != null
      }

      override def next(): ColumnarBatch = {
        if (!hasNext) throw new NoSuchElementException
        val ret = nextBatch
        nextBatch = null
        ret
      }
    }
  }
}

/** Pattern 1 & 3 RDD: Two-stage late-materialized Filter(Scan) or Project(Filter(Scan)) via LTO. */
private class GpuLencLtoFilterRdd(
    sc: SparkContext,
    fileGroups: Array[Array[String]],
    filterSpec: LtoExprSpec,
    projSpecOpt: Option[LtoExprSpec],
    outFileIndices: Array[Int],
    outDict32Mask: Array[Byte],
    outTypes: Array[DataType],
    metrics: Map[String, GpuMetric]) extends RDD[ColumnarBatch](sc, Nil) {

  override protected def getPartitions: Array[Partition] =
    fileGroups.zipWithIndex.map { case (fs, i) => LencLtoPartition(i, fs) }

  override def compute(split: Partition, context: TaskContext): Iterator[ColumnarBatch] = {
    val lp = split.asInstanceOf[LencLtoPartition]
    val opTime = metrics.getOrElse(OP_TIME_LEGACY, NoopMetric)
    val bufferTime = metrics.getOrElse(BUFFER_TIME, NoopMetric)
    val gpuDecodeTime = metrics.getOrElse(GPU_DECODE_TIME, NoopMetric)
    val numOutputRows = metrics.getOrElse(NUM_OUTPUT_ROWS, NoopMetric)
    val numOutputBatches = metrics.getOrElse(NUM_OUTPUT_BATCHES, NoopMetric)

    // Prefetch Stage 1 (filter columns only!) asynchronously OFF the GPU semaphore
    val prefetcher = new LencAsyncPrefetcher(lp.filePaths, filterSpec.fileColIndices)
    if (context != null) {
      context.addTaskCompletionListener[Unit](_ => prefetcher.close())
    }

    val dict32Arr: Array[Byte] =
      if (outDict32Mask.nonEmpty && outDict32Mask.exists(_ != 0)) outDict32Mask else null

    new Iterator[ColumnarBatch] {
      private var nextBatch: ColumnarBatch = _

      private def advance(): Unit = {
        while (nextBatch == null) {
          val itemOpt = opTime.ns { bufferTime.ns { prefetcher.next() } }
          if (itemOpt.isEmpty) {
            prefetcher.close()
            return
          }
          val item = itemOpt.get
          opTime.ns {
            val selectedOpt = withResource(item.plan) { fPlan =>
              val fEncs = fPlan.columnEncodings()
              val fTypes = fPlan.columnTypes()
              withResource(item.hostBuf) { fHb =>
                GpuSemaphore.acquireIfNecessary(TaskContext.get())
                gpuDecodeTime.ns {
                  val fKey = GpuLencLtoProgramPool.PoolKey(
                    filterSpec.compactExprs, fEncs.toVector, fTypes.toVector)
                  val (maskTbl, _) = GpuLencLtoProgramPool.withProgram(
                    fKey,
                    LencLtoFusion.buildLtoProgram(filterSpec.compactExprs, fEncs, fTypes)) { prog =>
                    prog.run(fPlan, fHb, filterSpec.dictLiteralsPerSlot)
                  }
                  withResource(maskTbl) { mt =>
                    val sel = LencNative.compactMaskIndices(mt.getColumn(0))
                    if (sel.getRowCount == 0L) {
                      sel.close()
                      None
                    } else {
                      Some(sel)
                    }
                  }
                }
              }
            }

            // Stage 2: ONLY if >0 rows survived, read & late-materialize payload / LTO project!
            selectedOpt.foreach { selCol =>
              withResource(selCol) { sel =>
                projSpecOpt match {
                  case Some(pSpec) =>
                    withResource(LencNative.planRead(item.filePath, pSpec.fileColIndices)) { pPlan =>
                      val pBytes = pPlan.totalPackedBytes()
                      val pEncs = pPlan.columnEncodings()
                      val pTypes = pPlan.columnTypes()
                      val pBuf = bufferTime.ns {
                        val b = HostMemoryBuffer.allocate(math.max(pBytes, 64L))
                        closeOnExcept(b) { buf =>
                          if (pBytes > 0L) pPlan.readToHost(item.filePath, buf)
                        }
                        b
                      }
                      withResource(pBuf) { hb =>
                        val tbl = gpuDecodeTime.ns {
                          val pKey = GpuLencLtoProgramPool.PoolKey(
                            pSpec.compactExprs, pEncs.toVector, pTypes.toVector)
                          val (t, _) = GpuLencLtoProgramPool.withProgram(
                            pKey,
                            LencLtoFusion.buildLtoProgram(pSpec.compactExprs, pEncs, pTypes)) { prog =>
                            prog.runSelected(pPlan, hb, pSpec.dictLiteralsPerSlot, sel)
                          }
                          t
                        }
                        withResource(tbl) { t =>
                          numOutputRows += t.getRowCount
                          numOutputBatches += 1
                          nextBatch = GpuColumnVector.from(t, outTypes)
                        }
                      }
                    }

                  case None =>
                    withResource(LencNative.planRead(item.filePath, outFileIndices)) { pPlan =>
                      val pBytes = pPlan.totalPackedBytes()
                      val pBuf = bufferTime.ns {
                        val b = HostMemoryBuffer.allocate(math.max(pBytes, 64L))
                        closeOnExcept(b) { buf =>
                          if (pBytes > 0L) pPlan.readToHost(item.filePath, buf)
                        }
                        b
                      }
                      withResource(pBuf) { hb =>
                        val tbl = gpuDecodeTime.ns {
                          pPlan.decodeFromHostSelected(hb, dict32Arr, sel)
                        }
                        withResource(tbl) { t =>
                          numOutputRows += t.getRowCount
                          numOutputBatches += 1
                          nextBatch = GpuColumnVector.from(t, outTypes)
                        }
                      }
                    }
                }
              }
            }
          }
        }
      }

      override def hasNext: Boolean = {
        if (nextBatch == null) advance()
        nextBatch != null
      }

      override def next(): ColumnarBatch = {
        if (!hasNext) throw new NoSuchElementException
        val ret = nextBatch
        nextBatch = null
        ret
      }
    }
  }
}

case class GpuLencLtoScanExec(
    output: Seq[Attribute],
    files: Array[String],
    projSpec: LtoExprSpec) extends LeafExecNode with GpuExec {

  override def supportsColumnar: Boolean = true

  override protected def doExecute(): RDD[org.apache.spark.sql.catalyst.InternalRow] =
    throw new IllegalStateException("GpuLencLtoScanExec must execute columnar on GPU")

  override lazy val additionalMetrics: Map[String, GpuMetric] = Map(
    GPU_DECODE_TIME -> createNanoTimingMetric(MODERATE_LEVEL, DESCRIPTION_GPU_DECODE_TIME),
    BUFFER_TIME -> createNanoTimingMetric(MODERATE_LEVEL, DESCRIPTION_BUFFER_TIME),
    OP_TIME_LEGACY -> createNanoTimingMetric(DEBUG_LEVEL, DESCRIPTION_OP_TIME_LEGACY)
  )

  override def internalDoExecuteColumnar(): RDD[ColumnarBatch] = {
    val outTypes = output.map(_.dataType).toArray
    val groups = LencPartitionHelper.groupFiles(files)
    new GpuLencLtoRdd(sparkContext, groups, projSpec, outTypes, allMetrics)
  }

  override def simpleString(maxFields: Int): String =
    s"GpuLencLtoScan(files=${files.length}, inCols=${projSpec.fileColIndices.length}, outCols=${output.length})"
}

case class GpuLencLtoFilterScanExec(
    output: Seq[Attribute],
    fullSchema: StructType,
    files: Array[String],
    filterSpec: LtoExprSpec,
    outFileIndices: Array[Int],
    outDict32Mask: Array[Byte]) extends LeafExecNode with GpuExec {

  override def supportsColumnar: Boolean = true

  override protected def doExecute(): RDD[org.apache.spark.sql.catalyst.InternalRow] =
    throw new IllegalStateException("GpuLencLtoFilterScanExec must execute columnar on GPU")

  override lazy val additionalMetrics: Map[String, GpuMetric] = Map(
    GPU_DECODE_TIME -> createNanoTimingMetric(MODERATE_LEVEL, DESCRIPTION_GPU_DECODE_TIME),
    BUFFER_TIME -> createNanoTimingMetric(MODERATE_LEVEL, DESCRIPTION_BUFFER_TIME),
    OP_TIME_LEGACY -> createNanoTimingMetric(DEBUG_LEVEL, DESCRIPTION_OP_TIME_LEGACY)
  )

  override def internalDoExecuteColumnar(): RDD[ColumnarBatch] = {
    val outTypes = output.map(_.dataType).toArray
    val groups = LencPartitionHelper.groupFiles(files)
    new GpuLencLtoFilterRdd(
      sparkContext, groups, filterSpec, None, outFileIndices, outDict32Mask, outTypes, allMetrics)
  }

  override def simpleString(maxFields: Int): String =
    s"GpuLencLtoFilterScan(files=${files.length}, filterCols=${filterSpec.fileColIndices.length}, outCols=${output.length})"
}

case class GpuLencLtoFilterProjectScanExec(
    output: Seq[Attribute],
    files: Array[String],
    filterSpec: LtoExprSpec,
    projSpec: LtoExprSpec) extends LeafExecNode with GpuExec {

  override def supportsColumnar: Boolean = true

  override protected def doExecute(): RDD[org.apache.spark.sql.catalyst.InternalRow] =
    throw new IllegalStateException("GpuLencLtoFilterProjectScanExec must execute columnar on GPU")

  override lazy val additionalMetrics: Map[String, GpuMetric] = Map(
    GPU_DECODE_TIME -> createNanoTimingMetric(MODERATE_LEVEL, DESCRIPTION_GPU_DECODE_TIME),
    BUFFER_TIME -> createNanoTimingMetric(MODERATE_LEVEL, DESCRIPTION_BUFFER_TIME),
    OP_TIME_LEGACY -> createNanoTimingMetric(DEBUG_LEVEL, DESCRIPTION_OP_TIME_LEGACY)
  )

  override def internalDoExecuteColumnar(): RDD[ColumnarBatch] = {
    val outTypes = output.map(_.dataType).toArray
    val groups = LencPartitionHelper.groupFiles(files)
    new GpuLencLtoFilterRdd(
      sparkContext, groups, filterSpec, Some(projSpec), Array.emptyIntArray, Array.emptyByteArray,
      outTypes, allMetrics)
  }

  override def simpleString(maxFields: Int): String =
    s"GpuLencLtoFilterProjectScan(files=${files.length}, fCols=${filterSpec.fileColIndices.length}, pCols=${projSpec.fileColIndices.length}, outCols=${output.length})"
}
