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
import org.apache.spark.sql.rapids._
import org.apache.spark.sql.types._
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * Simple file/chunk-level range predicate extracted from conjunction filters for zero-reconversion
 * .lenc Min-Max ZoneMap pruning (Track 3).
 * op: 0 = LT, 1 = LE, 2 = GT, 3 = GE, 4 = EQ.
 */
final case class LencRangePred(
    planSlot: Int,
    op: Int,
    isFloat: Boolean,
    intBound: Long,
    doubleBound: Double)

object LencRangePruner {
  val OP_LT = 0
  val OP_LE = 1
  val OP_GT = 2
  val OP_GE = 3
  val OP_EQ = 4

  private def pow10(e: Int): Double = e match {
    case 0 => 1.0
    case 1 => 10.0
    case 2 => 100.0
    case 3 => 1000.0
    case 4 => 10000.0
    case k => math.pow(10.0, k)
  }

  def canSkipFile(plan: LencNative.ReadPlan, preds: Array[LencRangePred]): Boolean = {
    if (preds.isEmpty) return false
    val hasMM = plan.colHasMinMax()
    val mins = plan.colMins()
    val maxs = plan.colMaxs()
    val exps = plan.colScaleExp10()
    var i = 0
    while (i < preds.length) {
      val pr = preds(i)
      val s = pr.planSlot
      if (s >= 0 && s < hasMM.length && hasMM(s) != 0) {
        if (pr.isFloat) {
          val inv = 1.0 / pow10(exps(s))
          val fmin = mins(s).toDouble * inv
          val fmax = maxs(s).toDouble * inv
          val b = pr.doubleBound
          // Conservative epsilon for float scaling rounding
          val eps = 1e-6 * math.max(1.0, math.abs(b))
          val skip = pr.op match {
            case OP_LT => fmin >= b + eps
            case OP_LE => fmin > b + eps
            case OP_GT => fmax <= b - eps
            case OP_GE => fmax < b - eps
            case OP_EQ => (b < fmin - eps) || (b > fmax + eps)
            case _ => false
          }
          if (skip) return true
        } else {
          val lo = mins(s)
          val hi = maxs(s)
          val b = pr.intBound
          val skip = pr.op match {
            case OP_LT => lo >= b
            case OP_LE => lo > b
            case OP_GT => hi <= b
            case OP_GE => hi < b
            case OP_EQ => (b < lo) || (b > hi)
            case _ => false
          }
          if (skip) return true
        }
      }
      i += 1
    }
    false
  }

  private def literalToBound(lit: GpuLiteral): Option[(Boolean, Long, Double)] = {
    if (lit.value == null) return None
    lit.dataType match {
      case IntegerType | DateType => Some((false, lit.value.asInstanceOf[Int].toLong, 0.0))
      case LongType | TimestampType => Some((false, lit.value.asInstanceOf[Long], 0.0))
      case ShortType => Some((false, lit.value.asInstanceOf[Short].toLong, 0.0))
      case ByteType => Some((false, lit.value.asInstanceOf[Byte].toLong, 0.0))
      case FloatType =>
        val v = lit.value.asInstanceOf[Float].toDouble
        if (v.isNaN) None else Some((true, 0L, v))
      case DoubleType =>
        val v = lit.value.asInstanceOf[Double]
        if (v.isNaN) None else Some((true, 0L, v))
      case _ => None
    }
  }

  /** Extract conjunctive range predicates mapped to slots in `planFileColIndices`. */
  def extract(
      filterExpr: Expression,
      filterFileColIndices: Array[Int],
      planFileColIndices: Array[Int]): Array[LencRangePred] = {
    val fileColToPlanSlot = planFileColIndices.zipWithIndex.toMap
    val out = mutable.ArrayBuffer.empty[LencRangePred]

    def splitAnd(e: Expression): Seq[Expression] = e match {
      case GpuAnd(l, r) => splitAnd(l) ++ splitAnd(r)
      case other => Seq(other)
    }

    def addCmp(ref: GpuBoundReference, lit: GpuLiteral, op: Int): Unit = {
      if (ref.ordinal >= 0 && ref.ordinal < filterFileColIndices.length) {
        val fci = filterFileColIndices(ref.ordinal)
        fileColToPlanSlot.get(fci).foreach { pslot =>
          literalToBound(lit).foreach { case (isF, ib, db) =>
            out += LencRangePred(pslot, op, isF, ib, db)
          }
        }
      }
    }

    splitAnd(filterExpr).foreach {
      case GpuLessThan(ref: GpuBoundReference, lit: GpuLiteral) => addCmp(ref, lit, OP_LT)
      case GpuLessThan(lit: GpuLiteral, ref: GpuBoundReference) => addCmp(ref, lit, OP_GT)
      case GpuLessThanOrEqual(ref: GpuBoundReference, lit: GpuLiteral) => addCmp(ref, lit, OP_LE)
      case GpuLessThanOrEqual(lit: GpuLiteral, ref: GpuBoundReference) => addCmp(ref, lit, OP_GE)
      case GpuGreaterThan(ref: GpuBoundReference, lit: GpuLiteral) => addCmp(ref, lit, OP_GT)
      case GpuGreaterThan(lit: GpuLiteral, ref: GpuBoundReference) => addCmp(ref, lit, OP_LT)
      case GpuGreaterThanOrEqual(ref: GpuBoundReference, lit: GpuLiteral) => addCmp(ref, lit, OP_GE)
      case GpuGreaterThanOrEqual(lit: GpuLiteral, ref: GpuBoundReference) => addCmp(ref, lit, OP_LE)
      case GpuEqualTo(ref: GpuBoundReference, lit: GpuLiteral) => addCmp(ref, lit, OP_EQ)
      case GpuEqualTo(lit: GpuLiteral, ref: GpuBoundReference) => addCmp(ref, lit, OP_EQ)
      case _ =>
    }
    out.toArray
  }
}

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

/**
 * Track 3 Unified Adaptive Filter+Project / Filter+Decode LTO RDD:
 * - Zero-reconversion file Min-Max ZoneMap pruning in LencAsyncPrefetcher!
 * - Mode A (Single-H2D Unified LTO): when `projSpecOpt.isDefined` OR payload has NO String
 *   columns (e.g. TPC-H Q6/Q14/Q19), prefetches union(filterCols, projCols) ONCE off-semaphore
 *   and executes BOTH filter LTO + selected-row project/decode on ONE single H2D buffer!
 * - Mode B (Two-Stage IO-Skipping LTO): when payload includes heavy String columns (e.g. NF_J0202),
 *   prefetches ONLY filter cols, skips Stage 2 disk read if 0 rows match, and releases the GPU
 *   semaphore during Stage 2 disk read!
 */
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

    val hasHeavyStringPayload = projSpecOpt.isEmpty && outTypes.contains(StringType)
    val useSingleH2D = !hasHeavyStringPayload

    val dict32Arr: Array[Byte] =
      if (outDict32Mask.nonEmpty && outDict32Mask.exists(_ != 0)) outDict32Mask else null

    if (useSingleH2D) {
      val stage2FileCols = projSpecOpt.map(_.fileColIndices).getOrElse(outFileIndices)
      val unionCols = (filterSpec.fileColIndices ++ stage2FileCols).distinct.sorted
      val colToUnionSlot = unionCols.zipWithIndex.toMap
      val filterCmap = filterSpec.fileColIndices.map(colToUnionSlot)
      val projCmap = stage2FileCols.map(colToUnionSlot)
      val rangePreds = LencRangePruner.extract(
        filterSpec.compactExprs.head, filterSpec.fileColIndices, unionCols)

      val prefetcher = new LencAsyncPrefetcher(lp.filePaths, unionCols, rangePreds)
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
              withResource(item.plan) { uPlan =>
                val uEncs = uPlan.columnEncodings()
                val uTypes = uPlan.columnTypes()
                val fEncs = filterCmap.map(uEncs)
                val fTypes = filterCmap.map(uTypes)
                withResource(item.hostBuf) { hb =>
                  GpuSemaphore.acquireIfNecessary(TaskContext.get())
                  val tblOpt = gpuDecodeTime.ns {
                    val fKey = GpuLencLtoProgramPool.PoolKey(
                      filterSpec.compactExprs, fEncs.toVector, fTypes.toVector)
                    val (res, _) = GpuLencLtoProgramPool.withProgram(
                      fKey,
                      LencLtoFusion.buildLtoProgram(filterSpec.compactExprs, fEncs, fTypes)) { fProg =>
                      projSpecOpt match {
                        case Some(pSpec) =>
                          val pEncs = projCmap.map(uEncs)
                          val pTypes = projCmap.map(uTypes)
                          val pKey = GpuLencLtoProgramPool.PoolKey(
                            pSpec.compactExprs, pEncs.toVector, pTypes.toVector)
                          val (t, _) = GpuLencLtoProgramPool.withProgram(
                            pKey,
                            LencLtoFusion.buildLtoProgram(pSpec.compactExprs, pEncs, pTypes)) { pProg =>
                            LencNative.runLtoFilterAndProject(
                              fProg, pProg, uPlan, hb,
                              filterCmap, filterSpec.dictLiteralsPerSlot,
                              projCmap, pSpec.dictLiteralsPerSlot, dict32Arr)
                          }
                          Option(t)
                        case None =>
                          Option(LencNative.runLtoFilterAndProject(
                            fProg, null, uPlan, hb,
                            filterCmap, filterSpec.dictLiteralsPerSlot,
                            projCmap, Array.empty[Array[String]], dict32Arr))
                      }
                    }
                    res
                  }
                  tblOpt.foreach { tbl =>
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
    } else {
      // Mode B: Two-stage IO-skipping for heavy String payloads (e.g. NF_J0202)
      val rangePreds = LencRangePruner.extract(
        filterSpec.compactExprs.head, filterSpec.fileColIndices, filterSpec.fileColIndices)
      val prefetcher = new LencAsyncPrefetcher(lp.filePaths, filterSpec.fileColIndices, rangePreds)
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

              selectedOpt.foreach { selCol =>
                withResource(selCol) { sel =>
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
