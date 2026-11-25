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
 * (`cudf::ast::jit::call` to `lenc_decode_*`), aligned with NVIDIA/cudf#24869.
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

  override def toString: String = s"$symbol(slot=$slot)"
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

object LencLtoFusion {
  val LTO_FUSION_CONF = "spark.rapids.sql.lenc.ltoFusionEnabled"

  // Must match lenc.hpp Encoding & ColType enum values
  private val ENC_FOR_BITPACK: Byte = 0
  private val ENC_DICT_BITPACK: Byte = 1
  private val ENC_RLE_INT: Byte = 2
  private val ENC_RAW_FIXED: Byte = 3
  private val ENC_SCALED_FLOAT_FOR: Byte = 4

  private val TYPE_INT32: Byte = 0
  private val TYPE_INT64: Byte = 1
  private val TYPE_FLOAT32: Byte = 2
  private val TYPE_FLOAT64: Byte = 3
  private val TYPE_DATE32: Byte = 4
  private val TYPE_STRING: Byte = 5

  def decodeSymbolAndType(encoding: Byte, colType: Byte): (String, DataType, DType) = {
    (encoding, colType) match {
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
      case (ENC_RAW_FIXED, TYPE_FLOAT32) => ("lenc_decode_raw_f32", FloatType, DType.FLOAT32)
      case (ENC_RAW_FIXED, TYPE_FLOAT64) => ("lenc_decode_raw_f64", DoubleType, DType.FLOAT64)
      case (e, t) =>
        throw new UnsupportedOperationException(s"Unsupported .lenc LTO (encoding=$e, type=$t)")
    }
  }

  private def unwrapAlias(expr: Expression): Expression = expr match {
    case GpuAlias(child, _) => child
    case other => other
  }

  /**
   * Automatically fuse `GpuProjectExec(gpuExprs, GpuBatchScanExec(GpuLencScan))` into a single
   * dynamically LTO-linked Decode-in-Register + AST-JIT scan (`GpuLencLtoScanExec`).
   */
  def tryFuse(gpuExprs: Seq[NamedExpression], gpuChild: SparkPlan): Option[GpuExec] = {
    val enabled = SQLConf.get.getConfString(LTO_FUSION_CONF, "false").toBoolean
    if (!enabled || gpuExprs.isEmpty) {
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

    val bound = GpuBindReferences.bindGpuReferences(gpuExprs, gpuChild.output, Map.empty)
    val unwrapped = bound.map(unwrapAlias)

    // Verify all expressions are AST-JIT eligible and have supported non-string output types
    val allEligible = unwrapped.forall {
      case g: GpuExpression =>
        g.supportsAstJit && g.dataType != StringType &&
          GpuAstJitExpression.isSupportedDataType(g.dataType)
      case _ => false
    }
    if (!allEligible) {
      return None
    }

    // Collect distinct referenced input ordinals in gpuChild.output (max 16 slots)
    val refOrds = unwrapped.flatMap(_.collect {
      case ref: GpuBoundReference => ref.ordinal
    }).distinct.sorted
    if (refOrds.isEmpty || refOrds.size > 16) {
      return None
    }

    val oldOrdToSlot = refOrds.zipWithIndex.toMap
    val fullNameToIdx = lencScan.fullSchema.fieldNames.zipWithIndex.toMap
    val fileColIndices = refOrds.map(ord => fullNameToIdx(gpuChild.output(ord).name)).toArray

    // Rebind ordinals to compact slots [0 .. K-1]
    var compactExprs: Vector[Expression] = unwrapped.map { e =>
      e.transformDown {
        case ref: GpuBoundReference =>
          GpuBoundReference(oldOrdToSlot(ref.ordinal), ref.dataType, ref.nullable)(
            ExprId(ref.exprId.id), ref.name)
      }
    }.toVector

    // Rewrite StringType dictionary columns into canonical INT32 code comparisons
    val dictLiteralsPerSlot = Array.fill[Array[String]](refOrds.size)(Array.empty[String])
    refOrds.zipWithIndex.foreach { case (oldOrd, slot) =>
      if (gpuChild.output(oldOrd).dataType == StringType) {
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

    val outAttrs = gpuExprs.map(_.toAttribute)
    Some(GpuLencLtoScanExec(
      outAttrs,
      lencScan.files,
      fileColIndices,
      compactExprs,
      dictLiteralsPerSlot))
  }
}

private case class LencLtoPartition(index: Int, filePath: String) extends Partition

private class GpuLencLtoRdd(
    sc: SparkContext,
    files: Array[String],
    fileColIndices: Array[Int],
    compactExprs: Vector[Expression],
    dictLiteralsPerSlot: Array[Array[String]],
    outTypes: Array[DataType],
    metrics: Map[String, GpuMetric]) extends RDD[ColumnarBatch](sc, Nil) {

  override protected def getPartitions: Array[Partition] =
    files.zipWithIndex.map { case (f, i) => LencLtoPartition(i, f) }

  override def compute(split: Partition, context: TaskContext): Iterator[ColumnarBatch] = {
    val lp = split.asInstanceOf[LencLtoPartition]
    val opTime = metrics.getOrElse(OP_TIME_LEGACY, NoopMetric)
    val bufferTime = metrics.getOrElse(BUFFER_TIME, NoopMetric)
    val gpuDecodeTime = metrics.getOrElse(GPU_DECODE_TIME, NoopMetric)
    val numOutputRows = metrics.getOrElse(NUM_OUTPUT_ROWS, NoopMetric)
    val numOutputBatches = metrics.getOrElse(NUM_OUTPUT_BATCHES, NoopMetric)

    val batch = opTime.ns {
      withResource(LencNative.planRead(lp.filePath, fileColIndices)) { plan =>
        val totalBytes = plan.totalPackedBytes()
        val encodings = plan.columnEncodings()
        val types = plan.columnTypes()

        // Step 1: Read compressed .lenc columns into Spark pinned host pool OFF GPU semaphore
        val hostBuf = bufferTime.ns {
          val buf = HostMemoryBuffer.allocate(math.max(totalBytes, 64L))
          closeOnExcept(buf) { b =>
            if (totalBytes > 0L) {
              plan.readToHost(lp.filePath, b)
            }
          }
          buf
        }

        // Step 2: Acquire GPU semaphore & run ONE dynamically LTO-linked Decode-in-Register kernel!
        withResource(hostBuf) { hb =>
          GpuSemaphore.acquireIfNecessary(TaskContext.get())
          val table = gpuDecodeTime.ns {
            val poolKey = GpuLencLtoProgramPool.PoolKey(
              compactExprs, encodings.toVector, types.toVector)
            val (tbl, _) = GpuLencLtoProgramPool.withProgram(poolKey, {
              val leafInfo = encodings.indices.map { slot =>
                LencLtoFusion.decodeSymbolAndType(encodings(slot), types(slot))
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
            }) { prog =>
              prog.run(plan, hb, dictLiteralsPerSlot)
            }
            tbl
          }
          withResource(table) { t =>
            numOutputRows += t.getRowCount
            numOutputBatches += 1
            GpuColumnVector.from(t, outTypes)
          }
        }
      }
    }
    Iterator.single(batch)
  }
}

case class GpuLencLtoScanExec(
    output: Seq[Attribute],
    files: Array[String],
    fileColIndices: Array[Int],
    compactExprs: Vector[Expression],
    dictLiteralsPerSlot: Array[Array[String]]) extends LeafExecNode with GpuExec {

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
    new GpuLencLtoRdd(
      sparkContext,
      files,
      fileColIndices,
      compactExprs,
      dictLiteralsPerSlot,
      outTypes,
      allMetrics)
  }

  override def simpleString(maxFields: Int): String =
    s"GpuLencLtoScan(files=${files.length}, inCols=${fileColIndices.length}, outCols=${output.length})"
}
