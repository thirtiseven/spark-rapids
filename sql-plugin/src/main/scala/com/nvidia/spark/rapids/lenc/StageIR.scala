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

import java.nio.charset.StandardCharsets

import scala.collection.mutable

import com.nvidia.spark.rapids._

import org.apache.spark.sql.catalyst.expressions.{Attribute, ExprId, Expression, Literal}
import org.apache.spark.sql.rapids.{GpuAdd, GpuAnd, GpuContains, GpuDivide, GpuEqualTo, GpuGreaterThan, GpuGreaterThanOrEqual, GpuLessThan, GpuLessThanOrEqual, GpuMultiply, GpuNot, GpuOr, GpuStartsWith, GpuSubtract}
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.types.UTF8String

/**
 * Clean Produce/Consume GPU Pipeline IR targeting libcudf's `cudf::whole_stage` NVRTC runtime.
 *
 * Architecture:
 *   [1 StageSource (Raw / Vortex Compressed / Hive CONSTANT / LSHF)]
 *     --> [0..N StageOp (Filter / Project / HashProbe(BHJ|SHJ) / Expand(ROLLUP))]
 *     --> [1 StageSink (HashAggSink / MaterializeSink / ShufflePartitionSink)]
 */
object StageIR {

  sealed trait StageJoinMode
  case object JoinInner extends StageJoinMode
  case object JoinLeftSemi extends StageJoinMode
  case object JoinLeftAnti extends StageJoinMode

  /** Variable binding inside a Produce/Consume pipeline's register environment. */
  sealed trait ColBinding {
    def dataType: DataType
  }
  final case class FactCol(slot: Int, dataType: DataType) extends ColBinding
  final case class DimCol(dimIdx: Int, slot: Int, dataType: DataType) extends ColBinding
  final case class RegVal(valVar: String, validVar: String, dataType: DataType) extends ColBinding

  type VarEnv = Map[ExprId, ColBinding]

  final class CodegenCtx(val isAggKernel: Boolean, val hasMultiMatch: Boolean) {
    private var idSeq = 0
    def nextId(): Int = {
      idSeq += 1
      idSeq
    }
    var failed: Boolean = false
    var usesFsst: Boolean = false
    val constBits: mutable.ArrayBuffer[Long] = mutable.ArrayBuffer.empty[Long]
    private val constIndexMap = mutable.HashMap.empty[Long, Int]
    def allocConst(bits: Long): Option[Int] = {
      constIndexMap.get(bits) match {
        case Some(idx) => Some(idx)
        case None if constBits.length < 64 =>
          val idx = constBits.length
          constBits += bits
          constIndexMap.put(bits, idx)
          Some(idx)
        case None => None
      }
    }
  }

  /** Composable register-level pipeline operator (Neumann Produce/Consume). */
  sealed trait StageOp {
    def emitConsume(
        ctx: CodegenCtx,
        env: VarEnv,
        indent: String,
        sb: StringBuilder,
        next: (VarEnv, String) => Unit): Unit
  }

  final case class FilterOp(pred: Expression) extends StageOp {
    override def emitConsume(
        ctx: CodegenCtx,
        env: VarEnv,
        indent: String,
        sb: StringBuilder,
        next: (VarEnv, String) => Unit): Unit = {
      val st = mutable.ArrayBuffer.empty[String]
      StageExprEmitter.emit(pred, env, st, ctx) match {
        case Some((fv, fok)) =>
          st.foreach(s => sb.append(s"$indent$s\n"))
          sb.append(s"${indent}if (($fok) && ($fv)) {\n")
          next(env, indent + "  ")
          sb.append(s"$indent}\n")
        case None =>
          ctx.failed = true
      }
    }
  }

  final case class ProjectOp(bindings: Seq[(ExprId, Expression)]) extends StageOp {
    override def emitConsume(
        ctx: CodegenCtx,
        env: VarEnv,
        indent: String,
        sb: StringBuilder,
        next: (VarEnv, String) => Unit): Unit = {
      var nextEnv = env
      bindings.foreach { case (eid, expr) =>
        val st = mutable.ArrayBuffer.empty[String]
        StageExprEmitter.emit(expr, env, st, ctx) match {
          case Some((v, ok)) =>
            st.foreach(s => sb.append(s"$indent$s\n"))
            nextEnv = nextEnv + (eid -> RegVal(v, ok, expr.dataType))
          case None =>
            ctx.failed = true
            return
        }
      }
      next(nextEnv, indent)
    }
  }

  /**
   * Hash probe operator shared uniformly by BroadcastHashJoin AND ShuffledHashJoin!
   * Supports Inner (1:N loop), LeftSemi (existence check), and LeftAnti (non-existence check).
   */
  final case class HashProbeOp(
      probeIdx: Int,
      keyExpr: Expression,
      dimBindings: Seq[(ExprId, Int, DataType)],
      mode: StageJoinMode = JoinInner) extends StageOp {
    override def emitConsume(
        ctx: CodegenCtx,
        env: VarEnv,
        indent: String,
        sb: StringBuilder,
        next: (VarEnv, String) => Unit): Unit = {
      val st = mutable.ArrayBuffer.empty[String]
      val keyRes = StageExprEmitter.emit(keyExpr, env, st, ctx)
      if (keyRes.isEmpty) { ctx.failed = true; return }
      val (kv, kok) = keyRes.get
      st.foreach(s => sb.append(s"$indent$s\n"))

      mode match {
        case JoinInner =>
          sb.append(s"${indent}if ($kok) {\n")
          val ind1 = indent + "  "
          sb.append(s"${ind1}int64_t pk$probeIdx = (int64_t)($kv);\n")
          val ovfCond = if (ctx.isAggKernel && ctx.hasMultiMatch) " && !row_overflow" else ""
          sb.append(s"${ind1}for (int32_t d$probeIdx = ws_probe_first(a->dims[$probeIdx], pk$probeIdx); d$probeIdx >= 0$ovfCond; d$probeIdx = ws_probe_next(a->dims[$probeIdx], d$probeIdx)) {\n")
          var nextEnv = env
          dimBindings.foreach { case (eid, dimSlot, dt) =>
            nextEnv = nextEnv + (eid -> DimCol(probeIdx, dimSlot, dt))
          }
          next(nextEnv, ind1 + "  ")
          sb.append(s"$ind1}\n")
          sb.append(s"$indent}\n")

        case JoinLeftSemi =>
          sb.append(s"${indent}if (($kok) && ws_probe_first(a->dims[$probeIdx], (int64_t)($kv)) >= 0) {\n")
          next(env, indent + "  ")
          sb.append(s"$indent}\n")

        case JoinLeftAnti =>
          // Spark LeftAnti keeps left rows with NULL key or non-matching key (unless null-aware anti)
          sb.append(s"${indent}if (!($kok) || ws_probe_first(a->dims[$probeIdx], (int64_t)($kv)) < 0) {\n")
          next(env, indent + "  ")
          sb.append(s"$indent}\n")
      }
    }
  }

  /**
   * ExpandOp for ROLLUP / CUBE / GROUPING SETS (e.g. TPC-DS Q67 9x row expansion!).
   * Eliminates GpuExpandExec's N-times HBM materialization completely by emitting an inner loop
   * over `eid = 0..branches.length - 1` right in GPU registers!
   * Supports both fixed-width columns AND broadcast-dim String dict columns (encoded as int32 codes
   * in `outNativeTypes` where rolled-up NULL branches emit `(int32_t)0, valid=false`).
   */
  final case class ExpandOp(
      outAttrs: Seq[Attribute],
      outNativeTypes: Seq[DataType],
      branches: Seq[Seq[Expression]]) extends StageOp {
    override def emitConsume(
        ctx: CodegenCtx,
        env: VarEnv,
        indent: String,
        sb: StringBuilder,
        next: (VarEnv, String) => Unit): Unit = {
      val id = ctx.nextId()
      val nBr = branches.length
      val nOut = outAttrs.length
      val ovfCond = if (ctx.isAggKernel && ctx.hasMultiMatch) " && !row_overflow" else ""
      val outVars = new Array[(String, String, DataType)](nOut)
      for (j <- 0 until nOut) {
        val dt = outNativeTypes(j)
        val ct = StageExprEmitter.cType(dt).getOrElse { ctx.failed = true; return }
        val vn = s"exv_${id}_$j"
        val ok = s"exok_${id}_$j"
        outVars(j) = (vn, ok, dt)
        sb.append(s"$indent$ct $vn = ($ct)0; bool $ok = false;\n")
      }
      sb.append(s"${indent}for (int32_t eid_$id = 0; eid_$id < $nBr$ovfCond; ++eid_$id) {\n")
      val ind1 = indent + "  "
      sb.append(s"${ind1}switch (eid_$id) {\n")
      branches.zipWithIndex.foreach { case (exprs, bIdx) =>
        sb.append(s"${ind1}  case $bIdx: {\n")
        val ind2 = ind1 + "    "
        for (j <- 0 until nOut) {
          val (vn, ok, dt) = outVars(j)
          exprs(j) match {
            case GpuLiteral(null, _) | Literal(null, _) =>
              val ct = StageExprEmitter.cType(dt).get
              sb.append(s"$ind2$vn = ($ct)0; $ok = false;\n")
            case other =>
              val st = mutable.ArrayBuffer.empty[String]
              StageExprEmitter.emit(other, env, st, ctx) match {
                case Some((ev, eok)) =>
                  st.foreach(s => sb.append(s"$ind2$s\n"))
                  sb.append(s"$ind2$vn = $ev; $ok = $eok;\n")
                case None =>
                  ctx.failed = true
                  return
              }
          }
        }
        sb.append(s"${ind2}break;\n")
        sb.append(s"${ind1}  }\n")
      }
      sb.append(s"$ind1}\n")
      var nextEnv = env
      for (j <- 0 until nOut) {
        val (vn, ok, dt) = outVars(j)
        nextEnv = nextEnv + (outAttrs(j).exprId -> RegVal(vn, ok, dt))
      }
      next(nextEnv, ind1)
      sb.append(s"$indent}\n")
    }
  }

  /** Terminal pipeline sinks. */
  sealed trait StageSink

  final case class HashAggSink(
      groupKeys: Seq[Expression],
      groupKeyDimStrSlot: Array[Int],
      groupKeyDimIdx: Array[Int],
      aggTerms: Seq[GenericAggTerm],
      useSmemHotCache: Boolean = false) extends StageSink

  final case class MaterializeSink(outExprs: Seq[Expression]) extends StageSink

  final case class ShufflePartitionSink(
      partExprs: Seq[Expression],
      outExprs: Seq[Expression]) extends StageSink

  /** Compiles a Produce/Consume pipeline into NVRTC CUDA kernel body source(s). */
  object StageCompiler {

    private def smemCacheKind(sink: HashAggSink): Int = {
      // 0 = none, 1 = I64 sum, 2 = I64 count, 3 = F64 sum
      if (!sink.useSmemHotCache || sink.aggTerms.length != 1) 0
      else sink.aggTerms.head.kind match {
        case GenericJitReduceCodegen.AGG_SUM_I64 => 1
        case GenericJitReduceCodegen.AGG_COUNT_I64 => 2
        case GenericJitReduceCodegen.AGG_SUM_F64 => 3
        case _ => 0
      }
    }

    private def emitAggUpdateBody(
        sink: HashAggSink,
        smemMode: Int,
        ctx: CodegenCtx,
        env: VarEnv,
        indent: String,
        sb: StringBuilder): Unit = {
      if (ctx.hasMultiMatch) {
        sb.append(s"${indent}if (match_idx < prog_cnt) { ++match_idx; continue; }\n")
      }
      val nk = sink.groupKeys.length
      if (nk == 0) {
        sb.append(s"${indent}uint64_t dummy_k[1] = {0ULL};\n")
        sb.append(s"${indent}int32_t slot = ws_hash_agg_upsert_n(a, dummy_k, 0u);\n")
      } else {
        sb.append(s"${indent}uint64_t gkeys[$nk]; uint16_t nbits = 0u;\n")
        sink.groupKeys.zipWithIndex.foreach { case (ge, gi) =>
          val regBound = ge match {
            case attr: Attribute => env.get(attr.exprId).exists(_.isInstanceOf[RegVal])
            case _ => false
          }
          if (!regBound && sink.groupKeyDimStrSlot(gi) >= 0) {
            val dSlot = sink.groupKeyDimStrSlot(gi)
            val dIdx = sink.groupKeyDimIdx(gi)
            sb.append(s"${indent}{\n")
            sb.append(s"$indent  bool gok = ws_valid(a->dim_cols[$dSlot], d$dIdx);\n")
            sb.append(s"$indent  if (gok) { gkeys[$gi] = (uint64_t)(int64_t)ws_load_i32(a->dim_cols[$dSlot], d$dIdx); } else { gkeys[$gi] = 0ULL; nbits |= (uint16_t)(1u << $gi); }\n")
            sb.append(s"${indent}}\n")
          } else {
            val st = mutable.ArrayBuffer.empty[String]
            StageExprEmitter.emit(ge, env, st, ctx) match {
              case Some((gv, gok)) =>
                sb.append(s"${indent}{\n")
                st.foreach(s => sb.append(s"$indent  $s\n"))
                sb.append(s"$indent  if ($gok) { gkeys[$gi] = (uint64_t)(int64_t)($gv); } else { gkeys[$gi] = 0ULL; nbits |= (uint16_t)(1u << $gi); }\n")
                sb.append(s"${indent}}\n")
              case None =>
                ctx.failed = true
                return
            }
          }
        }
        sb.append(s"${indent}int32_t slot = ws_hash_agg_upsert_n(a, gkeys, nbits);\n")
      }
      sb.append(s"${indent}if (slot < 0) { a->row_progress[row] = match_idx; row_overflow = true; break; }\n")
      sb.append(s"${indent}uint32_t base_off = (uint32_t)slot * a->num_aggs;\n")

      sink.aggTerms.zipWithIndex.foreach { case (term, ai) =>
        term.inputExpr match {
          case None =>
            if (smemMode == 2 && ai == 0) {
              sb.append(s"${indent}ws_smem_acc_i64(sm_hot, a, slot, 0u, 0LL, false);\n")
            } else {
              sb.append(s"${indent}atomicAdd(&a->ht_cnt[base_off + ${ai}u], 1ULL);\n")
            }
          case Some(ie) =>
            val st = mutable.ArrayBuffer.empty[String]
            StageExprEmitter.emit(ie, env, st, ctx) match {
              case Some((vv, vok)) =>
                sb.append(s"${indent}{\n")
                st.foreach(s => sb.append(s"$indent  $s\n"))
                sb.append(s"$indent  if ($vok) {\n")
                term.kind match {
                  case GenericJitReduceCodegen.AGG_SUM_F64 =>
                    sb.append(s"$indent    double dv = (double)($vv);\n")
                    if (smemMode == 3 && ai == 0) {
                      sb.append(s"$indent    if (!isnan(dv)) { ws_smem_acc_f64(sm_hot, a, slot, 0u, dv); }\n")
                    } else {
                      sb.append(s"$indent    if (!isnan(dv)) { atomicAdd(&a->ht_f64[base_off + ${ai}u], dv); atomicAdd(&a->ht_cnt[base_off + ${ai}u], 1ULL); }\n")
                    }
                  case GenericJitReduceCodegen.AGG_SUM_I64 =>
                    if (smemMode == 1 && ai == 0) {
                      sb.append(s"$indent    ws_smem_acc_i64(sm_hot, a, slot, 0u, (int64_t)($vv), true);\n")
                    } else {
                      sb.append(s"$indent    atomicAdd((unsigned long long*)&a->ht_i64[base_off + ${ai}u], (unsigned long long)(int64_t)($vv));\n")
                      sb.append(s"$indent    atomicAdd(&a->ht_cnt[base_off + ${ai}u], 1ULL);\n")
                    }
                  case GenericJitReduceCodegen.AGG_COUNT_I64 =>
                    if (smemMode == 2 && ai == 0) {
                      sb.append(s"$indent    ws_smem_acc_i64(sm_hot, a, slot, 0u, 0LL, false);\n")
                    } else {
                      sb.append(s"$indent    atomicAdd(&a->ht_cnt[base_off + ${ai}u], 1ULL);\n")
                    }
                  case GenericJitReduceCodegen.AGG_MIN_F64 =>
                    sb.append(s"$indent    double dv = (double)($vv); if (!isnan(dv)) { ws_atomic_min_f64(&a->ht_f64[base_off + ${ai}u], dv); atomicAdd(&a->ht_cnt[base_off + ${ai}u], 1ULL); }\n")
                  case GenericJitReduceCodegen.AGG_MAX_F64 =>
                    sb.append(s"$indent    double dv = (double)($vv); if (!isnan(dv)) { ws_atomic_max_f64(&a->ht_f64[base_off + ${ai}u], dv); atomicAdd(&a->ht_cnt[base_off + ${ai}u], 1ULL); }\n")
                  case GenericJitReduceCodegen.AGG_MIN_I64 =>
                    sb.append(s"$indent    ws_atomic_min_i64(&a->ht_i64[base_off + ${ai}u], (int64_t)($vv)); atomicAdd(&a->ht_cnt[base_off + ${ai}u], 1ULL);\n")
                  case GenericJitReduceCodegen.AGG_MAX_I64 =>
                    sb.append(s"$indent    ws_atomic_max_i64(&a->ht_i64[base_off + ${ai}u], (int64_t)($vv)); atomicAdd(&a->ht_cnt[base_off + ${ai}u], 1ULL);\n")
                  case GenericJitReduceCodegen.AGG_SUM_DEC128 =>
                    sb.append(s"$indent    ws_atomic_add_i128(a, base_off + ${ai}u, (int64_t)($vv));\n")
                    sb.append(s"$indent    atomicAdd(&a->ht_cnt[base_off + ${ai}u], 1ULL);\n")
                }
                sb.append(s"$indent  }\n")
                sb.append(s"${indent}}\n")
              case None =>
                ctx.failed = true
                return
            }
        }
      }
      if (ctx.hasMultiMatch) {
        sb.append(s"${indent}++match_idx;\n")
      }
    }

    private def chainOps(
        ops: List[StageOp],
        ctx: CodegenCtx,
        env: VarEnv,
        indent: String,
        sb: StringBuilder,
        sinkFn: (VarEnv, String) => Unit): Unit = ops match {
      case Nil =>
        sinkFn(env, indent)
      case op :: rest =>
        op.emitConsume(ctx, env, indent, sb, (nextEnv, nextIndent) =>
          chainOps(rest, ctx, nextEnv, nextIndent, sb, sinkFn))
    }

    def compileAgg(
        initialEnv: VarEnv,
        ops: Seq[StageOp],
        sink: HashAggSink,
        extraPreambleFlags: Int = 0): Option[(String, String, Array[Long], Int)] = {
      val kname = "ws_stage_agg"
      val hasMultiMatch = ops.exists {
        case HashProbeOp(_, _, _, JoinInner) => true
        case _: ExpandOp => true
        case _ => false
      }
      val smMode = smemCacheKind(sink)
      val ctx = new CodegenCtx(isAggKernel = true, hasMultiMatch = hasMultiMatch)
      val sb = new StringBuilder()
      sb.append("extern \"C\" __global__ void " + kname + "(const WsStageArgs* a) {\n")
      if (smMode == 1 || smMode == 2) {
        sb.append("  __shared__ WsSmemHotCacheI64 sm_hot;\n")
        sb.append("  ws_smem_init_i64(sm_hot);\n")
      } else if (smMode == 3) {
        sb.append("  __shared__ WsSmemHotCacheF64 sm_hot;\n")
        sb.append("  ws_smem_init_f64(sm_hot);\n")
      }
      sb.append("  uint32_t n = a->num_rows;\n")
      sb.append("  for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < n; row += gridDim.x * blockDim.x) {\n")
      sb.append("    uint32_t prog_cnt = a->row_progress[row];\n")
      sb.append("    if (prog_cnt == 0xFFFFFFFFu) continue;\n")
      sb.append("    uint32_t match_idx = 0u;\n")
      sb.append("    bool row_overflow = false;\n")
      chainOps(ops.toList, ctx, initialEnv, "    ", sb, (leafEnv, leafIndent) =>
        emitAggUpdateBody(sink, smMode, ctx, leafEnv, leafIndent, sb))
      if (ctx.failed) return None
      sb.append("    if (!row_overflow) { a->row_progress[row] = 0xFFFFFFFFu; }\n")
      sb.append("  }\n")
      if (smMode == 1) {
        sb.append("  ws_smem_flush_i64(sm_hot, a, 0u, true);\n")
      } else if (smMode == 2) {
        sb.append("  ws_smem_flush_i64(sm_hot, a, 0u, false);\n")
      } else if (smMode == 3) {
        sb.append("  ws_smem_flush_f64(sm_hot, a, 0u);\n")
      }
      sb.append("}\n")
      var pFlags = extraPreambleFlags
      if (ctx.usesFsst) pFlags |= com.nvidia.spark.rapids.jni.WholeStageNative.PREAMBLE_FSST
      if (smMode > 0) pFlags |= com.nvidia.spark.rapids.jni.WholeStageNative.PREAMBLE_SMEM
      Some((sb.toString(), kname, ctx.constBits.toArray, pFlags))
    }

    private def emitStoreCol(
        oe: Expression,
        ci: Int,
        ctx: CodegenCtx,
        env: VarEnv,
        indent: String,
        sb: StringBuilder): Unit = {
      val st = mutable.ArrayBuffer.empty[String]
      StageExprEmitter.emit(oe, env, st, ctx) match {
        case Some((ov, ook)) =>
          sb.append(s"${indent}{\n")
          st.foreach(s => sb.append(s"$indent  $s\n"))
          val storeStmt = oe.dataType match {
            case BooleanType | ByteType => s"((int8_t*)a->out_data[$ci])[dst] = (int8_t)($ov);"
            case ShortType => s"((int16_t*)a->out_data[$ci])[dst] = (int16_t)($ov);"
            case IntegerType | DateType => s"((int32_t*)a->out_data[$ci])[dst] = (int32_t)($ov);"
            case LongType => s"((int64_t*)a->out_data[$ci])[dst] = (int64_t)($ov);"
            case FloatType => s"((float*)a->out_data[$ci])[dst] = (float)($ov);"
            case DoubleType => s"((double*)a->out_data[$ci])[dst] = (double)($ov);"
            case d: DecimalType if d.precision <= 9 => s"((int32_t*)a->out_data[$ci])[dst] = (int32_t)($ov);"
            case d: DecimalType if d.precision <= 18 => s"((int64_t*)a->out_data[$ci])[dst] = (int64_t)($ov);"
            case _ => ctx.failed = true; return
          }
          sb.append(s"$indent  if ($ook) { $storeStmt atomicOr(&a->out_masks[$ci][dst >> 5], 1u << (dst & 31u)); }\n")
          sb.append(s"${indent}}\n")
        case None =>
          ctx.failed = true
      }
    }

    def compileTwoPass(
        initialEnv: VarEnv,
        ops: Seq[StageOp],
        partExprsOpt: Option[Seq[Expression]],
        outExprs: Seq[Expression]): Option[(String, String, Array[Long], Int)] = {
      val sharedConstCtx = new CodegenCtx(isAggKernel = false, hasMultiMatch = false)
      def emitOne(isScatter: Boolean): Option[(String, Boolean)] = {
        val kname = if (isScatter) "ws_part_scatter" else "ws_part_count"
        val isMat = partExprsOpt.isEmpty
        val ctx = sharedConstCtx
        ctx.failed = false
        val sb = new StringBuilder()
        sb.append("extern \"C\" __global__ void " + kname + "(const WsPartScatterArgs* a) {\n")
        if (isMat && !isScatter) {
          sb.append("  uint32_t local_cnt = 0u;\n")
        } else if (!isMat && !isScatter) {
          // Stage 4: Block-Private 256-bin Shared-Mem Histogram (~1 KiB SRAM) eliminating global atomic contention!
          sb.append("  __shared__ uint32_t s_hist[256];\n")
          sb.append("  bool use_smem_hist = (a->num_parts <= 256u);\n")
          sb.append("  if (use_smem_hist) {\n")
          sb.append("    for (uint32_t i = threadIdx.x; i < a->num_parts; i += blockDim.x) s_hist[i] = 0u;\n")
          sb.append("    __syncthreads();\n")
          sb.append("  }\n")
        }
        sb.append("  uint32_t n = a->num_rows;\n")
        sb.append("  for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < n; row += gridDim.x * blockDim.x) {\n")
        chainOps(ops.toList, ctx, initialEnv, "    ", sb, (leafEnv, leafIndent) => {
          val pidExpr = partExprsOpt match {
            case None => "0"
            case Some(partExprs) =>
              sb.append(s"${leafIndent}uint32_t h = 42u;\n")
              partExprs.foreach { pe =>
                if (!ctx.failed) {
                  val st = mutable.ArrayBuffer.empty[String]
                  StageExprEmitter.emit(pe, leafEnv, st, ctx) match {
                    case Some((pv, pok)) =>
                      st.foreach(s => sb.append(s"$leafIndent$s\n"))
                      if (pe.dataType == LongType) {
                        sb.append(s"${leafIndent}if ($pok) h = ws_spark_murmur3_i64((int64_t)($pv), h);\n")
                      } else {
                        sb.append(s"${leafIndent}if ($pok) h = ws_spark_murmur3_i32((int32_t)($pv), h);\n")
                      }
                    case None => ctx.failed = true
                  }
                }
              }
              sb.append(s"${leafIndent}int32_t pid = ws_spark_part_id(h, (int32_t)a->num_parts);\n")
              "pid"
          }
          if (!ctx.failed) {
            if (!isScatter) {
              if (isMat) sb.append(s"${leafIndent}++local_cnt;\n")
              else sb.append(s"${leafIndent}if (use_smem_hist) atomicAdd(&s_hist[$pidExpr], 1u); else atomicAdd(&a->part_counts[$pidExpr], 1u);\n")
            } else {
              if (isMat) sb.append(s"${leafIndent}uint32_t dst = ws_warp_alloc_slot(&a->part_counts[0]);\n")
              else sb.append(s"${leafIndent}uint32_t dst = atomicAdd(&a->part_counts[$pidExpr], 1u);\n")
              outExprs.zipWithIndex.foreach { case (oe, ci) =>
                if (!ctx.failed) emitStoreCol(oe, ci, ctx, leafEnv, leafIndent, sb)
              }
            }
          }
        })
        if (ctx.failed) return None
        sb.append("  }\n")
        if (isMat && !isScatter) {
          sb.append("  ws_warp_add_count(&a->part_counts[0], local_cnt);\n")
        } else if (!isMat && !isScatter) {
          sb.append("  if (use_smem_hist) {\n")
          sb.append("    __syncthreads();\n")
          sb.append("    for (uint32_t i = threadIdx.x; i < a->num_parts; i += blockDim.x) {\n")
          sb.append("      if (s_hist[i] > 0u) atomicAdd(&a->part_counts[i], s_hist[i]);\n")
          sb.append("    }\n")
          sb.append("  }\n")
        }
        sb.append("}\n")
        Some((sb.toString(), ctx.usesFsst))
      }
      for {
        (c, fsst1) <- emitOne(isScatter = false)
        (s, fsst2) <- emitOne(isScatter = true)
      } yield {
        var pFlags = 0
        if (fsst1 || fsst2) pFlags |= com.nvidia.spark.rapids.jni.WholeStageNative.PREAMBLE_FSST
        if (partExprsOpt.isDefined) pFlags |= com.nvidia.spark.rapids.jni.WholeStageNative.PREAMBLE_MURMUR3
        (c, s, sharedConstCtx.constBits.toArray, pFlags)
      }
    }
  }
}

/** Pure SSA register expression emitter for StageIR (shared by Agg, Materialize, ShufflePartition). */
object StageExprEmitter {

  import StageIR._

  def cType(dt: DataType): Option[String] = dt match {
    case BooleanType => Some("bool")
    case ByteType | ShortType | IntegerType | DateType => Some("int32_t")
    case LongType => Some("int64_t")
    case FloatType => Some("float")
    case DoubleType => Some("double")
    case d: DecimalType if d.precision <= 9 => Some("int32_t")
    case d: DecimalType if d.precision <= 18 => Some("int64_t")
    case _ => None
  }

  private def loadCall(src: String, rowVar: String, dt: DataType): Option[String] = dt match {
    case BooleanType => Some(s"(ws_load_i32($src, $rowVar) != 0)")
    case ByteType | ShortType | IntegerType | DateType => Some(s"ws_load_i32($src, $rowVar)")
    case LongType => Some(s"ws_load_i64($src, $rowVar)")
    case FloatType => Some(s"ws_load_f32($src, $rowVar)")
    case DoubleType => Some(s"ws_load_f64($src, $rowVar)")
    case d: DecimalType if d.precision <= 9 => Some(s"ws_load_i32($src, $rowVar)")
    case d: DecimalType if d.precision <= 18 => Some(s"ws_load_i64($src, $rowVar)")
    case _ => None
  }

  private def escapeCString(s: String): String = {
    val sb = new StringBuilder(s.length + 8)
    s.getBytes(StandardCharsets.UTF_8).foreach { b =>
      val u = b & 0xFF
      if (u >= 32 && u < 127 && u != '"' && u != '\\') sb.append(u.toChar)
      else sb.append(f"\\x$u%02x")
    }
    sb.toString()
  }

  private def emitStrPred(
      fnName: String,
      a: Attribute,
      lit: String,
      env: VarEnv,
      stmts: mutable.ArrayBuffer[String],
      ctx: CodegenCtx): Option[(String, String)] = {
    env.get(a.exprId).flatMap {
      case FactCol(slot, StringType) => Some((s"a->fact_cols[$slot]", "row"))
      case DimCol(dimIdx, slot, StringType) => Some((s"a->dim_cols[$slot]", s"d$dimIdx"))
      case _ => None
    }.map { case (srcExpr, rowVar) =>
      ctx.usesFsst = true
      val id = ctx.nextId()
      val ok = s"sok$id"
      val vn = s"sv$id"
      val bytesLen = lit.getBytes(StandardCharsets.UTF_8).length
      val esc = escapeCString(lit)
      stmts += s"bool $ok = ws_valid($srcExpr, $rowVar);"
      stmts += "bool " + vn + " = " + ok + " && " + fnName + "(" + srcExpr + ", " + rowVar + ", \"" + esc + "\", " + bytesLen + "u);"
      (vn, ok)
    }
  }

  private def emitCmp(
      op: String,
      l: Expression,
      r: Expression,
      env: VarEnv,
      stmts: mutable.ArrayBuffer[String],
      ctx: CodegenCtx): Option[(String, String)] = {
    for {
      (lv, lok) <- emit(l, env, stmts, ctx)
      (rv, rok) <- emit(r, env, stmts, ctx)
    } yield {
      val id = ctx.nextId()
      val ok = s"ok$id"
      val vn = s"v$id"
      val cmp = (l.dataType, r.dataType) match {
        case (d1: DecimalType, d2: DecimalType) if d1.scale != d2.scale =>
          val maxS = math.max(d1.scale, d2.scale)
          val lf = math.pow(10.0, maxS - d1.scale).toLong
          val rf = math.pow(10.0, maxS - d2.scale).toLong
          s"(((int64_t)($lv) * ${lf}LL) $op ((int64_t)($rv) * ${rf}LL))"
        case (FloatType | DoubleType, _) | (_, FloatType | DoubleType) =>
          s"((double)($lv) $op (double)($rv))"
        case _ =>
          s"((int64_t)($lv) $op (int64_t)($rv))"
      }
      stmts += s"bool $ok = ($lok) && ($rok);"
      stmts += s"bool $vn = $ok && $cmp;"
      (vn, ok)
    }
  }

  private def emitBinArith(
      op: String,
      outDt: DataType,
      l: Expression,
      r: Expression,
      env: VarEnv,
      stmts: mutable.ArrayBuffer[String],
      ctx: CodegenCtx): Option[(String, String)] = {
    for {
      ct <- cType(outDt)
      (lv, lok) <- emit(l, env, stmts, ctx)
      (rv, rok) <- emit(r, env, stmts, ctx)
    } yield {
      val id = ctx.nextId()
      val ok = s"ok$id"
      val vn = s"v$id"
      val exprStr = (outDt, l.dataType, r.dataType) match {
        case (od: DecimalType, ld: DecimalType, rd: DecimalType) if op == "+" || op == "-" =>
          val lf = math.pow(10.0, od.scale - ld.scale).toLong
          val rf = math.pow(10.0, od.scale - rd.scale).toLong
          s"($ct)(((int64_t)($lv) * ${lf}LL) $op ((int64_t)($rv) * ${rf}LL))"
        case (od: DecimalType, ld: DecimalType, rd: DecimalType) if op == "*" =>
          val diff = (ld.scale + rd.scale) - od.scale
          if (diff == 0) s"($ct)((int64_t)($lv) * (int64_t)($rv))"
          else if (diff > 0) {
            val div = math.pow(10.0, diff).toLong
            s"($ct)(((int64_t)($lv) * (int64_t)($rv)) / ${div}LL)"
          } else {
            val mul = math.pow(10.0, -diff).toLong
            s"($ct)(((int64_t)($lv) * (int64_t)($rv)) * ${mul}LL)"
          }
        case (DoubleType | FloatType, _, _) =>
          s"($ct)((double)($lv) $op (double)($rv))"
        case _ =>
          s"($ct)((int64_t)($lv) $op (int64_t)($rv))"
      }
      stmts += s"bool $ok = ($lok) && ($rok);"
      stmts += s"$ct $vn = $ok ? ($exprStr) : ($ct)0;"
      (vn, ok)
    }
  }

  private def emitLiteral(v: Any, dt: DataType, ctx: CodegenCtx): Option[(String, String)] = {
    if (v == null) {
      cType(dt).map(ct => (s"(($ct)0)", "false"))
    } else dt match {
      case BooleanType => Some((if (v.asInstanceOf[Boolean]) "true" else "false", "true"))
      case ByteType =>
        val raw = v.asInstanceOf[Byte].toLong
        Some((ctx.allocConst(raw).map(i => s"ws_const_i32(a->const_bits, ${i}u)").getOrElse(s"(int32_t)($raw)"), "true"))
      case ShortType =>
        val raw = v.asInstanceOf[Short].toLong
        Some((ctx.allocConst(raw).map(i => s"ws_const_i32(a->const_bits, ${i}u)").getOrElse(s"(int32_t)($raw)"), "true"))
      case IntegerType | DateType =>
        val raw = v.asInstanceOf[Int].toLong
        Some((ctx.allocConst(raw).map(i => s"ws_const_i32(a->const_bits, ${i}u)").getOrElse(s"(int32_t)($raw)"), "true"))
      case LongType =>
        val raw = v.asInstanceOf[Long]
        Some((ctx.allocConst(raw).map(i => s"ws_const_i64(a->const_bits, ${i}u)").getOrElse(s"(${raw}LL)"), "true"))
      case FloatType =>
        val bits = java.lang.Double.doubleToRawLongBits(v.asInstanceOf[Float].toDouble)
        Some((ctx.allocConst(bits).map(i => s"ws_const_f32(a->const_bits, ${i}u)").getOrElse(s"((float)${v.asInstanceOf[Float]})"), "true"))
      case DoubleType =>
        val bits = java.lang.Double.doubleToRawLongBits(v.asInstanceOf[Double])
        Some((ctx.allocConst(bits).map(i => s"ws_const_f64(a->const_bits, ${i}u)").getOrElse(s"((double)${v.asInstanceOf[Double]})"), "true"))
      case d: DecimalType if d.precision <= 18 =>
        val unscaled = v.asInstanceOf[Decimal].toUnscaledLong
        val fn = if (d.precision <= 9) "ws_const_i32" else "ws_const_i64"
        val fb = if (d.precision <= 9) s"(int32_t)(${unscaled}LL)" else s"(${unscaled}LL)"
        Some((ctx.allocConst(unscaled).map(i => s"$fn(a->const_bits, ${i}u)").getOrElse(fb), "true"))
      case _ => None
    }
  }

  def emit(
      e: Expression,
      env: VarEnv,
      stmts: mutable.ArrayBuffer[String],
      ctx: CodegenCtx): Option[(String, String)] = e match {
    case GpuAlias(child, _) => emit(child, env, stmts, ctx)
    case org.apache.spark.sql.catalyst.expressions.Alias(child, _) => emit(child, env, stmts, ctx)
    case j: GpuAstJitExpression => emit(j.child, env, stmts, ctx)
    case p: GpuPromotePrecision => emit(p.child, env, stmts, ctx)
    case co: GpuCheckOverflow => emit(co.child, env, stmts, ctx)

    case a: Attribute =>
      env.get(a.exprId).flatMap {
        case RegVal(v, ok, _) => Some((v, ok))
        case FactCol(slot, dt) =>
          for {
            ct <- cType(dt)
            ld <- loadCall(s"a->fact_cols[$slot]", "row", dt)
          } yield {
            val id = ctx.nextId()
            stmts += s"bool ok$id = ws_valid(a->fact_cols[$slot], row);"
            stmts += s"$ct v$id = ok$id ? ($ld) : ($ct)0;"
            (s"v$id", s"ok$id")
          }
        case DimCol(dimIdx, slot, dt) =>
          for {
            ct <- cType(dt)
            ld <- loadCall(s"a->dim_cols[$slot]", s"d$dimIdx", dt)
          } yield {
            val id = ctx.nextId()
            stmts += s"bool ok$id = ws_valid(a->dim_cols[$slot], d$dimIdx);"
            stmts += s"$ct v$id = ok$id ? ($ld) : ($ct)0;"
            (s"v$id", s"ok$id")
          }
      }

    case GpuLiteral(v, dt) => emitLiteral(v, dt, ctx)
    case Literal(v, dt) => emitLiteral(v, dt, ctx)

    case GpuCoalesce(children) if children.nonEmpty && children.length <= 8 =>
      cType(e.dataType).flatMap { ct =>
        val id = ctx.nextId()
        stmts += s"bool ok$id = false; $ct v$id = ($ct)0;"
        var allOk = true
        children.foreach { ch =>
          if (allOk) {
            emit(ch, env, stmts, ctx) match {
              case Some((cv, cok)) =>
                stmts += s"if (!ok$id && ($cok)) { ok$id = true; v$id = ($ct)($cv); }"
              case None => allOk = false
            }
          }
        }
        if (allOk) Some((s"v$id", s"ok$id")) else None
      }

    case GpuIsNull(child) =>
      emit(child, env, stmts, ctx).map { case (_, cok) =>
        val id = ctx.nextId()
        stmts += s"bool v$id = !($cok);"
        (s"v$id", "true")
      }

    case GpuIsNotNull(child) =>
      emit(child, env, stmts, ctx).map { case (_, cok) =>
        val id = ctx.nextId()
        stmts += s"bool v$id = ($cok);"
        (s"v$id", "true")
      }

    case GpuNot(child) =>
      emit(child, env, stmts, ctx).map { case (cv, cok) =>
        val id = ctx.nextId()
        stmts += s"bool ok$id = $cok;"
        stmts += s"bool v$id = $cok && !($cv);"
        (s"v$id", s"ok$id")
      }

    case GpuAnd(l, r) =>
      for {
        (lv, lok) <- emit(l, env, stmts, ctx)
        (rv, rok) <- emit(r, env, stmts, ctx)
      } yield {
        val id = ctx.nextId()
        stmts += s"bool ok$id = ($lok && !$lv) || ($rok && !$rv) || ($lok && $rok);"
        stmts += s"bool v$id = ($lok && $lv) && ($rok && $rv);"
        (s"v$id", s"ok$id")
      }

    case GpuOr(l, r) =>
      for {
        (lv, lok) <- emit(l, env, stmts, ctx)
        (rv, rok) <- emit(r, env, stmts, ctx)
      } yield {
        val id = ctx.nextId()
        stmts += s"bool ok$id = ($lok && $lv) || ($rok && $rv) || ($lok && $rok);"
        stmts += s"bool v$id = ($lok && $lv) || ($rok && $rv);"
        (s"v$id", s"ok$id")
      }

    case GpuEqualTo(a: Attribute, GpuLiteral(v: UTF8String, StringType)) if v != null =>
      emitStrPred("ws_str_equals", a, v.toString, env, stmts, ctx)
    case GpuEqualTo(GpuLiteral(v: UTF8String, StringType), a: Attribute) if v != null =>
      emitStrPred("ws_str_equals", a, v.toString, env, stmts, ctx)
    case GpuStartsWith(a: Attribute, GpuLiteral(v: UTF8String, StringType)) if v != null =>
      emitStrPred("ws_str_starts_with", a, v.toString, env, stmts, ctx)
    case GpuContains(a: Attribute, GpuLiteral(v: UTF8String, StringType)) if v != null && v.numBytes() <= 64 =>
      emitStrPred("ws_str_contains", a, v.toString, env, stmts, ctx)

    case GpuEqualTo(l, r) => emitCmp("==", l, r, env, stmts, ctx)
    case GpuLessThan(l, r) => emitCmp("<", l, r, env, stmts, ctx)
    case GpuLessThanOrEqual(l, r) => emitCmp("<=", l, r, env, stmts, ctx)
    case GpuGreaterThan(l, r) => emitCmp(">", l, r, env, stmts, ctx)
    case GpuGreaterThanOrEqual(l, r) => emitCmp(">=", l, r, env, stmts, ctx)

    case GpuInSet(child, hset, _) if hset.size <= 32 =>
      emit(child, env, stmts, ctx).flatMap { case (cv, cok) =>
        val hasNull = hset.exists(_ == null)
        val nonNulls = hset.filter(_ != null)
        val checksBuf = mutable.ArrayBuffer.empty[String]
        var allOk = true
        nonNulls.foreach {
          case i: java.lang.Integer => checksBuf += s"((int64_t)($cv) == ${i.longValue()}LL)"
          case l: java.lang.Long => checksBuf += s"((int64_t)($cv) == ${l.longValue()}LL)"
          case s: java.lang.Short => checksBuf += s"((int64_t)($cv) == ${s.longValue()}LL)"
          case b: java.lang.Byte => checksBuf += s"((int64_t)($cv) == ${b.longValue()}LL)"
          case d: Decimal => checksBuf += s"((int64_t)($cv) == ${d.toUnscaledLong}LL)"
          case _ => allOk = false
        }
        if (!allOk) None
        else {
          val id = ctx.nextId()
          val cond = if (checksBuf.isEmpty) "false" else checksBuf.mkString("(", " || ", ")")
          stmts += s"bool m$id = ($cok) && $cond;"
          stmts += s"bool ok$id = m$id || (($cok) && ${!hasNull});"
          stmts += s"bool v$id = m$id;"
          Some((s"v$id", s"ok$id"))
        }
      }

    case GpuAdd(l, r, _) => emitBinArith("+", e.dataType, l, r, env, stmts, ctx)
    case GpuSubtract(l, r, _) => emitBinArith("-", e.dataType, l, r, env, stmts, ctx)
    case GpuMultiply(l, r, _) => emitBinArith("*", e.dataType, l, r, env, stmts, ctx)
    case GpuDivide(l, r, _) if e.dataType == DoubleType || e.dataType == FloatType =>
      for {
        ct <- cType(e.dataType)
        (lv, lok) <- emit(l, env, stmts, ctx)
        (rv, rok) <- emit(r, env, stmts, ctx)
      } yield {
        val id = ctx.nextId()
        stmts += s"bool ok$id = ($lok) && ($rok) && ((double)($rv) != 0.0);"
        stmts += s"$ct v$id = ok$id ? ($ct)((double)($lv) / (double)($rv)) : ($ct)0;"
        (s"v$id", s"ok$id")
      }

    case GpuUnscaledValue(child) =>
      child.dataType match {
        case d: DecimalType if d.precision <= 18 =>
          emit(child, env, stmts, ctx).map { case (cv, cok) =>
            val id = ctx.nextId()
            stmts += s"bool ok$id = $cok;"
            stmts += s"int64_t v$id = ok$id ? (int64_t)($cv) : 0LL;"
            (s"v$id", s"ok$id")
          }
        case _ => None
      }

    case c: GpuCast =>
      val child = c.child
      val toDt = c.dataType
      for {
        ct <- cType(toDt)
        (cv, cok) <- emit(child, env, stmts, ctx)
      } yield {
        val id = ctx.nextId()
        val conv = (child.dataType, toDt) match {
          case (sd: DecimalType, DoubleType | FloatType) =>
            val inv = math.pow(10.0, -sd.scale)
            s"($ct)((double)($cv) * $inv)"
          case (sd: DecimalType, td: DecimalType) if sd.scale == td.scale =>
            s"($ct)($cv)"
          case (sd: DecimalType, td: DecimalType) if td.scale > sd.scale =>
            val mul = math.pow(10.0, td.scale - sd.scale).toLong
            s"($ct)((int64_t)($cv) * ${mul}LL)"
          case (ByteType | ShortType | IntegerType | LongType, td: DecimalType) =>
            val mul = math.pow(10.0, td.scale).toLong
            s"($ct)((int64_t)($cv) * ${mul}LL)"
          case _ =>
            s"($ct)($cv)"
        }
        stmts += s"bool ok$id = $cok;"
        stmts += s"$ct v$id = ok$id ? ($conv) : ($ct)0;"
        (s"v$id", s"ok$id")
      }

    case GpuIf(pred, tExpr, fExpr) =>
      for {
        ct <- cType(e.dataType)
        (pv, pok) <- emit(pred, env, stmts, ctx)
        (tv, tok) <- emit(tExpr, env, stmts, ctx)
        (fv, fok) <- emit(fExpr, env, stmts, ctx)
      } yield {
        val id = ctx.nextId()
        stmts += s"bool b$id = ($pok) && ($pv);"
        stmts += s"bool ok$id = b$id ? ($tok) : ($fok);"
        stmts += s"$ct v$id = b$id ? ($ct)($tv) : ($ct)($fv);"
        (s"v$id", s"ok$id")
      }

    case cw: GpuCaseWhen if cw.branches.nonEmpty && cw.branches.length <= 16 =>
      cType(e.dataType).flatMap { ct =>
        val id = ctx.nextId()
        stmts += s"bool ok$id = false; $ct v$id = ($ct)0; bool d$id = false;"
        var allOk = true
        cw.branches.foreach { case (cond, res) =>
          if (allOk) {
            val cRes = emit(cond, env, stmts, ctx)
            val rRes = emit(res, env, stmts, ctx)
            if (cRes.isEmpty || rRes.isEmpty) allOk = false
            else {
              val (cv, cok) = cRes.get
              val (rv, rok) = rRes.get
              stmts += s"if (!d$id && ($cok) && ($cv)) { d$id = true; ok$id = ($rok); v$id = ($ct)($rv); }"
            }
          }
        }
        if (allOk) {
          cw.elseValue.foreach { el =>
            emit(el, env, stmts, ctx) match {
              case Some((ev, eok)) => stmts += s"if (!d$id) { ok$id = ($eok); v$id = ($ct)($ev); }"
              case None => allOk = false
            }
          }
        }
        if (allOk) Some((s"v$id", s"ok$id")) else None
      }

    case _ => None
  }
}
