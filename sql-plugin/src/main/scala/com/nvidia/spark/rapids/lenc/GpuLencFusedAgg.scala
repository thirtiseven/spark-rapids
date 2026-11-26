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

import java.util.concurrent.ConcurrentHashMap

import scala.collection.mutable

import ai.rapids.cudf.{DType, Scalar, Table => CudfTable}
import com.nvidia.spark.rapids._
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.GpuMetric._
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.jni.LencNative

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate.Partial
import org.apache.spark.sql.execution.{LeafExecNode, SparkPlan, UnaryExecNode}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids._
import org.apache.spark.sql.rapids.aggregate._
import org.apache.spark.sql.types._
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * Item 2: 100% Generic Compiler-Level NVRTC/LTO Reduction + Conditional GroupBy Engine.
 * Completely replaces all workload-specific POCs (zero hardcoded Q1 or NF48 logic!):
 * Translates ANY Spark Filter + Project + Partial HashAggregate (SUM, COUNT, AVG, MIN, MAX)
 * into NVRTC-compiled CUDA device code with three generic execution modes:
 *   - Mode 0: Any Global Conditional Reduce (e.g. Q6, Q14/Q19 agg, W1, arbitrary global aggs)
 *   - Mode 1: Any Low-Cardinality Conditional GroupBy (<= 64 dict combinations, e.g. Q1 & any rollup)
 *   - Mode 2: Any High-Cardinality Conditional GroupBy (arbitrary group keys + multi-output aggs,
 *             e.g. NF48 36-way CASE WHEN & any wide/conditional GroupBy!)
 * Works on BOTH compressed .lenc (in-register + 1,024-row chunk Min-Max pruning) AND Zero-Rewrite Parquet!
 */
object GpuJitReducePool {
  private val MAX_KEYS = 512
  private val programs = new ConcurrentHashMap[String, LencNative.JitReduceProgram]()

  Runtime.getRuntime.addShutdownHook(new Thread(() => {
    val it = programs.values().iterator()
    while (it.hasNext) {
      it.next().safeClose()
    }
    programs.clear()
  }))

  def withProgram[T](
      cudaSrc: String,
      kernelName: String)(body: LencNative.JitReduceProgram => T): T = {
    val existing = programs.get(cudaSrc)
    if (existing != null) {
      body(existing)
    } else if (programs.size() >= MAX_KEYS) {
      withResource(LencNative.compileJitReduce(cudaSrc, kernelName))(body)
    } else {
      val prog = programs.computeIfAbsent(cudaSrc, _ => LencNative.compileJitReduce(cudaSrc, kernelName))
      body(prog)
    }
  }
}

final case class GenericAggTerm(
    kind: Byte, // 0=SUM_F64, 1=SUM_I64, 2=COUNT_I64, 3=MIN_F64, 4=MAX_F64, 5=MIN_I64, 6=MAX_I64
    inputExpr: Option[Expression]) // None for COUNT(*) / COUNT(1)

/** Generic Spark GpuExpression -> NVRTC CUDA C++ compiler (Spark ETL NaN + NULL semantics). */
object GenericJitReduceCodegen {
  val AGG_SUM_F64: Byte = 0
  val AGG_SUM_I64: Byte = 1
  val AGG_COUNT_I64: Byte = 2
  val AGG_MIN_F64: Byte = 3
  val AGG_MAX_F64: Byte = 4
  val AGG_MIN_I64: Byte = 5
  val AGG_MAX_I64: Byte = 6

  private val CUDA_PREAMBLE: String =
    """typedef unsigned char uint8_t;
      |typedef int int32_t;
      |typedef unsigned int uint32_t;
      |typedef long long int64_t;
      |typedef unsigned long long uint64_t;
      |struct JitColDesc {
      |  uint8_t mode;
      |  uint8_t has_nulls;
      |  uint8_t pad[6];
      |  const void* raw_ptr;
      |  const uint32_t* null_mask;
      |  const int64_t* min_vals;
      |  const uint8_t* bit_widths;
      |  const uint32_t* chunk_word_offs;
      |  const uint64_t* packed_words;
      |  const int32_t* dict_remap;
      |  const uint32_t* rle_chunk_start;
      |  const uint32_t* rle_run_ends;
      |  const int64_t* rle_run_vals;
      |  uint32_t rle_num_runs;
      |  uint32_t dict_size;
      |  double inv_scale;
      |};
      |struct JitReduceArgs {
      |  JitColDesc cols[16];
      |  uint32_t num_rows;
      |  int32_t max_groups;
      |  int32_t num_aggs;
      |  const uint32_t* active_chunk_mask;
      |  double* out_f64;
      |  long long* out_i64;
      |  unsigned long long* out_cnt;
      |  const int32_t* group_offsets;
      |  const int32_t* grouped_row_ids;
      |  int32_t num_groups;
      |  double* const* hc_out_f64;
      |  int64_t* const* hc_out_i64;
      |  uint8_t* const* hc_out_valid;
      |};
      |__device__ __forceinline__ bool col_valid(const JitColDesc& c, uint32_t row) {
      |  if (!c.has_nulls || c.null_mask == nullptr) return true;
      |  return ((c.null_mask[row >> 5] >> (row & 31u)) & 1u) != 0u;
      |}
      |__device__ __forceinline__ int64_t unpack_for(const JitColDesc& c, uint32_t row) {
      |  uint32_t ch = row >> 10;
      |  uint32_t bw = c.bit_widths[ch];
      |  int64_t base = c.min_vals[ch];
      |  if (bw == 0u) return base;
      |  uint32_t local = row & 1023u;
      |  uint32_t bp = local * bw;
      |  uint32_t wi = c.chunk_word_offs[ch] + (bp >> 6);
      |  uint32_t bo = bp & 63u;
      |  uint64_t v = c.packed_words[wi] >> bo;
      |  if (bo + bw > 64u) v |= c.packed_words[wi + 1u] << (64u - bo);
      |  uint64_t mask = (bw == 64u) ? ~0ULL : ((1ULL << bw) - 1ULL);
      |  return base + (int64_t)(v & mask);
      |}
      |__device__ __forceinline__ int64_t unpack_rle(const JitColDesc& c, uint32_t row) {
      |  uint32_t ch = row >> 10;
      |  uint32_t lo = c.rle_chunk_start[ch];
      |  uint32_t hi = c.rle_num_runs;
      |  while (lo + 1u < hi) {
      |    uint32_t mid = (lo + hi) >> 1;
      |    if (c.rle_run_ends[mid - 1u] <= row) lo = mid; else hi = mid;
      |  }
      |  return c.rle_run_vals[lo];
      |}
      |__device__ __forceinline__ int64_t read_i64(const JitColDesc& c, uint32_t row) {
      |  if (c.mode == 1) return unpack_for(c, row);
      |  if (c.mode == 2) return unpack_rle(c, row);
      |  return ((const int64_t*)c.raw_ptr)[row];
      |}
      |__device__ __forceinline__ int32_t read_i32(const JitColDesc& c, uint32_t row) {
      |  if (c.mode == 1) return (int32_t)unpack_for(c, row);
      |  if (c.mode == 2) return (int32_t)unpack_rle(c, row);
      |  if (c.mode == 3) {
      |    int32_t code = (int32_t)unpack_for(c, row);
      |    return (c.dict_remap != nullptr) ? c.dict_remap[code] : code;
      |  }
      |  return ((const int32_t*)c.raw_ptr)[row];
      |}
      |__device__ __forceinline__ float read_f32(const JitColDesc& c, uint32_t row) {
      |  if (c.mode == 1) return (float)((double)unpack_for(c, row) * c.inv_scale);
      |  return ((const float*)c.raw_ptr)[row];
      |}
      |__device__ __forceinline__ double read_f64(const JitColDesc& c, uint32_t row) {
      |  if (c.mode == 1) return (double)unpack_for(c, row) * c.inv_scale;
      |  return ((const double*)c.raw_ptr)[row];
      |}
      |// Spark ETL NaN comparison semantics (NaN == NaN, NaN > any non-NaN)
      |template <typename T> __device__ __forceinline__ bool etl_eq(T a, T b) {
      |  bool na = isnan(a), nb = isnan(b);
      |  return (na && nb) || (!na && !nb && a == b);
      |}
      |template <typename T> __device__ __forceinline__ bool etl_lt(T a, T b) {
      |  bool na = isnan(a), nb = isnan(b);
      |  return (!na && nb) || (!na && !nb && a < b);
      |}
      |template <typename T> __device__ __forceinline__ bool etl_le(T a, T b) {
      |  bool na = isnan(a), nb = isnan(b);
      |  return nb || (!na && a <= b);
      |}
      |template <typename T> __device__ __forceinline__ bool etl_gt(T a, T b) { return etl_lt(b, a); }
      |template <typename T> __device__ __forceinline__ bool etl_ge(T a, T b) { return etl_le(b, a); }
      |__device__ __forceinline__ void atomicMinDouble(double* addr, double val) {
      |  unsigned long long* u = (unsigned long long*)addr;
      |  unsigned long long old = *u, assumed;
      |  do {
      |    assumed = old;
      |    if (__longlong_as_double(assumed) <= val) break;
      |    old = atomicCAS(u, assumed, __double_as_longlong(val));
      |  } while (assumed != old);
      |}
      |__device__ __forceinline__ void atomicMaxDouble(double* addr, double val) {
      |  unsigned long long* u = (unsigned long long*)addr;
      |  unsigned long long old = *u, assumed;
      |  do {
      |    assumed = old;
      |    if (__longlong_as_double(assumed) >= val) break;
      |    old = atomicCAS(u, assumed, __double_as_longlong(val));
      |  } while (assumed != old);
      |}
      |__device__ __forceinline__ __int128 dec_div_round_half_up(__int128 num, __int128 den) {
      |  bool neg = (num < 0) ^ (den < 0);
      |  unsigned __int128 anum = (num < 0) ? (unsigned __int128)(-num) : (unsigned __int128)num;
      |  unsigned __int128 aden = (den < 0) ? (unsigned __int128)(-den) : (unsigned __int128)den;
      |  unsigned __int128 q = (anum + (aden >> 1)) / aden;
      |  return neg ? -(__int128)q : (__int128)q;
      |}
      |""".stripMargin

  private def pow10Long(e: Int): Long = {
    var r = 1L
    var i = 0
    while (i < e) { r *= 10L; i += 1 }
    r
  }

  private def sameScale(a: DataType, b: DataType): Boolean = (a, b) match {
    case (da: DecimalType, db: DecimalType) => da.scale == db.scale
    case (_: DecimalType, _) | (_, _: DecimalType) => false
    case _ => true
  }

  def cudaType(dt: DataType): Option[String] = dt match {
    case BooleanType => Some("bool")
    case ByteType => Some("int32_t")
    case ShortType => Some("int32_t")
    case IntegerType | DateType => Some("int32_t")
    case LongType | TimestampType => Some("int64_t")
    case FloatType => Some("float")
    case DoubleType => Some("double")
    case d: DecimalType if d.precision <= 18 => Some("int64_t")
    case d: DecimalType if d.precision <= 38 => Some("__int128")
    case _ => None
  }

  private def isFloatDt(dt: DataType): Boolean = dt == FloatType || dt == DoubleType

  /** Emits (valCode, validCode) into `stmts` for ANY Spark GpuExpression over compact slot refs! */
  def emitExpr(
      expr: Expression,
      stmts: mutable.ArrayBuffer[String],
      nextId: () => Int): Option[(String, String)] = {
    def fresh(prefix: String): String = s"${prefix}_${nextId()}"
    expr match {
      case GpuAlias(child, _) => emitExprFull(child, stmts, nextId)
      case jit: GpuAstJitExpression => emitExprFull(jit.child, stmts, nextId)
      case p: GpuPromotePrecision => emitExprFull(p.child, stmts, nextId)
      case u: GpuUnscaledValue =>
        u.child.dataType match {
          case d: DecimalType if d.precision <= 18 =>
            emitExprFull(u.child, stmts, nextId).map { case (cv, cok) =>
              val vn = fresh("unsc")
              stmts += s"int64_t $vn = (int64_t)($cv);"
              (vn, cok)
            }
          case _ => None
        }
      case co: GpuCheckOverflow =>
        if (!co.nullOnOverflow || co.dataType.precision > 38) None
        else {
          co.child.dataType match {
            case cd: DecimalType if cd.precision <= 38 =>
              emitExprFull(co.child, stmts, nextId).flatMap { case (cv, cok) =>
                val ct = cudaType(co.dataType).get
                val vn = fresh("ovf")
                val ok = fresh("ovfok")
                val diff = cd.scale - co.dataType.scale
                val adjExpr = if (diff == 0) {
                  s"(__int128)($cv)"
                } else if (diff > 0 && diff <= 18) {
                  s"dec_div_round_half_up((__int128)($cv), (__int128)${pow10Long(diff)}LL)"
                } else if (diff < 0 && -diff <= 18) {
                  s"((__int128)($cv) * (__int128)${pow10Long(-diff)}LL)"
                } else {
                  return None
                }
                if (co.dataType.precision <= 18) {
                  val bound = pow10Long(co.dataType.precision)
                  stmts += s"__int128 ${vn}_128 = $adjExpr;"
                  stmts += s"bool $ok = ($cok) && (${vn}_128 > -(__int128)${bound}LL) && (${vn}_128 < (__int128)${bound}LL);"
                  stmts += s"$ct $vn = ($ct)${vn}_128;"
                } else {
                  stmts += s"$ct $vn = ($ct)($adjExpr);"
                  stmts += s"bool $ok = ($cok);"
                }
                Some((vn, ok))
              }
            case _ => None
          }
        }
      case ref: GpuBoundReference =>
        val slot = ref.ordinal
        val vn = fresh("v")
        val ok = fresh("ok")
        stmts += s"bool $ok = col_valid(a->cols[$slot], row);"
        ref.dataType match {
          case IntegerType | DateType | ByteType | ShortType =>
            stmts += s"int32_t $vn = $ok ? read_i32(a->cols[$slot], row) : 0;"
            Some((vn, ok))
          case LongType | TimestampType =>
            stmts += s"int64_t $vn = $ok ? read_i64(a->cols[$slot], row) : 0LL;"
            Some((vn, ok))
          case FloatType =>
            stmts += s"float $vn = $ok ? read_f32(a->cols[$slot], row) : 0.0f;"
            Some((vn, ok))
          case DoubleType =>
            stmts += s"double $vn = $ok ? read_f64(a->cols[$slot], row) : 0.0;"
            Some((vn, ok))
          case BooleanType =>
            stmts += s"bool $vn = $ok ? (read_i32(a->cols[$slot], row) != 0) : false;"
            Some((vn, ok))
          case d: DecimalType if d.precision <= 9 =>
            stmts += s"int64_t $vn = $ok ? (int64_t)read_i32(a->cols[$slot], row) : 0LL;"
            Some((vn, ok))
          case d: DecimalType if d.precision <= 18 =>
            stmts += s"int64_t $vn = $ok ? read_i64(a->cols[$slot], row) : 0LL;"
            Some((vn, ok))
          case _ => None
        }

      case lit: GpuLiteral =>
        emitLit(lit.value, lit.dataType, stmts, fresh)
      case lit: Literal =>
        emitLit(lit.value, lit.dataType, stmts, fresh)

      case c: GpuCast =>
        (c.child.dataType, c.dataType) match {
          case (cd: DecimalType, td: DecimalType) if cd.precision <= 38 && td.precision <= 38 =>
            emitExprFull(c.child, stmts, nextId).flatMap { case (cv, cok) =>
              val ct = cudaType(td).get
              val vn = fresh("dcast")
              val diff = td.scale - cd.scale
              if (diff == 0) {
                stmts += s"$ct $vn = ($ct)($cv);"
                Some((vn, cok))
              } else if (diff > 0 && diff <= 18) {
                stmts += s"$ct $vn = ($ct)((__int128)($cv) * (__int128)${pow10Long(diff)}LL);"
                Some((vn, cok))
              } else if (diff < 0 && -diff <= 18) {
                stmts += s"$ct $vn = ($ct)dec_div_round_half_up((__int128)($cv), (__int128)${pow10Long(-diff)}LL);"
                Some((vn, cok))
              } else None
            }
          case (IntegerType | LongType | ShortType | ByteType, td: DecimalType) if td.precision <= 38 && td.scale <= 12 =>
            emitExprFull(c.child, stmts, nextId).map { case (cv, cok) =>
              val ct = cudaType(td).get
              val vn = fresh("idcast")
              stmts += s"$ct $vn = ($ct)((__int128)((int64_t)($cv)) * (__int128)${pow10Long(td.scale)}LL);"
              (vn, cok)
            }
          case (cd: DecimalType, DoubleType) if cd.precision <= 18 =>
            emitExprFull(c.child, stmts, nextId).map { case (cv, cok) =>
              val vn = fresh("dfcast")
              val inv = 1.0 / pow10Long(cd.scale).toDouble
              stmts += s"double $vn = ((double)($cv)) * ${java.lang.Double.toString(inv)};"
              (vn, cok)
            }
          case (cd: DecimalType, FloatType) if cd.precision <= 18 =>
            emitExprFull(c.child, stmts, nextId).map { case (cv, cok) =>
              val vn = fresh("ffcast")
              val inv = 1.0 / pow10Long(cd.scale).toDouble
              stmts += s"float $vn = (float)(((double)($cv)) * ${java.lang.Double.toString(inv)});"
              (vn, cok)
            }
          case (_: DecimalType, _) | (_, _: DecimalType) => None
          case _ =>
            for {
              ct <- cudaType(c.dataType)
              (cv, cok) <- emitExprFull(c.child, stmts, nextId)
            } yield {
              val vn = fresh("cast")
              stmts += s"$ct $vn = ($ct)($cv);"
              (vn, cok)
            }
        }

      case a: GpuAdd =>
        if (!sameScale(a.left.dataType, a.dataType) || !sameScale(a.right.dataType, a.dataType)) None
        else for {
          ct <- cudaType(a.dataType)
          (lv, lok) <- emitExprFull(a.left, stmts, nextId)
          (rv, rok) <- emitExprFull(a.right, stmts, nextId)
        } yield {
          val vn = fresh("add")
          val ok = fresh("aok")
          stmts += s"bool $ok = ($lok) && ($rok);"
          stmts += s"$ct $vn = ($ct)((($ct)($lv)) + (($ct)($rv)));"
          (vn, ok)
        }

      case s: GpuSubtract =>
        if (!sameScale(s.left.dataType, s.dataType) || !sameScale(s.right.dataType, s.dataType)) None
        else for {
          ct <- cudaType(s.dataType)
          (lv, lok) <- emitExprFull(s.left, stmts, nextId)
          (rv, rok) <- emitExprFull(s.right, stmts, nextId)
        } yield {
          val vn = fresh("sub")
          val ok = fresh("sok")
          stmts += s"bool $ok = ($lok) && ($rok);"
          stmts += s"$ct $vn = ($ct)((($ct)($lv)) - (($ct)($rv)));"
          (vn, ok)
        }

      case m: GpuDecimalMultiplyBase if !SQLConf.get.ansiEnabled =>
        emitDecimalMul(m.left, m.right, m.dataType, stmts, nextId, fresh)

      case m: GpuMultiply =>
        (m.left.dataType, m.right.dataType, m.dataType) match {
          case (_: DecimalType, _: DecimalType, _: DecimalType) =>
            emitDecimalMul(m.left, m.right, m.dataType, stmts, nextId, fresh)
          case (_: DecimalType, _, _) | (_, _: DecimalType, _) | (_, _, _: DecimalType) => None
          case _ =>
            for {
              ct <- cudaType(m.dataType)
              (lv, lok) <- emitExprFull(m.left, stmts, nextId)
              (rv, rok) <- emitExprFull(m.right, stmts, nextId)
            } yield {
              val vn = fresh("mul")
              val ok = fresh("mok")
              stmts += s"bool $ok = ($lok) && ($rok);"
              stmts += s"$ct $vn = ($ct)(($lv) * ($rv));"
              (vn, ok)
            }
        }

      case d: GpuDecimalDivideBase if !SQLConf.get.ansiEnabled =>
        emitDecimalDiv(d.left, d.right, d.dataType, stmts, nextId, fresh)

      case d: GpuDivide =>
        (d.left.dataType, d.right.dataType, d.dataType) match {
          case (_: DecimalType, _: DecimalType, _: DecimalType) =>
            emitDecimalDiv(d.left, d.right, d.dataType, stmts, nextId, fresh)
          case (_: DecimalType, _, _) | (_, _: DecimalType, _) | (_, _, _: DecimalType) => None
          case _ =>
            for {
              ct <- cudaType(d.dataType)
              (lv, lok) <- emitExprFull(d.left, stmts, nextId)
              (rv, rok) <- emitExprFull(d.right, stmts, nextId)
            } yield {
              val vn = fresh("div")
              val ok = fresh("dok")
              stmts += s"bool $ok = ($lok) && ($rok) && (($rv) != 0);"
              stmts += s"$ct $vn = $ok ? ($ct)(($lv) / ($rv)) : ($ct)0;"
              (vn, ok)
            }
        }

      case GpuEqualTo(l, r) =>
        if (!sameScale(l.dataType, r.dataType)) None
        else for {
          (lv, lok) <- emitExprFull(l, stmts, nextId)
          (rv, rok) <- emitExprFull(r, stmts, nextId)
        } yield {
          val vn = fresh("eq")
          val ok = fresh("eqok")
          stmts += s"bool $ok = ($lok) && ($rok);"
          val cmp = if (isFloatDt(l.dataType)) s"etl_eq($lv, $rv)" else s"(($lv) == ($rv))"
          stmts += s"bool $vn = $ok && $cmp;"
          (vn, ok)
        }

      case GpuLessThan(l, r) =>
        if (!sameScale(l.dataType, r.dataType)) None
        else for {
          (lv, lok) <- emitExprFull(l, stmts, nextId)
          (rv, rok) <- emitExprFull(r, stmts, nextId)
        } yield {
          val vn = fresh("lt")
          val ok = fresh("ltok")
          stmts += s"bool $ok = ($lok) && ($rok);"
          val cmp = if (isFloatDt(l.dataType)) s"etl_lt($lv, $rv)" else s"(($lv) < ($rv))"
          stmts += s"bool $vn = $ok && $cmp;"
          (vn, ok)
        }

      case GpuLessThanOrEqual(l, r) =>
        if (!sameScale(l.dataType, r.dataType)) None
        else for {
          (lv, lok) <- emitExprFull(l, stmts, nextId)
          (rv, rok) <- emitExprFull(r, stmts, nextId)
        } yield {
          val vn = fresh("le")
          val ok = fresh("leok")
          stmts += s"bool $ok = ($lok) && ($rok);"
          val cmp = if (isFloatDt(l.dataType)) s"etl_le($lv, $rv)" else s"(($lv) <= ($rv))"
          stmts += s"bool $vn = $ok && $cmp;"
          (vn, ok)
        }

      case GpuGreaterThan(l, r) =>
        if (!sameScale(l.dataType, r.dataType)) None
        else for {
          (lv, lok) <- emitExprFull(l, stmts, nextId)
          (rv, rok) <- emitExprFull(r, stmts, nextId)
        } yield {
          val vn = fresh("gt")
          val ok = fresh("gtok")
          stmts += s"bool $ok = ($lok) && ($rok);"
          val cmp = if (isFloatDt(l.dataType)) s"etl_gt($lv, $rv)" else s"(($lv) > ($rv))"
          stmts += s"bool $vn = $ok && $cmp;"
          (vn, ok)
        }

      case GpuGreaterThanOrEqual(l, r) =>
        if (!sameScale(l.dataType, r.dataType)) None
        else for {
          (lv, lok) <- emitExprFull(l, stmts, nextId)
          (rv, rok) <- emitExprFull(r, stmts, nextId)
        } yield {
          val vn = fresh("ge")
          val ok = fresh("geok")
          stmts += s"bool $ok = ($lok) && ($rok);"
          val cmp = if (isFloatDt(l.dataType)) s"etl_ge($lv, $rv)" else s"(($lv) >= ($rv))"
          stmts += s"bool $vn = $ok && $cmp;"
          (vn, ok)
        }

      case GpuAnd(l, r) =>
        for {
          (lv, lok) <- emitExprFull(l, stmts, nextId)
          (rv, rok) <- emitExprFull(r, stmts, nextId)
        } yield {
          val vn = fresh("and")
          val ok = fresh("andok")
          stmts += s"bool $vn = (($lok) && ($lv)) && (($rok) && ($rv));"
          stmts += s"bool $ok = $vn || (($lok) && !($lv)) || (($rok) && !($rv));"
          (vn, ok)
        }

      case GpuOr(l, r) =>
        for {
          (lv, lok) <- emitExprFull(l, stmts, nextId)
          (rv, rok) <- emitExprFull(r, stmts, nextId)
        } yield {
          val vn = fresh("or")
          val ok = fresh("orok")
          stmts += s"bool $vn = (($lok) && ($lv)) || (($rok) && ($rv));"
          stmts += s"bool $ok = $vn || (($lok) && ($rok));"
          (vn, ok)
        }

      case GpuNot(c) =>
        emitExprFull(c, stmts, nextId).map { case (cv, cok) =>
          val vn = fresh("not")
          stmts += s"bool $vn = !($cv);"
          (vn, cok)
        }

      case GpuIsNull(c) =>
        emitExprFull(c, stmts, nextId).map { case (_, cok) =>
          val vn = fresh("isn")
          stmts += s"bool $vn = !($cok);"
          (vn, "true")
        }

      case GpuIsNotNull(c) =>
        emitExprFull(c, stmts, nextId).map { case (_, cok) =>
          (cok, "true")
        }

      case GpuIf(cond, tv, fv) =>
        if (!sameScale(tv.dataType, fv.dataType)) None
        else for {
          ct <- cudaType(tv.dataType)
          (cv, cok) <- emitExprFull(cond, stmts, nextId)
          (tval, tok) <- emitExprFull(tv, stmts, nextId)
          (fval, fok) <- emitExprFull(fv, stmts, nextId)
        } yield {
          val vn = fresh("ifv")
          val ok = fresh("ifok")
          stmts += s"bool $ok = (($cok) && ($cv)) ? ($tok) : ($fok);"
          stmts += s"$ct $vn = (($cok) && ($cv)) ? ($tval) : ($fval);"
          (vn, ok)
        }

      case co: GpuCoalesce =>
        cudaType(co.dataType).flatMap { ct =>
          if (!co.children.forall(ch => sameScale(ch.dataType, co.dataType))) return None
          val vn = fresh("coal")
          val ok = fresh("coalok")
          stmts += s"$ct $vn = ($ct)0; bool $ok = false;"
          for (ch <- co.children) {
            stmts += s"if (!$ok) {"
            val r = emitExprFull(ch, stmts, nextId)
            if (r.isEmpty) return None
            val (cv, cok) = r.get
            stmts += s"  if ($cok) { $vn = ($ct)($cv); $ok = true; }"
            stmts += "}"
          }
          Some((vn, ok))
        }

      case is: GpuInSet =>
        emitExprFull(is.child, stmts, nextId).flatMap { case (cv, cok) =>
          val lits = is.list.toSeq.filter(_ != null)
          if (lits.isEmpty) None
          else {
            val vn = fresh("in")
            val checks = is.child.dataType match {
              case IntegerType | DateType | ByteType | ShortType =>
                lits.map(x => s"(($cv) == ${x.asInstanceOf[Number].intValue()})").mkString(" || ")
              case LongType | TimestampType =>
                lits.map(x => s"(($cv) == ${x.asInstanceOf[Number].longValue()}LL)").mkString(" || ")
              case d: DecimalType if d.precision <= 18 =>
                lits.map(x => s"(($cv) == ${x.asInstanceOf[org.apache.spark.sql.types.Decimal].toUnscaledLong}LL)").mkString(" || ")
              case _ => return None
            }
            stmts += s"bool $vn = ($cok) && ($checks);"
            Some((vn, cok))
          }
        }

      case _ => None
    }
  }

  private def emitDecimalMul(
      left: Expression,
      right: Expression,
      outType: DataType,
      stmts: mutable.ArrayBuffer[String],
      nextId: () => Int,
      fresh: String => String): Option[(String, String)] = {
    (left.dataType, right.dataType, outType) match {
      case (ld: DecimalType, rd: DecimalType, od: DecimalType)
          if ld.precision <= 38 && rd.precision <= 38 && od.precision <= 38 =>
        for {
          ct <- cudaType(od)
          (lv, lok) <- emitExprFull(left, stmts, nextId)
          (rv, rok) <- emitExprFull(right, stmts, nextId)
          res <- {
            val shift = ld.scale + rd.scale - od.scale
            if (shift == 0) {
              val vn = fresh("dmul")
              val ok = fresh("dmok")
              stmts += s"bool $ok = ($lok) && ($rok);"
              stmts += s"$ct $vn = ($ct)((__int128)($lv) * (__int128)($rv));"
              Some((vn, ok))
            } else if (shift > 0 && shift <= 18) {
              val vn = fresh("dmul")
              val ok = fresh("dmok")
              stmts += s"bool $ok = ($lok) && ($rok);"
              stmts += s"$ct $vn = ($ct)dec_div_round_half_up((__int128)($lv) * (__int128)($rv), (__int128)${pow10Long(shift)}LL);"
              Some((vn, ok))
            } else None
          }
        } yield res
      case _ => None
    }
  }

  private def emitDecimalDiv(
      left: Expression,
      right: Expression,
      outType: DataType,
      stmts: mutable.ArrayBuffer[String],
      nextId: () => Int,
      fresh: String => String): Option[(String, String)] = {
    (left.dataType, right.dataType, outType) match {
      case (ld: DecimalType, rd: DecimalType, od: DecimalType)
          if ld.precision <= 38 && rd.precision <= 38 && od.precision <= 38 =>
        val up = od.scale + rd.scale - ld.scale
        if (up < 0 || up > 18) None
        else for {
          ct <- cudaType(od)
          (lv, lok) <- emitExprFull(left, stmts, nextId)
          (rv, rok) <- emitExprFull(right, stmts, nextId)
        } yield {
          val vn = fresh("ddiv")
          val ok = fresh("ddok")
          stmts += s"bool $ok = ($lok) && ($rok) && ((__int128)($rv) != (__int128)0);"
          stmts += s"$ct $vn = $ok ? ($ct)dec_div_round_half_up((__int128)($lv) * (__int128)${pow10Long(up)}LL, (__int128)($rv)) : ($ct)0;"
          (vn, ok)
        }
      case _ => None
    }
  }

  private def emitLit(
      value: Any,
      dt: DataType,
      stmts: mutable.ArrayBuffer[String],
      fresh: String => String): Option[(String, String)] = {
    if (value == null) {
      cudaType(dt).map { ct =>
        val vn = fresh("lit")
        stmts += s"$ct $vn = ($ct)0;"
        (vn, "false")
      }
    } else {
      dt match {
        case BooleanType => Some((if (value.asInstanceOf[Boolean]) "true" else "false", "true"))
        case IntegerType | DateType => Some((s"((int32_t)${value.asInstanceOf[Int]})", "true"))
        case ShortType => Some((s"((int32_t)${value.asInstanceOf[Short]})", "true"))
        case ByteType => Some((s"((int32_t)${value.asInstanceOf[Byte]})", "true"))
        case LongType | TimestampType => Some((s"((int64_t)${value.asInstanceOf[Long]}LL)", "true"))
        case d: DecimalType if d.precision <= 38 =>
          val bi = value.asInstanceOf[org.apache.spark.sql.types.Decimal].toJavaBigDecimal.unscaledValue()
          if (bi.bitLength() <= 62) {
            val ct = cudaType(d).get
            Some((s"(($ct)${bi.longValue()}LL)", "true"))
          } else None
        case FloatType =>
          val f = value.asInstanceOf[Float]
          if (f.isNaN) Some(("__int_as_float(0x7fc00000)", "true"))
          else Some((s"((float)${java.lang.Float.toString(f)}f)", "true"))
        case DoubleType =>
          val d = value.asInstanceOf[Double]
          if (d.isNaN) Some(("__longlong_as_double(0x7ff8000000000000LL)", "true"))
          else Some((s"((double)${java.lang.Double.toString(d)})", "true"))
        case _ => None
      }
    }
  }

  /** Clean non-destructive emit supporting multi-branch GpuCaseWhen via matched flag! */
  def emitExprFull(
      expr: Expression,
      stmts: mutable.ArrayBuffer[String],
      nextId: () => Int): Option[(String, String)] = {
    expr match {
      case cw: GpuCaseWhen =>
        cudaType(cw.dataType).flatMap { ct =>
          if (!cw.branches.forall { case (_, t) => sameScale(t.dataType, cw.dataType) } ||
              !cw.elseValue.forall(ev => sameScale(ev.dataType, cw.dataType))) {
            return None
          }
          val vn = s"cwv_${nextId()}"
          val ok = s"cwok_${nextId()}"
          val hit = s"cwhit_${nextId()}"
          stmts += s"$ct $vn = ($ct)0; bool $ok = false; bool $hit = false;"
          for ((c, t) <- cw.branches) {
            stmts += s"if (!$hit) {"
            val cRes = emitExprFull(c, stmts, nextId)
            if (cRes.isEmpty) return None
            val (cv, cok) = cRes.get
            stmts += s"  if (($cok) && ($cv)) {"
            val tRes = emitExprFull(t, stmts, nextId)
            if (tRes.isEmpty) return None
            val (tv, tok) = tRes.get
            stmts += s"    $vn = ($ct)($tv); $ok = ($tok); $hit = true;"
            stmts += "  }"
            stmts += "}"
          }
          cw.elseValue.foreach { ev =>
            stmts += s"if (!$hit) {"
            val eRes = emitExprFull(ev, stmts, nextId)
            if (eRes.isEmpty) return None
            val (evv, eok) = eRes.get
            stmts += s"  $vn = ($ct)($evv); $ok = ($eok);"
            stmts += "}"
          }
          Some((vn, ok))
        }
      case other => emitExpr(other, stmts, nextId)
    }
  }

  /** Generate Mode 0 (Global) or Mode 1 (Low-Card GroupBy <= 64) generic CUDA kernel! */
  def generateLowCardKernel(
      filterExprOpt: Option[Expression],
      groupKeySlots: Array[Int],
      maxGroups: Int,
      aggTerms: Seq[GenericAggTerm]): Option[(String, String)] = {
    val na = aggTerms.length
    val kname = "jit_reduce_lowcard"
    var idCounter = 0
    val nextId = () => { idCounter += 1; idCounter }

    val fStmts = mutable.ArrayBuffer.empty[String]
    val fCheck = filterExprOpt match {
      case Some(fe) =>
        emitExprFull(fe, fStmts, nextId) match {
          case Some((fv, fok)) => s"if (!(($fok) && ($fv))) continue;"
          case None => return None
        }
      case None => ""
    }

    val aggUpdates = mutable.ArrayBuffer.empty[String]
    aggTerms.zipWithIndex.foreach { case (term, ai) =>
      term.kind match {
        case AGG_COUNT_I64 =>
          term.inputExpr match {
            case None =>
              aggUpdates += s"    if (gid < RLOC) { l_cnt[gid * NA + $ai] += 1ULL; } else { atomicAdd(&s_cnt[gid * NA + $ai], 1ULL); }"
            case Some(ie) =>
              val st = mutable.ArrayBuffer.empty[String]
              emitExprFull(ie, st, nextId) match {
                case Some((_, iok)) =>
                  aggUpdates += "    {"
                  aggUpdates ++= st.map("      " + _)
                  aggUpdates += s"      if ($iok) { if (gid < RLOC) { l_cnt[gid * NA + $ai] += 1ULL; } else { atomicAdd(&s_cnt[gid * NA + $ai], 1ULL); } }"
                  aggUpdates += "    }"
                case None => return None
              }
          }
        case AGG_SUM_F64 =>
          val st = mutable.ArrayBuffer.empty[String]
          emitExprFull(term.inputExpr.get, st, nextId) match {
            case Some((iv, iok)) =>
              aggUpdates += "    {"
              aggUpdates ++= st.map("      " + _)
              aggUpdates += s"      if ($iok) { double dv = (double)($iv); if (gid < RLOC) { l_f64[gid * NA + $ai] += dv; l_cnt[gid * NA + $ai] += 1ULL; } else { atomicAdd(&s_f64[gid * NA + $ai], dv); atomicAdd(&s_cnt[gid * NA + $ai], 1ULL); } }"
              aggUpdates += "    }"
            case None => return None
          }
        case AGG_SUM_I64 =>
          val st = mutable.ArrayBuffer.empty[String]
          emitExprFull(term.inputExpr.get, st, nextId) match {
            case Some((iv, iok)) =>
              aggUpdates += "    {"
              aggUpdates ++= st.map("      " + _)
              aggUpdates += s"      if ($iok) { long long lv = (long long)($iv); if (gid < RLOC) { l_i64[gid * NA + $ai] += lv; l_cnt[gid * NA + $ai] += 1ULL; } else { atomicAdd((unsigned long long*)&s_i64[gid * NA + $ai], (unsigned long long)lv); atomicAdd(&s_cnt[gid * NA + $ai], 1ULL); } }"
              aggUpdates += "    }"
            case None => return None
          }
        case AGG_MIN_F64 =>
          val st = mutable.ArrayBuffer.empty[String]
          emitExprFull(term.inputExpr.get, st, nextId) match {
            case Some((iv, iok)) =>
              aggUpdates += "    {"
              aggUpdates ++= st.map("      " + _)
              aggUpdates += s"      if ($iok) { double dv = (double)($iv); if (gid < RLOC) { if (l_cnt[gid * NA + $ai] == 0ULL || dv < l_f64[gid * NA + $ai]) l_f64[gid * NA + $ai] = dv; l_cnt[gid * NA + $ai] += 1ULL; } else { atomicMinDouble(&s_f64[gid * NA + $ai], dv); atomicAdd(&s_cnt[gid * NA + $ai], 1ULL); } }"
              aggUpdates += "    }"
            case None => return None
          }
        case AGG_MAX_F64 =>
          val st = mutable.ArrayBuffer.empty[String]
          emitExprFull(term.inputExpr.get, st, nextId) match {
            case Some((iv, iok)) =>
              aggUpdates += "    {"
              aggUpdates ++= st.map("      " + _)
              aggUpdates += s"      if ($iok) { double dv = (double)($iv); if (gid < RLOC) { if (l_cnt[gid * NA + $ai] == 0ULL || dv > l_f64[gid * NA + $ai]) l_f64[gid * NA + $ai] = dv; l_cnt[gid * NA + $ai] += 1ULL; } else { atomicMaxDouble(&s_f64[gid * NA + $ai], dv); atomicAdd(&s_cnt[gid * NA + $ai], 1ULL); } }"
              aggUpdates += "    }"
            case None => return None
          }
        case _ => return None
      }
    }

    val gidCode = if (groupKeySlots.isEmpty) {
      "int32_t gid = 0;"
    } else {
      val sb = new StringBuilder("int32_t gid = 0;\n")
      groupKeySlots.foreach { s =>
        sb.append(s"    { int32_t gk = read_i32(a->cols[$s], row); gid = gid * (int32_t)(a->cols[$s].dict_size > 0 ? a->cols[$s].dict_size : 1) + gk; }\n")
      }
      sb.append("    if (gid < 0 || gid >= a->max_groups) continue;")
      sb.toString()
    }

    val rloc = math.min(maxGroups, if (na <= 12) 8 else 1)
    val flushLocal = aggTerms.zipWithIndex.map { case (t, ai) =>
      t.kind match {
        case AGG_COUNT_I64 =>
          s"    if (l_cnt[g * NA + $ai] > 0ULL) atomicAdd(&s_cnt[g * NA + $ai], l_cnt[g * NA + $ai]);"
        case AGG_SUM_F64 =>
          s"    if (l_cnt[g * NA + $ai] > 0ULL) { atomicAdd(&s_f64[g * NA + $ai], l_f64[g * NA + $ai]); atomicAdd(&s_cnt[g * NA + $ai], l_cnt[g * NA + $ai]); }"
        case AGG_SUM_I64 =>
          s"    if (l_cnt[g * NA + $ai] > 0ULL) { atomicAdd((unsigned long long*)&s_i64[g * NA + $ai], (unsigned long long)l_i64[g * NA + $ai]); atomicAdd(&s_cnt[g * NA + $ai], l_cnt[g * NA + $ai]); }"
        case AGG_MIN_F64 =>
          s"    if (l_cnt[g * NA + $ai] > 0ULL) { atomicMinDouble(&s_f64[g * NA + $ai], l_f64[g * NA + $ai]); atomicAdd(&s_cnt[g * NA + $ai], l_cnt[g * NA + $ai]); }"
        case AGG_MAX_F64 =>
          s"    if (l_cnt[g * NA + $ai] > 0ULL) { atomicMaxDouble(&s_f64[g * NA + $ai], l_f64[g * NA + $ai]); atomicAdd(&s_cnt[g * NA + $ai], l_cnt[g * NA + $ai]); }"
      }
    }.mkString("\n")

    val initShared = aggTerms.zipWithIndex.map { case (t, ai) =>
      val initVal = t.kind match {
        case AGG_MIN_F64 => "__longlong_as_double(0x7ff0000000000000LL)"
        case AGG_MAX_F64 => "__longlong_as_double(0xfff0000000000000LL)"
        case _ => "0.0"
      }
      s"    s_f64[g * NA + $ai] = $initVal; s_i64[g * NA + $ai] = 0LL; s_cnt[g * NA + $ai] = 0ULL;"
    }.mkString("\n")

    val flushGlobal = aggTerms.zipWithIndex.map { case (t, ai) =>
      t.kind match {
        case AGG_COUNT_I64 =>
          s"    if (s_cnt[g * NA + $ai] > 0ULL) atomicAdd(&a->out_cnt[g * NA + $ai], s_cnt[g * NA + $ai]);"
        case AGG_SUM_F64 =>
          s"    if (s_cnt[g * NA + $ai] > 0ULL) { atomicAdd(&a->out_f64[g * NA + $ai], s_f64[g * NA + $ai]); atomicAdd(&a->out_cnt[g * NA + $ai], s_cnt[g * NA + $ai]); }"
        case AGG_SUM_I64 =>
          s"    if (s_cnt[g * NA + $ai] > 0ULL) { atomicAdd((unsigned long long*)&a->out_i64[g * NA + $ai], (unsigned long long)s_i64[g * NA + $ai]); atomicAdd(&a->out_cnt[g * NA + $ai], s_cnt[g * NA + $ai]); }"
        case AGG_MIN_F64 =>
          s"    if (s_cnt[g * NA + $ai] > 0ULL) { atomicMinDouble(&a->out_f64[g * NA + $ai], s_f64[g * NA + $ai]); atomicAdd(&a->out_cnt[g * NA + $ai], s_cnt[g * NA + $ai]); }"
        case AGG_MAX_F64 =>
          s"    if (s_cnt[g * NA + $ai] > 0ULL) { atomicMaxDouble(&a->out_f64[g * NA + $ai], s_f64[g * NA + $ai]); atomicAdd(&a->out_cnt[g * NA + $ai], s_cnt[g * NA + $ai]); }"
      }
    }.mkString("\n")

    val src =
      s"""$CUDA_PREAMBLE
         |extern "C" __global__ void $kname(const JitReduceArgs* __restrict__ a) {
         |  constexpr int NA = $na;
         |  constexpr int MG = $maxGroups;
         |  constexpr int RLOC = $rloc;
         |  __shared__ double s_f64[MG * NA];
         |  __shared__ long long s_i64[MG * NA];
         |  __shared__ unsigned long long s_cnt[MG * NA];
         |  for (int g = threadIdx.x; g < a->max_groups; g += blockDim.x) {
         |$initShared
         |  }
         |  __syncthreads();
         |  double l_f64[RLOC * NA];
         |  long long l_i64[RLOC * NA];
         |  unsigned long long l_cnt[RLOC * NA];
         |  for (int i = 0; i < RLOC * NA; ++i) { l_f64[i] = 0.0; l_i64[i] = 0LL; l_cnt[i] = 0ULL; }
         |  uint32_t stride = gridDim.x * blockDim.x;
         |  uint32_t n = a->num_rows;
         |  for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < n; row += stride) {
         |    if (a->active_chunk_mask != nullptr && a->active_chunk_mask[row >> 10] == 0u) continue;
         |${fStmts.map("    " + _).mkString("\n")}
         |    $fCheck
         |    $gidCode
         |${aggUpdates.mkString("\n")}
         |  }
         |  for (int g = 0; g < RLOC && g < a->max_groups; ++g) {
         |$flushLocal
         |  }
         |  __syncthreads();
         |  for (int g = threadIdx.x; g < a->max_groups; g += blockDim.x) {
         |$flushGlobal
         |  }
         |}
         |""".stripMargin
    Some((src, kname))
  }

  /** Generate Mode 2 (Arbitrary High-Cardinality Conditional GroupBy) generic CUDA kernel!
   * When `na > 8` (wide conditional CASE WHEN aggs), updates output arrays directly without
   * spilling a large `double acc_f64[na]` array to slow CUDA local memory! */
  def generateHighCardKernel(
      filterExprOpt: Option[Expression],
      aggTerms: Seq[GenericAggTerm]): Option[(String, String)] = {
    val na = aggTerms.length
    val kname = "jit_reduce_highcard"
    val directGlobal = na > 8
    var idCounter = 0
    val nextId = () => { idCounter += 1; idCounter }

    val fStmts = mutable.ArrayBuffer.empty[String]
    val fCheck = filterExprOpt match {
      case Some(fe) =>
        emitExprFull(fe, fStmts, nextId) match {
          case Some((fv, fok)) => s"if (!(($fok) && ($fv))) continue;"
          case None => return None
        }
      case None => ""
    }

    val aggUpdates = mutable.ArrayBuffer.empty[String]
    aggTerms.zipWithIndex.foreach { case (term, ai) =>
      val fRef = if (directGlobal) s"a->hc_out_f64[$ai][gid]" else s"acc_f64[$ai]"
      val iRef = if (directGlobal) s"a->hc_out_i64[$ai][gid]" else s"acc_i64[$ai]"
      val vRef = if (directGlobal) s"a->hc_out_valid[$ai][gid]" else s"has_v[$ai]"
      term.kind match {
        case AGG_COUNT_I64 =>
          term.inputExpr match {
            case None => aggUpdates += s"    $iRef += 1LL; $vRef = 1;"
            case Some(ie) =>
              val st = mutable.ArrayBuffer.empty[String]
              emitExprFull(ie, st, nextId) match {
                case Some((_, iok)) =>
                  aggUpdates += "    {"
                  aggUpdates ++= st.map("      " + _)
                  aggUpdates += s"      if ($iok) { $iRef += 1LL; $vRef = 1; }"
                  aggUpdates += "    }"
                case None => return None
              }
          }
        case AGG_SUM_F64 =>
          val st = mutable.ArrayBuffer.empty[String]
          emitExprFull(term.inputExpr.get, st, nextId) match {
            case Some((iv, iok)) =>
              aggUpdates += "    {"
              aggUpdates ++= st.map("      " + _)
              aggUpdates += s"      if ($iok) { $fRef += (double)($iv); $vRef = 1; }"
              aggUpdates += "    }"
            case None => return None
          }
        case AGG_SUM_I64 =>
          val st = mutable.ArrayBuffer.empty[String]
          emitExprFull(term.inputExpr.get, st, nextId) match {
            case Some((iv, iok)) =>
              aggUpdates += "    {"
              aggUpdates ++= st.map("      " + _)
              aggUpdates += s"      if ($iok) { $iRef += (int64_t)($iv); $vRef = 1; }"
              aggUpdates += "    }"
            case None => return None
          }
        case AGG_MIN_F64 =>
          val st = mutable.ArrayBuffer.empty[String]
          emitExprFull(term.inputExpr.get, st, nextId) match {
            case Some((iv, iok)) =>
              aggUpdates += "    {"
              aggUpdates ++= st.map("      " + _)
              aggUpdates += s"      if ($iok) { double dv = (double)($iv); if (!$vRef || dv < $fRef) $fRef = dv; $vRef = 1; }"
              aggUpdates += "    }"
            case None => return None
          }
        case AGG_MAX_F64 =>
          val st = mutable.ArrayBuffer.empty[String]
          emitExprFull(term.inputExpr.get, st, nextId) match {
            case Some((iv, iok)) =>
              aggUpdates += "    {"
              aggUpdates ++= st.map("      " + _)
              aggUpdates += s"      if ($iok) { double dv = (double)($iv); if (!$vRef || dv > $fRef) $fRef = dv; $vRef = 1; }"
              aggUpdates += "    }"
            case None => return None
          }
        case _ => return None
      }
    }

    val initCode = if (directGlobal) {
      aggTerms.zipWithIndex.map { case (t, ai) =>
        val isF64 = t.kind == AGG_SUM_F64 || t.kind == AGG_MIN_F64 || t.kind == AGG_MAX_F64
        if (isF64) s"  a->hc_out_f64[$ai][gid] = 0.0; a->hc_out_valid[$ai][gid] = 0;"
        else s"  a->hc_out_i64[$ai][gid] = 0LL; a->hc_out_valid[$ai][gid] = 0;"
      }.mkString("\n")
    } else {
      s"  double acc_f64[$na]; int64_t acc_i64[$na]; uint8_t has_v[$na];\n" +
      s"  for (int k = 0; k < $na; ++k) { acc_f64[k] = 0.0; acc_i64[k] = 0LL; has_v[k] = 0; }"
    }

    val writeOut = if (directGlobal) "" else {
      aggTerms.zipWithIndex.map { case (t, ai) =>
        val isF64 = t.kind == AGG_SUM_F64 || t.kind == AGG_MIN_F64 || t.kind == AGG_MAX_F64
        if (isF64) s"  a->hc_out_f64[$ai][gid] = acc_f64[$ai]; a->hc_out_valid[$ai][gid] = has_v[$ai];"
        else s"  a->hc_out_i64[$ai][gid] = acc_i64[$ai]; a->hc_out_valid[$ai][gid] = has_v[$ai];"
      }.mkString("\n")
    }

    // Also generate a block-parallel variant (128 threads per group) for low-cardinality
    // post-join aggs (e.g. Q66 20 warehouses x 24 CASE WHEN DECIMAL sums) so num_groups < 1024
    // saturates all SMs with zero atomics and shared-memory reduction!
    val blkIdCounter = new Array[Int](1)
    val nextBlkId = () => { blkIdCounter(0) += 1; blkIdCounter(0) }
    val blkFStmts = mutable.ArrayBuffer.empty[String]
    val blkFCheck = filterExprOpt match {
      case Some(fe) =>
        emitExprFull(fe, blkFStmts, nextBlkId) match {
          case Some((fv, fok)) => s"if (!(($fok) && ($fv))) continue;"
          case None => return None
        }
      case None => ""
    }
    val anyF64 = aggTerms.exists(t =>
      t.kind == AGG_SUM_F64 || t.kind == AGG_MIN_F64 || t.kind == AGG_MAX_F64)
    val anyI64 = aggTerms.exists(t => t.kind == AGG_SUM_I64 || t.kind == AGG_COUNT_I64)
    val blkUpdates = mutable.ArrayBuffer.empty[String]
    aggTerms.zipWithIndex.foreach { case (term, ai) =>
      term.kind match {
        case AGG_COUNT_I64 =>
          term.inputExpr match {
            case None => blkUpdates += s"    b_i64[$ai] += 1LL; b_v[$ai] = 1;"
            case Some(ie) =>
              val st = mutable.ArrayBuffer.empty[String]
              emitExprFull(ie, st, nextBlkId) match {
                case Some((_, iok)) =>
                  blkUpdates += "    {"
                  blkUpdates ++= st.map("      " + _)
                  blkUpdates += s"      if ($iok) { b_i64[$ai] += 1LL; b_v[$ai] = 1; }"
                  blkUpdates += "    }"
                case None => return None
              }
          }
        case AGG_SUM_F64 =>
          val st = mutable.ArrayBuffer.empty[String]
          emitExprFull(term.inputExpr.get, st, nextBlkId) match {
            case Some((iv, iok)) =>
              blkUpdates += "    {"
              blkUpdates ++= st.map("      " + _)
              blkUpdates += s"      if ($iok) { b_f64[$ai] += (double)($iv); b_v[$ai] = 1; }"
              blkUpdates += "    }"
            case None => return None
          }
        case AGG_SUM_I64 =>
          val st = mutable.ArrayBuffer.empty[String]
          emitExprFull(term.inputExpr.get, st, nextBlkId) match {
            case Some((iv, iok)) =>
              blkUpdates += "    {"
              blkUpdates ++= st.map("      " + _)
              blkUpdates += s"      if ($iok) { b_i64[$ai] += (int64_t)($iv); b_v[$ai] = 1; }"
              blkUpdates += "    }"
            case None => return None
          }
        case AGG_MIN_F64 =>
          val st = mutable.ArrayBuffer.empty[String]
          emitExprFull(term.inputExpr.get, st, nextBlkId) match {
            case Some((iv, iok)) =>
              blkUpdates += "    {"
              blkUpdates ++= st.map("      " + _)
              blkUpdates += s"      if ($iok) { double dv = (double)($iv); if (!b_v[$ai] || dv < b_f64[$ai]) b_f64[$ai] = dv; b_v[$ai] = 1; }"
              blkUpdates += "    }"
            case None => return None
          }
        case AGG_MAX_F64 =>
          val st = mutable.ArrayBuffer.empty[String]
          emitExprFull(term.inputExpr.get, st, nextBlkId) match {
            case Some((iv, iok)) =>
              blkUpdates += "    {"
              blkUpdates ++= st.map("      " + _)
              blkUpdates += s"      if ($iok) { double dv = (double)($iv); if (!b_v[$ai] || dv > b_f64[$ai]) b_f64[$ai] = dv; b_v[$ai] = 1; }"
              blkUpdates += "    }"
            case None => return None
          }
        case _ => return None
      }
    }
    val blkReduceSteps = aggTerms.zipWithIndex.map { case (t, ai) =>
      t.kind match {
        case AGG_SUM_F64 =>
          s"""  s_f64[tid] = b_v[$ai] ? b_f64[$ai] : 0.0; s_v[tid] = b_v[$ai];
             |  __syncthreads();
             |  for (int off = 64; off > 0; off >>= 1) {
             |    if (tid < off) { s_f64[tid] += s_f64[tid + off]; s_v[tid] |= s_v[tid + off]; }
             |    __syncthreads();
             |  }
             |  if (tid == 0) { a->hc_out_f64[$ai][gid] = s_f64[0]; a->hc_out_valid[$ai][gid] = s_v[0]; }
             |  __syncthreads();""".stripMargin
        case AGG_MIN_F64 =>
          s"""  s_f64[tid] = b_f64[$ai]; s_v[tid] = b_v[$ai];
             |  __syncthreads();
             |  for (int off = 64; off > 0; off >>= 1) {
             |    if (tid < off && s_v[tid + off]) {
             |      if (!s_v[tid] || s_f64[tid + off] < s_f64[tid]) s_f64[tid] = s_f64[tid + off];
             |      s_v[tid] = 1;
             |    }
             |    __syncthreads();
             |  }
             |  if (tid == 0) { a->hc_out_f64[$ai][gid] = s_f64[0]; a->hc_out_valid[$ai][gid] = s_v[0]; }
             |  __syncthreads();""".stripMargin
        case AGG_MAX_F64 =>
          s"""  s_f64[tid] = b_f64[$ai]; s_v[tid] = b_v[$ai];
             |  __syncthreads();
             |  for (int off = 64; off > 0; off >>= 1) {
             |    if (tid < off && s_v[tid + off]) {
             |      if (!s_v[tid] || s_f64[tid + off] > s_f64[tid]) s_f64[tid] = s_f64[tid + off];
             |      s_v[tid] = 1;
             |    }
             |    __syncthreads();
             |  }
             |  if (tid == 0) { a->hc_out_f64[$ai][gid] = s_f64[0]; a->hc_out_valid[$ai][gid] = s_v[0]; }
             |  __syncthreads();""".stripMargin
        case _ =>
          s"""  s_i64[tid] = b_v[$ai] ? b_i64[$ai] : 0LL; s_v[tid] = b_v[$ai];
             |  __syncthreads();
             |  for (int off = 64; off > 0; off >>= 1) {
             |    if (tid < off) { s_i64[tid] += s_i64[tid + off]; s_v[tid] |= s_v[tid + off]; }
             |    __syncthreads();
             |  }
             |  if (tid == 0) { a->hc_out_i64[$ai][gid] = s_i64[0]; a->hc_out_valid[$ai][gid] = s_v[0]; }
             |  __syncthreads();""".stripMargin
      }
    }.mkString("\n")
    val blkDecl = (if (anyF64) s"  double b_f64[$na]; __shared__ double s_f64[128];\n" else "") +
      (if (anyI64) s"  int64_t b_i64[$na]; __shared__ int64_t s_i64[128];\n" else "") +
      s"  uint8_t b_v[$na]; __shared__ uint8_t s_v[128];\n" +
      s"  for (int k = 0; k < $na; ++k) { " +
      (if (anyF64) "b_f64[k] = 0.0; " else "") +
      (if (anyI64) "b_i64[k] = 0LL; " else "") +
      "b_v[k] = 0; }"

    val src =
      s"""$CUDA_PREAMBLE
         |extern "C" __global__ void $kname(const JitReduceArgs* __restrict__ a) {
         |  int32_t gid = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
         |  if (gid >= a->num_groups) return;
         |  int32_t beg = a->group_offsets[gid];
         |  int32_t end = a->group_offsets[gid + 1];
         |$initCode
         |  for (int32_t pos = beg; pos < end; ++pos) {
         |    uint32_t row = (uint32_t)a->grouped_row_ids[pos];
         |${fStmts.map("    " + _).mkString("\n")}
         |    $fCheck
         |${aggUpdates.mkString("\n")}
         |  }
         |$writeOut
         |}
         |extern "C" __global__ void ${kname}_blk(const JitReduceArgs* __restrict__ a) {
         |  int32_t gid = (int32_t)blockIdx.x;
         |  if (gid >= a->num_groups) return;
         |  int tid = threadIdx.x;
         |  int32_t beg = a->group_offsets[gid];
         |  int32_t end = a->group_offsets[gid + 1];
         |$blkDecl
         |  for (int32_t pos = beg + tid; pos < end; pos += 128) {
         |    uint32_t row = (uint32_t)a->grouped_row_ids[pos];
         |${blkFStmts.map("    " + _).mkString("\n")}
         |    $blkFCheck
         |${blkUpdates.mkString("\n")}
         |  }
         |$blkReduceSteps
         |}
         |""".stripMargin
    Some((src, kname))
  }
}

/**
 * Generic Planner Rule: Automatically fuses ANY compatible Partial `GpuHashAggregateExec`
 * (Mode 0 Global Reduce, Mode 1 Low-Card GroupBy <= 64, Mode 2 High-Card GroupBy)
 * over EITHER `.lenc` OR Zero-Rewrite Parquet! Zero query-specific hardcoding!
 */
object GpuLencFusedAgg {
  val FUSED_AGG_CONF = "spark.rapids.sql.lenc.fusedAggEnabled"
  val BUF_NORMAL: Byte = 0
  val BUF_DEC_SUM: Byte = 1
  val BUF_DEC_IS_EMPTY: Byte = 2
  val BUF_COUNT_NON_NULL: Byte = 3
  val BUF_F64_SUM_NON_NULL: Byte = 4

  private def inlineProjectMap(e: Expression, projMap: Map[ExprId, Expression]): Expression = {
    e.transformDown {
      case a: Attribute if projMap.contains(a.exprId) => projMap(a.exprId)
    }
  }

  private def extractAggTermsAndBufferMap(
      gpuAggs: Seq[GpuAggregateExpression],
      projMap: Map[ExprId, Expression]): Option[(Seq[GenericAggTerm], Array[Int], Array[Byte])] = {
    // Each Spark partial aggregate may emit 1 (SUM/COUNT/MIN/MAX) or 2 (AVG: sum+count, DecimalSum: sum+isEmpty) buffer cols
    val terms = mutable.ArrayBuffer.empty[GenericAggTerm]
    val outBufToTerm = mutable.ArrayBuffer.empty[Int]
    val outBufModes = mutable.ArrayBuffer.empty[Byte]

    def addTerm(t: GenericAggTerm): Int = {
      val idx = terms.indexOf(t)
      if (idx >= 0) idx else { terms += t; terms.length - 1 }
    }

    def applyFilter(ch: Expression, filterOpt: Option[Expression]): Expression = filterOpt match {
      case Some(f) => GpuIf(inlineProjectMap(f, projMap), ch, GpuLiteral(null, ch.dataType))
      case None => ch
    }

    gpuAggs.foreach { gae =>
      if (gae.isDistinct) return None
      val (aggFn, fOpt) = gae.aggregateFunction match {
        case w: WrappedAggFunction =>
          val combined = gae.filter.map(GpuAnd(_, w.filter)).getOrElse(w.filter)
          (w.aggregateFunction, Some(combined))
        case other => (other, gae.filter)
      }
      aggFn match {
        case ds: GpuDecimalSum =>
          val failOnErr = ds match {
            case b: GpuBasicDecimalSum => b.failOnErrorOverride
            case d128: GpuDecimal128Sum => d128.failOnErrorOverride
            case _ => true
          }
          if (failOnErr) return None
          val rawCh = inlineProjectMap(ds.children.head, projMap)
          val bufDt = ds.dataType match {
            case d: DecimalType => d
            case _ => return None
          }
          // Safe for int64_t per-batch accumulation when output expression has scale == bufDt.scale
          // and all leaf decimal input columns fit in DECIMAL32 (p <= 9, e.g. NDS DECIMAL(7,2))!
          val leafDecimalsSafe = rawCh.collect {
            case a: Attribute => a.dataType
          }.forall {
            case d: DecimalType => d.precision <= 9
            case _ => true
          }
          rawCh.dataType match {
            case d: DecimalType if d.precision <= 38 && d.scale == bufDt.scale && leafDecimalsSafe =>
            case _ => return None
          }
          val ch = applyFilter(rawCh, fOpt)
          val tid = addTerm(GenericAggTerm(GenericJitReduceCodegen.AGG_SUM_I64, Some(ch)))
          outBufToTerm += tid; outBufModes += BUF_DEC_SUM
          outBufToTerm += tid; outBufModes += BUF_DEC_IS_EMPTY

        case s: GpuSum =>
          val ch = applyFilter(inlineProjectMap(s.children.head, projMap), fOpt)
          val kind = s.dataType match {
            case DoubleType | FloatType => GenericJitReduceCodegen.AGG_SUM_F64
            case LongType | IntegerType => GenericJitReduceCodegen.AGG_SUM_I64
            case _ => return None
          }
          outBufToTerm += addTerm(GenericAggTerm(kind, Some(ch)))
          outBufModes += BUF_NORMAL

        case da: GpuDecimalAverageBase =>
          if (SQLConf.get.ansiEnabled) return None
          val rawCh = inlineProjectMap(da.children.head, projMap)
          val sumDt = da.aggBufferAttributes.head.dataType match {
            case d: DecimalType => d
            case _ => return None
          }
          rawCh.dataType match {
            case d: DecimalType if d.precision <= 12 && d.scale == sumDt.scale =>
            case _ => return None
          }
          val ch = applyFilter(rawCh, fOpt)
          outBufToTerm += addTerm(GenericAggTerm(GenericJitReduceCodegen.AGG_SUM_I64, Some(ch)))
          outBufModes += BUF_DEC_SUM
          outBufToTerm += addTerm(GenericAggTerm(GenericJitReduceCodegen.AGG_COUNT_I64, Some(ch)))
          outBufModes += BUF_COUNT_NON_NULL

        case a: GpuAverage =>
          val ch = applyFilter(inlineProjectMap(a.children.head, projMap), fOpt)
          if (a.dataType != DoubleType && a.dataType != FloatType) return None
          outBufToTerm += addTerm(GenericAggTerm(GenericJitReduceCodegen.AGG_SUM_F64, Some(ch)))
          outBufModes += BUF_F64_SUM_NON_NULL
          outBufToTerm += addTerm(GenericAggTerm(GenericJitReduceCodegen.AGG_COUNT_I64, Some(ch)))
          outBufModes += BUF_COUNT_NON_NULL

        case c: GpuCount =>
          val chs = c.children.map(inlineProjectMap(_, projMap))
          if (chs.forall {
            case lit: GpuLiteral => lit.value != null
            case lit: Literal => lit.value != null
            case _ => false
          }) {
            val exprOpt = fOpt.map(f => GpuIf(inlineProjectMap(f, projMap), GpuLiteral(1, IntegerType), GpuLiteral(null, IntegerType)))
            outBufToTerm += addTerm(GenericAggTerm(GenericJitReduceCodegen.AGG_COUNT_I64, exprOpt))
            outBufModes += BUF_COUNT_NON_NULL
          } else if (chs.length == 1) {
            val ch = applyFilter(chs.head, fOpt)
            outBufToTerm += addTerm(GenericAggTerm(GenericJitReduceCodegen.AGG_COUNT_I64, Some(ch)))
            outBufModes += BUF_COUNT_NON_NULL
          } else {
            return None
          }

        case mn: GpuMin =>
          val ch = applyFilter(inlineProjectMap(mn.children.head, projMap), fOpt)
          if (mn.dataType != DoubleType && mn.dataType != FloatType) return None
          outBufToTerm += addTerm(GenericAggTerm(GenericJitReduceCodegen.AGG_MIN_F64, Some(ch)))
          outBufModes += BUF_NORMAL

        case mx: GpuMax =>
          val ch = applyFilter(inlineProjectMap(mx.children.head, projMap), fOpt)
          if (mx.dataType != DoubleType && mx.dataType != FloatType) return None
          outBufToTerm += addTerm(GenericAggTerm(GenericJitReduceCodegen.AGG_MAX_F64, Some(ch)))
          outBufModes += BUF_NORMAL

        case _ => return None
      }
    }
    if (terms.isEmpty || terms.length > 64) None
    else Some((terms.toSeq, outBufToTerm.toArray, outBufModes.toArray))
  }

  private def bindAndRewriteDicts(
      filterOpt: Option[Expression],
      terms: Seq[GenericAggTerm],
      baseOutput: Seq[Attribute],
      extraRequiredAttrs: Seq[Attribute]): Option[(
      Array[Int],
      Option[Expression],
      Seq[GenericAggTerm],
      Array[Array[String]])] = {
    val idToOrd = baseOutput.zipWithIndex.map { case (a, i) => a.exprId -> i }.toMap
    val refOrds = mutable.LinkedHashSet.empty[Int]

    def collectRefs(e: Expression): Boolean = {
      var ok = true
      e.foreach {
        case a: Attribute =>
          idToOrd.get(a.exprId) match {
            case Some(ord) => refOrds += ord
            case None => ok = false
          }
        case _ =>
      }
      ok
    }

    extraRequiredAttrs.foreach { a =>
      idToOrd.get(a.exprId) match {
        case Some(ord) => refOrds += ord
        case None => return None
      }
    }
    if (!filterOpt.forall(collectRefs)) return None
    if (!terms.forall(t => t.inputExpr.forall(collectRefs))) return None
    val sortedOrds = refOrds.toArray.sorted
    if (sortedOrds.isEmpty || sortedOrds.length > 16) return None
    val ordToSlot = sortedOrds.zipWithIndex.toMap

    def bind(e: Expression): Expression = e.transformDown {
      case a: Attribute =>
        val slot = ordToSlot(idToOrd(a.exprId))
        GpuBoundReference(slot, a.dataType, a.nullable)(a.exprId, a.name)
    }

    var boundFilter = filterOpt.map(bind)
    var boundTerms = terms.map(t => t.copy(inputExpr = t.inputExpr.map(bind)))
    val dictLits = Array.fill[Array[String]](sortedOrds.length)(Array.empty[String])

    // Rewrite String predicate columns (non-group-key) to canonical INT32 dict codes!
    val extraSet = extraRequiredAttrs.map(_.exprId).toSet
    sortedOrds.zipWithIndex.foreach { case (oldOrd, slot) =>
      val attr = baseOutput(oldOrd)
      if (attr.dataType == StringType && !extraSet.contains(attr.exprId)) {
        val litSet = mutable.HashSet.empty[String]
        val fOk = boundFilter.forall(GpuDictAwareAstJit.collectDictPredicates(_, slot, litSet))
        val tOk = boundTerms.forall(t => t.inputExpr.forall(GpuDictAwareAstJit.collectDictPredicates(_, slot, litSet)))
        if (!fOk || !tOk || litSet.isEmpty) {
          return None
        }
        val sortedLits = litSet.toArray.sorted
        val litToId = sortedLits.zipWithIndex.toMap
        dictLits(slot) = sortedLits
        boundFilter = boundFilter.map(GpuDictAwareAstJit.rewriteExprForDict(_, slot, litToId, sortedLits.length))
        boundTerms = boundTerms.map(t =>
          t.copy(inputExpr = t.inputExpr.map(GpuDictAwareAstJit.rewriteExprForDict(_, slot, litToId, sortedLits.length))))
      }
    }
    Some((sortedOrds, boundFilter, boundTerms, dictLits))
  }

  def tryFuse(
      gpuGrouping: Seq[NamedExpression],
      gpuAggs: Seq[GpuAggregateExpression],
      resultExprs: Seq[NamedExpression],
      gpuChild: SparkPlan): Option[GpuExec] = {
    if (!SQLConf.get.getConfString(FUSED_AGG_CONF, "false").toBoolean) {
      return None
    }
    if (gpuAggs.isEmpty || !gpuAggs.forall(_.mode == Partial)) {
      return None
    }

    def peelCoalesce(p: SparkPlan): SparkPlan = p match {
      case c: GpuCoalesceBatches => peelCoalesce(c.child)
      case other => other
    }

    // Peel optional GpuCoalesceBatches & GpuProjectExec (unwrapping GpuAstJitExpression if present!)
    val (afterProj, projMap) = peelCoalesce(gpuChild) match {
      case p: GpuProjectExec =>
        val m = p.projectList.collect {
          case a: GpuAlias =>
            val inner = a.child match {
              case j: GpuAstJitExpression => j.child
              case other => other
            }
            a.exprId -> inner
        }.toMap
        (peelCoalesce(p.child), m)
      case other =>
        (other, Map.empty[ExprId, Expression])
    }

    def unwrapJit(e: Expression): Expression = e.transformDown {
      case j: GpuAstJitExpression => j.child
    }

    val (rawTerms0, outBufToTerm, outBufModes) =
      extractAggTermsAndBufferMap(gpuAggs, projMap).getOrElse(return None)
    val rawTerms = rawTerms0.map(t => t.copy(inputExpr = t.inputExpr.map(unwrapJit)))
    if (resultExprs.length != gpuGrouping.length + outBufToTerm.length) {
      return None
    }
    val outAttrs = resultExprs.map(_.toAttribute)
    val outTypes = resultExprs.map(_.dataType).toArray

    // Unwrap optional GpuFilterExec directly over any GPU scan / child plan (Parquet, Vortex, etc.)
    val (basePlan, filterExprOpt): (SparkPlan, Option[Expression]) =
      afterProj match {
        case f: GpuFilterExec => (f.child, Some(unwrapJit(f.condition)))
        case other => (other, None)
      }

    // Mode 0: Global Conditional Reduce (0 group keys)
    if (gpuGrouping.isEmpty) {
      return tryBuildGlobalOrLowCard(
        outAttrs, outTypes, Seq.empty, filterExprOpt, rawTerms, outBufToTerm, outBufModes,
        basePlan.output, basePlan)
    }

    // Resolve grouping key attributes on basePlan.output
    val groupAttrs = gpuGrouping.map { g =>
      inlineProjectMap(g, projMap) match {
        case a: Attribute => a
        case _ => return None
      }
    }

    // On many-file zero-copy Vortex scans (>200 files) with low-card string group keys and modest
    // agg count (e.g. SF1000 Q01 with 1,982 files), per-batch D2H sync stalls GPU pipelining;
    // let AST-JIT + async GpuHashAggregate handle it cleanly without falling into Mode 2 high-card!
    if (groupAttrs.nonEmpty && groupAttrs.length <= 2 &&
        groupAttrs.forall(_.dataType == StringType) && rawTerms.length < 12) {
      val manyFileVortex = basePlan match {
        case bse: com.nvidia.spark.rapids.shims.GpuBatchScanExecBase =>
          bse.scan match {
            case vs: com.nvidia.spark.rapids.vortex.GpuVortexScan => vs.files.length > 200
            case _ => false
          }
        case _ => false
      }
      if (manyFileVortex) return None
    }

    // Mode 1: Low-Cardinality String GroupBy (1-2 string keys directly over LeafExecNode scan, e.g. Q1/U2!)
    // Restrict to direct LeafExecNode scans (not post-join high-card keys like Q21 s_name!).
    val isLeafScan = basePlan.isInstanceOf[LeafExecNode]
    val preferHighCard = !isLeafScan ||
      groupAttrs.length >= 3 ||
      groupAttrs.exists(_.dataType != StringType) ||
      rawTerms.length >= 12
    if (!preferHighCard && groupAttrs.length <= 2 && groupAttrs.forall(_.dataType == StringType)) {
      val m1 = tryBuildGlobalOrLowCard(
        outAttrs, outTypes, groupAttrs, filterExprOpt, rawTerms, outBufToTerm, outBufModes,
        basePlan.output, basePlan)
      if (m1.isDefined) return m1
    }

    // Mode 2: Arbitrary High-Card & Block-Parallel Low-Card Conditional GroupBy over BOTH
    // LeafExecNode scans (NF_J0048 36-way pivot) AND Post-Join plans (TPC-DS Q66 24-way CASE WHEN
    // DECIMAL pivot, Q04 complex DECIMAL expressions, etc.). Keep GpuFilterExec below Mode 2
    // (filterOpt = None) so cudf::groupby::get_groups never emits empty groups for filtered rows!
    def hasConditionalBranch(e: Expression): Boolean = {
      var hasBranch = false
      e.foreach {
        case _: GpuCaseWhen | _: GpuIf => hasBranch = true
        case _ =>
      }
      hasBranch
    }
    val hasBranchTerm = rawTerms.exists(t => t.inputExpr.exists(hasConditionalBranch))
    val admitMode2 = (isLeafScan && rawTerms.length >= 4) ||
      (!isLeafScan && (rawTerms.length >= 3 || hasBranchTerm))
    if (admitMode2) {
      val mode2Plan = afterProj
      val idToOrd = mode2Plan.output.zipWithIndex.map { case (a, i) => a.exprId -> i }.toMap
      val groupOrds = groupAttrs.map(a => idToOrd.getOrElse(a.exprId, return None)).toArray
      bindAndRewriteDicts(None, rawTerms, mode2Plan.output, Seq.empty).flatMap {
        case (valOrds, _, boundTerms, dictLits) =>
          GenericJitReduceCodegen.generateHighCardKernel(None, boundTerms).map {
            case (cudaSrc, kname) =>
              val aggKinds = boundTerms.map(_.kind).toArray
              val hcInput: SparkPlan = mode2Plan match {
                case bse: com.nvidia.spark.rapids.shims.GpuBatchScanExecBase
                    if bse.scan.isInstanceOf[com.nvidia.spark.rapids.vortex.GpuVortexScan] =>
                  GpuCoalesceBatches(mode2Plan, TargetSize(256L * 1024L * 1024L))
                case _ => mode2Plan
              }
              GpuGenericHighCardJitAggExec(
                outAttrs, hcInput, groupOrds, valOrds, dictLits, aggKinds, outBufToTerm,
                outBufModes, outTypes, cudaSrc, kname)
          }
      }
    } else {
      None
    }
  }

  private def tryBuildGlobalOrLowCard(
      outAttrs: Seq[Attribute],
      outTypes: Array[DataType],
      groupAttrs: Seq[Attribute],
      filterExprOpt: Option[Expression],
      rawTerms: Seq[GenericAggTerm],
      outBufToTerm: Array[Int],
      outBufModes: Array[Byte],
      baseOutput: Seq[Attribute],
      basePlan: SparkPlan): Option[GpuExec] = {
    val maxGroups = if (groupAttrs.isEmpty) 1 else 64
    bindAndRewriteDicts(filterExprOpt, rawTerms, baseOutput, groupAttrs).flatMap {
      case (inputOrds, boundFilter, boundTerms, dictLits) =>
        val idToOrd = baseOutput.zipWithIndex.map { case (a, i) => a.exprId -> i }.toMap
        val ordToSlot = inputOrds.zipWithIndex.toMap
        val gkSlots = groupAttrs.map(a => ordToSlot(idToOrd(a.exprId))).toArray
        GenericJitReduceCodegen.generateLowCardKernel(boundFilter, gkSlots, maxGroups, boundTerms).map {
          case (cudaSrc, kname) =>
            // Keep string group keys & literal-compared columns as zero-copy DICTIONARY32 directly
            // from GpuVortexScan (eliminates thousands of redundant GPU dictionary::encode sorts on Q01!)
            basePlan match {
              case bse: com.nvidia.spark.rapids.shims.GpuBatchScanExecBase =>
                bse.scan match {
                  case vs: com.nvidia.spark.rapids.vortex.GpuVortexScan =>
                    val strCols = scala.collection.mutable.HashSet[String]()
                    groupAttrs.foreach { a =>
                      if (a.dataType == StringType) strCols += a.name
                    }
                    inputOrds.zipWithIndex.foreach { case (ord, slot) =>
                      if (slot < dictLits.length && dictLits(slot).nonEmpty &&
                          baseOutput(ord).dataType == StringType) {
                        strCols += baseOutput(ord).name
                      }
                    }
                    if (strCols.nonEmpty) vs.addDict32Cols(strCols.toSet)
                  case _ =>
                }
              case _ =>
            }
            val aggKinds = boundTerms.map(_.kind).toArray
            val reduceInput: SparkPlan = basePlan match {
              case bse: com.nvidia.spark.rapids.shims.GpuBatchScanExecBase =>
                bse.scan match {
                  case vs: com.nvidia.spark.rapids.vortex.GpuVortexScan if vs.files.length > 64 =>
                    GpuCoalesceBatches(basePlan, TargetSize(256L * 1024L * 1024L))
                  case _ => basePlan
                }
              case _ => basePlan
            }
            GpuGenericTableJitReduceExec(
              outAttrs, reduceInput, inputOrds, dictLits, gkSlots, maxGroups,
              aggKinds, outBufToTerm, outBufModes, outTypes, cudaSrc, kname)
        }
    }
  }
}

private object GenericReduceBatchBuilder {
  def projectBuffers(
      rawTbl: CudfTable,
      numGroupKeys: Int,
      outBufToTerm: Array[Int],
      outBufModes: Array[Byte],
      outTypes: Array[DataType]): ColumnarBatch = {
    val totalOut = numGroupKeys + outBufToTerm.length
    val cols = (0 until totalOut).safeMap { i =>
      val nativeIdx = if (i < numGroupKeys) i else (numGroupKeys + outBufToTerm(i - numGroupKeys))
      val mode = if (i < numGroupKeys) GpuLencFusedAgg.BUF_NORMAL else outBufModes(i - numGroupKeys)
      val cv = rawTbl.getColumn(nativeIdx)
      val expectedDType = GpuColumnVector.getNonNestedRapidsType(outTypes(i))
      val typedCol = mode match {
        case GpuLencFusedAgg.BUF_DEC_SUM =>
          val scale = outTypes(i).asInstanceOf[DecimalType].scale
          val dec64Type = DType.create(DType.DTypeEnum.DECIMAL64, -scale)
          withResource(Scalar.fromLong(0L)) { zero =>
            withResource(cv.replaceNulls(zero)) { nonNullI64 =>
              withResource(nonNullI64.bitCastTo(dec64Type)) { dec64 =>
                if (expectedDType != dec64Type) dec64.castTo(expectedDType)
                else dec64.copyToColumnVector()
              }
            }
          }
        case GpuLencFusedAgg.BUF_DEC_IS_EMPTY =>
          cv.isNull()
        case GpuLencFusedAgg.BUF_COUNT_NON_NULL =>
          withResource(Scalar.fromLong(0L)) { zero =>
            withResource(cv.replaceNulls(zero)) { nonNullI64 =>
              if (nonNullI64.getType != expectedDType) nonNullI64.castTo(expectedDType)
              else nonNullI64.incRefCount()
            }
          }
        case GpuLencFusedAgg.BUF_F64_SUM_NON_NULL =>
          withResource(Scalar.fromDouble(0.0)) { zero =>
            withResource(cv.replaceNulls(zero)) { nonNullF64 =>
              if (nonNullF64.getType != expectedDType) nonNullF64.castTo(expectedDType)
              else nonNullF64.incRefCount()
            }
          }
        case _ =>
          if (cv.getType != expectedDType) cv.castTo(expectedDType) else cv.incRefCount()
      }
      GpuColumnVector.from(typedCol, outTypes(i))
    }
    new ColumnarBatch(
      cols.map(_.asInstanceOf[org.apache.spark.sql.vectorized.ColumnVector]).toArray,
      rawTbl.getRowCount.toInt)
  }
}

case class GpuGenericTableJitReduceExec(
    output: Seq[Attribute],
    child: SparkPlan,
    inputOrds: Array[Int],
    dictLits: Array[Array[String]],
    gkSlots: Array[Int],
    maxGroups: Int,
    aggKinds: Array[Byte],
    outBufToTerm: Array[Int],
    outBufModes: Array[Byte],
    outTypes: Array[DataType],
    cudaSrc: String,
    kernelName: String) extends UnaryExecNode with GpuExec {

  override def supportsColumnar: Boolean = true

  override protected def doExecute(): RDD[org.apache.spark.sql.catalyst.InternalRow] =
    throw new IllegalStateException("GpuGenericTableJitReduceExec must execute columnar on GPU")

  override lazy val additionalMetrics: Map[String, GpuMetric] = Map(
    OP_TIME_LEGACY -> createNanoTimingMetric(DEBUG_LEVEL, DESCRIPTION_OP_TIME_LEGACY)
  )

  override def internalDoExecuteColumnar(): RDD[ColumnarBatch] = {
    val opTime = allMetrics.getOrElse(OP_TIME_LEGACY, NoopMetric)
    val numOutputRows = allMetrics.getOrElse(NUM_OUTPUT_ROWS, NoopMetric)
    val numOutputBatches = allMetrics.getOrElse(NUM_OUTPUT_BATCHES, NoopMetric)
    val ords = inputOrds
    val dlits = dictLits
    val gks = gkSlots
    val mg = maxGroups
    val ak = aggKinds
    val obm = outBufToTerm
    val obModes = outBufModes
    val oTypes = outTypes
    val src = cudaSrc
    val kn = kernelName

    child.executeColumnar().mapPartitions { iter =>
      iter.flatMap { batch =>
        opTime.ns {
          withResource(batch) { b =>
            if (b.numRows() == 0) None
            else {
              val cols = ords.map(o => b.column(o).asInstanceOf[GpuColumnVector].getBase)
              withResource(new CudfTable(cols: _*)) { t =>
                val res = GpuJitReducePool.withProgram(src, kn) { prog =>
                  prog.runOnTable(t, dlits, gks, mg, ak)
                }
                withResource(res) { r =>
                  if (r.getRowCount > 0) {
                    numOutputRows += r.getRowCount
                    numOutputBatches += 1
                    Some(GenericReduceBatchBuilder.projectBuffers(r, gks.length, obm, obModes, oTypes))
                  } else None
                }
              }
            }
          }
        }
      }
    }
  }

  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan =
    copy(child = newChild)

  override def simpleString(maxFields: Int): String =
    s"GpuGenericTableJitReduce(gk=${gkSlots.length}, aggs=${aggKinds.length})"
}

case class GpuGenericHighCardJitAggExec(
    output: Seq[Attribute],
    child: SparkPlan,
    groupOrds: Array[Int],
    valOrds: Array[Int],
    dictLits: Array[Array[String]],
    aggKinds: Array[Byte],
    outBufToTerm: Array[Int],
    outBufModes: Array[Byte],
    outTypes: Array[DataType],
    cudaSrc: String,
    kernelName: String) extends UnaryExecNode with GpuExec {

  override def supportsColumnar: Boolean = true

  override protected def doExecute(): RDD[org.apache.spark.sql.catalyst.InternalRow] =
    throw new IllegalStateException("GpuGenericHighCardJitAggExec must execute columnar on GPU")

  override lazy val additionalMetrics: Map[String, GpuMetric] = Map(
    OP_TIME_LEGACY -> createNanoTimingMetric(DEBUG_LEVEL, DESCRIPTION_OP_TIME_LEGACY)
  )

  override def internalDoExecuteColumnar(): RDD[ColumnarBatch] = {
    val opTime = allMetrics.getOrElse(OP_TIME_LEGACY, NoopMetric)
    val numOutputRows = allMetrics.getOrElse(NUM_OUTPUT_ROWS, NoopMetric)
    val numOutputBatches = allMetrics.getOrElse(NUM_OUTPUT_BATCHES, NoopMetric)
    val gOrds = groupOrds
    val vOrds = valOrds
    val dlits = dictLits
    val ak = aggKinds
    val obm = outBufToTerm
    val obModes = outBufModes
    val oTypes = outTypes
    val src = cudaSrc
    val kn = kernelName

    child.executeColumnar().mapPartitions { iter =>
      iter.flatMap { batch =>
        opTime.ns {
          withResource(batch) { b =>
            if (b.numRows() == 0) None
            else {
              val gCols = gOrds.map(o => b.column(o).asInstanceOf[GpuColumnVector].getBase)
              val vCols = vOrds.map(o => b.column(o).asInstanceOf[GpuColumnVector].getBase)
              withResource(new CudfTable(gCols: _*)) { gkTbl =>
                withResource(new CudfTable(vCols: _*)) { valTbl =>
                  val res = GpuJitReducePool.withProgram(src, kn) { prog =>
                    prog.runHighCardGroupBy(gkTbl, valTbl, dlits, ak)
                  }
                  withResource(res) { r =>
                    if (r.getRowCount > 0) {
                      numOutputRows += r.getRowCount
                      numOutputBatches += 1
                      Some(GenericReduceBatchBuilder.projectBuffers(r, gOrds.length, obm, obModes, oTypes))
                    } else None
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan =
    copy(child = newChild)

  override def simpleString(maxFields: Int): String =
    s"GpuGenericHighCardJitAgg(keys=${groupOrds.length}, aggs=${aggKinds.length})"
}
