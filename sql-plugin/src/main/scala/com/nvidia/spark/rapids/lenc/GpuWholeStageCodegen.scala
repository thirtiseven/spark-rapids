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

import ai.rapids.cudf.{ColumnVector, ColumnView, OutOfBoundsPolicy, Scalar, Table => CudfTable}
import com.nvidia.spark.rapids._
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.GpuMetric._
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.jni.{VortexNative, WholeStageNative}

import org.apache.spark.TaskContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Alias, Attribute, ExprId, Expression, NamedExpression}
import org.apache.spark.sql.catalyst.expressions.aggregate.Partial
import org.apache.spark.sql.catalyst.plans.Inner
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.datasources.FilePartition
import org.apache.spark.sql.execution.rapids.shims.FilePartitionShims
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids.{GpuAdd, GpuAnd, GpuContains, GpuDivide, GpuEqualTo, GpuFileSourceScanExec, GpuGreaterThan, GpuGreaterThanOrEqual, GpuLessThan, GpuLessThanOrEqual, GpuMultiply, GpuNot, GpuOr, GpuStartsWith, GpuSubstring, GpuSubtract}
import org.apache.spark.sql.rapids.execution.{GpuBroadcastHashJoinExecBase, GpuBroadcastHelper, GpuShuffleExchangeExecBase}
import org.apache.spark.sql.types._
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.unsafe.types.UTF8String

/**
 * Thin Whole-Stage GPU Codegen physical-plan matcher & NVRTC stage-body generator backed by
 * libcudf's cudf::whole_stage execution primitives (pluggable Sources + L2 LUT/Hash multi-way
 * Broadcast Probe + Native Single-Pass GPU Open-Addressing Hash-Agg Sink with auto-flush).
 *
 * Eliminates cudf::gather wide-table materialization AND cudf::groupby::get_groups sorting on:
 *   - K = 1..6 Star Broadcast Equi-Join + Filter + Project + Partial Hash-Agg (including
 *     broadcast-dimension String group keys via per-task dictionary encoding + tiny final gather!)
 *   - K = 0 Standalone Partial Hash-Agg (1..4 fixed-width group keys) over any child plan.
 */
object GpuWholeStageCodegenRewrite {
  private val MAX_DIMS = 6
  private val MAX_FACT_COLS = 24
  private val MAX_DIM_COLS = 16
  private val MAX_GROUP_KEYS = 8

  private def isJoinKeyType(dt: DataType): Boolean = dt match {
    case IntegerType | LongType | DateType => true
    case _ => false
  }

  private def isFixedGroupKeyType(dt: DataType): Boolean = dt match {
    case ByteType | ShortType | IntegerType | LongType | DateType => true
    case d: DecimalType if d.precision <= 18 => true
    case _ => false
  }

  private def peelCoalesce(p: SparkPlan): SparkPlan = p match {
    case c: GpuCoalesceBatches => peelCoalesce(c.child)
    case other => other
  }

  private def unwrapJit(e: Expression): Expression = e.transformDown {
    case j: GpuAstJitExpression => j.child
  }

  private def extractProjectMap(projList: Seq[NamedExpression]): Map[ExprId, Expression] = {
    projList.collect {
      case a: GpuAlias => a.exprId -> unwrapJit(a.child)
      case a: Alias => a.exprId -> unwrapJit(a.child)
    }.toMap
  }

  private def isPassThroughProject(projList: Seq[NamedExpression]): Option[Map[ExprId, Attribute]] = {
    val m = mutable.HashMap.empty[ExprId, Attribute]
    projList.foreach {
      case a: Attribute => m.put(a.exprId, a)
      case GpuAlias(a: Attribute, _) => m.put(a.exprId, a)
      case Alias(a: Attribute, _) => m.put(a.exprId, a)
      case _ => return None
    }
    Some(m.toMap)
  }

  def apply(plan: SparkPlan): SparkPlan = {
    if (!SQLConf.get.getConfString(GpuLencFusedAgg.FUSED_AGG_CONF, "false").toBoolean) {
      return plan
    }
    plan.transformUp {
      case agg: GpuHashAggregateExec =>
        tryFuseWholeStageAgg(agg).getOrElse(agg)
      case shuf: GpuShuffleExchangeExecBase =>
        tryFuseWholeStageShuffle(shuf).getOrElse(shuf)
    }
  }

  /**
   * Checks if `plan` is a V1 GpuFileSourceScanExec backed by GpuReadVortexFileFormat whose
   * compressed segments (FSST / FastLanes BitPacked+FoR / ALP / Dict) can be streamed directly
   * into libcudf cudf::whole_stage without HBM decompression!
   */
  private[lenc] def isDirectCompressedVortexScan(plan: SparkPlan): Boolean = plan match {
    case fsse: GpuFileSourceScanExec =>
      fsse.relation.fileFormat.isInstanceOf[com.nvidia.spark.rapids.vortex.GpuReadVortexFileFormat] &&
        fsse.relation.partitionSchema.isEmpty &&
        !fsse.bucketedScan &&
        !fsse.partitionFilters.exists(_.find(_.isInstanceOf[org.apache.spark.sql.catalyst.expressions.PlanExpression[_]]).isDefined)
    case _ => false
  }

  private def hasStringPredicate(e: Expression): Boolean = {
    e.find {
      case _: GpuContains | _: GpuStartsWith => true
      case GpuEqualTo(l, r) if l.dataType == StringType && r.dataType == StringType => true
      case _ => false
    }.isDefined
  }

  /**
   * Unwinds 0..MAX_DIMS Inner BuildRight single-key GpuBroadcastHashJoins below an optional
   * post-join Project/Filter, plus an optional fact-side Filter at the bottom.
   */
  private case class StageColRef(
      tableIdx: Int,     // -1 = fact table, 0..K-1 = broadcast dim table
      colOrd: Int,       // ordinal in that table's output
      dataType: DataType,
      isDimStrDict: Boolean = false)

  private def tryFuseWholeStageAgg(agg: GpuHashAggregateExec): Option[SparkPlan] = {
    if (agg.aggregateExpressions.isEmpty || !agg.aggregateExpressions.forall(_.mode == Partial)) {
      return None
    }
    if (agg.groupingExpressions.length > MAX_GROUP_KEYS) {
      return None
    }
    if (agg.groupingExpressions.exists(_.isInstanceOf[GpuAlias])) {
      return None
    }

    // 1. Peel optional GpuCoalesceBatches -> GpuProjectExec -> GpuFilterExec above join chain
    var cur = peelCoalesce(agg.child)
    var topProjMap = Map.empty[ExprId, Expression]
    cur match {
      case p: GpuProjectExec =>
        topProjMap = extractProjectMap(p.projectList)
        cur = peelCoalesce(p.child)
      case _ =>
    }
    var postFilterOpt: Option[Expression] = None
    cur match {
      case f: GpuFilterExec =>
        postFilterOpt = Some(GpuLencFusedAgg.inlineProjectMap(unwrapJit(f.condition), topProjMap))
        cur = peelCoalesce(f.child)
      case _ =>
    }

    // 2. Unwind K = 0..MAX_DIMS Inner BuildRight single-key GpuBroadcastHashJoins
    val joins = mutable.ArrayBuffer.empty[GpuBroadcastHashJoinExecBase]
    val midAliasMaps = mutable.ArrayBuffer.empty[Map[ExprId, Attribute]]
    var keep = true
    while (keep && joins.length < MAX_DIMS) {
      cur match {
        case bhj: GpuBroadcastHashJoinExecBase
            if bhj.joinType == Inner &&
               bhj.buildSide == GpuBuildRight &&
               bhj.condition.isEmpty &&
               bhj.leftKeys.length == 1 &&
               bhj.rightKeys.length == 1 &&
               isJoinKeyType(bhj.leftKeys.head.dataType) &&
               bhj.leftKeys.head.dataType == bhj.rightKeys.head.dataType &&
               bhj.leftKeys.head.isInstanceOf[Attribute] &&
               bhj.rightKeys.head.isInstanceOf[Attribute] &&
               GpuFusedStarJoinRewrite.resolveBroadcastExchange(bhj.right).isDefined =>
          joins += bhj
          cur = peelCoalesce(bhj.left)
        case p: GpuProjectExec if joins.nonEmpty =>
          isPassThroughProject(p.projectList) match {
            case Some(am) =>
              midAliasMaps += am
              cur = peelCoalesce(p.child)
            case None => keep = false
          }
        case _ => keep = false
      }
    }

    // Peel optional fact-side GpuFilterExec at bottom of stream chain
    var factFilterOpt: Option[Expression] = None
    cur match {
      case f: GpuFilterExec =>
        factFilterOpt = Some(unwrapJit(f.condition))
        cur = peelCoalesce(f.child)
      case _ =>
    }
    val factPlan = cur

    // Cost-based admission:
    //   1. K == 0 Global Agg directly over V1 Vortex (zero-HBM FastLanes BitPacked+FoR / ALP / FSST!)
    //      or containing string predicates (GpuContains / GpuStartsWith / GpuEqualTo).
    //   2. K >= 2 multi-way star-join + low/medium-card agg (0..4 group keys originating from
    //      broadcast dimensions so the open-addressing hash table stays L2-resident with zero
    //      overflow flushes, avoiding high-card fact-key regressions like 300M-group Q65/Q73!).
    val directVortex = isDirectCompressedVortexScan(factPlan)
    if (joins.isEmpty) {
      if (agg.groupingExpressions.nonEmpty) return None
      val anyStrPred = factFilterOpt.exists(hasStringPredicate) || postFilterOpt.exists(hasStringPredicate) ||
        agg.aggregateExpressions.exists(hasStringPredicate) ||
        topProjMap.values.exists(hasStringPredicate)
      if (!directVortex && !anyStrPred) return None
    } else if (joins.length < 2 || agg.groupingExpressions.length > 4) {
      return None
    }

    // Resolve alias chains through midAliasMaps
    def resolveAttr(a: Attribute): Attribute = {
      var curA = a
      var changed = true
      var guard = 0
      while (changed && guard < 16) {
        changed = false
        guard += 1
        midAliasMaps.foreach { m =>
          m.get(curA.exprId).foreach { nextA =>
            if (nextA.exprId != curA.exprId) {
              curA = nextA
              changed = true
            }
          }
        }
      }
      curA
    }

    // Order dims bottom-to-top (0..K-1)
    val orderedJoins = joins.reverse.toArray
    val k = orderedJoins.length
    val dimPlans = orderedJoins.map(_.right)

    val factOut = factPlan.output
    val factIdToOrd = factOut.zipWithIndex.map { case (a, i) => a.exprId -> i }.toMap
    val dimIdToLoc = mutable.HashMap.empty[ExprId, (Int, Int)]
    dimPlans.zipWithIndex.foreach { case (dp, dIdx) =>
      dp.output.zipWithIndex.foreach { case (a, ord) =>
        dimIdToLoc.put(a.exprId, (dIdx, ord))
      }
    }

    // Every broadcast probe key must originate from factPlan (star schema) or an earlier dim!
    val probeKeySrcTbl = new Array[Int](k)
    val probeKeySrcOrd = new Array[Int](k)
    val dimKeyOrds = new Array[Int](k)
    for (d <- 0 until k) {
      val lk = resolveAttr(orderedJoins(d).leftKeys.head.asInstanceOf[Attribute])
      val rk = orderedJoins(d).rightKeys.head.asInstanceOf[Attribute]
      factIdToOrd.get(lk.exprId) match {
        case Some(ord) =>
          probeKeySrcTbl(d) = -1
          probeKeySrcOrd(d) = ord
        case None =>
          dimIdToLoc.get(lk.exprId) match {
            case Some((prevD, ord)) if prevD < d =>
              probeKeySrcTbl(d) = prevD
              probeKeySrcOrd(d) = ord
            case _ => return None
          }
      }
      val dOrd = dimPlans(d).output.indexWhere(_.exprId == rk.exprId)
      if (dOrd < 0) return None
      dimKeyOrds(d) = dOrd
    }

    // 3. Extract aggregate terms & output buffer mapping (reusing battle-tested GpuLencFusedAgg contract!)
    val (rawTerms0, outBufToTerm, outBufModes) =
      GpuLencFusedAgg.extractAggTermsAndBufferMap(agg.aggregateExpressions, topProjMap).getOrElse(return None)
    val rawTerms = rawTerms0.map(t => t.copy(inputExpr = t.inputExpr.map(unwrapJit)))
    val resultExprs = agg.resultExpressions
    if (resultExprs.length != agg.groupingExpressions.length + outBufToTerm.length) {
      return None
    }

    // Inline topProjMap into grouping expressions
    val groupExprs = agg.groupingExpressions.map { ge =>
      GpuLencFusedAgg.inlineProjectMap(unwrapJit(ge), topProjMap)
    }

    // Validate every group key: either fixed-width expr OR broadcast-dim StringType attribute / substring(dimStr, pos, len)!
    val groupKeyIsDimStr = new Array[Boolean](groupExprs.length)
    val groupKeyDimStrRef = new Array[(Int, Int, Int, Int)](groupExprs.length) // (dimIdx, colOrd, substrStart0, substrLen)
    groupExprs.zipWithIndex.foreach { case (ge, gi) =>
      if (isFixedGroupKeyType(ge.dataType)) {
        groupKeyIsDimStr(gi) = false
      } else if (ge.dataType == StringType) {
        ge match {
          case a: Attribute =>
            val ra = resolveAttr(a)
            dimIdToLoc.get(ra.exprId) match {
              case Some((dIdx, ord)) =>
                groupKeyIsDimStr(gi) = true
                groupKeyDimStrRef(gi) = (dIdx, ord, 0, -1)
              case None => return None
            }
          case GpuSubstring(a: Attribute, GpuLiteral(pos: Int, IntegerType), GpuLiteral(len: Int, IntegerType))
              if pos >= 1 && len >= 0 =>
            val ra = resolveAttr(a)
            dimIdToLoc.get(ra.exprId) match {
              case Some((dIdx, ord)) =>
                groupKeyIsDimStr(gi) = true
                groupKeyDimStrRef(gi) = (dIdx, ord, pos - 1, len)
              case None => return None
            }
          case _ => return None
        }
      } else {
        return None
      }
    }

    // 4. Collect all referenced fact & dim columns into compact slot tables (fact_cols[], dim_cols[])
    val factSlotMap = mutable.LinkedHashMap.empty[Int, Int] // factOrd -> factSlot
    // (dIdx, ord, isStrDict, substrStart0, substrLen) -> dimSlot
    val dimSlotMap = mutable.LinkedHashMap.empty[(Int, Int, Boolean, Int, Int), Int]

    def allocFactSlot(ord: Int): Int = factSlotMap.getOrElseUpdate(ord, factSlotMap.size)
    def allocDimSlot(dIdx: Int, ord: Int, isStrDict: Boolean, ss0: Int = 0, ssl: Int = -1): Int =
      dimSlotMap.getOrElseUpdate((dIdx, ord, isStrDict, ss0, ssl), dimSlotMap.size)

    // Allocate slots for all probe keys
    val probeKeyFactOrDimSlot = new Array[(Int, Int)](k) // (srcTbl: -1=fact or prevD, slot)
    for (d <- 0 until k) {
      if (probeKeySrcTbl(d) == -1) {
        probeKeyFactOrDimSlot(d) = (-1, allocFactSlot(probeKeySrcOrd(d)))
      } else {
        probeKeyFactOrDimSlot(d) =
          (probeKeySrcTbl(d), allocDimSlot(probeKeySrcTbl(d), probeKeySrcOrd(d), isStrDict = false))
      }
    }

    // Map every Attribute inside an expression to a StageColRef slot!
    def collectExprCols(e: Expression, allowDim: Boolean): Boolean = {
      var ok = true
      e.foreach {
        case a: Attribute =>
          val ra = resolveAttr(a)
          factIdToOrd.get(ra.exprId) match {
            case Some(ord) =>
              if (!isCodegenLeafType(ra.dataType) && ra.dataType != StringType) ok = false
              else allocFactSlot(ord)
            case None if allowDim =>
              dimIdToLoc.get(ra.exprId) match {
                case Some((dIdx, ord)) =>
                  if (!isCodegenLeafType(ra.dataType) && ra.dataType != StringType) ok = false
                  else allocDimSlot(dIdx, ord, isStrDict = false)
                case None => ok = false
              }
            case None => ok = false
          }
        case _ =>
      }
      ok
    }

    if (factFilterOpt.exists(e => !collectExprCols(e, allowDim = false))) return None
    if (postFilterOpt.exists(e => !collectExprCols(e, allowDim = true))) return None
    // Require star-join group keys to originate from broadcast dimension tables (bounded cardinality)
    // so open-addressing hash-agg stays L2-resident without overflow flush churn on fact-table IDs.
    val factAttrIds = factIdToOrd.keySet
    if (k >= 1 && groupExprs.exists(ge => ge.references.exists(a => factAttrIds.contains(resolveAttr(a).exprId)))) {
      return None
    }
    groupExprs.zipWithIndex.foreach { case (ge, gi) =>
      if (groupKeyIsDimStr(gi)) {
        val (dIdx, ord, ss0, ssl) = groupKeyDimStrRef(gi)
        allocDimSlot(dIdx, ord, isStrDict = true, ss0, ssl)
      } else if (!collectExprCols(ge, allowDim = true)) {
        return None
      }
    }
    rawTerms.foreach { t =>
      if (t.inputExpr.exists(e => !collectExprCols(e, allowDim = true))) return None
    }

    if (factSlotMap.size > MAX_FACT_COLS || dimSlotMap.size > MAX_DIM_COLS) {
      return None
    }

    // 5. Generate NVRTC Whole-Stage CUDA kernel body calling libcudf cudf::whole_stage preamble!
    val factOrds = factSlotMap.keys.toArray
    val dimColSpecs = dimSlotMap.keys.toArray // (dIdx, ord, isStrDict, ss0, ssl)
    val dimSlotLookup = dimSlotMap.toMap
    val factSlotLookup = factSlotMap.toMap

    def attrAccess(a: Attribute): Option[(String, String, DataType)] = {
      val ra = resolveAttr(a)
      factIdToOrd.get(ra.exprId).flatMap(factSlotLookup.get).map { fSlot =>
        (s"a->fact_cols[$fSlot]", "row", ra.dataType)
      }.orElse {
        dimIdToLoc.get(ra.exprId).flatMap { case (dIdx, ord) =>
          dimSlotLookup.get((dIdx, ord, false, 0, -1)).map { dSlot =>
            (s"a->dim_cols[$dSlot]", s"d$dIdx", ra.dataType)
          }
        }
      }
    }

    val cudaBodyOpt = WholeStageKernelGen.generateStarJoinAggBody(
      k = k,
      probeKeySpecs = probeKeyFactOrDimSlot.zipWithIndex.map { case ((srcT, slot), d) =>
        val dt = orderedJoins(d).leftKeys.head.dataType
        (srcT, slot, dt)
      },
      factFilterOpt = factFilterOpt,
      postFilterOpt = postFilterOpt,
      groupExprs = groupExprs,
      groupKeyDimStrSlot = groupExprs.indices.map { gi =>
        if (groupKeyIsDimStr(gi)) {
          val (dIdx, ord, ss0, ssl) = groupKeyDimStrRef(gi)
          Some((dIdx, dimSlotLookup((dIdx, ord, true, ss0, ssl))))
        } else None
      }.toArray,
      aggTerms = rawTerms,
      attrAccess = attrAccess)

    val (cudaBody, kernelName) = cudaBodyOpt.getOrElse(return None)

    val outAttrs = resultExprs.map(_.toAttribute)
    val outTypes = resultExprs.map(_.dataType).toArray
    val nativeKeyTypes = groupExprs.zipWithIndex.map { case (ge, gi) =>
      if (groupKeyIsDimStr(gi)) IntegerType else ge.dataType
    }.toArray
    val aggKinds = rawTerms.map(_.kind).toArray
    val groupKeyDimStrSlotIdx = groupExprs.indices.map { gi =>
      if (groupKeyIsDimStr(gi)) {
        val (dIdx, ord, ss0, ssl) = groupKeyDimStrRef(gi)
        dimSlotLookup((dIdx, ord, true, ss0, ssl))
      } else -1
    }.toArray

    Some(GpuWholeStageStarJoinAggExec(
      output = outAttrs,
      factPlan = factPlan,
      dimPlans = dimPlans.toSeq,
      factOrds = factOrds,
      dimKeyOrds = dimKeyOrds,
      dimColTableIdx = dimColSpecs.map(_._1),
      dimColOrd = dimColSpecs.map(_._2),
      dimColIsStrDict = dimColSpecs.map(_._3),
      dimColSubstrStart0 = dimColSpecs.map(_._4),
      dimColSubstrLen = dimColSpecs.map(_._5),
      nativeKeyTypes = nativeKeyTypes,
      groupKeyDimStrSlotIdx = groupKeyDimStrSlotIdx,
      aggKinds = aggKinds,
      outBufToTerm = outBufToTerm,
      outBufModes = outBufModes,
      outTypes = outTypes,
      cudaBody = cudaBody,
      kernelName = kernelName))
  }

  private def isCodegenLeafType(dt: DataType): Boolean = dt match {
    case BooleanType | ByteType | ShortType | IntegerType | LongType |
         FloatType | DoubleType | DateType => true
    case d: DecimalType if d.precision <= 18 => true
    case _ => false
  }

  /**
   * Phase 4 (Pillar 4-Write): Fuses Stage (Filter + 0..6 BroadcastHashJoin + Project) directly
   * into Spark Murmur3 HashPartition Scatter right below GpuShuffleExchangeExec so LSHF Shuffle
   * receives pre-partitioned columns without intermediate Filter/Project/HashPartition HBM tables!
   */
  private def tryFuseWholeStageShuffle(shuf: GpuShuffleExchangeExecBase): Option[SparkPlan] = {
    // Opt-in flag (spark.rapids.sql.lenc.fusedShuffleWriteEnabled) so AQE broadcast-join
    // reordering (e.g. TPC-H Q08) is never disturbed by hiding BroadcastHashJoin inside Shuffle.
    if (!SQLConf.get.getConfString("spark.rapids.sql.lenc.fusedShuffleWriteEnabled", "false").toBoolean) {
      return None
    }
    val hp = shuf.outputPartitioning match {
      case h: GpuHashPartitioningBase if h.numPartitions > 1 && h.children.nonEmpty &&
          h.children.length <= 4 &&
          h.children.forall(e => e.isInstanceOf[Attribute] && isJoinKeyType(e.dataType)) => h
      case _ => return None
    }
    if (shuf.output.isEmpty || shuf.output.length > MAX_FACT_COLS ||
        !shuf.output.forall(a => isCodegenLeafType(a.dataType))) {
      return None
    }
    if (shuf.child.isInstanceOf[GpuWholeStageFusedPartitionExec]) return None

    var cur = peelCoalesce(shuf.child)
    var topProjMap = Map.empty[ExprId, Expression]
    var hadProj = false
    cur match {
      case p: GpuProjectExec =>
        topProjMap = extractProjectMap(p.projectList)
        cur = peelCoalesce(p.child)
        hadProj = true
      case _ =>
    }
    var postFilterOpt: Option[Expression] = None
    cur match {
      case f: GpuFilterExec =>
        postFilterOpt = Some(GpuLencFusedAgg.inlineProjectMap(unwrapJit(f.condition), topProjMap))
        cur = peelCoalesce(f.child)
      case _ =>
    }
    val joins = mutable.ArrayBuffer.empty[GpuBroadcastHashJoinExecBase]
    val midAliasMaps = mutable.ArrayBuffer.empty[Map[ExprId, Attribute]]
    var keep = true
    while (keep && joins.length < MAX_DIMS) {
      cur match {
        case bhj: GpuBroadcastHashJoinExecBase
            if bhj.joinType == Inner &&
               bhj.buildSide == GpuBuildRight &&
               bhj.condition.isEmpty &&
               bhj.leftKeys.length == 1 &&
               bhj.rightKeys.length == 1 &&
               isJoinKeyType(bhj.leftKeys.head.dataType) &&
               bhj.leftKeys.head.dataType == bhj.rightKeys.head.dataType &&
               bhj.leftKeys.head.isInstanceOf[Attribute] &&
               bhj.rightKeys.head.isInstanceOf[Attribute] &&
               GpuFusedStarJoinRewrite.resolveBroadcastExchange(bhj.right).isDefined =>
          joins += bhj
          cur = peelCoalesce(bhj.left)
        case p: GpuProjectExec if joins.nonEmpty =>
          isPassThroughProject(p.projectList) match {
            case Some(am) =>
              midAliasMaps += am
              cur = peelCoalesce(p.child)
            case None => keep = false
          }
        case _ => keep = false
      }
    }
    var factFilterOpt: Option[Expression] = None
    cur match {
      case f: GpuFilterExec =>
        factFilterOpt = Some(unwrapJit(f.condition))
        cur = peelCoalesce(f.child)
      case _ =>
    }
    // Only rewrite when we actually fuse at least 1 BroadcastJoin OR a Filter+Project!
    if (joins.isEmpty && factFilterOpt.isEmpty && postFilterOpt.isEmpty) {
      return None
    }
    val factPlan = cur

    def resolveAttr(a: Attribute): Attribute = {
      var curA = a
      var changed = true
      var guard = 0
      while (changed && guard < 16) {
        changed = false
        guard += 1
        midAliasMaps.foreach { m =>
          m.get(curA.exprId).foreach { nextA =>
            if (nextA.exprId != curA.exprId) { curA = nextA; changed = true }
          }
        }
      }
      curA
    }

    val orderedJoins = joins.reverse.toArray
    val k = orderedJoins.length
    val dimPlans = orderedJoins.map(_.right)
    val factIdToOrd = factPlan.output.zipWithIndex.map { case (a, i) => a.exprId -> i }.toMap
    val dimIdToLoc = mutable.HashMap.empty[ExprId, (Int, Int)]
    dimPlans.zipWithIndex.foreach { case (dp, dIdx) =>
      dp.output.zipWithIndex.foreach { case (a, ord) => dimIdToLoc.put(a.exprId, (dIdx, ord)) }
    }

    val probeKeySrcTbl = new Array[Int](k)
    val probeKeySrcOrd = new Array[Int](k)
    val dimKeyOrds = new Array[Int](k)
    for (d <- 0 until k) {
      val lk = resolveAttr(orderedJoins(d).leftKeys.head.asInstanceOf[Attribute])
      val rk = orderedJoins(d).rightKeys.head.asInstanceOf[Attribute]
      factIdToOrd.get(lk.exprId) match {
        case Some(ord) => probeKeySrcTbl(d) = -1; probeKeySrcOrd(d) = ord
        case None =>
          dimIdToLoc.get(lk.exprId) match {
            case Some((prevD, ord)) if prevD < d => probeKeySrcTbl(d) = prevD; probeKeySrcOrd(d) = ord
            case _ => return None
          }
      }
      val dOrd = dimPlans(d).output.indexWhere(_.exprId == rk.exprId)
      if (dOrd < 0) return None
      dimKeyOrds(d) = dOrd
    }

    val factSlotMap = mutable.LinkedHashMap.empty[Int, Int]
    val dimSlotMap = mutable.LinkedHashMap.empty[(Int, Int), Int]
    def allocFact(ord: Int): Int = factSlotMap.getOrElseUpdate(ord, factSlotMap.size)
    def allocDim(dIdx: Int, ord: Int): Int = dimSlotMap.getOrElseUpdate((dIdx, ord), dimSlotMap.size)

    val probeSlots = new Array[(Int, Int, DataType)](k)
    for (d <- 0 until k) {
      val dt = orderedJoins(d).leftKeys.head.dataType
      if (probeKeySrcTbl(d) == -1) probeSlots(d) = (-1, allocFact(probeKeySrcOrd(d)), dt)
      else probeSlots(d) = (probeKeySrcTbl(d), allocDim(probeKeySrcTbl(d), probeKeySrcOrd(d)), dt)
    }

    val outExprs = shuf.output.map(a => GpuLencFusedAgg.inlineProjectMap(a, topProjMap))
    val partExprs = hp.children.map(e => GpuLencFusedAgg.inlineProjectMap(e, topProjMap))

    def collectCols(e: Expression, allowDim: Boolean): Boolean = {
      var ok = true
      e.foreach {
        case a: Attribute =>
          val ra = resolveAttr(a)
          factIdToOrd.get(ra.exprId) match {
            case Some(ord) =>
              if (!isCodegenLeafType(ra.dataType) && ra.dataType != StringType) ok = false
              else allocFact(ord)
            case None if allowDim =>
              dimIdToLoc.get(ra.exprId) match {
                case Some((dIdx, ord)) =>
                  if (!isCodegenLeafType(ra.dataType) && ra.dataType != StringType) ok = false
                  else allocDim(dIdx, ord)
                case None => ok = false
              }
            case None => ok = false
          }
        case _ =>
      }
      ok
    }

    if (factFilterOpt.exists(e => !collectCols(e, allowDim = false))) return None
    if (postFilterOpt.exists(e => !collectCols(e, allowDim = true))) return None
    if (outExprs.exists(e => !collectCols(e, allowDim = true))) return None
    if (partExprs.exists(e => !collectCols(e, allowDim = true))) return None
    if (factSlotMap.size > MAX_FACT_COLS || dimSlotMap.size > MAX_DIM_COLS) return None

    val fLookup = factSlotMap.toMap
    val dLookup = dimSlotMap.toMap
    def attrAccess(a: Attribute): Option[(String, String, DataType)] = {
      val ra = resolveAttr(a)
      factIdToOrd.get(ra.exprId).flatMap(fLookup.get).map { fSlot =>
        (s"a->fact_cols[$fSlot]", "row", ra.dataType)
      }.orElse {
        dimIdToLoc.get(ra.exprId).flatMap { case (dIdx, ord) =>
          dLookup.get((dIdx, ord)).map { dSlot =>
            (s"a->dim_cols[$dSlot]", s"d$dIdx", ra.dataType)
          }
        }
      }
    }

    val (countBody, scatterBody) = WholeStageKernelGen.generatePartitionScatterBodies(
      k, probeSlots, factFilterOpt, postFilterOpt, partExprs, outExprs, attrAccess).getOrElse(return None)

    val dimSpecs = dimSlotMap.keys.toArray
    val fusedChild = GpuWholeStageFusedPartitionExec(
      output = shuf.output,
      factPlan = factPlan,
      dimPlans = dimPlans.toSeq,
      factOrds = factSlotMap.keys.toArray,
      dimKeyOrds = dimKeyOrds,
      dimColTableIdx = dimSpecs.map(_._1),
      dimColOrd = dimSpecs.map(_._2),
      outTypes = shuf.output.map(_.dataType).toArray,
      numPartitions = hp.numPartitions,
      countCudaBody = countBody,
      scatterCudaBody = scatterBody)

    Some(shuf.withNewChildren(Seq(fusedChild)))
  }
}

private object WholeStageKernelGen {
  private def cType(dt: DataType): Option[String] = dt match {
    case BooleanType => Some("bool")
    case ByteType | ShortType | IntegerType | DateType => Some("int32_t")
    case LongType => Some("int64_t")
    case FloatType => Some("float")
    case DoubleType => Some("double")
    case d: DecimalType if d.precision <= 18 => Some("int64_t")
    case _ => None
  }

  private def pow10i64(n: Int): Long = {
    var r = 1L
    var i = 0
    while (i < n) { r *= 10L; i += 1 }
    r
  }

  private def emitExpr(
      e: Expression,
      stmts: mutable.ArrayBuffer[String],
      nextId: () => Int,
      attrAccess: Attribute => Option[(String, String, DataType)]): Option[(String, String)] = {
    e match {
      case GpuLiteral(v, dt) =>
        if (v == null) {
          cType(dt).map(ct => (s"($ct)0", "false"))
        } else dt match {
          case BooleanType => Some((if (v.asInstanceOf[Boolean]) "true" else "false", "true"))
          case ByteType | ShortType | IntegerType | DateType =>
            Some((s"((int32_t)${v.asInstanceOf[Number].intValue()})", "true"))
          case LongType => Some((s"((int64_t)${v.asInstanceOf[Number].longValue()}LL)", "true"))
          case FloatType =>
            val f = v.asInstanceOf[Number].floatValue()
            if (f.isNaN || f.isInfinite) None else Some((s"((float)${f}f)", "true"))
          case DoubleType =>
            val d = v.asInstanceOf[Number].doubleValue()
            if (d.isNaN || d.isInfinite) None else Some((s"((double)$d)", "true"))
          case d: DecimalType if d.precision <= 18 =>
            val unscaled = v.asInstanceOf[Decimal].toUnscaledLong
            Some((s"((int64_t)${unscaled}LL)", "true"))
          case _ => None
        }

      case a: Attribute =>
        attrAccess(a).flatMap { case (srcExpr, rowVar, dt) =>
          val id = nextId()
          val vn = s"v$id"
          val ok = s"ok$id"
          stmts += s"bool $ok = ws_valid($srcExpr, $rowVar);"
          dt match {
            case BooleanType =>
              stmts += s"bool $vn = $ok ? (ws_load_i32($srcExpr, $rowVar) != 0) : false;"
              Some((vn, ok))
            case ByteType | ShortType | IntegerType | DateType =>
              stmts += s"int32_t $vn = $ok ? ws_load_i32($srcExpr, $rowVar) : 0;"
              Some((vn, ok))
            case LongType =>
              stmts += s"int64_t $vn = $ok ? ws_load_i64($srcExpr, $rowVar) : 0LL;"
              Some((vn, ok))
            case FloatType =>
              stmts += s"float $vn = $ok ? ws_load_f32($srcExpr, $rowVar) : 0.0f;"
              Some((vn, ok))
            case DoubleType =>
              stmts += s"double $vn = $ok ? ws_load_f64($srcExpr, $rowVar) : 0.0;"
              Some((vn, ok))
            case d: DecimalType if d.precision <= 18 =>
              stmts += s"int64_t $vn = $ok ? ws_load_i64($srcExpr, $rowVar) : 0LL;"
              Some((vn, ok))
            case _ => None
          }
        }

      case u: GpuUnscaledValue =>
        emitExpr(u.child, stmts, nextId, attrAccess)

      case p: GpuPromotePrecision =>
        emitExpr(p.child, stmts, nextId, attrAccess)

      case co: GpuCheckOverflow if co.nullOnOverflow && co.dataType.isInstanceOf[DecimalType] &&
          co.dataType.asInstanceOf[DecimalType].precision <= 18 =>
        emitExpr(co.child, stmts, nextId, attrAccess)

      case c: GpuCast if !c.ansiMode =>
        val ch = c.child
        val toDt = c.dataType
        emitExpr(ch, stmts, nextId, attrAccess).flatMap { case (cv, cok) =>
          (ch.dataType, toDt) match {
            case (fromDt, td) if fromDt == td => Some((cv, cok))
            case (fd: DecimalType, DoubleType) if fd.precision <= 18 =>
              val scaleDiv = math.pow(10.0, fd.scale)
              Some((s"((double)($cv) / $scaleDiv)", cok))
            case (fd: DecimalType, td: DecimalType) if fd.precision <= 18 && td.precision <= 18 =>
              if (fd.scale == td.scale) Some((s"((int64_t)($cv))", cok))
              else if (td.scale > fd.scale) Some((s"((int64_t)($cv) * ${pow10i64(td.scale - fd.scale)}LL)", cok))
              else None
            case (ByteType | ShortType | IntegerType | LongType, td: DecimalType) if td.precision <= 18 =>
              val mult = pow10i64(td.scale)
              Some((s"((int64_t)($cv) * ${mult}LL)", cok))
            case (_, DoubleType) => Some((s"((double)($cv))", cok))
            case (_, FloatType) => Some((s"((float)($cv))", cok))
            case (_, LongType) => Some((s"((int64_t)($cv))", cok))
            case (_, IntegerType) => Some((s"((int32_t)($cv))", cok))
            case _ => None
          }
        }

      case GpuIsNull(ch) =>
        emitExpr(ch, stmts, nextId, attrAccess).map { case (_, cok) => (s"(!$cok)", "true") }

      case GpuIsNotNull(ch) =>
        emitExpr(ch, stmts, nextId, attrAccess).map { case (_, cok) => (cok, "true") }

      case GpuNot(ch) =>
        emitExpr(ch, stmts, nextId, attrAccess).map { case (cv, cok) => (s"(!$cv)", cok) }

      case GpuAnd(l, r) =>
        for {
          (lv, lok) <- emitExpr(l, stmts, nextId, attrAccess)
          (rv, rok) <- emitExpr(r, stmts, nextId, attrAccess)
        } yield {
          val id = nextId()
          val vn = s"andv$id"
          val ok = s"andok$id"
          stmts += s"bool $ok = ($lok && $rok) || ($lok && !$lv) || ($rok && !$rv);"
          stmts += s"bool $vn = ($lok && $lv) && ($rok && $rv);"
          (vn, ok)
        }

      case GpuOr(l, r) =>
        for {
          (lv, lok) <- emitExpr(l, stmts, nextId, attrAccess)
          (rv, rok) <- emitExpr(r, stmts, nextId, attrAccess)
        } yield {
          val id = nextId()
          val vn = s"orv$id"
          val ok = s"orok$id"
          stmts += s"bool $ok = ($lok && $rok) || ($lok && $lv) || ($rok && $rv);"
          stmts += s"bool $vn = ($lok && $lv) || ($rok && $rv);"
          (vn, ok)
        }

      case GpuEqualTo(a: Attribute, GpuLiteral(v: UTF8String, StringType)) if v != null =>
        emitStrPred("ws_str_equals", a, v.toString, stmts, nextId, attrAccess)
      case GpuEqualTo(GpuLiteral(v: UTF8String, StringType), a: Attribute) if v != null =>
        emitStrPred("ws_str_equals", a, v.toString, stmts, nextId, attrAccess)
      case GpuStartsWith(a: Attribute, GpuLiteral(v: UTF8String, StringType)) if v != null =>
        emitStrPred("ws_str_starts_with", a, v.toString, stmts, nextId, attrAccess)
      case GpuContains(a: Attribute, GpuLiteral(v: UTF8String, StringType)) if v != null && v.numBytes() <= 64 =>
        emitStrPred("ws_str_contains", a, v.toString, stmts, nextId, attrAccess)
      case GpuEqualTo(l, r) => emitCmp("==", l, r, stmts, nextId, attrAccess)
      case GpuLessThan(l, r) => emitCmp("<", l, r, stmts, nextId, attrAccess)
      case GpuLessThanOrEqual(l, r) => emitCmp("<=", l, r, stmts, nextId, attrAccess)
      case GpuGreaterThan(l, r) => emitCmp(">", l, r, stmts, nextId, attrAccess)
      case GpuGreaterThanOrEqual(l, r) => emitCmp(">=", l, r, stmts, nextId, attrAccess)

      case ins: GpuInSet if ins.list.size <= 32 =>
        val ch = ins.child
        emitExpr(ch, stmts, nextId, attrAccess).flatMap { case (cv, cok) =>
          val lits = ins.list.filter(_ != null).map { v =>
            ch.dataType match {
              case ByteType | ShortType | IntegerType | DateType =>
                s"((int32_t)${v.asInstanceOf[Number].intValue()})"
              case LongType => s"((int64_t)${v.asInstanceOf[Number].longValue()}LL)"
              case d: DecimalType if d.precision <= 18 =>
                s"((int64_t)${v.asInstanceOf[Decimal].toUnscaledLong}LL)"
              case _ => return None
            }
          }
          if (lits.isEmpty) None
          else {
            val id = nextId()
            val vn = s"inv$id"
            val checks = lits.map(lit => s"(($cv) == $lit)").mkString(" || ")
            stmts += s"bool $vn = $cok && ($checks);"
            Some((vn, cok))
          }
        }

      case a: GpuAdd if !a.failOnError => emitBinArith("+", a.left, a.right, a.dataType, stmts, nextId, attrAccess)
      case s: GpuSubtract if !s.failOnError => emitBinArith("-", s.left, s.right, s.dataType, stmts, nextId, attrAccess)
      case m: GpuMultiply if !m.failOnError => emitBinArith("*", m.left, m.right, m.dataType, stmts, nextId, attrAccess)
      case d: GpuDivide if !d.failOnError && (d.dataType == DoubleType || d.dataType == FloatType) =>
        for {
          (lv, lok) <- emitExpr(d.left, stmts, nextId, attrAccess)
          (rv, rok) <- emitExpr(d.right, stmts, nextId, attrAccess)
        } yield {
          val id = nextId()
          val ct = if (d.dataType == DoubleType) "double" else "float"
          val vn = s"div$id"
          val ok = s"divok$id"
          stmts += s"bool $ok = ($lok) && ($rok) && (($rv) != 0);"
          stmts += s"$ct $vn = $ok ? (($ct)($lv) / ($ct)($rv)) : ($ct)0;"
          (vn, ok)
        }

      case GpuIf(pred, tv, fv) =>
        for {
          ct <- cType(e.dataType)
          (pv, pok) <- emitExpr(pred, stmts, nextId, attrAccess)
          (tval, tok) <- emitExpr(tv, stmts, nextId, attrAccess)
          (fval, fok) <- emitExpr(fv, stmts, nextId, attrAccess)
        } yield {
          val id = nextId()
          val vn = s"ifv$id"
          val ok = s"ifok$id"
          stmts += s"bool $ok = ($pok && $pv) ? ($tok) : ($fok);"
          stmts += s"$ct $vn = ($pok && $pv) ? ($ct)($tval) : ($ct)($fval);"
          (vn, ok)
        }

      case cw: GpuCaseWhen =>
        cType(cw.dataType).flatMap { ct =>
          val id = nextId()
          val vn = s"cwv$id"
          val ok = s"cwok$id"
          val hit = s"cwhit$id"
          stmts += s"$ct $vn = ($ct)0; bool $ok = false; bool $hit = false;"
          cw.branches.foreach { case (cond, br) =>
            val cRes = emitExpr(cond, stmts, nextId, attrAccess)
            if (cRes.isEmpty) return None
            val (cv, cok) = cRes.get
            stmts += s"if (!$hit && ($cok) && ($cv)) {"
            val bStmts = mutable.ArrayBuffer.empty[String]
            val bRes = emitExpr(br, bStmts, nextId, attrAccess)
            if (bRes.isEmpty) return None
            val (bv, bok) = bRes.get
            stmts ++= bStmts.map("  " + _)
            stmts += s"  $vn = ($ct)($bv); $ok = ($bok); $hit = true;"
            stmts += "}"
          }
          cw.elseValue.foreach { ev =>
            stmts += s"if (!$hit) {"
            val eStmts = mutable.ArrayBuffer.empty[String]
            val eRes = emitExpr(ev, eStmts, nextId, attrAccess)
            if (eRes.isEmpty) return None
            val (evv, eok) = eRes.get
            stmts ++= eStmts.map("  " + _)
            stmts += s"  $vn = ($ct)($evv); $ok = ($eok);"
            stmts += "}"
          }
          Some((vn, ok))
        }

      case _ => None
    }
  }

  private def escapeCString(s: String): String = {
    val sb = new StringBuilder(s.length + 8)
    s.getBytes(java.nio.charset.StandardCharsets.UTF_8).foreach { b =>
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
      stmts: mutable.ArrayBuffer[String],
      nextId: () => Int,
      attrAccess: Attribute => Option[(String, String, DataType)]): Option[(String, String)] = {
    attrAccess(a).filter(_._3 == StringType).map { case (srcExpr, rowVar, _) =>
      val id = nextId()
      val ok = s"sok$id"
      val vn = s"sv$id"
      val bytesLen = lit.getBytes(java.nio.charset.StandardCharsets.UTF_8).length
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
      stmts: mutable.ArrayBuffer[String],
      nextId: () => Int,
      attrAccess: Attribute => Option[(String, String, DataType)]): Option[(String, String)] = {
    if (l.dataType != r.dataType) return None
    for {
      (lv, lok) <- emitExpr(l, stmts, nextId, attrAccess)
      (rv, rok) <- emitExpr(r, stmts, nextId, attrAccess)
    } yield {
      val id = nextId()
      val ok = s"cok$id"
      val vn = s"cv$id"
      stmts += s"bool $ok = ($lok) && ($rok);"
      stmts += s"bool $vn = $ok && (($lv) $op ($rv));"
      (vn, ok)
    }
  }

  private def emitBinArith(
      op: String,
      l: Expression,
      r: Expression,
      outDt: DataType,
      stmts: mutable.ArrayBuffer[String],
      nextId: () => Int,
      attrAccess: Attribute => Option[(String, String, DataType)]): Option[(String, String)] = {
    // Only allow exact matching decimal scale on +/-, and unscaled scale sum on * when <= 18
    outDt match {
      case od: DecimalType =>
        if (od.precision > 18) return None
        (l.dataType, r.dataType) match {
          case (ld: DecimalType, rd: DecimalType) if ld.precision <= 18 && rd.precision <= 18 =>
            if ((op == "+" || op == "-") && (ld.scale != od.scale || rd.scale != od.scale)) return None
            if (op == "*" && (ld.scale + rd.scale != od.scale)) return None
          case _ => return None
        }
      case _ =>
    }
    for {
      ct <- cType(outDt)
      (lv, lok) <- emitExpr(l, stmts, nextId, attrAccess)
      (rv, rok) <- emitExpr(r, stmts, nextId, attrAccess)
    } yield {
      val id = nextId()
      val ok = s"aok$id"
      val vn = s"av$id"
      stmts += s"bool $ok = ($lok) && ($rok);"
      if (outDt.isInstanceOf[DecimalType] && op == "*") {
        stmts += s"int64_t $vn = $ok ? (int64_t)((__int128)($lv) * (__int128)($rv)) : 0LL;"
      } else {
        stmts += s"$ct $vn = $ok ? (($ct)($lv) $op ($ct)($rv)) : ($ct)0;"
      }
      (vn, ok)
    }
  }

  def generateStarJoinAggBody(
      k: Int,
      probeKeySpecs: Array[(Int, Int, DataType)],
      factFilterOpt: Option[Expression],
      postFilterOpt: Option[Expression],
      groupExprs: Seq[Expression],
      groupKeyDimStrSlot: Array[Option[(Int, Int)]],
      aggTerms: Seq[GenericAggTerm],
      attrAccess: Attribute => Option[(String, String, DataType)]): Option[(String, String)] = {
    val kname = "ws_star_join_agg"
    var idCounter = 0
    val nextId = () => { idCounter += 1; idCounter }

    val sb = new StringBuilder()
    sb.append("extern \"C\" __global__ void " + kname + "(const WsStageArgs* a) {\n")
    sb.append("  uint32_t n = a->num_rows;\n")
    sb.append("  uint32_t na = a->num_aggs;\n")
    sb.append("  for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < n; row += gridDim.x * blockDim.x) {\n")
    sb.append("    uint32_t prog_cnt = a->row_progress[row];\n")
    sb.append("    if (prog_cnt == 0xFFFFFFFFu) continue;\n")

    // 1. Fact-side filter (before probing any dimension table!)
    factFilterOpt.foreach { fe =>
      val fStmts = mutable.ArrayBuffer.empty[String]
      val fRes = emitExpr(fe, fStmts, nextId, attrAccess)
      if (fRes.isEmpty) return None
      val (fv, fok) = fRes.get
      fStmts.foreach(s => sb.append(s"    $s\n"))
      sb.append(s"    if (!(($fok) && ($fv))) { a->row_progress[row] = 0xFFFFFFFFu; continue; }\n")
    }

    sb.append("    uint32_t match_idx = 0u;\n")
    sb.append("    bool row_overflow = false;\n")

    // 2. Nested K-way Broadcast Equi-Join Probe loops (supports both L2 LUT and 1:N Chained Hash!)
    var indent = "    "
    for (d <- 0 until k) {
      val (srcTbl, slot, dt) = probeKeySpecs(d)
      val srcRef = if (srcTbl == -1) s"a->fact_cols[$slot]" else s"a->dim_cols[$slot]"
      val rowRef = if (srcTbl == -1) "row" else s"d$srcTbl"
      sb.append(s"${indent}if (ws_valid($srcRef, $rowRef)) {\n")
      indent += "  "
      val loadCall = if (dt == LongType) s"ws_load_i64($srcRef, $rowRef)" else s"(int64_t)ws_load_i32($srcRef, $rowRef)"
      sb.append(s"${indent}int64_t pk$d = $loadCall;\n")
      sb.append(s"${indent}for (int32_t d$d = ws_probe_first(a->dims[$d], pk$d); d$d >= 0 && !row_overflow; d$d = ws_probe_next(a->dims[$d], d$d)) {\n")
      indent += "  "
    }

    // 3. Post-join filter (if any)
    postFilterOpt.foreach { pe =>
      val pStmts = mutable.ArrayBuffer.empty[String]
      val pRes = emitExpr(pe, pStmts, nextId, attrAccess)
      if (pRes.isEmpty) return None
      val (pv, pok) = pRes.get
      pStmts.foreach(s => sb.append(s"$indent$s\n"))
      sb.append(s"${indent}if (!(($pok) && ($pv))) continue;\n")
    }

    // Exact 1:N overflow-retry skip guard (zero double-counting!)
    sb.append(s"${indent}if (match_idx < prog_cnt) { ++match_idx; continue; }\n")

    // 4. Compute composite group key (up to 8 x 64-bit keys + null_bits)
    sb.append(s"${indent}uint64_t gkeys[WS_MAX_GROUP_KEYS];\n")
    sb.append(s"${indent}uint8_t nbits = 0u;\n")
    groupExprs.zipWithIndex.foreach { case (ge, gi) =>
      groupKeyDimStrSlot(gi) match {
        case Some((dIdx, dSlot)) =>
          sb.append(s"${indent}if (ws_valid(a->dim_cols[$dSlot], d$dIdx)) { gkeys[$gi] = (uint64_t)(int64_t)ws_load_i32(a->dim_cols[$dSlot], d$dIdx); } else { gkeys[$gi] = 0ULL; nbits |= (uint8_t)(1u << $gi); }\n")
        case None =>
          val gStmts = mutable.ArrayBuffer.empty[String]
          val gRes = emitExpr(ge, gStmts, nextId, attrAccess)
          if (gRes.isEmpty) return None
          val (gv, gok) = gRes.get
          gStmts.foreach(s => sb.append(s"$indent$s\n"))
          sb.append(s"${indent}if ($gok) { gkeys[$gi] = (uint64_t)(int64_t)($gv); } else { gkeys[$gi] = 0ULL; nbits |= (uint8_t)(1u << $gi); }\n")
      }
    }

    // 5. Upsert into Native GPU Open-Addressing Hash-Agg Sink (returns -1 on >65% load -> host flushes!)
    sb.append(s"${indent}int32_t slot = ws_hash_agg_upsert_n(a, gkeys, nbits);\n")
    sb.append(s"${indent}if (slot < 0) { a->row_progress[row] = match_idx; row_overflow = true; break; }\n")
    sb.append(s"${indent}uint32_t base = (uint32_t)slot * na;\n")

    // 6. Compute aggregate inputs in registers and atomically update slot accumulators!
    aggTerms.zipWithIndex.foreach { case (t, ai) =>
      t.kind match {
        case GenericJitReduceCodegen.AGG_COUNT_I64 =>
          t.inputExpr match {
            case None =>
              sb.append(s"${indent}atomicAdd(&a->ht_cnt[base + $ai], 1ULL);\n")
            case Some(ie) =>
              val st = mutable.ArrayBuffer.empty[String]
              val iRes = emitExpr(ie, st, nextId, attrAccess)
              if (iRes.isEmpty) return None
              val (_, iok) = iRes.get
              sb.append(s"${indent}{\n")
              st.foreach(s => sb.append(s"$indent  $s\n"))
              sb.append(s"${indent}  if ($iok) atomicAdd(&a->ht_cnt[base + $ai], 1ULL);\n")
              sb.append(s"${indent}}\n")
          }
        case GenericJitReduceCodegen.AGG_SUM_F64 =>
          val st = mutable.ArrayBuffer.empty[String]
          val iRes = t.inputExpr.flatMap(emitExpr(_, st, nextId, attrAccess))
          if (iRes.isEmpty) return None
          val (iv, iok) = iRes.get
          sb.append(s"${indent}{\n")
          st.foreach(s => sb.append(s"$indent  $s\n"))
          sb.append(s"${indent}  if ($iok) { atomicAdd(&a->ht_f64[base + $ai], (double)($iv)); atomicAdd(&a->ht_cnt[base + $ai], 1ULL); }\n")
          sb.append(s"${indent}}\n")
        case GenericJitReduceCodegen.AGG_SUM_I64 =>
          val st = mutable.ArrayBuffer.empty[String]
          val iRes = t.inputExpr.flatMap(emitExpr(_, st, nextId, attrAccess))
          if (iRes.isEmpty) return None
          val (iv, iok) = iRes.get
          sb.append(s"${indent}{\n")
          st.foreach(s => sb.append(s"$indent  $s\n"))
          sb.append(s"${indent}  if ($iok) { atomicAdd((unsigned long long*)&a->ht_i64[base + $ai], (unsigned long long)(int64_t)($iv)); atomicAdd(&a->ht_cnt[base + $ai], 1ULL); }\n")
          sb.append(s"${indent}}\n")
        case GenericJitReduceCodegen.AGG_MIN_F64 =>
          val st = mutable.ArrayBuffer.empty[String]
          val iRes = t.inputExpr.flatMap(emitExpr(_, st, nextId, attrAccess))
          if (iRes.isEmpty) return None
          val (iv, iok) = iRes.get
          sb.append(s"${indent}{\n")
          st.foreach(s => sb.append(s"$indent  $s\n"))
          sb.append(s"${indent}  if ($iok) { ws_atomic_min_f64(&a->ht_f64[base + $ai], (double)($iv)); atomicAdd(&a->ht_cnt[base + $ai], 1ULL); }\n")
          sb.append(s"${indent}}\n")
        case GenericJitReduceCodegen.AGG_MAX_F64 =>
          val st = mutable.ArrayBuffer.empty[String]
          val iRes = t.inputExpr.flatMap(emitExpr(_, st, nextId, attrAccess))
          if (iRes.isEmpty) return None
          val (iv, iok) = iRes.get
          sb.append(s"${indent}{\n")
          st.foreach(s => sb.append(s"$indent  $s\n"))
          sb.append(s"${indent}  if ($iok) { ws_atomic_max_f64(&a->ht_f64[base + $ai], (double)($iv)); atomicAdd(&a->ht_cnt[base + $ai], 1ULL); }\n")
          sb.append(s"${indent}}\n")
        case GenericJitReduceCodegen.AGG_MIN_I64 =>
          val st = mutable.ArrayBuffer.empty[String]
          val iRes = t.inputExpr.flatMap(emitExpr(_, st, nextId, attrAccess))
          if (iRes.isEmpty) return None
          val (iv, iok) = iRes.get
          sb.append(s"${indent}{\n")
          st.foreach(s => sb.append(s"$indent  $s\n"))
          sb.append(s"${indent}  if ($iok) { ws_atomic_min_i64(&a->ht_i64[base + $ai], (long long)(int64_t)($iv)); atomicAdd(&a->ht_cnt[base + $ai], 1ULL); }\n")
          sb.append(s"${indent}}\n")
        case GenericJitReduceCodegen.AGG_MAX_I64 =>
          val st = mutable.ArrayBuffer.empty[String]
          val iRes = t.inputExpr.flatMap(emitExpr(_, st, nextId, attrAccess))
          if (iRes.isEmpty) return None
          val (iv, iok) = iRes.get
          sb.append(s"${indent}{\n")
          st.foreach(s => sb.append(s"$indent  $s\n"))
          sb.append(s"${indent}  if ($iok) { ws_atomic_max_i64(&a->ht_i64[base + $ai], (long long)(int64_t)($iv)); atomicAdd(&a->ht_cnt[base + $ai], 1ULL); }\n")
          sb.append(s"${indent}}\n")
        case _ => return None
      }
    }

    sb.append(s"${indent}++match_idx;\n")

    // Close nested K-way probe loops
    for (_ <- 0 until k) {
      indent = indent.dropRight(2)
      sb.append(s"$indent}\n")
      indent = indent.dropRight(2)
      sb.append(s"$indent}\n")
    }
    sb.append("    if (!row_overflow) { a->row_progress[row] = 0xFFFFFFFFu; }\n")
    sb.append("  }\n}\n")
    Some((sb.toString(), kname))
  }

  def generatePartitionScatterBodies(
      k: Int,
      probeKeySpecs: Array[(Int, Int, DataType)],
      factFilterOpt: Option[Expression],
      postFilterOpt: Option[Expression],
      partExprs: Seq[Expression],
      outExprs: Seq[Expression],
      attrAccess: Attribute => Option[(String, String, DataType)]): Option[(String, String)] = {
    def emitOne(isScatter: Boolean): Option[String] = {
      val kname = if (isScatter) "ws_part_scatter" else "ws_part_count"
      var idCounter = 0
      val nextId = () => { idCounter += 1; idCounter }
      val sb = new StringBuilder()
      sb.append("extern \"C\" __global__ void " + kname + "(const WsPartScatterArgs* a) {\n")
      sb.append("  uint32_t n = a->num_rows;\n")
      sb.append("  for (uint32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < n; row += gridDim.x * blockDim.x) {\n")

      factFilterOpt.foreach { fe =>
        val st = mutable.ArrayBuffer.empty[String]
        val res = emitExpr(fe, st, nextId, attrAccess)
        if (res.isEmpty) return None
        val (fv, fok) = res.get
        st.foreach(s => sb.append(s"    $s\n"))
        sb.append(s"    if (!(($fok) && ($fv))) continue;\n")
      }

      var indent = "    "
      for (d <- 0 until k) {
        val (srcTbl, slot, dt) = probeKeySpecs(d)
        val srcRef = if (srcTbl == -1) s"a->fact_cols[$slot]" else s"a->dim_cols[$slot]"
        val rowRef = if (srcTbl == -1) "row" else s"d$srcTbl"
        sb.append(s"${indent}if (ws_valid($srcRef, $rowRef)) {\n")
        indent += "  "
        val loadCall = if (dt == LongType) s"ws_load_i64($srcRef, $rowRef)" else s"(int64_t)ws_load_i32($srcRef, $rowRef)"
        sb.append(s"${indent}int64_t pk$d = $loadCall;\n")
        sb.append(s"${indent}for (int32_t d$d = ws_probe_first(a->dims[$d], pk$d); d$d >= 0; d$d = ws_probe_next(a->dims[$d], d$d)) {\n")
        indent += "  "
      }

      postFilterOpt.foreach { pe =>
        val st = mutable.ArrayBuffer.empty[String]
        val res = emitExpr(pe, st, nextId, attrAccess)
        if (res.isEmpty) return None
        val (pv, pok) = res.get
        st.foreach(s => sb.append(s"$indent$s\n"))
        sb.append(s"${indent}if (!(($pok) && ($pv))) continue;\n")
      }

      // Compute exact Spark Murmur3_x86_32 (seed = 42) hash across partition expressions
      sb.append(s"${indent}uint32_t h = 42u;\n")
      partExprs.foreach { pe =>
        val st = mutable.ArrayBuffer.empty[String]
        val res = emitExpr(pe, st, nextId, attrAccess)
        if (res.isEmpty) return None
        val (pv, pok) = res.get
        st.foreach(s => sb.append(s"$indent$s\n"))
        if (pe.dataType == LongType) {
          sb.append(s"${indent}if ($pok) h = ws_spark_murmur3_i64((int64_t)($pv), h);\n")
        } else {
          sb.append(s"${indent}if ($pok) h = ws_spark_murmur3_i32((int32_t)($pv), h);\n")
        }
      }
      sb.append(s"${indent}int32_t pid = ws_spark_part_id(h, (int32_t)a->num_parts);\n")

      if (!isScatter) {
        sb.append(s"${indent}atomicAdd(&a->part_counts[pid], 1u);\n")
      } else {
        sb.append(s"${indent}uint32_t dst = atomicAdd(&a->part_counts[pid], 1u);\n")
        outExprs.zipWithIndex.foreach { case (oe, ci) =>
          val st = mutable.ArrayBuffer.empty[String]
          val res = emitExpr(oe, st, nextId, attrAccess)
          if (res.isEmpty) return None
          val (ov, ook) = res.get
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
            case _ => return None
          }
          sb.append(s"$indent  if ($ook) { $storeStmt atomicOr(&a->out_masks[$ci][dst >> 5], 1u << (dst & 31u)); }\n")
          sb.append(s"${indent}}\n")
        }
      }

      for (_ <- 0 until k) {
        indent = indent.dropRight(2)
        sb.append(s"$indent}\n")
        indent = indent.dropRight(2)
        sb.append(s"$indent}\n")
      }
      sb.append("  }\n}\n")
      Some(sb.toString())
    }
    for {
      c <- emitOne(isScatter = false)
      s <- emitOne(isScatter = true)
    } yield (c, s)
  }
}

private object WholeStageProgramPool {
  private val cache = new ConcurrentHashMap[String, WholeStageNative.Program]()

  def getProgram(cudaBody: String, kernelName: String): WholeStageNative.Program = {
    val key = kernelName + "\n" + cudaBody
    cache.computeIfAbsent(key, _ => WholeStageNative.compileProgram(cudaBody, kernelName))
  }
}

/**
 * Whole-Stage GPU operator fusing 0..6 Broadcast Equi-Joins + Filter + Project + Native GPU
 * Open-Addressing Hash-Agg Sink into ONE single NVRTC kernel across all batches of a Spark task!
 */
case class GpuWholeStageStarJoinAggExec(
    output: Seq[Attribute],
    factPlan: SparkPlan,
    dimPlans: Seq[SparkPlan],
    factOrds: Array[Int],
    dimKeyOrds: Array[Int],
    dimColTableIdx: Array[Int],
    dimColOrd: Array[Int],
    dimColIsStrDict: Array[Boolean],
    dimColSubstrStart0: Array[Int],
    dimColSubstrLen: Array[Int],
    nativeKeyTypes: Array[DataType],
    groupKeyDimStrSlotIdx: Array[Int],
    aggKinds: Array[Byte],
    outBufToTerm: Array[Int],
    outBufModes: Array[Byte],
    outTypes: Array[DataType],
    cudaBody: String,
    kernelName: String) extends SparkPlan with GpuExec {

  override def children: Seq[SparkPlan] = factPlan +: dimPlans

  override protected def withNewChildrenInternal(newChildren: IndexedSeq[SparkPlan]): SparkPlan =
    copy(factPlan = newChildren.head, dimPlans = newChildren.tail)

  override def supportsColumnar: Boolean = true

  override protected def doExecute(): RDD[InternalRow] =
    throw new IllegalStateException("GpuWholeStageStarJoinAggExec executes columnar on GPU only")

  override lazy val additionalMetrics: Map[String, GpuMetric] = Map(
    OP_TIME_LEGACY -> createNanoTimingMetric(MODERATE_LEVEL, DESCRIPTION_OP_TIME_LEGACY),
    JOIN_TIME -> createNanoTimingMetric(DEBUG_LEVEL, DESCRIPTION_JOIN_TIME)
  )

  private def gatherDictStrings(keysCv: ColumnVector, codeCv: ColumnVector): ColumnVector = {
    val nKeys = keysCv.getRowCount.toInt
    val gatherMap = withResource(Scalar.fromInt(nKeys)) { oob =>
      codeCv.replaceNulls(oob)
    }
    withResource(gatherMap) { gm =>
      withResource(new CudfTable(keysCv)) { keyTbl =>
        withResource(keyTbl.gather(gm, OutOfBoundsPolicy.NULLIFY)) { gathered =>
          gathered.getColumn(0).incRefCount()
        }
      }
    }
  }

  private def buildBatchFromWholeStageTable(
      rawTbl: CudfTable,
      dimStrDictKeys: Array[ColumnVector],
      numKeys: Int,
      gkDimStrSlots: Array[Int],
      obm: Array[Int],
      obModes: Array[Byte],
      oTypes: Array[DataType]): ColumnarBatch = {
    val hasStrKeys = gkDimStrSlots.exists(_ >= 0)
    if (!hasStrKeys) {
      GenericReduceBatchBuilder.projectBuffers(rawTbl, numKeys, obm, obModes, oTypes)
    } else {
      val totalOut = numKeys + obm.length
      val cols = (0 until totalOut).safeMap { i =>
        if (i < numKeys && gkDimStrSlots(i) >= 0) {
          val keysCv = dimStrDictKeys(gkDimStrSlots(i))
          val codeCv = rawTbl.getColumn(i)
          GpuColumnVector.from(gatherDictStrings(keysCv, codeCv), StringType)
        } else {
          GenericReduceBatchBuilder.projectSingleColumn(
            rawTbl, i, numKeys, obm, obModes, oTypes)
        }
      }
      new ColumnarBatch(cols.toArray, rawTbl.getRowCount.toInt)
    }
  }

  override def internalDoExecuteColumnar(): RDD[ColumnarBatch] = {
    val opTime = allMetrics.getOrElse(OP_TIME_LEGACY, NoopMetric)
    val numOutputRows = allMetrics.getOrElse(NUM_OUTPUT_ROWS, NoopMetric)
    val numOutputBatches = allMetrics.getOrElse(NUM_OUTPUT_BATCHES, NoopMetric)

    val k = dimPlans.length
    val bcasts = dimPlans.map { dp =>
      GpuFusedStarJoinRewrite.resolveBroadcastExchange(dp)
        .getOrElse(throw new IllegalStateException(s"Expected GpuBroadcastExchangeExec in $dp"))
        .executeColumnarBroadcast[Any]()
    }.toArray
    val dimSchemas = dimPlans.map(_.schema).toArray

    val fOrds = factOrds
    val dKeyOrds = dimKeyOrds
    val dColTbl = dimColTableIdx
    val dColOrd = dimColOrd
    val dColIsStr = dimColIsStrDict
    val dColSs0 = dimColSubstrStart0
    val dColSsl = dimColSubstrLen
    val nKeyTypes = nativeKeyTypes
    val gkDimStrSlots = groupKeyDimStrSlotIdx
    val aKinds = aggKinds
    val obm = outBufToTerm
    val obModes = outBufModes
    val oTypes = outTypes
    val srcBody = cudaBody
    val kName = kernelName
    val numKeys = nativeKeyTypes.length

    // Phase 2 + Phase 3-Vortex: When factPlan is a direct non-partitioned V1 Vortex scan,
    // stream compressed Vortex segments (FSST / FastLanes BitPacked+FoR / ALP / Dict) straight
    // into libcudf cudf::whole_stage WITHOUT decompressing to HBM!
    val directVortexSplitsOpt: Option[(Array[Array[String]], Array[String])] =
      if (GpuWholeStageCodegenRewrite.isDirectCompressedVortexScan(factPlan)) {
        val fsse = factPlan.asInstanceOf[GpuFileSourceScanExec]
        val splits = fsse.inputRDD.partitions.collect {
          case fp: FilePartition =>
            FilePartitionShims.getFiles(fp).map(pf => new java.net.URI(pf.filePath.toString).getPath)
        }.filter(_.nonEmpty)
        val colNames = fOrds.map(o => fsse.output(o).name)
        if (splits.nonEmpty) Some((splits, colNames)) else None
      } else None

    def runCompressedVortexTask(
        paths: Array[String],
        colNames: Array[String]): Iterator[ColumnarBatch] = {
      val nKeyDTypes = nKeyTypes.map(GpuColumnVector.getNonNestedRapidsType)
      val dimBatches = new Array[ColumnarBatch](k)
      val probeTables = new Array[WholeStageNative.ProbeTable](k)
      val dimEncodedCols = new Array[ColumnVector](dColTbl.length)
      val dimStrDictKeys = new Array[ColumnVector](dColTbl.length)
      var taskState: WholeStageNative.HashAggTask = null
      var outBatch: ColumnarBatch = null
      try {
        GpuSemaphore.acquireIfNecessary(TaskContext.get())
        var anyDimEmpty = false
        if (k > 0) {
          opTime.ns {
            var d = 0
            while (d < k && !anyDimEmpty) {
              val db = GpuBroadcastHelper.getBroadcastBatch(bcasts(d), dimSchemas(d))
              dimBatches(d) = db
              if (db.numRows() == 0) anyDimEmpty = true
              else {
                val keyCv = db.column(dKeyOrds(d)).asInstanceOf[GpuColumnVector].getBase
                probeTables(d) = WholeStageNative.buildProbeTable(keyCv)
              }
              d += 1
            }
          }
        }
        val prog = WholeStageProgramPool.getProgram(srcBody, kName)
        if (anyDimEmpty) {
          if (numKeys == 0) {
            taskState = prog.openHashAggTask(nKeyDTypes, aKinds, 1)
            withResource(taskState.finish()) { r =>
              if (r.getRowCount > 0) {
                numOutputRows += r.getRowCount
                numOutputBatches += 1
                outBatch = GenericReduceBatchBuilder.projectBuffers(r, 0, obm, obModes, oTypes)
              }
            }
          }
        } else {
          val dimViews = new Array[ColumnView](dColTbl.length)
          if (dColTbl.nonEmpty) {
            opTime.ns {
              for (i <- dColTbl.indices) {
                val baseCv = dimBatches(dColTbl(i)).column(dColOrd(i)).asInstanceOf[GpuColumnVector].getBase
                if (dColIsStr(i)) {
                  val pair = WholeStageNative.encodeDimStringDict(baseCv, dColSs0(i), dColSsl(i))
                  dimEncodedCols(i) = pair(0)
                  dimStrDictKeys(i) = pair(1)
                  dimViews(i) = dimEncodedCols(i)
                } else dimViews(i) = baseCv
              }
            }
          }
          val probeHandles = probeTables.map(_.getNativeHandle)
          opTime.ns {
            val initCap = if (numKeys == 0) 1 else 262144
            taskState = prog.openHashAggTask(nKeyDTypes, aKinds, initCap)
            VortexNative.streamCompressedPathsToWholeStageAgg(
              paths, colNames, taskState, probeHandles, dimViews)
            withResource(taskState.finish()) { r =>
              if (r.getRowCount > 0) {
                numOutputRows += r.getRowCount
                numOutputBatches += 1
                outBatch = buildBatchFromWholeStageTable(
                  r, dimStrDictKeys, numKeys, gkDimStrSlots, obm, obModes, oTypes)
              }
            }
          }
        }
      } finally {
        if (taskState != null) taskState.close()
        dimEncodedCols.foreach(c => if (c != null) c.safeClose())
        dimStrDictKeys.foreach(c => if (c != null) c.safeClose())
        probeTables.foreach(p => if (p != null) p.safeClose())
        dimBatches.foreach(db => if (db != null) db.safeClose())
      }
      if (outBatch != null) Iterator.single(outBatch) else Iterator.empty
    }

    directVortexSplitsOpt match {
      case Some((fileSplits, colNames)) =>
        sparkContext.parallelize(fileSplits.toSeq, fileSplits.length).mapPartitions { spIter =>
          spIter.flatMap(paths => runCompressedVortexTask(paths, colNames))
        }
      case None =>
        factPlan.executeColumnar().mapPartitions { iter =>
      new Iterator[ColumnarBatch] {
        private var finished = false
        private var pending: ColumnarBatch = _
        Option(TaskContext.get()).foreach { tc =>
          tc.addTaskCompletionListener[Unit](_ => {
            if (pending != null) {
              pending.safeClose()
              pending = null
            }
          })
        }

        private def computeOnce(): Unit = {
          finished = true
          val nKeyDTypes = nKeyTypes.map(GpuColumnVector.getNonNestedRapidsType)
          val dimBatches = new Array[ColumnarBatch](k)
          val probeTables = new Array[WholeStageNative.ProbeTable](k)
          val dimEncodedCols = new Array[ColumnVector](dColTbl.length)
          val dimStrDictKeys = new Array[ColumnVector](dColTbl.length)
          var taskState: WholeStageNative.HashAggTask = null
          try {
            // 1. Deserialize broadcast dims & build L2 LUT / Chained Hash Probe tables ONCE per task!
            var anyDimEmpty = false
            if (k > 0) {
              GpuSemaphore.acquireIfNecessary(TaskContext.get())
              opTime.ns {
                var d = 0
                while (d < k && !anyDimEmpty) {
                  val db = GpuBroadcastHelper.getBroadcastBatch(bcasts(d), dimSchemas(d))
                  dimBatches(d) = db
                  if (db.numRows() == 0) {
                    anyDimEmpty = true
                  } else {
                    val keyCv = db.column(dKeyOrds(d)).asInstanceOf[GpuColumnVector].getBase
                    probeTables(d) = WholeStageNative.buildProbeTable(keyCv)
                  }
                  d += 1
                }
              }
            }

            if (anyDimEmpty) {
              // Drain upstream iterator cleanly
              while (iter.hasNext) { withResource(iter.next()) { _ => } }
              if (numKeys == 0) {
                // Global aggregate on 0 surviving rows emits 1 row of initial values (e.g. COUNT=0, SUM=NULL)
                val prog = WholeStageProgramPool.getProgram(srcBody, kName)
                taskState = prog.openHashAggTask(nKeyDTypes, aKinds, 64)
                withResource(taskState.finish()) { r =>
                  if (r.getRowCount > 0) {
                    numOutputRows += r.getRowCount
                    numOutputBatches += 1
                    pending = GenericReduceBatchBuilder.projectBuffers(r, 0, obm, obModes, oTypes)
                  }
                }
              }
              return
            }

            // Build dim payload ColumnView array (dictionary-encoding any broadcast String group keys!)
            val dimViews = new Array[ColumnView](dColTbl.length)
            if (dColTbl.nonEmpty) {
              opTime.ns {
                for (i <- dColTbl.indices) {
                  val baseCv = dimBatches(dColTbl(i)).column(dColOrd(i)).asInstanceOf[GpuColumnVector].getBase
                  if (dColIsStr(i)) {
                    val pair = WholeStageNative.encodeDimStringDict(baseCv, dColSs0(i), dColSsl(i))
                    dimEncodedCols(i) = pair(0)
                    dimStrDictKeys(i) = pair(1)
                    dimViews(i) = dimEncodedCols(i)
                  } else {
                    dimViews(i) = baseCv
                  }
                }
              }
            }

            val probeHandles = probeTables.map(_.getNativeHandle)
            val prog = WholeStageProgramPool.getProgram(srcBody, kName)

            // 2. Stream ALL fact batches through ONE persistent Native GPU Open-Addressing Hash-Agg Sink!
            while (iter.hasNext) {
              withResource(iter.next()) { b =>
                if (b.numRows() > 0) {
                  opTime.ns {
                    if (taskState == null) {
                      val initCap = if (numKeys == 0) 1 else 262144
                      taskState = prog.openHashAggTask(nKeyDTypes, aKinds, initCap)
                    }
                    val factViews = fOrds.map(o => b.column(o).asInstanceOf[GpuColumnVector].getBase.asInstanceOf[ColumnView])
                    taskState.addBatch(factViews, b.numRows(), probeHandles, dimViews)
                  }
                }
              }
            }

            // 3. Compact & emit ONE partial-aggregate ColumnarBatch per Spark Task!
            if (taskState != null) {
              opTime.ns {
                withResource(taskState.finish()) { r =>
                  if (r.getRowCount > 0) {
                    numOutputRows += r.getRowCount
                    numOutputBatches += 1
                    pending = buildBatchFromWholeStageTable(
                      r, dimStrDictKeys, numKeys, gkDimStrSlots, obm, obModes, oTypes)
                  }
                }
              }
            }
          } finally {
            if (taskState != null) taskState.close()
            dimEncodedCols.foreach(c => if (c != null) c.safeClose())
            dimStrDictKeys.foreach(c => if (c != null) c.safeClose())
            probeTables.foreach(p => if (p != null) p.safeClose())
            dimBatches.foreach(db => if (db != null) db.safeClose())
          }
        }

        override def hasNext: Boolean = {
          if (!finished) computeOnce()
          pending != null
        }

        override def next(): ColumnarBatch = {
          if (!hasNext) throw new NoSuchElementException("empty whole-stage agg result")
          val out = pending
          pending = null
          out
        }
      }
    }
    }
  }

  override def simpleString(maxFields: Int): String =
    s"GpuWholeStageStarJoinAgg(dims=${dimPlans.length}, keys=${nativeKeyTypes.length}, aggs=${aggKinds.length})"
}

/**
 * Phase 4 (Pillar 4-Write): Fuses Stage (Filter + 0..6 BroadcastHashJoin + Project) directly
 * into Spark Murmur3 HashPartition Scatter right underneath GpuShuffleExchangeExec so LSHF
 * Shuffle receives pre-partitioned output columns without intermediate HBM tables!
 */
case class GpuWholeStageFusedPartitionExec(
    output: Seq[Attribute],
    factPlan: SparkPlan,
    dimPlans: Seq[SparkPlan],
    factOrds: Array[Int],
    dimKeyOrds: Array[Int],
    dimColTableIdx: Array[Int],
    dimColOrd: Array[Int],
    outTypes: Array[DataType],
    numPartitions: Int,
    countCudaBody: String,
    scatterCudaBody: String) extends SparkPlan with GpuExec {

  override def children: Seq[SparkPlan] = factPlan +: dimPlans

  override protected def withNewChildrenInternal(newChildren: IndexedSeq[SparkPlan]): SparkPlan =
    copy(factPlan = newChildren.head, dimPlans = newChildren.tail)

  override def supportsColumnar: Boolean = true

  override protected def doExecute(): RDD[InternalRow] =
    throw new IllegalStateException("GpuWholeStageFusedPartitionExec executes columnar on GPU only")

  override lazy val additionalMetrics: Map[String, GpuMetric] = Map(
    OP_TIME_LEGACY -> createNanoTimingMetric(MODERATE_LEVEL, DESCRIPTION_OP_TIME_LEGACY)
  )

  /**
   * Executes Filter + BroadcastProbe + Project + Murmur3 Partition Scatter in ONE fused GPU pass,
   * returning `(partitionedColumnarBatch, splits)` directly to GpuShuffleExchangeExec!
   */
  def executePrePartitioned(): RDD[(ColumnarBatch, Array[Int])] = {
    val opTime = allMetrics.getOrElse(OP_TIME_LEGACY, NoopMetric)
    val numOutputRows = allMetrics.getOrElse(NUM_OUTPUT_ROWS, NoopMetric)
    val numOutputBatches = allMetrics.getOrElse(NUM_OUTPUT_BATCHES, NoopMetric)
    val k = dimPlans.length
    val bcasts = dimPlans.map { dp =>
      GpuFusedStarJoinRewrite.resolveBroadcastExchange(dp)
        .getOrElse(throw new IllegalStateException(s"Expected GpuBroadcastExchangeExec in $dp"))
        .executeColumnarBroadcast[Any]()
    }.toArray
    val dimSchemas = dimPlans.map(_.schema).toArray
    val fOrds = factOrds
    val dKeyOrds = dimKeyOrds
    val dColTbl = dimColTableIdx
    val dColOrd = dimColOrd
    val oTypes = outTypes
    val nParts = numPartitions
    val cBody = countCudaBody
    val sBody = scatterCudaBody

    factPlan.executeColumnar().mapPartitions { iter =>
      new Iterator[(ColumnarBatch, Array[Int])] {
        private val dimBatches = new Array[ColumnarBatch](k)
        private val probeTables = new Array[WholeStageNative.ProbeTable](k)
        private var dimViews: Array[ColumnView] = Array.empty
        private var probeHandles: Array[Long] = Array.empty
        private var initialized = false
        private var anyEmpty = false
        private var closed = false
        private var nextItem: (ColumnarBatch, Array[Int]) = _

        private def cleanup(): Unit = {
          if (!closed) {
            closed = true
            probeTables.foreach(p => if (p != null) p.safeClose())
            dimBatches.foreach(db => if (db != null) db.safeClose())
          }
        }

        private def initDims(): Unit = {
          initialized = true
          if (k > 0) {
            GpuSemaphore.acquireIfNecessary(TaskContext.get())
            opTime.ns {
              var d = 0
              while (d < k && !anyEmpty) {
                val db = GpuBroadcastHelper.getBroadcastBatch(bcasts(d), dimSchemas(d))
                dimBatches(d) = db
                if (db.numRows() == 0) anyEmpty = true
                else {
                  val keyCv = db.column(dKeyOrds(d)).asInstanceOf[GpuColumnVector].getBase
                  probeTables(d) = WholeStageNative.buildProbeTable(keyCv)
                }
                d += 1
              }
              if (!anyEmpty) {
                probeHandles = probeTables.map(_.getNativeHandle)
                dimViews = dColTbl.indices.map { i =>
                  dimBatches(dColTbl(i)).column(dColOrd(i)).asInstanceOf[GpuColumnVector].getBase.asInstanceOf[ColumnView]
                }.toArray
              }
            }
          }
          if (anyEmpty) {
            while (iter.hasNext) { withResource(iter.next()) { _ => } }
            cleanup()
          }
        }

        private def advance(): Unit = {
          if (!initialized) initDims()
          if (anyEmpty || closed) return
          val outDTypes = oTypes.map(GpuColumnVector.getNonNestedRapidsType)
          val cProg = WholeStageProgramPool.getProgram(cBody, "ws_part_count")
          val sProg = WholeStageProgramPool.getProgram(sBody, "ws_part_scatter")
          while (nextItem == null && iter.hasNext) {
            withResource(iter.next()) { b =>
              if (b.numRows() > 0) {
                opTime.ns {
                  val fViews = fOrds.map(o => b.column(o).asInstanceOf[GpuColumnVector].getBase.asInstanceOf[ColumnView])
                  val splits = new Array[Int](nParts)
                  withResource(WholeStageNative.runStagePartitionScatter(
                    cProg, sProg, fViews, b.numRows(), probeHandles, dimViews, outDTypes, nParts, splits)) { tbl =>
                    if (tbl.getRowCount > 0) {
                      numOutputRows += tbl.getRowCount
                      numOutputBatches += 1
                      val outBatch = GpuColumnVector.from(tbl, oTypes)
                      // Convert exclusive ends [end_0, ..., end_{P-1}] to start offsets [0, end_0, ..., end_{P-2}]
                      val starts = new Array[Int](nParts)
                      var p = 1
                      while (p < nParts) { starts(p) = splits(p - 1); p += 1 }
                      nextItem = (outBatch, starts)
                    }
                  }
                }
              }
            }
          }
          if (nextItem == null) cleanup()
        }

        override def hasNext: Boolean = {
          if (nextItem == null && !closed) advance()
          nextItem != null
        }

        override def next(): (ColumnarBatch, Array[Int]) = {
          if (!hasNext) throw new NoSuchElementException("empty fused partition")
          val r = nextItem
          nextItem = null
          r
        }
      }
    }
  }

  override def internalDoExecuteColumnar(): RDD[ColumnarBatch] =
    executePrePartitioned().map(_._1)

  override def simpleString(maxFields: Int): String =
    s"GpuWholeStageFusedPartition(dims=${dimPlans.length}, parts=$numPartitions)"
}
