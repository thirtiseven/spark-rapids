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

import ai.rapids.cudf.{ColumnVector, ColumnView, DType, OutOfBoundsPolicy, Scalar, Table => CudfTable}
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
import org.apache.spark.sql.rapids.{GpuContains, GpuEqualTo, GpuFileSourceScanExec, GpuStartsWith, GpuSubstring}
import org.apache.spark.sql.rapids.execution.{GpuBroadcastHashJoinExecBase, GpuBroadcastHelper, GpuShuffleExchangeExecBase}
import org.apache.spark.sql.types._
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * Whole-Stage GPU Codegen physical-plan planner backed by `StageIR` (Produce/Consume pipeline IR)
 * and libcudf's `cudf::whole_stage` execution engine (pluggable Raw/Vortex/CONSTANT/LSHF Sources,
 * L2 LUT/Hash multi-way Broadcast/Shuffled probes, Expand(ROLLUP), native GPU Open-Addressing
 * Hash-Agg Sink, MaterializeSink, and ShufflePartitionSink).
 */
object GpuWholeStageCodegenRewrite {

  import StageIR._

  private val MAX_DIMS = 6
  private val MAX_FACT_COLS = 24
  private val MAX_DIM_COLS = 24
  private val MAX_GROUP_KEYS = 8
  private val MAX_AGGS = 32

  private def isJoinKeyType(dt: DataType): Boolean = dt match {
    case IntegerType | LongType | DateType => true
    case _ => false
  }

  private def isFixedGroupKeyType(dt: DataType): Boolean = dt match {
    case BooleanType | ByteType | ShortType | IntegerType | LongType | DateType => true
    case d: DecimalType if d.precision <= 18 => true
    case _ => false
  }

  private def isPartitionConstType(dt: DataType): Boolean = dt match {
    case BooleanType | ByteType | ShortType | IntegerType | LongType | DateType => true
    case _ => false
  }

  private def isCodegenLeafType(dt: DataType): Boolean = dt match {
    case BooleanType | ByteType | ShortType | IntegerType | LongType |
         FloatType | DoubleType | DateType => true
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

  private def extractProjectMap(projectList: Seq[NamedExpression]): Map[ExprId, Expression] = {
    projectList.collect {
      case a: GpuAlias => a.exprId -> unwrapJit(a.child)
      case a: Alias => a.exprId -> unwrapJit(a.child)
    }.toMap
  }

  private def isPassThroughProject(projectList: Seq[NamedExpression]): Option[Map[ExprId, Attribute]] = {
    val m = mutable.HashMap.empty[ExprId, Attribute]
    projectList.foreach {
      case a: Attribute => m.put(a.exprId, a)
      case a: Alias =>
        a.child match {
          case attr: Attribute => m.put(a.exprId, attr)
          case _ => return None
        }
      case g: GpuAlias =>
        g.child match {
          case attr: Attribute => m.put(g.exprId, attr)
          case _ => return None
        }
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
   * compressed segments (FSST / FastLanes BitPacked+FoR / ALP / Dict) + optional scalar Hive
   * partition CONSTANT columns can stream directly into libcudf cudf::whole_stage!
   */
  private[lenc] def isDirectCompressedVortexScan(plan: SparkPlan): Boolean = plan match {
    case fsse: GpuFileSourceScanExec =>
      fsse.relation.fileFormat.isInstanceOf[com.nvidia.spark.rapids.vortex.GpuReadVortexFileFormat] &&
        !fsse.bucketedScan &&
        fsse.relation.partitionSchema.fields.forall(f => isPartitionConstType(f.dataType)) &&
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

  /** Unwinds a star-join + optional filter/project chain into clean StageIR operators! */
  private def tryFuseWholeStageAgg(agg: GpuHashAggregateExec): Option[SparkPlan] = {
    if (agg.aggregateExpressions.isEmpty ||
        !agg.aggregateExpressions.forall(_.mode == Partial) ||
        agg.aggregateExpressions.exists(_.isDistinct) ||
        agg.groupingExpressions.length > MAX_GROUP_KEYS) {
      return None
    }
    if (agg.groupingExpressions.exists(g => !isFixedGroupKeyType(g.dataType) && g.dataType != StringType)) {
      return None
    }

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
    val factPlan = cur

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

    def rewriteAliases(e: Expression): Expression = e.transformUp {
      case a: Attribute => resolveAttr(a)
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

    val (rawTerms, outBufToTerm, outBufModes) =
      GpuLencFusedAgg.extractAggTermsAndBufferMap(agg.aggregateExpressions, topProjMap).getOrElse(return None)
    if (rawTerms.isEmpty || rawTerms.length > MAX_AGGS) return None

    val groupExprs = agg.groupingExpressions.map(g => rewriteAliases(GpuLencFusedAgg.inlineProjectMap(g, topProjMap)))
    val factAttrIds = factIdToOrd.keySet
    if (k >= 1 && groupExprs.exists(ge => ge.references.exists(a => factAttrIds.contains(a.exprId)))) {
      return None
    }

    // Identify Broadcast-Dim String / Substring(String, pos, len) group keys
    val groupKeyIsDimStr = new Array[Boolean](groupExprs.length)
    val groupKeyDimStrRef = new Array[(Int, Int, Int, Int)](groupExprs.length)
    val nativeKeyTypes = new Array[DataType](groupExprs.length)
    for (gi <- groupExprs.indices) {
      val ge = groupExprs(gi)
      if (ge.dataType == StringType) {
        ge match {
          case a: Attribute =>
            dimIdToLoc.get(a.exprId) match {
              case Some((dIdx, ord)) =>
                groupKeyIsDimStr(gi) = true
                groupKeyDimStrRef(gi) = (dIdx, ord, 0, -1)
                nativeKeyTypes(gi) = IntegerType
              case None => return None
            }
          case GpuSubstring(a: Attribute, GpuLiteral(posObj, IntegerType), GpuLiteral(lenObj, IntegerType))
              if posObj != null && lenObj != null =>
            val pos1 = posObj.asInstanceOf[Int]
            val slen = lenObj.asInstanceOf[Int]
            if (pos1 < 1 || slen < 0) return None
            dimIdToLoc.get(a.exprId) match {
              case Some((dIdx, ord)) =>
                groupKeyIsDimStr(gi) = true
                groupKeyDimStrRef(gi) = (dIdx, ord, pos1 - 1, slen)
                nativeKeyTypes(gi) = IntegerType
              case None => return None
            }
          case _ => return None
        }
      } else {
        nativeKeyTypes(gi) = ge.dataType
      }
    }

    val factSlotMap = mutable.LinkedHashMap.empty[Int, Int]
    val dimSlotMap = mutable.LinkedHashMap.empty[(Int, Int, Boolean, Int, Int), Int]
    def allocFactSlot(ord: Int): Int = factSlotMap.getOrElseUpdate(ord, factSlotMap.size)
    def allocDimSlot(dIdx: Int, ord: Int, isStrDict: Boolean, ss0: Int = 0, ssl: Int = -1): Int =
      dimSlotMap.getOrElseUpdate((dIdx, ord, isStrDict, ss0, ssl), dimSlotMap.size)

    for (d <- 0 until k) {
      if (probeKeySrcTbl(d) == -1) allocFactSlot(probeKeySrcOrd(d))
      else allocDimSlot(probeKeySrcTbl(d), probeKeySrcOrd(d), isStrDict = false)
    }

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

    val factFilterClean = factFilterOpt.map(rewriteAliases)
    val postFilterClean = postFilterOpt.map(rewriteAliases)
    val cleanTerms = rawTerms.map(t => t.copy(inputExpr = t.inputExpr.map(rewriteAliases)))

    if (factFilterClean.exists(e => !collectExprCols(e, allowDim = false))) return None
    if (postFilterClean.exists(e => !collectExprCols(e, allowDim = true))) return None
    groupExprs.zipWithIndex.foreach { case (ge, gi) =>
      if (groupKeyIsDimStr(gi)) {
        val (dIdx, ord, ss0, ssl) = groupKeyDimStrRef(gi)
        allocDimSlot(dIdx, ord, isStrDict = true, ss0, ssl)
      } else if (!collectExprCols(ge, allowDim = true)) {
        return None
      }
    }
    cleanTerms.foreach { t =>
      if (t.inputExpr.exists(e => !collectExprCols(e, allowDim = true))) return None
    }
    if (factSlotMap.size > MAX_FACT_COLS || dimSlotMap.size > MAX_DIM_COLS) return None

    // Build initial VarEnv (fact columns) + clean Produce/Consume StageOp chain!
    val initialEnv: VarEnv = factSlotMap.map { case (ord, slot) =>
      val attr = factPlan.output(ord)
      attr.exprId -> FactCol(slot, attr.dataType)
    }.toMap

    val ops = mutable.ArrayBuffer.empty[StageOp]
    factFilterClean.foreach(fe => ops += FilterOp(fe))
    for (d <- 0 until k) {
      val keyAttr = resolveAttr(orderedJoins(d).leftKeys.head.asInstanceOf[Attribute])
      val dBindings = dimSlotMap.collect {
        case ((dIdx, ord, false, _, _), slot) if dIdx == d =>
          val a = dimPlans(d).output(ord)
          (a.exprId, slot, a.dataType)
      }.toSeq
      ops += HashProbeOp(d, keyAttr, dBindings, JoinInner)
    }
    postFilterClean.foreach(pe => ops += FilterOp(pe))

    val gkDimStrSlot = Array.tabulate(groupExprs.length) { gi =>
      if (groupKeyIsDimStr(gi)) {
        val (dIdx, ord, ss0, ssl) = groupKeyDimStrRef(gi)
        dimSlotMap((dIdx, ord, true, ss0, ssl))
      } else -1
    }
    val gkDimIdx = Array.tabulate(groupExprs.length) { gi =>
      if (groupKeyIsDimStr(gi)) groupKeyDimStrRef(gi)._1 else -1
    }

    val aggSink = HashAggSink(groupExprs, gkDimStrSlot, gkDimIdx, cleanTerms)
    val (cudaBody, kernelName) = StageCompiler.compileAgg(initialEnv, ops.toSeq, aggSink).getOrElse(return None)

    val outAttrs = agg.output
    val outTypes = outAttrs.map(_.dataType).toArray
    val dimSpecs = dimSlotMap.keys.toArray

    Some(GpuWholeStageStarJoinAggExec(
      output = outAttrs,
      factPlan = factPlan,
      dimPlans = dimPlans.toSeq,
      factOrds = factSlotMap.keys.toArray,
      dimKeyOrds = dimKeyOrds,
      dimColTableIdx = dimSpecs.map(_._1),
      dimColOrd = dimSpecs.map(_._2),
      dimColIsStrDict = dimSpecs.map(_._3),
      dimColSubstrStart0 = dimSpecs.map(_._4),
      dimColSubstrLen = dimSpecs.map(_._5),
      nativeKeyTypes = nativeKeyTypes,
      groupKeyDimStrSlotIdx = gkDimStrSlot,
      aggKinds = cleanTerms.map(_.kind).toArray,
      outBufToTerm = outBufToTerm,
      outBufModes = outBufModes,
      outTypes = outTypes,
      cudaBody = cudaBody,
      kernelName = kernelName))
  }

  /**
   * Opt-in Fused Stage -> Spark Murmur3 HashPartition Scatter compiled via `StageIR`.
   */
  private def tryFuseWholeStageShuffle(shuf: GpuShuffleExchangeExecBase): Option[SparkPlan] = {
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
    cur match {
      case p: GpuProjectExec =>
        topProjMap = extractProjectMap(p.projectList)
        cur = peelCoalesce(p.child)
      case _ =>
    }
    var factFilterOpt: Option[Expression] = None
    cur match {
      case f: GpuFilterExec =>
        factFilterOpt = Some(unwrapJit(f.condition))
        cur = peelCoalesce(f.child)
      case _ =>
    }
    // Keep K == 0 only so AQE dynamic broadcast-join optimization is never disturbed!
    if (factFilterOpt.isEmpty) return None
    val factPlan = cur
    val factIdToOrd = factPlan.output.zipWithIndex.map { case (a, i) => a.exprId -> i }.toMap
    val factSlotMap = mutable.LinkedHashMap.empty[Int, Int]
    def allocFact(ord: Int): Int = factSlotMap.getOrElseUpdate(ord, factSlotMap.size)

    val outExprs = shuf.output.map(a => GpuLencFusedAgg.inlineProjectMap(a, topProjMap))
    val partExprs = hp.children.map(e => GpuLencFusedAgg.inlineProjectMap(e, topProjMap))

    def collectCols(e: Expression): Boolean = {
      var ok = true
      e.foreach {
        case a: Attribute =>
          factIdToOrd.get(a.exprId) match {
            case Some(ord) if isCodegenLeafType(a.dataType) || a.dataType == StringType => allocFact(ord)
            case _ => ok = false
          }
        case _ =>
      }
      ok
    }
    if (!collectCols(factFilterOpt.get)) return None
    if (outExprs.exists(e => !collectCols(e))) return None
    if (partExprs.exists(e => !collectCols(e))) return None
    if (factSlotMap.size > MAX_FACT_COLS) return None

    val initialEnv: VarEnv = factSlotMap.map { case (ord, slot) =>
      val a = factPlan.output(ord)
      a.exprId -> FactCol(slot, a.dataType)
    }.toMap
    val ops = Seq(FilterOp(factFilterOpt.get))
    val (countBody, scatterBody) = StageCompiler.compileTwoPass(
      initialEnv, ops, Some(partExprs), outExprs).getOrElse(return None)

    val fusedChild = GpuWholeStageFusedPartitionExec(
      output = shuf.output,
      factPlan = factPlan,
      dimPlans = Seq.empty,
      factOrds = factSlotMap.keys.toArray,
      dimKeyOrds = Array.empty,
      dimColTableIdx = Array.empty,
      dimColOrd = Array.empty,
      outTypes = shuf.output.map(_.dataType).toArray,
      numPartitions = hp.numPartitions,
      countCudaBody = countBody,
      scatterCudaBody = scatterBody)

    Some(shuf.withNewChildren(Seq(fusedChild)))
  }
}

private object WholeStageProgramPool {
  private val cache = new ConcurrentHashMap[String, WholeStageNative.Program]()

  def getProgram(cudaBody: String, kernelName: String): WholeStageNative.Program = {
    val key = kernelName + ":" + cudaBody
    var p = cache.get(key)
    if (p == null) {
      val created = WholeStageNative.compileProgram(cudaBody, kernelName)
      val existing = cache.putIfAbsent(key, created)
      if (existing != null) {
        created.close()
        p = existing
      } else {
        p = created
      }
    }
    p
  }
}

/**
 * Serializable per-task compressed Vortex file split specification including optional
 * zero-HBM Hive partition CONSTANT column values (`source_kind::CONSTANT`).
 */
final case class CompressedVortexSplitSpec(
    paths: Array[String],
    partConstVals: Array[Long],
    partConstNulls: Array[Byte])

/**
 * Whole-Stage GPU Star-Join + Native Open-Addressing Hash-Agg Exec backed by libcudf
 * `cudf::whole_stage` and compiled via `StageIR`. Supports both standard columnar inputs
 * and direct zero-HBM compressed Vortex (non-partitioned & Hive-partitioned!) streaming.
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
    OP_TIME_LEGACY -> createNanoTimingMetric(MODERATE_LEVEL, DESCRIPTION_OP_TIME_LEGACY)
  )

  private def gatherDictStrings(keysCv: ColumnVector, codesCv: ColumnView): ColumnVector = {
    val nKeys = keysCv.getRowCount.toInt
    val gatherMap = withResource(Scalar.fromInt(nKeys)) { oob =>
      if (codesCv.getType == DType.INT32) {
        codesCv.replaceNulls(oob)
      } else {
        withResource(codesCv.castTo(DType.INT32)) { c32 =>
          c32.replaceNulls(oob)
        }
      }
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
    val totalOut = numKeys + obm.length
    val cols = (0 until totalOut).safeMap { i =>
      if (i < numKeys && gkDimStrSlots(i) >= 0) {
        val keysCv = dimStrDictKeys(gkDimStrSlots(i))
        val codeCv = rawTbl.getColumn(i)
        GpuColumnVector.from(gatherDictStrings(keysCv, codeCv), StringType)
      } else {
        GenericReduceBatchBuilder.projectSingleColumn(rawTbl, i, numKeys, obm, obModes, oTypes)
      }
    }.toArray
    new ColumnarBatch(cols.asInstanceOf[Array[org.apache.spark.sql.vectorized.ColumnVector]], rawTbl.getRowCount.toInt)
  }

  private def extractScalarPartValue(row: InternalRow, ord: Int, dt: DataType): (Long, Byte) = {
    if (row.isNullAt(ord)) (0L, 1.toByte)
    else dt match {
      case BooleanType => (if (row.getBoolean(ord)) 1L else 0L, 0.toByte)
      case ByteType => (row.getByte(ord).toLong, 0.toByte)
      case ShortType => (row.getShort(ord).toLong, 0.toByte)
      case IntegerType | DateType => (row.getInt(ord).toLong, 0.toByte)
      case LongType => (row.getLong(ord), 0.toByte)
      case _ => (0L, 1.toByte)
    }
  }

  private def elemWidthOf(dt: DataType): Int = dt match {
    case BooleanType | ByteType => 1
    case ShortType => 2
    case IntegerType | DateType => 4
    case LongType => 8
    case _ => 4
  }

  private def buildCompressedVortexPlan(
      fsse: GpuFileSourceScanExec,
      fOrds: Array[Int]): Option[(Array[CompressedVortexSplitSpec], Array[String], Array[Int], Array[Int], Array[Int])] = {
    val partSchema = fsse.relation.partitionSchema
    val partNameToIdx = partSchema.fields.zipWithIndex.map { case (f, i) => f.name -> i }.toMap

    val fileColNamesBuf = mutable.ArrayBuffer.empty[String]
    val fileColSlotsBuf = mutable.ArrayBuffer.empty[Int]
    val partConstSlotsBuf = mutable.ArrayBuffer.empty[Int]
    val partConstWidthsBuf = mutable.ArrayBuffer.empty[Int]
    val partOrdInRowBuf = mutable.ArrayBuffer.empty[(Int, DataType)]

    fOrds.zipWithIndex.foreach { case (ord, slotIdx) =>
      val attr = fsse.output(ord)
      partNameToIdx.get(attr.name) match {
        case Some(pIdx) =>
          partConstSlotsBuf += slotIdx
          partConstWidthsBuf += elemWidthOf(attr.dataType)
          partOrdInRowBuf += ((pIdx, attr.dataType))
        case None =>
          fileColNamesBuf += attr.name
          fileColSlotsBuf += slotIdx
      }
    }
    // Need at least 1 Vortex file column so Vortex yields exact batch row counts
    if (fileColNamesBuf.isEmpty) return None

    val nPart = partOrdInRowBuf.length
    val splits = fsse.inputRDD.partitions.collect {
      case fp: FilePartition =>
        val pfs = FilePartitionShims.getFiles(fp)
        val paths = new Array[String](pfs.length)
        val pVals = new Array[Long](pfs.length * nPart)
        val pNulls = new Array[Byte](pfs.length * nPart)
        var fi = 0
        while (fi < pfs.length) {
          val pf = pfs(fi)
          paths(fi) = new java.net.URI(pf.filePath.toString).getPath
          var pi = 0
          while (pi < nPart) {
            val (pOrd, pDt) = partOrdInRowBuf(pi)
            val (v, isNull) = extractScalarPartValue(pf.partitionValues, pOrd, pDt)
            pVals(fi * nPart + pi) = v
            pNulls(fi * nPart + pi) = isNull
            pi += 1
          }
          fi += 1
        }
        CompressedVortexSplitSpec(paths, pVals, pNulls)
    }.filter(_.paths.nonEmpty)

    if (splits.isEmpty) None
    else Some((
      splits,
      fileColNamesBuf.toArray,
      fileColSlotsBuf.toArray,
      partConstSlotsBuf.toArray,
      partConstWidthsBuf.toArray))
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

    val directVortexOpt =
      if (GpuWholeStageCodegenRewrite.isDirectCompressedVortexScan(factPlan)) {
        buildCompressedVortexPlan(factPlan.asInstanceOf[GpuFileSourceScanExec], fOrds)
      } else None

    def runTask(
        vortexSplitOpt: Option[(CompressedVortexSplitSpec, Array[String], Array[Int], Array[Int], Array[Int])],
        batchIter: Iterator[ColumnarBatch]): Option[ColumnarBatch] = {
      val nKeyDTypes = nKeyTypes.map(GpuColumnVector.getNonNestedRapidsType)
      val dimBatches = new Array[ColumnarBatch](k)
      val probeTables = new Array[WholeStageNative.ProbeTable](k)
      val dimEncodedCols = new Array[ColumnVector](dColTbl.length)
      val dimStrDictKeys = new Array[ColumnVector](dColTbl.length)
      var taskState: WholeStageNative.HashAggTask = null
      var outBatch: Option[ColumnarBatch] = None
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
          while (batchIter.hasNext) { withResource(batchIter.next()) { _ => } }
          if (numKeys == 0) {
            taskState = prog.openHashAggTask(nKeyDTypes, aKinds, 1)
            withResource(taskState.finish()) { r =>
              if (r.getRowCount > 0) {
                numOutputRows += r.getRowCount
                numOutputBatches += 1
                outBatch = Some(GenericReduceBatchBuilder.projectBuffers(r, 0, obm, obModes, oTypes))
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
          val initCap = if (numKeys == 0) 1 else 262144
          vortexSplitOpt match {
            case Some((sp, fColNames, fColSlots, pConstSlots, pConstWidths)) =>
              opTime.ns {
                taskState = prog.openHashAggTask(nKeyDTypes, aKinds, initCap)
                VortexNative.streamCompressedPathsToWholeStageAgg(
                  sp.paths, fColNames, fOrds.length, fColSlots, pConstSlots, pConstWidths,
                  sp.partConstVals, sp.partConstNulls, taskState, probeHandles, dimViews)
                withResource(taskState.finish()) { r =>
                  if (r.getRowCount > 0) {
                    numOutputRows += r.getRowCount
                    numOutputBatches += 1
                    outBatch = Some(buildBatchFromWholeStageTable(
                      r, dimStrDictKeys, numKeys, gkDimStrSlots, obm, obModes, oTypes))
                  }
                }
              }
            case None =>
              while (batchIter.hasNext) {
                withResource(batchIter.next()) { b =>
                  if (b.numRows() > 0) {
                    opTime.ns {
                      if (taskState == null) {
                        taskState = prog.openHashAggTask(nKeyDTypes, aKinds, initCap)
                      }
                      val factViews = fOrds.map(o => b.column(o).asInstanceOf[GpuColumnVector].getBase.asInstanceOf[ColumnView])
                      taskState.addBatch(factViews, b.numRows(), probeHandles, dimViews)
                    }
                  }
                }
              }
              if (taskState == null && numKeys == 0) {
                taskState = prog.openHashAggTask(nKeyDTypes, aKinds, 1)
              }
              if (taskState != null) {
                opTime.ns {
                  withResource(taskState.finish()) { r =>
                    if (r.getRowCount > 0) {
                      numOutputRows += r.getRowCount
                      numOutputBatches += 1
                      outBatch = Some(buildBatchFromWholeStageTable(
                        r, dimStrDictKeys, numKeys, gkDimStrSlots, obm, obModes, oTypes))
                    }
                  }
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
      outBatch
    }

    directVortexOpt match {
      case Some((splits, fColNames, fColSlots, pConstSlots, pConstWidths)) =>
        sparkContext.parallelize(splits.toSeq, splits.length).mapPartitions { spIter =>
          new Iterator[ColumnarBatch] {
            private var pending: ColumnarBatch = _
            Option(TaskContext.get()).foreach { tc =>
              tc.addTaskCompletionListener[Unit](_ => if (pending != null) { pending.safeClose(); pending = null })
            }
            override def hasNext: Boolean = {
              while (pending == null && spIter.hasNext) {
                val sp = spIter.next()
                runTask(Some((sp, fColNames, fColSlots, pConstSlots, pConstWidths)), Iterator.empty)
                  .foreach(b => pending = b)
              }
              pending != null
            }
            override def next(): ColumnarBatch = {
              if (!hasNext) throw new NoSuchElementException("empty whole-stage vortex result")
              val out = pending
              pending = null
              out
            }
          }
        }

      case None =>
        factPlan.executeColumnar().mapPartitions { iter =>
          new Iterator[ColumnarBatch] {
            private var finished = false
            private var pending: ColumnarBatch = _
            Option(TaskContext.get()).foreach { tc =>
              tc.addTaskCompletionListener[Unit](_ => if (pending != null) { pending.safeClose(); pending = null })
            }
            override def hasNext: Boolean = {
              if (!finished) {
                finished = true
                runTask(None, iter).foreach(b => pending = b)
              }
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
 * Opt-in Fused Stage -> Spark Murmur3 HashPartition Scatter right underneath GpuShuffleExchangeExec.
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

        Option(TaskContext.get()).foreach { tc =>
          tc.addTaskCompletionListener[Unit](_ => {
            if (nextItem != null) { nextItem._1.safeClose(); nextItem = null }
            cleanup()
          })
        }

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
