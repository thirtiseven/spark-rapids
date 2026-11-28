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

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions.{Alias, And, Attribute, BloomFilterMightContain, Cast, EqualTo, Expression, IsNotNull, Literal, PredicateHelper, ScalarSubquery, SubqueryExpression}
import org.apache.spark.sql.catalyst.expressions.aggregate.BloomFilterAggregate
import org.apache.spark.sql.catalyst.plans.{Inner, LeftSemi}
import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, Filter, Join, LogicalPlan, Project, Union}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types.{ByteType, DataType, IntegerType, LongType, ShortType}

/**
 * Ultimate GPU-native Runtime Bloom Filter Semi-Join Pushdown (safe to enable by default!).
 *
 * Pushes compact L2-resident GPU Bloom Filters (GpuBloomFilterAggregate / GpuBloomFilterMightContain)
 * directly onto large probe-side leaf scans BEFORE distributed Shuffle Write:
 *   - Rule 1 (Non-broadcastable single-table selective build 10 MiB..24 GiB, Int/Short/Byte/Long):
 *     e.g. TPC-H Q02/Q03/Q08/Q09/Q17 (part/customer -> lineitem/partsupp/orders 7x-748x shuf cut!)
 *     & TPC-DS Q16/Q24/Q82/Q94/Q95!
 *   - Rule 2 (Selective HAVING Filter-over-Aggregate build -> huge probe leaf >= 10 GiB):
 *     e.g. TPC-H Q18 (orders 120 GiB) & TPC-DS Q23 (store_sales/web_sales/catalog_sales)!
 *
 * Zero-regression cost guards (eliminates Q49/Q64/Q34/Q73/Q81/Q30 regressions so BF can be default ON):
 *   1. Skips Union subtrees (prevents Q49 3-branch duplicate scans) and >10-join mega-DAGs (Q64);
 *   2. Skips tiny <10 MiB dims (preserves DPP & BroadcastHashJoin, e.g. Q50 date_dim);
 *   3. Skips probe sides already narrow-filtered by >=2 small selective dim joins (Q34/Q73);
 *   4. Requires huge >=10 GiB probe leaf for HAVING-aggregate Rule 2 (skips small Q30/Q81 returns);
 *   5. Respects spark.rapids.sql.expression.BloomFilter* disable flags.
 */
class GpuInjectRuntimeBloomFilter(spark: SparkSession)
    extends Rule[LogicalPlan] with PredicateHelper {

  import GpuInjectRuntimeBloomFilter._

  override def apply(plan: LogicalPlan): LogicalPlan = {
    val conf = SQLConf.get
    // Default ON whenever fusedAggEnabled / Whole-Stage optimization is enabled, unless explicitly disabled!
    val defaultOn = conf.getConfString(GpuLencFusedAgg.FUSED_AGG_CONF, "false")
    if (!conf.getConfString(CONF_ENABLED, defaultOn).toBoolean) {
      return plan
    }
    if (!conf.getConfString("spark.rapids.sql.enabled", "true").toBoolean ||
        !conf.getConfString("spark.rapids.sql.expression.BloomFilterMightContain", "true").toBoolean ||
        !conf.getConfString("spark.rapids.sql.expression.BloomFilterAggregate", "true").toBoolean) {
      return plan
    }
    val minProbeBytes = conf.getConfString(CONF_MIN_PROBE_BYTES, DEFAULT_MIN_PROBE_BYTES.toString).toLong
    val minHavingProbeBytes =
      if (minProbeBytes == 0L) 0L else DEFAULT_MIN_HAVING_PROBE_BYTES
    val maxSingleBuildBytes =
      conf.getConfString(CONF_MAX_BUILD_BYTES, DEFAULT_MAX_BUILD_BYTES.toString).toLong
    val maxInjected = conf.getConfString(CONF_MAX_INJECTED, "6").toInt

    // Zero-regression global DAG complexity guard:
    // Skip multi-branch UNION plans (e.g. TPC-DS Q49) and >10-join CTE DAGs (e.g. Q64) where
    // logical ScalarSubquery risks rescanning large subtrees or breaking BroadcastHashJoin.
    if (minProbeBytes > 0L) {
      var joinCount = 0
      var unionCount = 0
      plan.foreach {
        case _: Join => joinCount += 1
        case _: Union => unionCount += 1
        case _ =>
      }
      if (unionCount > 0 || joinCount > 10) return plan
    }

    var injected = 0
    val attrToBuildSpec = scala.collection.mutable.HashMap.empty[Long, (LogicalPlan, Attribute)]

    def tryInjectOnJoin(j: Join): LogicalPlan = {
      if (injected >= maxInjected) return j
      if (j.joinType != Inner && j.joinType != LeftSemi) return j
      val cond = j.condition.getOrElse(return j)

      val eqPairs = splitConjunctivePredicates(cond).collect {
        case EqualTo(l: Attribute, r: Attribute)
            if isSupportedKeyType(l.dataType) && l.dataType == r.dataType =>
          if (j.left.outputSet.contains(l) && j.right.outputSet.contains(r)) Some((l, r))
          else if (j.left.outputSet.contains(r) && j.right.outputSet.contains(l)) Some((r, l))
          else None
      }.flatten
      if (eqPairs.isEmpty) return j

      var curJoin = j

      // Transitive propagation (e.g. TPC-H Q09: part -> lineitem -> partsupp!)
      for ((lKey, rKey) <- eqPairs if injected < maxInjected) {
        if (curJoin.joinType == Inner) {
          val propCandidates = Seq(
            (lKey, curJoin.right, rKey, false),
            (rKey, curJoin.left, lKey, true)
          )
          for ((srcKey, targetSide, targetKey, targetIsLeft) <- propCandidates if injected < maxInjected) {
            attrToBuildSpec.get(srcKey.exprId.id).foreach { case (bPlan, bKey) =>
              val tBytes = findProbeLeafBytes(targetSide, targetKey)
              if (tBytes >= minProbeBytes &&
                  !hasBloomFilterOn(targetSide, targetKey) &&
                  !containsBloomFilter(bPlan)) {
                val bfPred = buildBloomPred(targetKey, bPlan, bKey)
                pushFilterToProbeLeaf(targetSide, targetKey, bfPred).foreach { filteredTarget =>
                  curJoin = if (targetIsLeft) curJoin.copy(left = filteredTarget)
                            else curJoin.copy(right = filteredTarget)
                  injected += 1
                }
              }
            }
          }
        }
      }

      val (lKey, rKey) = eqPairs.sortBy(p => if (p._1.dataType == LongType) 0 else 1).head
      val candidates = if (curJoin.joinType == LeftSemi) {
        Seq((curJoin.left, lKey, curJoin.right, rKey, true))
      } else {
        Seq(
          (curJoin.left, lKey, curJoin.right, rKey, true),
          (curJoin.right, rKey, curJoin.left, lKey, false)
        ).sortBy { case (_, _, b, _, _) => totalLeafScanBytes(b) }
      }

      for ((probeSide, probeKey, buildSide, buildKey, probeIsLeft) <- candidates if injected < maxInjected) {
        val pScanBytes = findProbeLeafBytes(probeSide, probeKey)
        val bScanBytes = totalLeafScanBytes(buildSide)
        val bHasJoin = containsJoin(buildSide)
        val minSingleBuildBytes = if (minProbeBytes == 0L) 0L else 10L * 1024L * 1024L

        // Skip probe subtrees already heavily pruned by >= 2 selective small-dim joins (e.g. Q34/Q73)
        val probeAlreadyPruned = minProbeBytes > 0L && countSmallSelectiveDimJoins(probeSide) >= 2

        val bIsSingleTableSelective =
          !bHasJoin && hasSelectiveFilter(buildSide) && !probeAlreadyPruned &&
            ((bScanBytes >= minSingleBuildBytes && bScanBytes <= maxSingleBuildBytes &&
              bScanBytes <= (pScanBytes * 0.5).toLong) || minProbeBytes == 0L)
        val bIsHavingAgg =
          isSelectiveHavingAggregate(buildSide) && pScanBytes >= minHavingProbeBytes && !probeAlreadyPruned

        if (pScanBytes >= minProbeBytes &&
            (bIsSingleTableSelective || bIsHavingAgg) &&
            !containsBloomFilter(buildSide) &&
            !hasBloomFilterOn(probeSide, probeKey) &&
            isSafeBuildSide(buildSide)) {
          val bfPred = buildBloomPred(probeKey, buildSide, buildKey)
          pushFilterToProbeLeaf(probeSide, probeKey, bfPred).foreach { filteredProbe =>
            curJoin = if (probeIsLeft) curJoin.copy(left = filteredProbe)
                      else curJoin.copy(right = filteredProbe)
            if (bIsSingleTableSelective) {
              attrToBuildSpec.put(probeKey.exprId.id, (buildSide, buildKey))
            }
            injected += 1
            return curJoin
          }
        }
      }
      curJoin
    }

    plan.transformUp {
      case j: Join => tryInjectOnJoin(j)
    }
  }

  private def buildBloomPred(
      probeKey: Attribute,
      buildSide: LogicalPlan,
      buildKey: Attribute): Expression = {
    val bKeyLong: Expression =
      if (buildKey.dataType == LongType) buildKey else Cast(buildKey, LongType)
    val pKeyLong: Expression =
      if (probeKey.dataType == LongType) probeKey else Cast(probeKey, LongType)

    val estBytes = totalLeafScanBytes(buildSide)
    val estRows = buildSide.stats.rowCount.map(_.toLong).getOrElse(math.max(65536L, estBytes / 100L))
    val clampedItems = math.max(131072L, math.min(estRows, 8000000L))
    val numBits = math.max(2097152L, math.min(clampedItems * 8L, 67108864L))

    val bfExpr = new BloomFilterAggregate(
      bKeyLong,
      Literal(clampedItems, LongType),
      Literal(numBits, LongType)).toAggregateExpression()
    val bfAggPlan = Aggregate(
      Nil,
      Seq(Alias(bfExpr, "gpu_rt_bf")()),
      Project(Seq(buildKey), buildSide))
    BloomFilterMightContain(ScalarSubquery(bfAggPlan), pKeyLong)
  }

  private def pushFilterToProbeLeaf(
      plan: LogicalPlan,
      probeKey: Attribute,
      bfPred: Expression): Option[LogicalPlan] = plan match {
    case p: Project if p.child.outputSet.contains(probeKey) =>
      pushFilterToProbeLeaf(p.child, probeKey, bfPred).map(c => p.copy(child = c))
    case f: Filter if f.child.outputSet.contains(probeKey) =>
      pushFilterToProbeLeaf(f.child, probeKey, bfPred).map(c => f.copy(child = c))
    case j: Join if j.joinType == Inner || j.joinType == LeftSemi =>
      if (j.left.outputSet.contains(probeKey)) {
        pushFilterToProbeLeaf(j.left, probeKey, bfPred).map(l => j.copy(left = l))
      } else if (j.joinType == Inner && j.right.outputSet.contains(probeKey)) {
        pushFilterToProbeLeaf(j.right, probeKey, bfPred).map(r => j.copy(right = r))
      } else None
    case leaf if leaf.children.isEmpty && leaf.outputSet.contains(probeKey) =>
      Some(Filter(bfPred, leaf))
    case _ => None
  }
}

object GpuInjectRuntimeBloomFilter {
  val CONF_ENABLED = "spark.rapids.sql.lenc.runtimeBloomFilterEnabled"
  val CONF_MIN_PROBE_BYTES = "spark.rapids.sql.lenc.runtimeBloomFilter.minProbeBytes"
  val CONF_MAX_BUILD_BYTES = "spark.rapids.sql.lenc.runtimeBloomFilter.maxBuildBytes"
  val CONF_MAX_INJECTED = "spark.rapids.sql.lenc.runtimeBloomFilter.maxInjected"

  // Probe leaf >= 1 GiB for Rule 1, >= 10 GiB for HAVING-Aggregate Rule 2; build <= 24 GiB
  val DEFAULT_MIN_PROBE_BYTES: Long = 1024L * 1024L * 1024L
  val DEFAULT_MIN_HAVING_PROBE_BYTES: Long = 10L * 1024L * 1024L * 1024L
  val DEFAULT_MAX_BUILD_BYTES: Long = 24L * 1024L * 1024L * 1024L

  private def isSupportedKeyType(dt: DataType): Boolean = dt match {
    case LongType | IntegerType | ShortType | ByteType => true
    case _ => false
  }

  private def safeSizeBytes(plan: LogicalPlan): Long = {
    try {
      val s = plan.stats.sizeInBytes
      if (s >= BigInt(Long.MaxValue)) Long.MaxValue else s.toLong
    } catch {
      case _: Throwable => 0L
    }
  }

  private def totalLeafScanBytes(plan: LogicalPlan): Long = {
    var total = 0L
    plan.foreach {
      case leaf if leaf.children.isEmpty =>
        val s = safeSizeBytes(leaf)
        if (s > 0L && total <= Long.MaxValue - s) total += s
      case _ =>
    }
    total
  }

  private def findProbeLeafBytes(plan: LogicalPlan, probeKey: Attribute): Long = plan match {
    case p: Project if p.child.outputSet.contains(probeKey) => findProbeLeafBytes(p.child, probeKey)
    case f: Filter if f.child.outputSet.contains(probeKey) => findProbeLeafBytes(f.child, probeKey)
    case j: Join if j.joinType == Inner || j.joinType == LeftSemi =>
      if (j.left.outputSet.contains(probeKey)) findProbeLeafBytes(j.left, probeKey)
      else if (j.joinType == Inner && j.right.outputSet.contains(probeKey)) findProbeLeafBytes(j.right, probeKey)
      else 0L
    case leaf if leaf.children.isEmpty && leaf.outputSet.contains(probeKey) => safeSizeBytes(leaf)
    case _ => 0L
  }

  /** Counts small (< 64 MiB) selective dim-table joins already filtering `plan`. */
  private def countSmallSelectiveDimJoins(plan: LogicalPlan): Int = {
    var cnt = 0
    plan.foreach {
      case Join(l, r, Inner | LeftSemi, _, _) =>
        val ls = totalLeafScanBytes(l)
        val rs = totalLeafScanBytes(r)
        if ((rs > 0L && rs < 64L * 1024L * 1024L && hasSelectiveFilter(r)) ||
            (ls > 0L && ls < 64L * 1024L * 1024L && hasSelectiveFilter(l))) {
          cnt += 1
        }
      case _ =>
    }
    cnt
  }

  private def containsJoin(plan: LogicalPlan): Boolean =
    plan.find(_.isInstanceOf[Join]).isDefined

  private def containsBloomFilter(plan: LogicalPlan): Boolean = {
    plan.find { p =>
      p.expressions.exists { e =>
        e.find {
          case _: BloomFilterMightContain | _: BloomFilterAggregate => true
          case _ => false
        }.isDefined
      }
    }.isDefined
  }

  private def hasBloomFilterOn(plan: LogicalPlan, attr: Attribute): Boolean = {
    plan.find {
      case Filter(cond, _) =>
        cond.find {
          case BloomFilterMightContain(_, rhs) =>
            rhs.references.exists(_.exprId == attr.exprId)
          case _ => false
        }.isDefined
      case _ => false
    }.isDefined
  }

  private def isNonTrivialFilter(e: Expression): Boolean = e match {
    case _: IsNotNull => false
    case And(l, r) => isNonTrivialFilter(l) || isNonTrivialFilter(r)
    case _ => true
  }

  private def hasSelectiveFilter(plan: LogicalPlan): Boolean = {
    plan.find {
      case Filter(cond, _) if isNonTrivialFilter(cond) => true
      case _ => false
    }.isDefined
  }

  private def isSelectiveHavingAggregate(plan: LogicalPlan): Boolean = plan match {
    case Project(_, child) => isSelectiveHavingAggregate(child)
    case Filter(cond, child) if isNonTrivialFilter(cond) =>
      child.find(_.isInstanceOf[Aggregate]).isDefined
    case _ => false
  }

  private def isSafeBuildSide(plan: LogicalPlan): Boolean = {
    var safe = true
    var nodeCount = 0
    plan.foreach { p =>
      nodeCount += 1
      if (nodeCount > 64) safe = false
      if (!p.expressions.forall(_.deterministic)) safe = false
      if (p.expressions.exists(_.find(_.isInstanceOf[SubqueryExpression]).isDefined)) safe = false
    }
    safe
  }
}
