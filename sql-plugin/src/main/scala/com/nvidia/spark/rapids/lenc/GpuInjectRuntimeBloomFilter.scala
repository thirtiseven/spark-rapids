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
import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, Filter, Join, LogicalPlan, Project}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types.{ByteType, DataType, IntegerType, LongType, ShortType}

/**
 * Phase III (Distributed Opt #1): Opt-in GPU-native Runtime Bloom Filter Semi-Join Pushdown.
 *
 * Pushes compact L2-resident GPU Bloom Filters (1-8 MiB, built via GpuBloomFilterAggregate and
 * probed via GpuBloomFilterMightContain) directly onto large probe-side leaf scans BEFORE
 * distributed Shuffle Write, targeting two high-ROI cases missed by Spark 3.5's InjectRuntimeFilter:
 *   Rule 1 (Selective Single-Table Build <= 20 GiB, Int/Short/Byte/Long keys):
 *     e.g. TPC-H Q08 (p_type = ... -> lineitem 150x reduction), Q09 (p_name LIKE '%hot%' ->
 *     lineitem & partsupp 18x reduction!), Q03 (c_mktsegment = ... -> orders 5x reduction),
 *     TPC-DS Q24/Q64/Q93/Q95 IntType *_sk joins!
 *   Rule 2 (Selective HAVING Filter-over-Aggregate Build):
 *     e.g. TPC-H Q18 (HAVING sum(l_quantity) > 314 -> orders 30,000x reduction!),
 *     TPC-DS Q23 (frequent_ss_items / best_ss_customer)!
 *
 * Explicitly skips multi-join un-aggregated build subtrees (preventing large join rescans like Q50).
 * Controlled independently by `spark.rapids.sql.lenc.runtimeBloomFilterEnabled` (default false).
 */
class GpuInjectRuntimeBloomFilter(spark: SparkSession)
    extends Rule[LogicalPlan] with PredicateHelper {

  import GpuInjectRuntimeBloomFilter._

  override def apply(plan: LogicalPlan): LogicalPlan = {
    val conf = SQLConf.get
    if (!conf.getConfString(CONF_ENABLED, "false").toBoolean) {
      return plan
    }
    if (!conf.getConfString("spark.rapids.sql.enabled", "true").toBoolean ||
        !conf.getConfString("spark.rapids.sql.expression.BloomFilterMightContain", "true").toBoolean ||
        !conf.getConfString("spark.rapids.sql.expression.BloomFilterAggregate", "true").toBoolean) {
      return plan
    }
    val minProbeBytes = conf.getConfString(CONF_MIN_PROBE_BYTES, DEFAULT_MIN_PROBE_BYTES.toString).toLong
    val maxSingleBuildBytes =
      conf.getConfString(CONF_MAX_BUILD_BYTES, DEFAULT_MAX_BUILD_BYTES.toString).toLong
    val maxInjected = conf.getConfString(CONF_MAX_INJECTED, "8").toInt

    var injected = 0

    // Track single-table selective Bloom Filters built on inner joins so outer equi-joins on the
    // same transitive key (e.g. TPC-H Q09: p_partkey = l_partkey AND ps_partkey = l_partkey) can
    // ALSO prune the outer table (partsupp!) using the exact same Bloom Filter!
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

      // First: transitive propagation from previously discovered selective build filters
      // (e.g. TPC-H Q09: lineitem already got part's p_partkey BF -> propagate onto partsupp.ps_partkey!)
      for ((lKey, rKey) <- eqPairs if injected < maxInjected) {
        if (curJoin.joinType == Inner) {
          val propCandidates = Seq(
            (lKey, curJoin.right, rKey, false),
            (rKey, curJoin.left, lKey, true)
          )
          for ((srcKey, targetSide, targetKey, targetIsLeft) <- propCandidates if injected < maxInjected) {
            attrToBuildSpec.get(srcKey.exprId.id).foreach { case (bPlan, bKey) =>
              val tBytes = totalLeafScanBytes(targetSide)
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

      // Second: direct injection for this Join (Rule 1: single-table selective filter; Rule 2: HAVING agg)
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
        // Skip tiny <10 MiB broadcast dims (date_dim/region/nation/store) so Spark DPP & BroadcastHashJoin
        // stay undisturbed (prevents Q50 29->52 GiB rescan), while targeting non-broadcastable 10 MiB..24 GiB
        // selective builds (part, customer, item, etc.) and selective HAVING aggregates!
        val minSingleBuildBytes = if (minProbeBytes == 0L) 0L else 10L * 1024L * 1024L
        val bIsSingleTableSelective =
          !bHasJoin && hasSelectiveFilter(buildSide) &&
            ((bScanBytes >= minSingleBuildBytes && bScanBytes <= maxSingleBuildBytes &&
              bScanBytes <= (pScanBytes * 0.5).toLong) || minProbeBytes == 0L)
        val bIsHavingAgg = isSelectiveHavingAggregate(buildSide)

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

    // Estimate build keys conservatively: 8 bits/key capped at 64M bits (8 MiB, L2-resident on GPU!)
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

  /** Push BloomFilterMightContain directly down to the leaf scan producing `probeKey` (before Shuffle Write!). */
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

  // Default: probe leaf >= 256 MiB, single-table selective build <= 24 GiB (covers SF1000 part/customer!)
  val DEFAULT_MIN_PROBE_BYTES: Long = 256L * 1024L * 1024L
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

  /** Detects selective HAVING patterns: Filter(non-trivial) above Aggregate (e.g. TPC-H Q18, TPC-DS Q23). */
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
