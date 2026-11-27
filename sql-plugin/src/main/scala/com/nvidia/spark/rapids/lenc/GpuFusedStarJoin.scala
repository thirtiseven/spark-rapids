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

import scala.collection.mutable.ArrayBuffer

import ai.rapids.cudf.{ColumnVector, Scalar, Table => CudfTable}
import com.nvidia.spark.rapids._
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.GpuMetric._
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.jni.LencNative

import org.apache.spark.TaskContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Alias, Attribute, ExprId, Expression, Literal, NamedExpression}
import org.apache.spark.sql.catalyst.expressions.aggregate.Partial
import org.apache.spark.sql.catalyst.plans.Inner
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.adaptive.BroadcastQueryStageExec
import org.apache.spark.sql.execution.exchange.ReusedExchangeExec
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids.GpuAnd
import org.apache.spark.sql.rapids.aggregate.GpuCount
import org.apache.spark.sql.rapids.execution.{GpuBroadcastExchangeExec, GpuBroadcastHashJoinExecBase, GpuBroadcastHelper, GpuDriverlessBroadcastHashJoinExec}
import org.apache.spark.sql.types._
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * Path C: Fused Multi-Way Star Broadcast Equi-Join (L2 LUT / Open-Addressing Hash Probe).
 *
 * Fuses K >= 2 chained Inner GpuBroadcastHashJoin(BuildRight, single INT32/INT64/DATE key,
 * no non-equi condition) + intervening pass-through GpuProject + bottom pure-IsNotNull GpuFilter
 * into ONE single GPU multi-probe pass:
 *   - Mode C1 (GpuFusedStarJoinCountExec): when directly feeding global partial COUNT(*),
 *     eliminates 100% of join materialization, gather maps, and GpuHashAggregate (e.g. Q88/Q96/Q90).
 *   - Mode C2 (GpuFusedStarJoinExec): probes all K broadcast dimensions in L2 cache (sorted most
 *     selective first) with 1-pass warp-aggregated compaction and 1 final gather on surviving rows
 *     (e.g. Q99/Q62/Q50/Q70/Q47/Q57/Q66/Q03/Q07/Q19/Q26/Q27/Q43/Q59).
 */
object GpuFusedStarJoinRewrite {
  private val MAX_DIMS = 6
  private val MAX_EXTRA_NN = 8

  private def isSupportedKeyType(dt: DataType): Boolean = dt match {
    case IntegerType | LongType | DateType => true
    case _ => false
  }

  private def extractPassThroughMap(projList: Seq[NamedExpression]): Option[Map[ExprId, Attribute]] = {
    val m = scala.collection.mutable.HashMap.empty[ExprId, Attribute]
    projList.foreach {
      case a: Attribute => m.put(a.exprId, a)
      case Alias(a: Attribute, _) => m.put(a.exprId, a)
      case GpuAlias(a: Attribute, _) => m.put(a.exprId, a)
      case _ => return None
    }
    Some(m.toMap)
  }

  private def extractPureIsNotNullAttrs(cond: Expression): Option[Seq[Attribute]] = {
    val out = new ArrayBuffer[Attribute]()
    def collect(e: Expression): Boolean = e match {
      case GpuAnd(l, r) => collect(l) && collect(r)
      case GpuIsNotNull(a: Attribute) =>
        out += a
        true
      case _ => false
    }
    if (collect(cond)) Some(out.toSeq) else None
  }

  private[lenc] def resolveBroadcastExchange(plan: SparkPlan): Option[GpuBroadcastExchangeExec] = plan match {
    case b: GpuBroadcastExchangeExec => Some(b)
    case bqs: BroadcastQueryStageExec => resolveBroadcastExchange(bqs.plan)
    case re: ReusedExchangeExec => resolveBroadcastExchange(re.child)
    case _ => None
  }

  private case class StarChain(
      factPlan: SparkPlan,
      dimPlans: Seq[SparkPlan],
      probeKeyOrds: Array[Int],
      dimKeyOrds: Array[Int],
      extraNotNullOrds: Array[Int],
      attrToSource: Map[ExprId, (Int, Int)]) // ExprId -> (srcTable: -1=fact or 0..K-1, colOrd)

  /**
   * Unwinds a chain of K >= 2 eligible Inner BuildRight single-key GpuBroadcastHashJoins
   * (with optional attribute-only GpuProjects between them and pure-IsNotNull GpuFilter below).
   */
  private def extractStarChain(root: SparkPlan): Option[StarChain] = {
    val joins = new ArrayBuffer[GpuBroadcastHashJoinExecBase]()
    val aliasMaps = new ArrayBuffer[Map[ExprId, Attribute]]()
    var cur = root

    // Collect joins from top to bottom
    var keepGoing = true
    while (keepGoing && joins.length < MAX_DIMS) {
      cur match {
        case p: GpuProjectExec =>
          extractPassThroughMap(p.projectList) match {
            case Some(am) =>
              aliasMaps += am
              cur = p.child
            case None => keepGoing = false
          }
        case j: GpuBroadcastHashJoinExecBase
            if !j.isInstanceOf[GpuDriverlessBroadcastHashJoinExec] &&
              j.joinType == Inner &&
              j.buildSide == GpuBuildRight &&
              j.condition.isEmpty &&
              j.leftKeys.length == 1 &&
              j.rightKeys.length == 1 &&
              resolveBroadcastExchange(j.right).isDefined =>
          (j.leftKeys.head, j.rightKeys.head) match {
            case (lk: Attribute, rk: Attribute)
                if isSupportedKeyType(lk.dataType) && isSupportedKeyType(rk.dataType) =>
              joins += j
              cur = j.left
            case _ => keepGoing = false
          }
        case _ => keepGoing = false
      }
    }

    if (joins.length < 2) return None

    // Optionally absorb a pure-IsNotNull GpuFilterExec + pass-through GpuProject below bottom join
    // (preserving any GpuCoalesceBatches around the scan so downstream batches stay coalesced!).
    val extraNnAttrs = new ArrayBuffer[Attribute]()
    var wrapCoalesce: Option[GpuCoalesceBatches] = None
    var absorbing = true
    while (absorbing) {
      cur match {
        case p: GpuProjectExec =>
          extractPassThroughMap(p.projectList) match {
            case Some(am) =>
              aliasMaps += am
              cur = p.child
            case None => absorbing = false
          }
        case c: GpuCoalesceBatches if wrapCoalesce.isEmpty =>
          c.child match {
            case f: GpuFilterExec if extractPureIsNotNullAttrs(f.condition).isDefined =>
              wrapCoalesce = Some(c)
              extraNnAttrs ++= extractPureIsNotNullAttrs(f.condition).get
              cur = f.child
            case _ => absorbing = false
          }
        case f: GpuFilterExec =>
          extractPureIsNotNullAttrs(f.condition) match {
            case Some(attrs) =>
              extraNnAttrs ++= attrs
              cur = f.child
            case None => absorbing = false
          }
        case _ => absorbing = false
      }
    }

    val factPlan = wrapCoalesce match {
      case Some(c) => c.copy(child = cur)
      case None => cur
    }
    val factOrdById = factPlan.output.zipWithIndex.map { case (a, i) => a.exprId -> i }.toMap

    // Resolve any ExprId alias chain top-down to its original underlying ExprId
    def resolveOrigin(id: ExprId): ExprId = {
      var x = id
      var i = 0
      while (i < aliasMaps.length) {
        aliasMaps(i).get(x).foreach(a => x = a.exprId)
        i += 1
      }
      x
    }

    // Joins were collected top-down; reverse to bottom-up dim index 0..K-1
    val orderedJoins = joins.reverse.toArray
    val k = orderedJoins.length
    val dimPlans = new Array[SparkPlan](k)
    val probeKeyOrds = new Array[Int](k)
    val dimKeyOrds = new Array[Int](k)
    val attrToSrc = scala.collection.mutable.HashMap.empty[ExprId, (Int, Int)]

    factPlan.output.zipWithIndex.foreach { case (a, i) =>
      attrToSrc.put(a.exprId, (-1, i))
    }

    var d = 0
    while (d < k) {
      val j = orderedJoins(d)
      val lk = j.leftKeys.head.asInstanceOf[Attribute]
      val rk = j.rightKeys.head.asInstanceOf[Attribute]
      val lkOrig = resolveOrigin(lk.exprId)
      val fOrd = factOrdById.getOrElse(lkOrig, return None)
      val dOrdById = j.right.output.zipWithIndex.map { case (a, i) => a.exprId -> i }.toMap
      val dOrd = dOrdById.getOrElse(rk.exprId, return None)
      dimPlans(d) = j.right
      probeKeyOrds(d) = fOrd
      dimKeyOrds(d) = dOrd
      j.right.output.zipWithIndex.foreach { case (a, i) =>
        attrToSrc.put(a.exprId, (d, i))
      }
      d += 1
    }

    // Add alias mappings back so any top-level output ExprId resolves cleanly
    for (i <- aliasMaps.indices.reverse) {
      aliasMaps(i).foreach { case (aliasId, origAttr) =>
        attrToSrc.get(origAttr.exprId).foreach(src => attrToSrc.put(aliasId, src))
      }
    }

    // Only keep extra IsNotNull checks that are NOT already covered by join probe keys
    val probeKeyOrdSet = probeKeyOrds.toSet
    val extraOrds = extraNnAttrs.flatMap { a =>
      val orig = resolveOrigin(a.exprId)
      factOrdById.get(orig)
    }.distinct.filterNot(probeKeyOrdSet.contains).toArray
    if (extraOrds.length > MAX_EXTRA_NN) return None

    Some(StarChain(factPlan, dimPlans.toSeq, probeKeyOrds, dimKeyOrds, extraOrds, attrToSrc.toMap))
  }

  private def isGlobalPartialCountOne(agg: GpuHashAggregateExec): Boolean = {
    if (agg.groupingExpressions.nonEmpty || agg.aggregateExpressions.length != 1) return false
    if (agg.resultExpressions.length != 1) return false
    val gae = agg.aggregateExpressions.head
    if (gae.mode != Partial || gae.isDistinct || gae.filter.isDefined) return false
    gae.aggregateFunction match {
      case c: GpuCount =>
        c.children.forall {
          case lit: GpuLiteral => lit.value != null
          case lit: Literal => lit.value != null
          case _ => false
        }
      case _ => false
    }
  }

  def apply(plan: SparkPlan): SparkPlan = {
    val enabled = SQLConf.get
      .getConfString("spark.rapids.sql.joinAstJitEnabled", "false").toBoolean
    if (!enabled) return plan

    // Top-down rewrite so maximal chains (e.g. Agg(count) -> 3x Join, or 4x Join) fuse completely
    // in one operator instead of stopping at dims=2 from bottom-up matching!
    plan.transformDown {
      // Mode C1: Global Partial COUNT(*) over K-way Star Broadcast Join (Q88 x8, Q96, Q90!)
      case agg: GpuHashAggregateExec if isGlobalPartialCountOne(agg) =>
        val sub = agg.child match {
          case p: GpuProjectExec if p.projectList.isEmpty => p.child
          case other => other
        }
        extractStarChain(sub) match {
          case Some(ch) =>
            GpuFusedStarJoinCountExec(
              agg.output, ch.factPlan, ch.dimPlans, ch.probeKeyOrds.toVector,
              ch.dimKeyOrds.toVector, ch.extraNotNullOrds.toVector)
          case None => agg
        }

      // Mode C2: General K-way Star Broadcast Join with optional top pass-through GpuProject!
      case p: GpuProjectExec if extractPassThroughMap(p.projectList).isDefined &&
          p.child.isInstanceOf[GpuBroadcastHashJoinExecBase] =>
        extractStarChain(p) match {
          case Some(ch) =>
            buildMaterializeExec(p.output, ch).getOrElse(p)
          case None => p
        }

      case j: GpuBroadcastHashJoinExecBase =>
        extractStarChain(j) match {
          case Some(ch) =>
            buildMaterializeExec(j.output, ch).getOrElse(j)
          case None => j
        }
    }
  }

  private def buildMaterializeExec(
      outAttrs: Seq[Attribute],
      ch: StarChain): Option[GpuFusedStarJoinExec] = {
    if (outAttrs.isEmpty) return None
    val srcTbl = new Array[Int](outAttrs.length)
    val srcCol = new Array[Int](outAttrs.length)
    var i = 0
    while (i < outAttrs.length) {
      ch.attrToSource.get(outAttrs(i).exprId) match {
        case Some((t, c)) =>
          srcTbl(i) = t
          srcCol(i) = c
        case None => return None
      }
      i += 1
    }
    Some(GpuFusedStarJoinExec(
      outAttrs, ch.factPlan, ch.dimPlans, ch.probeKeyOrds.toVector, ch.dimKeyOrds.toVector,
      ch.extraNotNullOrds.toVector, srcTbl.toVector, srcCol.toVector))
  }
}

case class GpuFusedStarJoinCountExec(
    output: Seq[Attribute],
    factPlan: SparkPlan,
    dimPlans: Seq[SparkPlan],
    probeKeyOrds: Seq[Int],
    dimKeyOrds: Seq[Int],
    extraNotNullOrds: Seq[Int]) extends SparkPlan with GpuExec {

  override def children: Seq[SparkPlan] = factPlan +: dimPlans

  override protected def withNewChildrenInternal(newChildren: IndexedSeq[SparkPlan]): SparkPlan =
    copy(factPlan = newChildren.head, dimPlans = newChildren.tail)

  override def supportsColumnar: Boolean = true

  override protected def doExecute(): RDD[InternalRow] =
    throw new IllegalStateException("GpuFusedStarJoinCountExec executes columnar on GPU only")

  override lazy val additionalMetrics: Map[String, GpuMetric] = Map(
    OP_TIME_LEGACY -> createNanoTimingMetric(MODERATE_LEVEL, DESCRIPTION_OP_TIME_LEGACY),
    JOIN_TIME -> createNanoTimingMetric(DEBUG_LEVEL, DESCRIPTION_JOIN_TIME)
  )

  override def internalDoExecuteColumnar(): RDD[ColumnarBatch] = {
    val opTime = allMetrics.getOrElse(OP_TIME_LEGACY, NoopMetric)
    val joinTime = allMetrics.getOrElse(JOIN_TIME, NoopMetric)
    val numOutputRows = allMetrics.getOrElse(NUM_OUTPUT_ROWS, NoopMetric)
    val numOutputBatches = allMetrics.getOrElse(NUM_OUTPUT_BATCHES, NoopMetric)

    val bcasts = dimPlans.map { dp =>
      GpuFusedStarJoinRewrite.resolveBroadcastExchange(dp)
        .getOrElse(throw new IllegalStateException(s"Expected GpuBroadcastExchangeExec in $dp"))
        .executeColumnarBroadcast[Any]()
    }.toArray
    val dimSchemas = dimPlans.map(_.schema).toArray
    val pOrds = probeKeyOrds.toArray
    val dOrds = dimKeyOrds.toArray
    val nnOrds = extraNotNullOrds.toArray

    factPlan.executeColumnar().mapPartitions { iter =>
      val buffered = new CloseableBufferedIterator(iter)
      var totalCount = 0L
      val k = bcasts.length
      val dimBatches = new Array[ColumnarBatch](k)
      val buildTables = new Array[LencNative.StarJoinBuildTable](k)
      try {
        // Trigger first fact batch BEFORE acquiring GpuSemaphore so off-semaphore I/O runs freely!
        if (buffered.hasNext) {
          buffered.head
          GpuSemaphore.acquireIfNecessary(TaskContext.get())
          opTime.ns {
            var anyEmpty = false
            var d = 0
            while (d < k && !anyEmpty) {
              val db = GpuBroadcastHelper.getBroadcastBatch(bcasts(d), dimSchemas(d))
              dimBatches(d) = db
              if (db.numRows() == 0) {
                anyEmpty = true
              } else {
                val keyCol = db.column(dOrds(d)).asInstanceOf[GpuColumnVector].getBase
                buildTables(d) = LencNative.buildStarJoinTable(keyCol)
              }
              d += 1
            }

            while (buffered.hasNext) {
              withResource(buffered.next()) { fb =>
                if (!anyEmpty && fb.numRows() > 0) {
                  val fCols = (0 until fb.numCols()).map(c =>
                    fb.column(c).asInstanceOf[GpuColumnVector].getBase)
                  withResource(new CudfTable(fCols: _*)) { fTbl =>
                    val cnt = joinTime.ns {
                      LencNative.runStarJoinCount(fTbl, pOrds, buildTables, nnOrds)
                    }
                    totalCount += cnt
                  }
                }
              }
            }
          }
        }
      } finally {
        buffered.close()
        buildTables.foreach(bt => if (bt != null) bt.safeClose())
        dimBatches.foreach(db => if (db != null) db.safeClose())
      }

      GpuSemaphore.acquireIfNecessary(TaskContext.get())
      val outBatch = withResource(Scalar.fromLong(totalCount)) { s =>
        val cv = ColumnVector.fromScalar(s, 1)
        new ColumnarBatch(
          Array[org.apache.spark.sql.vectorized.ColumnVector](GpuColumnVector.from(cv, LongType)),
          1)
      }
      numOutputRows += 1
      numOutputBatches += 1
      Iterator.single(outBatch)
    }
  }

  override def simpleString(maxFields: Int): String =
    s"GpuFusedStarJoinCount(dims=${dimPlans.length})"
}

case class GpuFusedStarJoinExec(
    output: Seq[Attribute],
    factPlan: SparkPlan,
    dimPlans: Seq[SparkPlan],
    probeKeyOrds: Seq[Int],
    dimKeyOrds: Seq[Int],
    extraNotNullOrds: Seq[Int],
    outSrcTable: Seq[Int],
    outSrcCol: Seq[Int]) extends SparkPlan with GpuExec {

  override def children: Seq[SparkPlan] = factPlan +: dimPlans

  override protected def withNewChildrenInternal(newChildren: IndexedSeq[SparkPlan]): SparkPlan =
    copy(factPlan = newChildren.head, dimPlans = newChildren.tail)

  override def supportsColumnar: Boolean = true

  override protected def doExecute(): RDD[InternalRow] =
    throw new IllegalStateException("GpuFusedStarJoinExec executes columnar on GPU only")

  override lazy val additionalMetrics: Map[String, GpuMetric] = Map(
    OP_TIME_LEGACY -> createNanoTimingMetric(MODERATE_LEVEL, DESCRIPTION_OP_TIME_LEGACY),
    JOIN_TIME -> createNanoTimingMetric(DEBUG_LEVEL, DESCRIPTION_JOIN_TIME)
  )

  override def internalDoExecuteColumnar(): RDD[ColumnarBatch] = {
    val opTime = allMetrics.getOrElse(OP_TIME_LEGACY, NoopMetric)
    val joinTime = allMetrics.getOrElse(JOIN_TIME, NoopMetric)
    val numOutputRows = allMetrics.getOrElse(NUM_OUTPUT_ROWS, NoopMetric)
    val numOutputBatches = allMetrics.getOrElse(NUM_OUTPUT_BATCHES, NoopMetric)

    val bcasts = dimPlans.map { dp =>
      GpuFusedStarJoinRewrite.resolveBroadcastExchange(dp)
        .getOrElse(throw new IllegalStateException(s"Expected GpuBroadcastExchangeExec in $dp"))
        .executeColumnarBroadcast[Any]()
    }.toArray
    val dimSchemas = dimPlans.map(_.schema).toArray
    val pOrds = probeKeyOrds.toArray
    val dOrds = dimKeyOrds.toArray
    val nnOrds = extraNotNullOrds.toArray
    val ost = outSrcTable.toArray
    val osc = outSrcCol.toArray
    val outTypes = output.map(_.dataType).toArray

    factPlan.executeColumnar().mapPartitions { iter =>
      new Iterator[ColumnarBatch] {
        private val streamBuf = new CloseableBufferedIterator(iter)
        private val k = bcasts.length
        private val dimBatches = new Array[ColumnarBatch](k)
        private val dimTables = new Array[CudfTable](k)
        private val buildTables = new Array[LencNative.StarJoinBuildTable](k)
        private var initialized = false
        private var anyEmpty = false
        private var curFactBatch: ColumnarBatch = _
        private var curFactTable: CudfTable = _
        private var curRowOffset: Int = 0
        private var curChunkRows: Int = Int.MaxValue
        private var maxDimStrBytesPerRow: Long = 0L
        private var nextOut: ColumnarBatch = _
        private var closed = false

        TaskContext.get().addTaskCompletionListener[Unit](_ => cleanup())

        private def releaseCurFact(): Unit = {
          if (curFactTable != null) { curFactTable.safeClose(); curFactTable = null }
          if (curFactBatch != null) { curFactBatch.safeClose(); curFactBatch = null }
          curRowOffset = 0
        }

        private def cleanup(): Unit = {
          if (!closed) {
            closed = true
            if (nextOut != null) { nextOut.safeClose(); nextOut = null }
            releaseCurFact()
            streamBuf.close()
            buildTables.foreach(bt => if (bt != null) bt.safeClose())
            dimTables.foreach(dt => if (dt != null) dt.safeClose())
            dimBatches.foreach(db => if (db != null) db.safeClose())
          }
        }

        private def initBuildTables(): Unit = {
          if (initialized) return
          initialized = true
          GpuSemaphore.acquireIfNecessary(TaskContext.get())
          var d = 0
          while (d < k && !anyEmpty) {
            val db = GpuBroadcastHelper.getBroadcastBatch(bcasts(d), dimSchemas(d))
            dimBatches(d) = db
            if (db.numRows() == 0) {
              anyEmpty = true
            } else {
              val keyCol = db.column(dOrds(d)).asInstanceOf[GpuColumnVector].getBase
              buildTables(d) = LencNative.buildStarJoinTable(keyCol)
              val neededForDim = ost.contains(d)
              if (neededForDim) {
                val dCols = (0 until db.numCols()).map(c =>
                  db.column(c).asInstanceOf[GpuColumnVector].getBase)
                dimTables(d) = new CudfTable(dCols: _*)
                // Estimate max string column bytes/row so chunked gather never hits 2 GiB overflow!
                var o = 0
                while (o < ost.length) {
                  if (ost(o) == d && outTypes(o) == StringType) {
                    val cv = dCols(osc(o))
                    val avgB = (cv.getDeviceMemorySize / math.max(db.numRows().toLong, 1L)) + 16L
                    maxDimStrBytesPerRow = math.max(maxDimStrBytesPerRow, avgB)
                  }
                  o += 1
                }
              }
            }
            d += 1
          }
        }

        private def computeChunkRows(fb: ColumnarBatch): Int = {
          val n = fb.numRows()
          if (n <= 0) return 1
          var maxStrB = maxDimStrBytesPerRow
          var o = 0
          while (o < ost.length) {
            if (ost(o) < 0 && outTypes(o) == StringType) {
              val cv = fb.column(osc(o)).asInstanceOf[GpuColumnVector].getBase
              val avgB = (cv.getDeviceMemorySize / math.max(n.toLong, 1L)) + 16L
              maxStrB = math.max(maxStrB, avgB)
            }
            o += 1
          }
          if (maxStrB <= 0L) {
            n
          } else {
            // Keep each gathered string column safely under 512 MiB (4x safety margin below 2 GiB!)
            val cap = ((512L * 1024L * 1024L) / math.max(maxStrB * 2L, 32L)).toInt
            math.max(65536, math.min(n, cap))
          }
        }

        override def hasNext: Boolean = {
          if (nextOut != null) return true
          if (closed) return false
          while (curFactBatch != null || streamBuf.hasNext) {
            if (curFactBatch == null) {
              // Peek first stream batch BEFORE acquiring GpuSemaphore on init!
              if (!initialized) {
                streamBuf.head
                initBuildTables()
              }
              if (anyEmpty) {
                withResource(streamBuf.next()) { _ => }
              } else {
                val fb = streamBuf.next()
                if (fb.numRows() <= 0) {
                  fb.safeClose()
                } else {
                  curFactBatch = fb
                  val fCols = (0 until fb.numCols()).map(c =>
                    fb.column(c).asInstanceOf[GpuColumnVector].getBase)
                  curFactTable = new CudfTable(fCols: _*)
                  curRowOffset = 0
                  curChunkRows = computeChunkRows(fb)
                }
              }
            }
            if (curFactBatch != null) {
              val totalRows = curFactBatch.numRows()
              val start = curRowOffset
              val end = math.min(totalRows, start + curChunkRows)
              curRowOffset = end
              val outOpt = opTime.ns {
                val resTbl = joinTime.ns {
                  LencNative.runStarJoinMaterialize(
                    curFactTable, start, end, pOrds, buildTables, dimTables, nnOrds, ost, osc)
                }
                withResource(resTbl) { rt =>
                  if (rt.getRowCount > 0) {
                    numOutputRows += rt.getRowCount
                    numOutputBatches += 1
                    Some(GpuColumnVector.from(rt, outTypes))
                  } else None
                }
              }
              if (curRowOffset >= totalRows) {
                releaseCurFact()
              }
              if (outOpt.isDefined) {
                nextOut = outOpt.get
                return true
              }
            }
          }
          cleanup()
          false
        }

        override def next(): ColumnarBatch = {
          if (!hasNext) throw new NoSuchElementException("GpuFusedStarJoinExec exhausted")
          val ret = nextOut
          nextOut = null
          ret
        }
      }
    }
  }

  override def simpleString(maxFields: Int): String =
    s"GpuFusedStarJoin(dims=${dimPlans.length}, cols=${output.length})"
}
