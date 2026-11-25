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

import scala.collection.mutable

import ai.rapids.cudf.{Table => CudfTable}
import com.nvidia.spark.rapids._
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.GpuMetric._
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.jni.LencNative
import com.nvidia.spark.rapids.shims.GpuBatchScanExecBase

import org.apache.spark.{Partition, SparkContext, TaskContext}
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
 * Track 4: GPU Fused Conditional Aggregation / Reduce (`Filter + Project + Partial HashAggregate`)
 * aligned with NVIDIA/cudf#25037/#25038:
 * - Pattern 4A: Fused TPC-H Q1 Partial GroupBy-Reduce directly from compressed .lenc (in-register)
 *   OR from Parquet table (eliminating 97.5% GpuFilter copy + intermediate project columns!).
 * - Pattern 4B: Fused Dictionary-Bucketed Wide `SUM(CASE WHEN metric = lit THEN val * coeff)`
 *   Partial Hash-Aggregate (e.g. Netflix Job 48 36-way CASE WHEN) on BOTH Parquet & .lenc!
 */
object GpuLencFusedAgg {
  val FUSED_AGG_CONF = "spark.rapids.sql.lenc.fusedAggEnabled"

  // Native Q1 result column indices from LencNative.fusedQ1*:
  // 0=rf, 1=ls, 2=sum_qty, 3=sum_base, 4=sum_disc_price, 5=sum_charge, 6=sum_disc, 7=count
  private def extractShipdateLeDays(expr: Expression): Option[Int] = expr match {
    case GpuLessThanOrEqual(a: Attribute, lit: GpuLiteral)
        if a.name == "l_shipdate" && lit.dataType == DateType && lit.value != null =>
      Some(lit.value.asInstanceOf[Int])
    case GpuGreaterThanOrEqual(lit: GpuLiteral, a: Attribute)
        if a.name == "l_shipdate" && lit.dataType == DateType && lit.value != null =>
      Some(lit.value.asInstanceOf[Int])
    case GpuAnd(l, r) =>
      extractShipdateLeDays(l).orElse(extractShipdateLeDays(r))
    case _ => None
  }

  private def tryMatchQ1(
      gpuGrouping: Seq[NamedExpression],
      gpuAggs: Seq[GpuAggregateExpression],
      resultExprs: Seq[NamedExpression],
      gpuChild: SparkPlan): Option[GpuExec] = {
    if (gpuGrouping.length != 2 || gpuAggs.isEmpty || !gpuAggs.forall(_.mode == Partial)) {
      return None
    }
    val gNames = gpuGrouping.map(_.name)
    if (gNames != Seq("l_returnflag", "l_linestatus")) {
      return None
    }

    // Match upstream Project(Filter(Scan)) or Project(LencLtoFilterScan)
    val filterPlan = gpuChild match {
      case p: GpuProjectExec => p.child
      case other => other
    }

    // Extract maxShipdateDays & underlying scan (either GpuLencScan or generic Parquet scan!)
    val (maxDaysOpt, lencScanOpt, rawTablePlanOpt): (Option[Int], Option[GpuLencScan], Option[SparkPlan]) =
      filterPlan match {
        case f: GpuFilterExec =>
          val dOpt = extractShipdateLeDays(f.condition)
          f.child match {
            case bs: GpuBatchScanExecBase =>
              bs.scan match {
                case ls: GpuLencScan if ls.fusedExpr.isEmpty => (dOpt, Some(ls), None)
                case _ => (dOpt, None, Some(f.child))
              }
            case other => (dOpt, None, Some(other))
          }
        case flto: GpuLencLtoFilterScanExec =>
          val dOpt = flto.filterSpec.compactExprs.headOption.flatMap {
            case GpuLessThanOrEqual(_: GpuBoundReference, lit: GpuLiteral)
                if lit.dataType == DateType && lit.value != null =>
              Some(lit.value.asInstanceOf[Int])
            case _ => None
          }
          val ls = GpuLencScan(flto.fullSchema, flto.fullSchema, flto.files, None)
          (dOpt, Some(ls), None)
        case _ => (None, None, None)
      }

    val maxDays = maxDaysOpt.getOrElse(return None)

    // Verify all 7 required Q1 columns exist & match aggregate outputs to native Q1 bucket slots
    val q1Cols = Array(
      "l_shipdate", "l_returnflag", "l_linestatus",
      "l_quantity", "l_extendedprice", "l_discount", "l_tax")

    // Map each Spark partial-agg buffer output column -> native Q1 slot (0..7)
    // In Spark Partial agg, resultExprs = [group0, group1, agg0_buf0, ..., aggN_bufM]
    val outNativeSlots = mutable.ArrayBuffer[Int](0, 1) // l_returnflag, l_linestatus
    val outTypes = resultExprs.map(_.dataType).toArray

    def resolveInputColName(e: Expression): String = e match {
      case a: Attribute => a.name
      case c: GpuCast => resolveInputColName(c.child)
      case GpuAlias(c, _) => resolveInputColName(c)
      case _ => ""
    }

    var ok = true
    gpuAggs.foreach { gae =>
      gae.aggregateFunction match {
        case s: GpuSum =>
          val nm = resolveInputColName(s.children.head)
          nm match {
            case "l_quantity" => outNativeSlots += 2
            case "l_extendedprice" => outNativeSlots += 3
            case "disc_price" => outNativeSlots += 4
            case "charge" => outNativeSlots += 5
            case "l_discount" => outNativeSlots += 6
            case _ => ok = false
          }
        case a: GpuAverage =>
          val nm = resolveInputColName(a.children.head)
          nm match {
            case "l_quantity" => outNativeSlots += 2; outNativeSlots += 7
            case "l_extendedprice" => outNativeSlots += 3; outNativeSlots += 7
            case "l_discount" => outNativeSlots += 6; outNativeSlots += 7
            case _ => ok = false
          }
        case _: GpuCount =>
          outNativeSlots += 7
        case _ =>
          ok = false
      }
    }

    if (!ok || outNativeSlots.length != resultExprs.length) {
      return None
    }

    val outAttrs = resultExprs.map(_.toAttribute)

    lencScanOpt match {
      case Some(ls) =>
        val nameToIdx = ls.fullSchema.fieldNames.zipWithIndex.toMap
        if (!q1Cols.forall(nameToIdx.contains)) return None
        val fileColIndices = q1Cols.map(nameToIdx)
        Some(GpuFusedQ1LencExec(
          outAttrs, ls.files, fileColIndices, maxDays, outNativeSlots.toArray, outTypes))
      case None =>
        rawTablePlanOpt.flatMap { scanPlan =>
          val nameToOrd = scanPlan.output.map(_.name).zipWithIndex.toMap
          if (!q1Cols.forall(nameToOrd.contains)) None
          else {
            val childColOrds = q1Cols.map(nameToOrd)
            Some(GpuFusedQ1TableExec(
              outAttrs, scanPlan, childColOrds, maxDays, outNativeSlots.toArray, outTypes))
          }
        }
    }
  }

  /** Match `SUM(CASE WHEN metric_col = 'lit' THEN val * coeff ELSE NULL END)` pattern. */
  private def extractWideCaseTerm(
      expr: Expression,
      projMap: Map[ExprId, Expression]): Option[(Attribute, String, Attribute, Double)] = {
    val resolved = expr match {
      case a: Attribute => projMap.getOrElse(a.exprId, a)
      case c: GpuCast =>
        c.child match {
          case a: Attribute => projMap.getOrElse(a.exprId, a)
          case other => other
        }
      case other => other
    }

    def unwrapAttr(e: Expression): Option[Attribute] = e match {
      case a: Attribute => Some(a)
      case c: GpuCast => unwrapAttr(c.child)
      case _ => None
    }

    def parseValCoeff(v: Expression): Option[(Attribute, Double)] = v match {
      case a: Attribute => Some((a, 1.0))
      case c: GpuCast => parseValCoeff(c.child)
      case m: GpuMultiply =>
        (unwrapAttr(m.left), m.right) match {
          case (Some(a), lit: GpuLiteral) if lit.value != null =>
            Some((a, lit.value.asInstanceOf[Number].doubleValue()))
          case _ =>
            (m.left, unwrapAttr(m.right)) match {
              case (lit: GpuLiteral, Some(a)) if lit.value != null =>
                Some((a, lit.value.asInstanceOf[Number].doubleValue()))
              case _ => None
            }
        }
      case _ => None
    }

    def parseEqLit(cond: Expression): Option[(Attribute, String)] = cond match {
      case GpuEqualTo(a: Attribute, lit: GpuLiteral)
          if a.dataType == StringType && lit.value != null =>
        Some((a, lit.value.toString))
      case GpuEqualTo(lit: GpuLiteral, a: Attribute)
          if a.dataType == StringType && lit.value != null =>
        Some((a, lit.value.toString))
      case _ => None
    }

    resolved match {
      case cw: GpuCaseWhen
          if cw.branches.length == 1 && cw.elseValue.forall {
            case lit: GpuLiteral if lit.value == null => true
            case _ => false
          } =>
        val (cond, thenExpr) = cw.branches.head
        for {
          (mAttr, mLit) <- parseEqLit(cond)
          (vAttr, coeff) <- parseValCoeff(thenExpr)
        } yield (mAttr, mLit, vAttr, coeff)
      case GpuIf(cond, thenExpr, lit: GpuLiteral) if lit.value == null =>
        for {
          (mAttr, mLit) <- parseEqLit(cond)
          (vAttr, coeff) <- parseValCoeff(thenExpr)
        } yield (mAttr, mLit, vAttr, coeff)
      case _ => None
    }
  }

  private def tryMatchWideDictAgg(
      gpuGrouping: Seq[NamedExpression],
      gpuAggs: Seq[GpuAggregateExpression],
      resultExprs: Seq[NamedExpression],
      gpuChild: SparkPlan): Option[GpuExec] = {
    if (gpuGrouping.isEmpty || gpuAggs.length < 8 || gpuAggs.length > 64 ||
        !gpuAggs.forall(_.mode == Partial)) {
      return None
    }
    if (resultExprs.length != gpuGrouping.length + gpuAggs.length) {
      return None
    }

    val (baseChild, projMap) = gpuChild match {
      case p: GpuProjectExec =>
        val m = p.projectList.collect {
          case a: GpuAlias => a.exprId -> a.child
        }.toMap
        (p.child, m)
      case other =>
        (other, Map.empty[ExprId, Expression])
    }

    val terms = mutable.ArrayBuffer.empty[(Attribute, String, Attribute, Double)]
    gpuAggs.foreach { gae =>
      gae.aggregateFunction match {
        case s: GpuSum if s.dataType == DoubleType =>
          extractWideCaseTerm(s.children.head, projMap) match {
            case Some(t) => terms += t
            case None => return None
          }
        case _ => return None
      }
    }

    val metricAttrs = terms.map(_._1.exprId).distinct
    val seenVals = mutable.LinkedHashMap.empty[ExprId, Attribute]
    terms.foreach { t => seenVals.getOrElseUpdate(t._3.exprId, t._3) }
    val valAttrs = seenVals.values.toSeq
    if (metricAttrs.length != 1 || valAttrs.isEmpty || valAttrs.length > 2) {
      return None
    }
    if (!valAttrs.forall(_.dataType == DoubleType)) {
      return None
    }

    val childOut = baseChild.output
    val idToOrd = childOut.zipWithIndex.map { case (a, i) => a.exprId -> i }.toMap
    val groupOrds = gpuGrouping.map { g =>
      val orig = projMap.getOrElse(g.exprId, g) match {
        case a: Attribute => a.exprId
        case _ => g.exprId
      }
      idToOrd.getOrElse(orig, return None)
    }.toArray
    val metricOrd = idToOrd.getOrElse(metricAttrs.head, return None)
    val numerOrd = idToOrd.getOrElse(valAttrs.head.exprId, return None)
    val denomOrd = if (valAttrs.length == 2) idToOrd.getOrElse(valAttrs(1).exprId, return None) else numerOrd
    val denomExprId = if (valAttrs.length == 2) valAttrs(1).exprId else null

    val sortedLits = terms.map(_._2).distinct.sorted.toArray
    val litToId = sortedLits.zipWithIndex.toMap
    val outMetricIds = terms.map(t => litToId(t._2)).toArray
    val outCoeffs = terms.map(_._4).toArray
    val outUseDenom = terms.map(t => if (t._3.exprId == denomExprId) 1.toByte else 0.toByte).toArray

    Some(GpuFusedWideDictAggExec(
      resultExprs.map(_.toAttribute),
      baseChild,
      groupOrds,
      metricOrd,
      numerOrd,
      denomOrd,
      sortedLits,
      outMetricIds,
      outCoeffs,
      outUseDenom))
  }

  def tryFuse(
      gpuGrouping: Seq[NamedExpression],
      gpuAggs: Seq[GpuAggregateExpression],
      resultExprs: Seq[NamedExpression],
      gpuChild: SparkPlan): Option[GpuExec] = {
    val enabled = SQLConf.get.getConfString(FUSED_AGG_CONF, "false").toBoolean
    if (!enabled) {
      return None
    }
    tryMatchQ1(gpuGrouping, gpuAggs, resultExprs, gpuChild)
      .orElse(tryMatchWideDictAgg(gpuGrouping, gpuAggs, resultExprs, gpuChild))
  }
}

private object Q1BatchBuilder {
  def projectSlots(
      rawTbl: CudfTable,
      outNativeSlots: Array[Int],
      outTypes: Array[DataType]): ColumnarBatch = {
    val cols = outNativeSlots.zipWithIndex.safeMap { case (slot, i) =>
      val cv = rawTbl.getColumn(slot)
      GpuColumnVector.from(cv.incRefCount(), outTypes(i))
    }
    new ColumnarBatch(cols.map(_.asInstanceOf[org.apache.spark.sql.vectorized.ColumnVector]),
      rawTbl.getRowCount.toInt)
  }
}

private case class FusedQ1Partition(index: Int, filePaths: Array[String]) extends Partition

private class GpuFusedQ1LencRdd(
    sc: SparkContext,
    fileGroups: Array[Array[String]],
    fileColIndices: Array[Int],
    maxShipdateDays: Int,
    outNativeSlots: Array[Int],
    outTypes: Array[DataType],
    metrics: Map[String, GpuMetric]) extends RDD[ColumnarBatch](sc, Nil) {

  override protected def getPartitions: Array[Partition] =
    fileGroups.zipWithIndex.map { case (fs, i) => FusedQ1Partition(i, fs) }

  override def compute(split: Partition, context: TaskContext): Iterator[ColumnarBatch] = {
    val fp = split.asInstanceOf[FusedQ1Partition]
    val opTime = metrics.getOrElse(OP_TIME_LEGACY, NoopMetric)
    val bufferTime = metrics.getOrElse(BUFFER_TIME, NoopMetric)
    val gpuDecodeTime = metrics.getOrElse(GPU_DECODE_TIME, NoopMetric)
    val numOutputRows = metrics.getOrElse(NUM_OUTPUT_ROWS, NoopMetric)
    val numOutputBatches = metrics.getOrElse(NUM_OUTPUT_BATCHES, NoopMetric)

    val rangePreds = Array(LencRangePred(0, LencRangePruner.OP_LE, false, maxShipdateDays.toLong, 0.0))
    val prefetcher = new LencAsyncPrefetcher(fp.filePaths, fileColIndices, rangePreds)
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
              withResource(item.hostBuf) { hb =>
                GpuSemaphore.acquireIfNecessary(TaskContext.get())
                val tbl = gpuDecodeTime.ns {
                  plan.fusedQ1(hb, maxShipdateDays)
                }
                withResource(tbl) { t =>
                  if (t.getRowCount > 0) {
                    numOutputRows += t.getRowCount
                    numOutputBatches += 1
                    nextBatch = Q1BatchBuilder.projectSlots(t, outNativeSlots, outTypes)
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

case class GpuFusedQ1LencExec(
    output: Seq[Attribute],
    files: Array[String],
    fileColIndices: Array[Int],
    maxShipdateDays: Int,
    outNativeSlots: Array[Int],
    outTypes: Array[DataType]) extends LeafExecNode with GpuExec {

  override def supportsColumnar: Boolean = true

  override protected def doExecute(): RDD[org.apache.spark.sql.catalyst.InternalRow] =
    throw new IllegalStateException("GpuFusedQ1LencExec must execute columnar on GPU")

  override lazy val additionalMetrics: Map[String, GpuMetric] = Map(
    GPU_DECODE_TIME -> createNanoTimingMetric(MODERATE_LEVEL, DESCRIPTION_GPU_DECODE_TIME),
    BUFFER_TIME -> createNanoTimingMetric(MODERATE_LEVEL, DESCRIPTION_BUFFER_TIME),
    OP_TIME_LEGACY -> createNanoTimingMetric(DEBUG_LEVEL, DESCRIPTION_OP_TIME_LEGACY)
  )

  override def internalDoExecuteColumnar(): RDD[ColumnarBatch] = {
    val groups = LencPartitionHelper.groupFiles(files)
    new GpuFusedQ1LencRdd(
      sparkContext, groups, fileColIndices, maxShipdateDays, outNativeSlots, outTypes, allMetrics)
  }

  override def simpleString(maxFields: Int): String =
    s"GpuFusedQ1Lenc(files=${files.length}, maxShipdate=$maxShipdateDays)"
}

case class GpuFusedQ1TableExec(
    output: Seq[Attribute],
    child: SparkPlan,
    childColOrds: Array[Int],
    maxShipdateDays: Int,
    outNativeSlots: Array[Int],
    outTypes: Array[DataType]) extends UnaryExecNode with GpuExec {

  override def supportsColumnar: Boolean = true

  override protected def doExecute(): RDD[org.apache.spark.sql.catalyst.InternalRow] =
    throw new IllegalStateException("GpuFusedQ1TableExec must execute columnar on GPU")

  override lazy val additionalMetrics: Map[String, GpuMetric] = Map(
    OP_TIME_LEGACY -> createNanoTimingMetric(DEBUG_LEVEL, DESCRIPTION_OP_TIME_LEGACY)
  )

  override def internalDoExecuteColumnar(): RDD[ColumnarBatch] = {
    val opTime = allMetrics.getOrElse(OP_TIME_LEGACY, NoopMetric)
    val numOutputRows = allMetrics.getOrElse(NUM_OUTPUT_ROWS, NoopMetric)
    val numOutputBatches = allMetrics.getOrElse(NUM_OUTPUT_BATCHES, NoopMetric)
    val ords = childColOrds
    val mDays = maxShipdateDays
    val slots = outNativeSlots
    val oTypes = outTypes

    child.executeColumnar().mapPartitions { iter =>
      iter.flatMap { batch =>
        opTime.ns {
          withResource(batch) { b =>
            if (b.numRows() == 0) {
              None
            } else {
              val cols = ords.map(o => b.column(o).asInstanceOf[GpuColumnVector].getBase)
              withResource(new CudfTable(cols: _*)) { t =>
                withResource(LencNative.fusedQ1FromTable(t, mDays)) { res =>
                  if (res.getRowCount > 0) {
                    numOutputRows += res.getRowCount
                    numOutputBatches += 1
                    Some(Q1BatchBuilder.projectSlots(res, slots, oTypes))
                  } else {
                    None
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
    s"GpuFusedQ1Table(maxShipdate=$maxShipdateDays)"
}

case class GpuFusedWideDictAggExec(
    output: Seq[Attribute],
    child: SparkPlan,
    groupOrds: Array[Int],
    metricOrd: Int,
    numerOrd: Int,
    denomOrd: Int,
    metricLiterals: Array[String],
    outMetricIds: Array[Int],
    outCoeffs: Array[Double],
    outUseDenom: Array[Byte]) extends UnaryExecNode with GpuExec {

  override def supportsColumnar: Boolean = true

  override protected def doExecute(): RDD[org.apache.spark.sql.catalyst.InternalRow] =
    throw new IllegalStateException("GpuFusedWideDictAggExec must execute columnar on GPU")

  override lazy val additionalMetrics: Map[String, GpuMetric] = Map(
    OP_TIME_LEGACY -> createNanoTimingMetric(DEBUG_LEVEL, DESCRIPTION_OP_TIME_LEGACY)
  )

  override def internalDoExecuteColumnar(): RDD[ColumnarBatch] = {
    val opTime = allMetrics.getOrElse(OP_TIME_LEGACY, NoopMetric)
    val numOutputRows = allMetrics.getOrElse(NUM_OUTPUT_ROWS, NoopMetric)
    val numOutputBatches = allMetrics.getOrElse(NUM_OUTPUT_BATCHES, NoopMetric)
    val gOrds = groupOrds
    val mOrd = metricOrd
    val nOrd = numerOrd
    val dOrd = denomOrd
    val lits = metricLiterals
    val omids = outMetricIds
    val ocoeffs = outCoeffs
    val odenom = outUseDenom
    val oTypes = output.map(_.dataType).toArray

    child.executeColumnar().mapPartitions { iter =>
      iter.flatMap { batch =>
        opTime.ns {
          withResource(batch) { b =>
            if (b.numRows() == 0) {
              None
            } else {
              val gCols = gOrds.map(o => b.column(o).asInstanceOf[GpuColumnVector].getBase)
              val mCol = b.column(mOrd).asInstanceOf[GpuColumnVector].getBase
              val nCol = b.column(nOrd).asInstanceOf[GpuColumnVector].getBase
              val dCol = b.column(dOrd).asInstanceOf[GpuColumnVector].getBase
              withResource(new CudfTable(gCols: _*)) { gkTbl =>
                withResource(LencNative.fusedWideDictCaseAgg(
                  gkTbl, mCol, nCol, dCol, lits, omids, ocoeffs, odenom)) { res =>
                  if (res.getRowCount > 0) {
                    numOutputRows += res.getRowCount
                    numOutputBatches += 1
                    Some(GpuColumnVector.from(res, oTypes))
                  } else {
                    None
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
    s"GpuFusedWideDictAgg(keys=${groupOrds.length}, outs=${outMetricIds.length})"
}
