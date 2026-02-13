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

package org.apache.spark.sql.rapids

import com.nvidia.spark.rapids.{CoalesceGoal, CoalesceSizeGoal, GpuExec, GpuRowToColumnarExec}

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, SortOrder}
import org.apache.spark.sql.catalyst.plans.physical.Partitioning
import org.apache.spark.sql.execution.{SparkPlan, UnaryExecNode}
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * Physical replacement for SequenceFile RDD scans.
 *
 * This wraps an RDDScanExec and materializes columnar output through the
 * existing row-to-columnar GPU transition path.
 */
case class GpuSequenceFileRDDScanExec(child: SparkPlan, goal: CoalesceSizeGoal)
  extends UnaryExecNode with GpuExec {

  override def output: Seq[Attribute] = child.output
  override def outputPartitioning: Partitioning = child.outputPartitioning
  override def outputOrdering: Seq[SortOrder] = child.outputOrdering
  override def outputBatching: CoalesceGoal = goal

  override def doExecute(): RDD[InternalRow] = child.execute()

  override def internalDoExecuteColumnar(): RDD[ColumnarBatch] = {
    // Reuse the existing GPU row->columnar converter for stable behavior.
    GpuRowToColumnarExec(child, goal).executeColumnar()
  }

  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan = {
    copy(child = newChild)
  }
}

