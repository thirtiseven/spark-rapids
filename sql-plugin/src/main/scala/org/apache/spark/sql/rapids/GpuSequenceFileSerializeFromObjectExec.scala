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
import org.apache.spark.sql.catalyst.expressions.{Attribute, GenericInternalRow, SortOrder, UnsafeProjection}
import org.apache.spark.sql.catalyst.plans.physical.Partitioning
import org.apache.spark.sql.execution.{SparkPlan, UnaryExecNode}
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.vectorized.ColumnarBatch

/**
 * GPU replacement for SerializeFromObjectExec over SequenceFile object scans.
 */
case class GpuSequenceFileSerializeFromObjectExec(
    outputAttrs: Seq[Attribute],
    child: SparkPlan,
    goal: CoalesceSizeGoal)
  extends UnaryExecNode with GpuExec {

  override def output: Seq[Attribute] = outputAttrs
  override def outputPartitioning: Partitioning = child.outputPartitioning
  override def outputOrdering: Seq[SortOrder] = child.outputOrdering
  override def outputBatching: CoalesceGoal = goal

  override def doExecute(): RDD[InternalRow] = {
    val localOutput = output
    val childObjType = child.output.head.dataType
    val numOutCols = localOutput.length
    val outSchema = StructType(localOutput.map(a =>
      StructField(a.name, a.dataType, a.nullable)))
    child.execute().mapPartitionsWithIndexInternal { (index, it) =>
      val unsafeProj = UnsafeProjection.create(outSchema)
      unsafeProj.initialize(index)
      it.map { row =>
        val obj = row.get(0, childObjType)
        val outRow = new GenericInternalRow(numOutCols)
        if (numOutCols == 1) {
          outRow.update(0, obj.asInstanceOf[Array[Byte]])
        } else {
          val tuple = obj.asInstanceOf[Product]
          outRow.update(0,
            tuple.productElement(0).asInstanceOf[Array[Byte]])
          outRow.update(1,
            tuple.productElement(1).asInstanceOf[Array[Byte]])
        }
        unsafeProj(outRow).copy()
      }
    }
  }

  override def internalDoExecuteColumnar(): RDD[ColumnarBatch] = {
    // Reuse existing row->columnar path to avoid duplicating converter internals.
    GpuRowToColumnarExec(this, goal).executeColumnar()
  }

  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan = {
    copy(child = newChild)
  }
}

