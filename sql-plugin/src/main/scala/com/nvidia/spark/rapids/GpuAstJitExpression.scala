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

package com.nvidia.spark.rapids

import scala.collection.mutable.ArrayBuffer

import ai.rapids.cudf.{Scalar, Table}
import ai.rapids.cudf.ast.CompiledExpression
import com.nvidia.spark.Retryable
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.ScalableTaskCompletion.onTaskCompletion
import com.nvidia.spark.rapids.shims.ShimUnaryExpression

import org.apache.spark.TaskContext
import org.apache.spark.sql.catalyst.expressions.{Expression, NamedExpression}
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.{ColumnarBatch, ColumnVector}

object GpuAstJitExpression {
  private def wrapMaximalSubtrees(expression: Expression): Expression = expression match {
    case gpuExpression: GpuExpression
        if gpuExpression.supportsAstJit && gpuExpression.containsAstJitOperator =>
      GpuAstJitExpression(gpuExpression)
    case gpuExpression: GpuExpression =>
      gpuExpression.mapChildren {
        case child: GpuExpression => wrapMaximalSubtrees(child)
        case child => child
      }
    case other => other
  }

  private[rapids] def wrapProjectExpressions(
      expressions: List[NamedExpression]): List[NamedExpression] = {
    expressions.map(wrapMaximalSubtrees(_).asInstanceOf[NamedExpression])
  }

  private[rapids] def tableFromBatch(batch: ColumnarBatch): Table = {
    if (batch.numCols() != 0) {
      GpuColumnVector.from(batch)
    } else {
      withResource(Scalar.fromBool(false)) { falseScalar =>
        withResource(ai.rapids.cudf.ColumnVector.fromScalar(falseScalar, batch.numRows())) {
          falseColumn => new Table(falseColumn)
        }
      }
    }
  }
}

private[rapids] object GpuAstJitFusion {
  private case class JitInProject(outputIndex: Int, expression: GpuAstJitExpression)

  private case class JitGroup(members: Seq[JitInProject]) {
    val startIndex: Int = members.head.outputIndex
    val outputIndexes: Seq[Int] = members.map(_.outputIndex)
  }

  private[rapids] def project(
      batch: ColumnarBatch,
      boundExprs: Seq[Expression]): Option[ColumnarBatch] = {
    val fusedGroups = findFusedGroups(boundExprs)
    if (fusedGroups.isEmpty) {
      None
    } else {
      val groupsByStartIndex = fusedGroups.map(group => group.startIndex -> group).toMap
      Some(projectWithFusedGroups(batch, boundExprs, groupsByStartIndex))
    }
  }

  private[rapids] def findFusedGroupIndexes(
      boundExprs: Seq[Expression]): Seq[Seq[Int]] =
    findFusedGroups(boundExprs).map(_.outputIndexes)

  private def findFusedGroups(boundExprs: Seq[Expression]): Seq[JitGroup] = {
    val fusedGroups = ArrayBuffer[JitGroup]()
    val candidates = ArrayBuffer[JitInProject]()

    def flushCandidates(): Unit = {
      if (candidates.length > 1) {
        fusedGroups += JitGroup(candidates.toList)
      }
      candidates.clear()
    }

    boundExprs.zipWithIndex.foreach { case (expr, index) =>
      extractJitExpression(expr).filter(canFuse) match {
        case Some(jit) => candidates += JitInProject(index, jit)
        case None if !canReorderExpression(expr) => flushCandidates()
        case None =>
      }
    }
    flushCandidates()
    fusedGroups.toSeq
  }

  private def extractJitExpression(expr: Expression): Option[GpuAstJitExpression] = expr match {
    case GpuAlias(child, _) => extractJitExpression(child)
    case jit: GpuAstJitExpression => Some(jit)
    case _ => None
  }

  private def canFuse(jit: GpuAstJitExpression): Boolean =
    jit.deterministic && !jit.hasSideEffects

  private def canReorderExpression(expr: Expression): Boolean =
    expr.deterministic && (expr match {
      case gpuExpr: GpuExpression => !gpuExpr.hasSideEffects
      case _ => false
    })

  private def projectWithFusedGroups(
      batch: ColumnarBatch,
      boundExprs: Seq[Expression],
      groupsByStartIndex: Map[Int, JitGroup]): ColumnarBatch = {
    val outputColumns = new Array[ColumnVector](boundExprs.length)
    closeOnExcept(outputColumns) { _ =>
      boundExprs.indices.foreach { index =>
        if (outputColumns(index) == null) {
          groupsByStartIndex.get(index) match {
            case Some(group) => evaluateFusedGroup(batch, group, outputColumns)
            case None => outputColumns(index) = boundExprs(index).columnarEval(batch)
          }
        }
      }
      new ColumnarBatch(outputColumns, batch.numRows())
    }
  }

  private def evaluateFusedGroup(
      batch: ColumnarBatch,
      group: JitGroup,
      outputColumns: Array[ColumnVector]): Unit = {
    val compiledExpressions = group.members.map(_.expression.getCompiledExpression).toArray
    withResource(GpuAstJitExpression.tableFromBatch(batch)) { table =>
      withResource(CompiledExpression.computeColumnsJit(table, compiledExpressions)) { result =>
        if (result.getNumberOfColumns != group.members.length) {
          throw new IllegalStateException(
            s"AST JIT returned ${result.getNumberOfColumns} columns for " +
              s"${group.members.length} expressions")
        }
        if (result.getRowCount != batch.numRows()) {
          throw new IllegalStateException(
            s"AST JIT returned ${result.getRowCount} rows for ${batch.numRows()} input rows")
        }
        group.members.zipWithIndex.foreach { case (member, resultIndex) =>
          outputColumns(member.outputIndex) = GpuColumnVector.from(
            result.getColumn(resultIndex).incRefCount(), member.expression.dataType)
        }
      }
    }
  }
}

case class GpuAstJitExpression(child: Expression)
    extends ShimUnaryExpression with GpuExpression with Retryable with AutoCloseable {
  require(child.isInstanceOf[GpuExpression], "AST JIT child must be a GPU expression")

  @transient private[this] var compiledExpression: CompiledExpression = _
  @transient private[this] var completionRegistered = false

  override def dataType: DataType = child.dataType

  override def nullable: Boolean = child.nullable

  override def disableTieredProjectCombine: Boolean = true

  override def toString: String = s"AST_JIT($child)"

  override def checkpoint(): Unit = {
    getCompiledExpression
  }

  override def restore(): Unit = {
    // The existing task callback closes the expression recompiled after a retry.
    closeCompiledExpression()
  }

  override def close(): Unit = closeCompiledExpression()

  override def columnarEval(batch: ColumnarBatch): GpuColumnVector = {
    withResource(GpuAstJitExpression.tableFromBatch(batch)) { table =>
      closeOnExcept(getCompiledExpression.computeColumnJit(table)) { result =>
        GpuColumnVector.from(result, dataType)
      }
    }
  }

  private[rapids] def getCompiledExpression: CompiledExpression = synchronized {
    if (compiledExpression == null) {
      compiledExpression = child.asInstanceOf[GpuExpression]
        .convertToAst(Int.MaxValue)
        .compile()
    }
    if (!completionRegistered) {
      Option(TaskContext.get()).foreach { taskContext =>
        onTaskCompletion(taskContext) {
          close()
        }
        completionRegistered = true
      }
    }
    compiledExpression
  }

  private def closeCompiledExpression(): Unit = synchronized {
    Option(compiledExpression).foreach(_.safeClose())
    compiledExpression = null
  }
}
