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

import scala.reflect.ClassTag

import ai.rapids.cudf.{ast, BinaryOp}

import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.types.DataType

/**
 * Test-only GPU expression that supports interpreted AST but intentionally opts out of AST JIT,
 * so cross-backend planner tests keep exercising legacy-vs-JIT arbitration as real operators gain
 * JIT support.
 */
case class TestLegacyOnlyBinary(left: Expression, right: Expression)
    extends CudfBinaryExpression {
  override def dataType: DataType = left.dataType
  override def binaryOp: BinaryOp = BinaryOp.ADD
  override def astOperator: Option[ast.BinaryOperator] = Some(ast.BinaryOperator.ADD)
  override def selfConvertsToAst: Boolean = true
  override protected def astJitCompatible: Boolean = false
}

object ProjectAstTestUtils {
  def legacyOnlyBinary(left: Expression, right: Expression): TestLegacyOnlyBinary = {
    val expr = TestLegacyOnlyBinary(left, right)
    GpuProjectAstPlanner.tagLegacySupport(expr, supported = true)
    expr
  }

  def collectExpressions[T <: Expression : ClassTag](
      expressions: Seq[Expression]): Seq[T] = {
    expressions.flatMap(_.collect {
      case expression: T => expression
    })
  }

  def tierReferences(expression: Expression): Seq[GpuBoundReference] = {
    expression.collect {
      case reference: GpuBoundReference if reference.name.startsWith("tiered_input_") => reference
    }
  }
}
