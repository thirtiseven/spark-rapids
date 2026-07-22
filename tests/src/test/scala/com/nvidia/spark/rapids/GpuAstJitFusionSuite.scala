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

import ai.rapids.cudf.ColumnVector
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.jni.RmmSpark

import org.apache.spark.sql.catalyst.expressions.{ExprId, NamedExpression}
import org.apache.spark.sql.rapids.{GpuAdd, GpuMultiply, GpuSubtract}
import org.apache.spark.sql.types.{DataType, IntegerType}
import org.apache.spark.sql.vectorized.ColumnarBatch

class GpuAstJitFusionSuite extends RmmSparkRetrySuiteBase {
  private case class NonDeterministicExpr() extends GpuLeafExpression {
    override lazy val deterministic: Boolean = false
    override def dataType: DataType = IntegerType
    override def nullable: Boolean = false
    override def columnarEval(batch: ColumnarBatch): GpuColumnVector =
      throw new UnsupportedOperationException("test-only expression")
  }

  private case class SideEffectExpr() extends GpuLeafExpression {
    override def hasSideEffects: Boolean = true
    override def dataType: DataType = IntegerType
    override def nullable: Boolean = false
    override def columnarEval(batch: ColumnarBatch): GpuColumnVector =
      throw new UnsupportedOperationException("test-only expression")
  }

  private def ref(ordinal: Int = 0): GpuBoundReference =
    GpuBoundReference(ordinal, IntegerType, nullable = true)(ExprId(ordinal), s"c$ordinal")

  private def addJit(value: Int): GpuAstJitExpression =
    GpuAstJitExpression(GpuAdd(
      ref(), GpuLiteral(value, IntegerType), failOnError = false)())

  private def multiplyJit(value: Int): GpuAstJitExpression =
    GpuAstJitExpression(GpuMultiply(
      ref(), GpuLiteral(value, IntegerType), failOnError = false)())

  private def alias(expression: GpuExpression, name: String): GpuAlias =
    GpuAlias(expression, name)(NamedExpression.newExprId)

  private def collectInts(batch: ColumnarBatch, columnIndex: Int): Seq[Int] = {
    val column = batch.column(columnIndex).asInstanceOf[GpuColumnVector]
    withResource(column.copyToHost()) { host =>
      (0 until host.getRowCount.toInt).map(host.getInt)
    }
  }

  override def afterEach(): Unit = {
    RmmSpark.getAndResetNumRetryThrow(1)
    super.afterEach()
  }

  test("groups top-level AST JIT expressions across safe projections and honors barriers") {
    val expressions = Seq(
      alias(addJit(1), "before_left"),
      alias(GpuLiteral(7, IntegerType), "safe"),
      alias(multiplyJit(2), "before_right"),
      alias(NonDeterministicExpr(), "non_deterministic"),
      alias(addJit(3), "singleton"),
      alias(SideEffectExpr(), "side_effect"),
      alias(addJit(4), "after_left"),
      alias(multiplyJit(5), "after_right"))

    assertResult(Seq(Seq(0, 2), Seq(6, 7))) {
      GpuAstJitFusion.findFusedGroupIndexes(expressions)
    }
  }

  test("keeps nested AST JIT expressions on the single-output path") {
    val nested = GpuSubtract(
      addJit(1), GpuLiteral(1, IntegerType), failOnError = false)()
    val expressions = Seq(
      alias(addJit(2), "left"),
      alias(nested, "nested"),
      alias(multiplyJit(3), "right"))

    assertResult(Seq(Seq(0, 2))) {
      GpuAstJitFusion.findFusedGroupIndexes(expressions)
    }
  }

  test("multi-output evaluation preserves project order and resource ownership") {
    val add = addJit(10)
    val multiply = multiplyJit(3)
    try {
      val input = new ColumnarBatch(Array(
        GpuColumnVector.from(ColumnVector.fromInts(1, 2, 3, 4), IntegerType)), 4)
      withResource(input) { input =>
        val expected = new ColumnarBatch(Array(
          GpuColumnVector.from(ColumnVector.fromInts(11, 12, 13, 14), IntegerType),
          GpuColumnVector.from(ColumnVector.fromInts(1, 2, 3, 4), IntegerType),
          GpuColumnVector.from(ColumnVector.fromInts(3, 6, 9, 12), IntegerType)), 4)
        withResource(expected) { expected =>
          val expressions = Seq(
            alias(add, "add"),
            alias(ref(), "input"),
            alias(multiply, "multiply"))
          withResource(GpuProjectExec.project(input, expressions)) { actual =>
            TestUtils.compareBatches(expected, actual)
          }
        }
        withResource(input.column(0).asInstanceOf[GpuColumnVector].copyToHost()) { host =>
          assertResult(Seq(1, 2, 3, 4)) {
            (0 until host.getRowCount.toInt).map(host.getInt)
          }
        }
      }
    } finally {
      add.close()
      multiply.close()
    }
  }

  test("restore discards compiled expressions before retry") {
    val jit = addJit(1)
    try {
      jit.checkpoint()
      val beforeRestore = jit.getCompiledExpression
      assert(beforeRestore.getNativeHandle != 0)

      jit.restore()
      assertResult(0L)(beforeRestore.getNativeHandle)

      jit.checkpoint()
      val afterRestore = jit.getCompiledExpression
      assert(afterRestore.getNativeHandle != 0)
      assert(!(beforeRestore eq afterRestore))
    } finally {
      jit.close()
    }
  }

  test("multi-output evaluation retries as one project group") {
    val add = addJit(10)
    val multiply = multiplyJit(3)
    try {
      val input = new ColumnarBatch(Array(
        GpuColumnVector.from(ColumnVector.fromInts(1, 2, 3, 4), IntegerType)), 4)
      val spillable = closeOnExcept(input) { input =>
        SpillableColumnarBatch(input, SpillPriorities.ACTIVE_ON_DECK_PRIORITY)
      }
      RmmSpark.forceRetryOOM(RmmSpark.getCurrentThreadId, 1,
        RmmSpark.OomInjectionType.GPU.ordinal, 0)

      val expressions = Seq(alias(add, "add"), alias(multiply, "multiply"))
      withResource(GpuProjectExec.projectAndCloseWithRetrySingleBatch(
          spillable, expressions)) { actual =>
        assertResult(Seq(11, 12, 13, 14))(collectInts(actual, 0))
        assertResult(Seq(3, 6, 9, 12))(collectInts(actual, 1))
      }
      assert(RmmSpark.getAndResetNumRetryThrow(1) > 0)
    } finally {
      add.close()
      multiply.close()
    }
  }
}
