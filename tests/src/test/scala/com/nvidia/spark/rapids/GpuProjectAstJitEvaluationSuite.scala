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

import ai.rapids.cudf.{BinaryOp, ColumnVector, CudfException, DType, Table}
import com.nvidia.spark.rapids.Arm.withResource

import org.apache.spark.sql.catalyst.expressions.NamedExpression
import org.apache.spark.sql.rapids.{GpuAbs, GpuAdd, GpuMultiply, GpuSubtract, GpuUnaryMinus}
import org.apache.spark.sql.types.{ByteType, DataType, IntegerType, LongType, ShortType}

/** Evaluates AST JIT programs on the GPU, with and without the regular-path fallback. */
class GpuProjectAstJitEvaluationSuite extends GpuUnitTests {
  private def ref(ordinal: Int, dataType: DataType): GpuBoundReference =
    GpuBoundReference(ordinal, dataType, nullable = true)(NamedExpression.newExprId, s"c$ordinal")

  /** Applies `nulls`' validity without touching the data, so null rows keep their values. */
  private def withNullsOver(data: ColumnVector, nulls: ColumnVector): ColumnVector =
    withResource(data) { _ =>
      withResource(nulls) { _ => data.mergeAndSetValidity(BinaryOp.BITWISE_AND, nulls) }
    }

  private def longs(values: java.lang.Long*): Seq[java.lang.Long] = values

  /** Builds a table that owns `columns`. */
  private def table(columns: ColumnVector*): Table =
    withResource(columns) { _ => new Table(columns: _*) }

  private def collect(column: ai.rapids.cudf.ColumnView): Seq[java.lang.Long] =
    withResource(column.copyToHost()) { host =>
      (0 until host.getRowCount.toInt).map { row =>
        if (host.isNull(row)) {
          null
        } else {
          java.lang.Long.valueOf(column.getType match {
            case DType.INT8 => host.getByte(row).toLong
            case DType.INT16 => host.getShort(row).toLong
            case DType.INT32 => host.getInt(row).toLong
            case _ => host.getLong(row)
          })
        }
      }
    }

  /** Runs the JIT program directly, so a JIT failure is not hidden by the fallback. */
  private def jitOnly(expressions: Seq[GpuExpression], table: Table): Seq[Seq[java.lang.Long]] = {
    val roots = expressions.map(GpuAstJitExpression(_))
    try {
      withResource(roots.head.getJitProgram(roots, table).computeTable(table)) { result =>
        (0 until result.getNumberOfColumns).map(i => collect(result.getColumn(i)))
      }
    } finally {
      roots.foreach(_.close())
    }
  }

  private def nullableLongTable(): Table = {
    // Null rows hold values that overflow every checked operation below.
    val left = withNullsOver(
      ColumnVector.fromLongs(Long.MaxValue, Long.MinValue, 3L, 4L),
      ColumnVector.fromBoxedLongs(null, null, 1L, 1L))
    val right = withNullsOver(
      ColumnVector.fromLongs(1L, -1L, Long.MaxValue, 2L),
      ColumnVector.fromBoxedLongs(1L, 1L, null, 1L))
    table(left, right)
  }

  test("ANSI AST JIT ignores overflowing values in null rows") {
    withResource(nullableLongTable()) { input =>
      val left = ref(0, LongType)
      val right = ref(1, LongType)
      val results = jitOnly(Seq(
        GpuAdd(left, right, failOnError = true)(),
        GpuSubtract(left, right, failOnError = true)(),
        GpuMultiply(left, right, failOnError = true)(),
        GpuUnaryMinus(left, failOnError = true),
        GpuAbs(left, failOnError = true)), input)
      assertResult(longs(null, null, null, 6L))(results(0))
      assertResult(longs(null, null, null, 2L))(results(1))
      assertResult(longs(null, null, null, 8L))(results(2))
      assertResult(longs(null, null, -3L, -4L))(results(3))
      assertResult(longs(null, null, 3L, 4L))(results(4))
    }
  }

  test("ANSI AST JIT reports overflow from the kernel") {
    val cases: Seq[(GpuBoundReference => GpuExpression, Long)] = Seq(
      (r => GpuAdd(r, GpuLiteral(1L, LongType), failOnError = true)(), Long.MaxValue),
      (r => GpuSubtract(r, GpuLiteral(1L, LongType), failOnError = true)(), Long.MinValue),
      (r => GpuMultiply(r, GpuLiteral(2L, LongType), failOnError = true)(), Long.MaxValue),
      (r => GpuUnaryMinus(r, failOnError = true), Long.MinValue),
      (r => GpuAbs(r, failOnError = true), Long.MinValue))
    cases.foreach { case (build, overflowing) =>
      withResource(table(ColumnVector.fromLongs(0L, overflowing))) { input =>
        val error = intercept[CudfException] { jitOnly(Seq(build(ref(0, LongType))), input) }
        assert(error.getMessage.contains("ARITHMETIC_OVERFLOW"), error.getMessage)
      }
    }
  }

  test("non-ANSI AST JIT wraps like the regular GPU path") {
    val values = Seq(Long.MinValue, -1L, 0L, Long.MaxValue)
    withResource(table(
        ColumnVector.fromLongs(values: _*), ColumnVector.fromLongs(1L, 2L, 3L, -1L))) { input =>
      val left = ref(0, LongType)
      val right = ref(1, LongType)
      val expressions = Seq(
        GpuAdd(left, right, failOnError = false)(),
        GpuSubtract(left, right, failOnError = false)(),
        GpuMultiply(left, right, failOnError = false)(),
        GpuUnaryMinus(left, failOnError = false),
        GpuAbs(left, failOnError = false))
      val jit = jitOnly(expressions, input)
      val batch = GpuColumnVector.from(input, Array[DataType](LongType, LongType))
      withResource(batch) { _ =>
        expressions.zip(jit).foreach { case (expression, jitValues) =>
          val regular = withResource(expression.columnarEval(batch)) { c => collect(c.getBase) }
          assertResult(regular, expression)(jitValues)
        }
      }
    }
  }

  test("AST JIT inlines an integral widening cast") {
    withResource(table(
        ColumnVector.fromBoxedInts(Int.MaxValue, null, -3),
        ColumnVector.fromLongs(1L, 2L, 3L))) { input =>
      val widened = GpuCast(ref(0, IntegerType), LongType)()
      val sum = GpuAdd(widened, ref(1, LongType), failOnError = true)()
      assert(GpuAstJitExpression.canUseAstJit(sum))
      assert(!GpuAstJitExpression.canUseAstJit(widened))
      assertResult(Seq(longs(Int.MaxValue + 1L, null, 0L)))(jitOnly(Seq(sum), input))
    }
  }

  test("a JIT overflow throws the regular GPU error") {
    withResource(table(
        ColumnVector.fromLongs(1L, Long.MaxValue), ColumnVector.fromLongs(1L, 1L))) { input =>
      val left = ref(0, LongType)
      val right = ref(1, LongType)
      // Only the second output overflows; the group reports it like the regular path does.
      val roots = Seq(
        GpuAstJitExpression(GpuMultiply(left, right, failOnError = true)()),
        GpuAstJitExpression(GpuAdd(left, right, failOnError = true)()))
      val batch = GpuColumnVector.from(input, Array[DataType](LongType, LongType))
      try {
        withResource(batch) { _ =>
          val error = intercept[ArithmeticException] {
            GpuAstJitExpression.computeColumns(roots, input, batch).close()
          }
          assert(error.getMessage.contains("overflow"), error.getMessage)
          assert(error.getSuppressed.exists(_.isInstanceOf[CudfException]))
        }
      } finally {
        roots.foreach(_.close())
      }
    }
  }

  test("a JIT group without errors returns the JIT columns") {
    withResource(nullableLongTable()) { input =>
      val left = ref(0, LongType)
      val right = ref(1, LongType)
      val roots = Seq(
        GpuAstJitExpression(GpuAdd(left, right, failOnError = true)()),
        GpuAstJitExpression(GpuUnaryMinus(right, failOnError = true)))
      val batch = GpuColumnVector.from(input, Array[DataType](LongType, LongType))
      try {
        withResource(batch) { _ =>
          withResource(GpuAstJitExpression.computeColumns(roots, input, batch)) { result =>
            def column(i: Int) = result.column(i).asInstanceOf[GpuColumnVector].getBase
            assertResult(longs(null, null, null, 6L))(collect(column(0)))
            assertResult(longs(-1L, 1L, null, -2L))(collect(column(1)))
          }
        }
      } finally {
        roots.foreach(_.close())
      }
    }
  }

  // Spark wraps or checks byte and short arithmetic at every step. A kernel that kept a promoted
  // intermediate would differ for abs of a wrapped value and miss an overflow undone by a later
  // operation.
  for ((dataType, min, max) <- Seq(
      (ByteType, Byte.MinValue.toLong, Byte.MaxValue.toLong),
      (ShortType, Short.MinValue.toLong, Short.MaxValue.toLong))) {
    def column(values: Long*): ColumnVector = dataType match {
      case ByteType => ColumnVector.fromBytes(values.map(_.toByte): _*)
      case _ => ColumnVector.fromShorts(values.map(_.toShort): _*)
    }

    test(s"non-ANSI AST JIT wraps $dataType at each operation like the regular GPU path") {
      withResource(table(
          column(max, min, max - 27, -1L, 0L, 3L),
          column(1L, -1L, max - 27, min, 5L, -7L))) { input =>
        val left = ref(0, dataType)
        val right = ref(1, dataType)
        val sum = GpuAdd(left, right, failOnError = false)()
        val expressions = Seq(
          sum,
          GpuSubtract(left, right, failOnError = false)(),
          GpuMultiply(left, right, failOnError = false)(),
          GpuUnaryMinus(left, failOnError = false),
          GpuAbs(left, failOnError = false),
          GpuAbs(sum, failOnError = false),
          GpuSubtract(GpuMultiply(sum, right, failOnError = false)(), left,
            failOnError = false)())
        val jit = jitOnly(expressions, input)
        val batch = GpuColumnVector.from(input, Array[DataType](dataType, dataType))
        withResource(batch) { _ =>
          expressions.zip(jit).foreach { case (expression, jitValues) =>
            val regular = withResource(expression.columnarEval(batch)) { c => collect(c.getBase) }
            assertResult(regular, expression)(jitValues)
          }
        }
      }
    }

    test(s"ANSI AST JIT reports $dataType overflow in an intermediate result") {
      // max + 1 overflows even though (max + 1) - 1 is back in range.
      withResource(table(column(0L, max), column(1L, 1L))) { input =>
        val left = ref(0, dataType)
        val right = ref(1, dataType)
        val expression = GpuSubtract(
          GpuAdd(left, right, failOnError = true)(), right, failOnError = true)()
        val error = intercept[CudfException] { jitOnly(Seq(expression), input) }
        assert(error.getMessage.contains("ARITHMETIC_OVERFLOW"), error.getMessage)
      }
    }
  }
}
