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

import ai.rapids.cudf.{ColumnVector, Cuda, Scalar, Table}
import ai.rapids.cudf.ast._

val rows = sys.env.getOrElse("BENCH_ROWS", "100000").toInt
val variant = sys.env.getOrElse("BENCH_VARIANT", "lto")
val iterations = sys.env.getOrElse("BENCH_ITERS", "1").toInt

def sequence(initial: Long, step: Long): ColumnVector = {
  val initialScalar = Scalar.fromLong(initial)
  val stepScalar = Scalar.fromLong(step)
  try {
    ColumnVector.sequence(initialScalar, stepScalar, rows)
  } finally {
    stepScalar.close()
    initialScalar.close()
  }
}

val a = sequence(0L, 1L)
val b = sequence(1L, 3L)
val table = new Table(a, b)
a.close()
b.close()

val aRef = new ColumnReference(0)
val bRef = new ColumnReference(1)
val add0 = new BinaryOperation(BinaryOperator.ADD, aRef, bRef)
val mul0 = new BinaryOperation(BinaryOperator.MUL, add0, aRef)
val add1 = new BinaryOperation(BinaryOperator.ADD, mul0, bRef)
val base = new BinaryOperation(BinaryOperator.MUL, add1, aRef)
val expression: AstExpression = variant match {
  case "lto" => base
  // IDENTITY preserves the output while selecting the source-JIT fallback.
  case "source" => new UnaryOperation(UnaryOperator.IDENTITY, base)
  case other => throw new IllegalArgumentException(s"Unknown BENCH_VARIANT: $other")
}

val compiled = expression.compile()
try {
  (0 until iterations).foreach { iteration =>
    val start = System.nanoTime()
    val result = compiled.computeColumnJit(table)
    try {
      Cuda.DEFAULT_STREAM.sync()
      val wallMs = (System.nanoTime() - start) / 1e6
      val checksum = result.sum()
      try {
        println(f"variant=$variant rows=$rows iteration=$iteration wallMs=$wallMs%.3f " +
          s"checksum=${checksum.getLong()}")
      } finally {
        checksum.close()
      }
    } finally {
      result.close()
    }
  }
} finally {
  compiled.close()
  table.close()
}
