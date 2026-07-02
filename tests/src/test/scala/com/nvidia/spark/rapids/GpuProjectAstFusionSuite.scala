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

import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.sql.catalyst.expressions.NamedExpression
import org.apache.spark.sql.rapids.GpuAdd
import org.apache.spark.sql.types.LongType

class GpuProjectAstFusionSuite extends AnyFunSuite {
  private def ref(ordinal: Int, nullable: Boolean): GpuBoundReference =
    GpuBoundReference(ordinal, LongType, nullable)(NamedExpression.newExprId, s"c$ordinal")

  private def add(input: GpuBoundReference): GpuExpression =
    GpuAdd(input, GpuLiteral(1L, LongType), failOnError = false)()

  test("fusion groups do not split on nullable input dependencies") {
    val expressions = Seq(add(ref(0, nullable = true)), add(ref(0, nullable = true)),
      add(ref(1, nullable = true)))

    val groups = GpuProjectAstExec.planFusionGroups(
      expressions, GpuProjectAstExec.AstFusionLimits(Int.MaxValue, Int.MaxValue, Int.MaxValue))

    assert(groups.map(_.size) === Seq(3))
  }

  test("fusion groups honor output, node, and live-value limits") {
    val expressions = Seq.fill(3)(add(ref(0, nullable = true)))

    val outputLimited = GpuProjectAstExec.planFusionGroups(
      expressions, GpuProjectAstExec.AstFusionLimits(1, Int.MaxValue, Int.MaxValue))
    val nodeLimited = GpuProjectAstExec.planFusionGroups(
      expressions, GpuProjectAstExec.AstFusionLimits(Int.MaxValue, 1, Int.MaxValue))
    val liveValueLimited = GpuProjectAstExec.planFusionGroups(
      expressions, GpuProjectAstExec.AstFusionLimits(Int.MaxValue, Int.MaxValue, 1))

    assert(outputLimited.map(_.size) === Seq(1, 1, 1))
    assert(nodeLimited.map(_.size) === Seq(1, 1, 1))
    assert(liveValueLimited.map(_.size) === Seq(1, 1, 1))
  }
}
