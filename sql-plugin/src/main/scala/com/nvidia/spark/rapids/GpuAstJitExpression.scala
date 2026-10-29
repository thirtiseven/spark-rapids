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

import java.util.concurrent.{CompletableFuture, ConcurrentHashMap, ExecutionException,
  ThreadPoolExecutor}
import java.util.concurrent.atomic.AtomicBoolean
import java.util.function.BiConsumer

import scala.collection.mutable
import scala.util.control.NonFatal

import ai.rapids.cudf.{ColumnVector, DType, Scalar, Table}
import ai.rapids.cudf.ast.{AstExpression, AstJitProgram, CompiledExpression}
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.GpuMetric._
import com.nvidia.spark.rapids.RapidsPluginImplicits._

import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.expressions.{Expression, Literal}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids.catalyst.expressions.GpuExpressionEquals
import org.apache.spark.sql.rapids.execution.TrampolineUtil
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.ColumnarBatch

object GpuAstJitExpression {
  /** Extracts an AST JIT wrapper after unwrapping any top-level aliases. */
  private[rapids] def extractTopLevel(expression: Expression): Option[GpuAstJitExpression] =
    GpuProjectAstExpressionBase.extractTopLevel(expression).collect {
      case jitExpression: GpuAstJitExpression => jitExpression
    }

  private final case class JitInputColumn(
      ordinal: Int,
      dataType: DType,
      nullable: Boolean)

  // Set by the first precompile on this executor, so tasks without one never load the precompiler.
  @volatile private[rapids] var precompileSubmitted = false

  /** A column referenced by a JIT group, as the bound plan describes it. */
  private[rapids] final case class JitPlanInput(
      ordinal: Int,
      dataType: DataType,
      nullable: Boolean)

  private[rapids] final class JitProgramGroup(
      val expressions: Vector[GpuAstJitExpression],
      val planInputs: Vector[JitPlanInput]) {
    val referencedOrdinals: Vector[Int] = planInputs.map(_.ordinal)

    def matches(other: Seq[GpuAstJitExpression]): Boolean = {
      expressions.size == other.size &&
        expressions.iterator.zip(other.iterator).forall { case (left, right) => left eq right }
    }

    /**
     * Identifies the native kernel this group compiles to. cuDF gives a multi-output group a
     * null-aware kernel when any input column carries a null mask, whether or not it holds
     * nulls; otherwise mask presence does not change the kernel.
     */
    def kernelKey(anyInputMasked: Boolean): GpuAstJitPrecompiler.Key =
      GpuAstJitPrecompiler.Key(expressions.map(_.child), anyInputMasked && expressions.size > 1)
  }

  private[rapids] object JitProgramGroup {
    def apply(expressions: Seq[GpuAstJitExpression]): JitProgramGroup = {
      val roots = expressions.toVector
      val references = roots.flatMap(_.child.collect {
        case reference: GpuBoundReference => reference
      })
      val planInputs = references.groupBy(_.ordinal).toVector.sortBy(_._1).map {
        case (ordinal, sameColumn) =>
          JitPlanInput(ordinal, sameColumn.head.dataType, sameColumn.exists(_.nullable))
      }
      new JitProgramGroup(roots, planInputs)
    }
  }

  /**
   * Builds a one-row table with each input's plan type, where `masked` gives every input a null
   * mask. Types and mask presence are all a group's kernel depends on, so this table compiles
   * the kernel a batch will need before any batch exists. Unreferenced ordinals get placeholders.
   */
  private[rapids] def loweringTable(inputs: Vector[JitPlanInput], masked: Boolean): Table = {
    val byOrdinal = inputs.map(input => input.ordinal -> input).toMap
    val numColumns = inputs.lastOption.map(_.ordinal + 1).getOrElse(1)
    val columns = (0 until numColumns).safeMap { ordinal =>
      val scalar = byOrdinal.get(ordinal) match {
        // A column made from a null scalar carries a null mask; one from a valid scalar does not.
        case Some(input) =>
          val value = if (masked) null else Literal.default(input.dataType).value
          GpuScalar.from(value, input.dataType)
        case None => Scalar.fromBool(false)
      }
      withResource(scalar)(ColumnVector.fromScalar(_, 1))
    }
    withResource(columns)(new Table(_: _*))
  }

  private def inputSchema(
      group: JitProgramGroup,
      table: Table): Vector[JitInputColumn] = {
    group.referencedOrdinals.map { ordinal =>
      val column = table.getColumn(ordinal)
      JitInputColumn(ordinal, column.getType, column.hasValidityVector)
    }
  }

  private[rapids] def canUseAstJit(expression: Expression): Boolean = expression match {
    case gpuExpression: GpuExpression =>
      GpuBatchUtils.isFixedWidth(expression.dataType) &&
        gpuExpression.supportsAstJit && gpuExpression.containsAstJitOperator
    case _ => false
  }

  private def astJitChild(child: GpuExpression): Option[GpuExpression] = child match {
    case jitExpression: GpuAstJitExpression => Some(jitExpression.child)
    case other if canUseAstJit(other) => Some(other)
    case _ => None
  }

  private def operationSet(expression: GpuExpression): Set[GpuExpressionEquals] = {
    expression.collect {
      case gpuExpression: GpuExpression if gpuExpression.selfIsAstJitOperator =>
        GpuExpressionEquals(gpuExpression)
    }.toSet
  }

  private[rapids] def wrapTierExpressions(
      expressions: Seq[Expression],
      conf: SQLConf): Seq[Expression] = {
    val jitChildren = expressions.zipWithIndex.flatMap { case (expression, index) =>
      expression match {
        case GpuAlias(child: GpuExpression, _) if
            !child.isInstanceOf[GpuProjectAstExpression] =>
          astJitChild(child).map(index -> _)
        case _ => None
      }
    }
    val groupIds = GpuAstJitGrouping.assign(
      jitChildren.map { case (index, child) =>
        GpuAstJitGrouping.Root(index, operationSet(child))
      },
      GpuAstJitGrouping.Limits(
        maxOps = RapidsConf.PROJECT_AST_JIT_MAX_GROUP_OPS.get(conf),
        maxOutputs = RapidsConf.PROJECT_AST_JIT_MAX_GROUP_OUTPUTS.get(conf)))
    val childrenByIndex = jitChildren.toMap
    expressions.zipWithIndex.map { case (expression, index) =>
      (expression, childrenByIndex.get(index), groupIds.get(index)) match {
        case (alias: GpuAlias, Some(child), Some(groupId)) =>
          GpuProjectAstExpressionBase.replaceChild(alias, GpuAstJitExpression(child, groupId))
        case _ => expression
      }
    }
  }

  private[rapids] def outputGroups(
      expressions: Seq[GpuAstJitExpression]): Seq[Seq[GpuAstJitExpression]] = {
    val groups = mutable.LinkedHashMap.empty[Int, mutable.ArrayBuffer[GpuAstJitExpression]]
    expressions.foreach { expression =>
      groups.getOrElseUpdate(expression.groupId, mutable.ArrayBuffer.empty) += expression
    }
    groups.values.map(_.toSeq).toSeq
  }

  private[rapids] def executionGroups(
      expressions: Seq[GpuAstJitExpression],
      multiOutputEnabled: Boolean): Seq[Seq[GpuAstJitExpression]] = {
    if (multiOutputEnabled) {
      outputGroups(expressions)
    } else {
      expressions.map(Seq(_))
    }
  }

  private[rapids] def computeColumns(
      expressions: Seq[GpuAstJitExpression],
      table: Table): ColumnarBatch = {
    require(expressions.nonEmpty, "AST JIT requires at least one expression")
    val owner = expressions.head
    val program = owner.getJitProgram(expressions, table)
    owner.withComputeMetrics(table.getRowCount) {
      withResource(program.computeTable(table)) { result =>
        GpuColumnVector.from(result, expressions.map(_.dataType).toArray)
      }
    }
  }

  private def hasJitCandidate(expression: Expression): Boolean = expression.find {
    case gpuExpression: GpuExpression =>
      gpuExpression.supportsAstJit && gpuExpression.containsAstJitOperator
    case _ => false
  }.isDefined

  private def finalBackend(expression: Expression): String = {
    GpuProjectAstExpressionBase.extractTopLevel(expression) match {
      case Some(_: GpuAstJitExpression) => "AST JIT"
      case Some(_: GpuProjectAstExpression) => "AST Interpreted"
      case _ => "the regular GPU projection"
    }
  }

  private[rapids] def explainFinalSelections(
      expressionTiers: Seq[Seq[Expression]],
      all: Boolean,
      multiOutputEnabled: Boolean = true,
      astJitEnabled: Boolean = true): String = {
    expressionTiers.zipWithIndex.flatMap { case (expressions, tier) =>
      val explanations = expressions.iterator.zipWithIndex.collect {
        case (expression, index) if all || (astJitEnabled &&
            extractTopLevel(expression).isEmpty && hasJitCandidate(expression)) =>
          s"    output[$index] $expression final backend: ${finalBackend(expression)}\n"
      }.mkString
      if (explanations.nonEmpty) {
        val structure = if (all) {
          val indexedRoots = expressions.zipWithIndex.flatMap { case (expression, index) =>
            extractTopLevel(expression).map(index -> _)
          }
          val groups = executionGroups(indexedRoots.map(_._2), multiOutputEnabled)
          val remainingRoots = mutable.ArrayBuffer(indexedRoots: _*)
          val groupDetails = groups.zipWithIndex.map { case (group, index) =>
            val inputs = JitProgramGroup(group).referencedOrdinals.mkString("[", ",", "]")
            val outputs = group.map { root =>
              val position = remainingRoots.indexWhere(_._2 eq root)
              remainingRoots.remove(position)._1
            }.mkString("[", ",", "]")
            val occurrences = group.flatMap(_.child.collect {
              case expression: GpuExpression if expression.selfIsAstJitOperator =>
                GpuExpressionEquals(expression)
            })
            val uniqueOps = occurrences.distinct.size
            val sharedOps = occurrences.groupBy(identity).count(_._2.size > 1)
            s"    JIT GROUP $index inputs=$inputs outputs=$outputs " +
              s"plugin_unique_ops=$uniqueOps plugin_shared_ops=$sharedOps\n"
          }.mkString
          def forwarded(expression: Expression): Boolean = expression match {
            case alias: GpuAlias => forwarded(alias.child)
            case _: GpuBoundReference => true
            case _ => false
          }
          val materialized = if (tier + 1 < expressionTiers.size) {
            expressions.indices.filterNot(i => forwarded(expressions(i)))
          } else {
            Seq.empty
          }
          val inputs = expressions.flatMap(_.collect {
            case reference: GpuBoundReference => reference.ordinal
          }).distinct.sorted.mkString("[", ",", "]")
          val forwardedOutputs = expressions.indices.filter(i => forwarded(expressions(i)))
              .mkString("[", ",", "]")
          s"    inputs=$inputs outputs=${expressions.size} forwarded_outputs=$forwardedOutputs " +
            s"materialized_outputs_to_next_tier=${materialized.mkString("[", ",", "]")}\n" +
            groupDetails
        } else {
          ""
        }
        Some(s"  TIER $tier\n$structure$explanations")
      } else {
        None
      }
    }.mkString
  }
}

case class GpuAstJitExpression(child: GpuExpression, groupId: Int = 0)
    extends GpuProjectAstExpressionBase {

  @transient private[this] var jitProgram: AstJitProgram = _
  @transient private[this] var jitProgramGroup: GpuAstJitExpression.JitProgramGroup = _
  @transient private[this] var jitProgramSchema: Vector[GpuAstJitExpression.JitInputColumn] = _

  private[this] var programBuildAttempts: GpuMetric = NoopMetric
  private[this] var programCacheHits: GpuMetric = NoopMetric
  private[this] var programBuildTime: GpuMetric = NoopMetric
  private[this] var precompileWaitTime: GpuMetric = NoopMetric
  private[this] var evalAttempts: GpuMetric = NoopMetric
  private[this] var evalRows: GpuMetric = NoopMetric
  private[this] var evalTime: GpuMetric = NoopMetric

  override def injectMetrics(metrics: Map[String, GpuMetric]): Unit = {
    super.injectMetrics(metrics)
    programBuildAttempts = metrics.getOrElse(AST_JIT_PROGRAM_BUILD_ATTEMPTS, NoopMetric)
    programCacheHits = metrics.getOrElse(AST_JIT_PROGRAM_CACHE_HITS, NoopMetric)
    programBuildTime = metrics.getOrElse(AST_JIT_PROGRAM_BUILD_TIME, NoopMetric)
    precompileWaitTime = metrics.getOrElse(AST_JIT_PRECOMPILE_WAIT_TIME, NoopMetric)
    evalAttempts = metrics.getOrElse(AST_JIT_EVAL_ATTEMPTS, NoopMetric)
    evalRows = metrics.getOrElse(AST_JIT_EVAL_ROWS, NoopMetric)
    evalTime = metrics.getOrElse(AST_JIT_EVAL_TIME, NoopMetric)
  }

  override private[rapids] def withComputeMetrics[T](rows: Long)(body: => T): T = {
    // Count group work, including retries, rather than unique Project input rows.
    evalAttempts += 1
    evalRows += rows
    evalTime.ns { super.withComputeMetrics(rows)(body) }
  }

  override def disableTieredProjectCombine: Boolean = true

  override protected def compileNvtxId: NvtxId = NvtxRegistry.COMPILE_AST_JIT

  override protected def computeNvtxId: NvtxId = NvtxRegistry.PROJECT_AST_JIT

  override protected def compileAst(ast: AstExpression): CompiledExpression = ast.compileJit()

  protected def compileJitProgram(
      table: Table,
      expressions: Array[CompiledExpression]): AstJitProgram = {
    AstJitProgram.compile(table, expressions: _*)
  }

  private[rapids] def getJitProgram(
      expressions: Seq[GpuAstJitExpression],
      table: Table): AstJitProgram = synchronized {
    require(expressions.head eq this, "The first AST JIT expression must own the group program")
    val groupMatches = jitProgramGroup != null && jitProgramGroup.matches(expressions)
    val group = if (groupMatches) {
      jitProgramGroup
    } else {
      GpuAstJitExpression.JitProgramGroup(expressions)
    }
    val schema = GpuAstJitExpression.inputSchema(group, table)
    if (jitProgram == null || !groupMatches || jitProgramSchema != schema) {
      if (GpuAstJitExpression.precompileSubmitted) {
        // If this kernel is still being precompiled, the build below hits the native cache after.
        precompileWaitTime.ns {
          GpuAstJitPrecompiler.awaitInFlight(group.kernelKey(schema.exists(_.nullable)))
        }
      }
      val compiledExpressions = group.expressions.map(_.getCompiledExpression).toArray
      programBuildAttempts += 1
      // A program build may hit the native kernel cache without invoking NVRTC.
      val replacement = programBuildTime.ns {
        withCompileMetrics { compileJitProgram(table, compiledExpressions) }
      }
      val previous = jitProgram
      jitProgram = replacement
      jitProgramGroup = group
      jitProgramSchema = schema
      Option(previous).foreach(_.safeClose())
    } else {
      programCacheHits += 1
    }
    jitProgram
  }

  override protected def releaseAdditionalResources(): Seq[AutoCloseable] = {
    val current = jitProgram
    jitProgram = null
    jitProgramGroup = null
    jitProgramSchema = null
    Option(current).toSeq
  }

  override def toString: String = s"AST_JIT($child)"
}

/**
 * Compiles a Project's AST JIT kernels on executor threads when its task starts, so NVRTC runs
 * while the task reads its input instead of after the first batch arrives. A precompile only
 * fills the process-wide native kernel cache: each task still builds its program from the
 * batch's actual schema, and that build then hits the cache.
 */
private[rapids] object GpuAstJitPrecompiler extends Logging {
  /** A native kernel: the group's expressions and whether it is the multi-output null-aware one. */
  final case class Key(roots: Vector[Expression], nullAware: Boolean)

  /**
   * One kernel compile. Whichever thread claims it first runs it: an executor thread that reaches
   * it in the queue, or a task that needs it before that happens.
   */
  private final class Compile(
      key: Key,
      group: GpuAstJitExpression.JitProgramGroup,
      masked: Boolean) extends Runnable {
    private val claimed = new AtomicBoolean(false)
    val finished = new CompletableFuture[Unit]()

    def claim(): Boolean = claimed.compareAndSet(false, true)

    override def run(): Unit = {
      if (claim()) {
        try {
          compileKernel(group, masked)
          if (compiled.size >= MAX_COMPILED_KEYS) {
            compiled.clear()
          }
          compiled.add(key)
          finished.complete(())
        } catch {
          case t: Throwable =>
            finished.completeExceptionally(t)
            if (!NonFatal(t)) throw t
        } finally {
          inFlight.remove(key, this)
        }
      }
    }
  }

  // Only queued and running compiles: a finished one leaves, and the native cache answers after.
  private val inFlight = new ConcurrentHashMap[Key, Compile]()
  // Kernels precompiled in this process, so later tasks and queries skip lowering them again. The
  // native cache still holds them, so forgetting keys at the cap costs only a cache-hit lowering.
  private val compiled = ConcurrentHashMap.newKeySet[Key]()
  private val MAX_COMPILED_KEYS = 4096
  @volatile private var pool: ThreadPoolExecutor = _

  private def getPool(threads: Int): ThreadPoolExecutor = {
    if (pool == null) {
      synchronized {
        if (pool == null) {
          pool = TrampolineUtil.newDaemonCachedThreadPool("ast-jit-precompile", threads)
        }
      }
    }
    pool
  }

  /**
   * Starts compiling the kernels for every JIT group in `tiers`, for the null masks the plan
   * predicts. Speculative kernels for the other mask presence start only after those finish.
   */
  def submit(
      tiers: Seq[Seq[Expression]],
      multiOutputEnabled: Boolean,
      speculative: Boolean,
      threads: Int): Unit = {
    GpuAstJitExpression.precompileSubmitted = true
    val groups = tiers.flatMap { tier =>
      GpuAstJitExpression.executionGroups(
        tier.flatMap(GpuAstJitExpression.extractTopLevel), multiOutputEnabled)
    }.map(GpuAstJitExpression.JitProgramGroup(_)).filter { group =>
      group.planInputs.forall(input => GpuBatchUtils.isFixedWidth(input.dataType))
    }
    // Plan nullability only predicts masks: file scans report every column nullable, Parquet
    // REQUIRED columns arrive without a mask, and filtered columns can keep one without nulls.
    val predicted = groups.map(group => group -> group.planInputs.exists(_.nullable))
    val pending = predicted.flatMap { case (group, masked) => start(group, masked, threads) }
    val others = predicted.collect {
      case (group, masked) if speculative && group.expressions.size > 1 => group -> !masked
    }
    if (others.nonEmpty) {
      // NVRTC is CPU bound and busier cores run at lower clocks, so a speculative compile running
      // next to a predicted one slows it down.
      CompletableFuture.allOf(pending: _*).whenComplete(new BiConsumer[Void, Throwable] {
        override def accept(ignored: Void, failure: Throwable): Unit =
          others.foreach { case (group, masked) => start(group, masked, threads) }
      })
    }
  }

  /** Starts a compile unless it already ran in this process; returns what to wait for, if any. */
  private def start(
      group: GpuAstJitExpression.JitProgramGroup,
      masked: Boolean,
      threads: Int): Option[CompletableFuture[Unit]] = {
    val key = group.kernelKey(masked)
    if (compiled.contains(key)) {
      None
    } else {
      val compile = new Compile(key, group, masked)
      val existing = inFlight.putIfAbsent(key, compile)
      if (existing == null) {
        getPool(threads).execute(compile)
        Some(compile.finished)
      } else {
        Some(existing.finished)
      }
    }
  }

  private def compileKernel(
      group: GpuAstJitExpression.JitProgramGroup,
      masked: Boolean): Unit = NvtxRegistry.PRECOMPILE_AST_JIT {
    withResource(GpuAstJitExpression.loweringTable(group.planInputs, masked)) { table =>
      withResource(group.expressions.safeMap(_.child.convertToAst(Int.MaxValue).compileJit())) {
        compiled => AstJitProgram.compile(table, compiled: _*).close()
      }
    }
  }

  /**
   * Waits for a running precompile of `key`. One still queued is claimed instead, so the caller
   * compiles it now; tasks already waiting on it then wait in the native compile cache.
   */
  def awaitInFlight(key: Key): Unit = {
    val compile = inFlight.get(key)
    if (compile != null) {
      if (compile.claim()) {
        inFlight.remove(key, compile)
        compile.finished.complete(())
      } else {
        NvtxRegistry.WAIT_PRECOMPILE_AST_JIT {
          try {
            compile.finished.get()
          } catch {
            // The caller's own build reports a real failure.
            case e: ExecutionException => logDebug("AST JIT precompile failed", e.getCause)
          }
        }
      }
    }
  }
}
