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

import java.io.PrintWriter
import java.nio.file.{Files, Paths}

import scala.collection.mutable.ArrayBuffer

import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object ProjectAstJitUltimateBench {
  private val spark = SparkSession.active

  private def env(name: String, default: String): String =
    sys.env.get(name).filter(_.nonEmpty).getOrElse(default)

  private def ints(name: String, default: String): Seq[Int] =
    env(name, default).split(',').map(_.trim).filter(_.nonEmpty).map(_.toInt).toSeq

  private def strings(name: String, default: String): Seq[String] =
    env(name, default).split(',').map(_.trim.toLowerCase).filter(_.nonEmpty).toSeq

  private val rows = env("BENCH_ROWS", "100000000").toLong
  private val partitions = env("BENCH_PARTITIONS", "16").toInt
  private val outputCounts = ints("BENCH_OUTPUT_COUNTS", "2,4,8")
  private val depths = ints("BENCH_DEPTHS", "8,16")
  private val workloads = strings("BENCH_WORKLOADS", "independent,shared_prefix")
  private val modes = strings("BENCH_MODES", "gpu_project,legacy_ast,ast_jit")
  private val warmups = env("BENCH_WARMUPS", "2").toInt
  private val iterations = env("BENCH_ITERS", "5").toInt
  private val resultPath = env(
    "BENCH_RESULT_CSV_PATH", s"/tmp/project_ast_jit_${System.currentTimeMillis()}.csv")
  private val nativeBackend = env("LIBCUDF_AST_JIT_BACKEND", "source")
  require(Set("source", "lto", "lto-wrapper-required").contains(nativeBackend),
    s"Unsupported LIBCUDF_AST_JIT_BACKEND: $nativeBackend")
  require(nativeBackend != "lto-wrapper-required" || outputCounts.forall(Set(2, 8)),
    "lto-wrapper-required only has precompiled 3-input wrappers for 2 and 8 outputs")

  private case class Result(
      backend: String,
      mode: String,
      workload: String,
      outputCount: Int,
      depth: Int,
      iteration: Int,
      warmup: Boolean,
      cold: Boolean,
      wallMs: Double,
      checksum: Long,
      astJitExpressions: Int,
      legacyAstProject: Boolean,
      projectNodes: String)

  private val results = ArrayBuffer[Result]()

  private def withMode[T](mode: String)(body: => T): T = {
    val values = mode match {
      case "gpu_project" => Map(
        "spark.rapids.sql.projectAstEnabled" -> "false",
        "spark.rapids.sql.projectAstJitEnabled" -> "false")
      case "legacy_ast" => Map(
        "spark.rapids.sql.projectAstEnabled" -> "true",
        "spark.rapids.sql.projectAstJitEnabled" -> "false")
      case "ast_jit" => Map(
        "spark.rapids.sql.projectAstEnabled" -> "false",
        "spark.rapids.sql.projectAstJitEnabled" -> "true")
      case other => throw new IllegalArgumentException(s"Unknown BENCH_MODES entry: $other")
    }
    val old = values.keys.map(key => key -> spark.conf.getOption(key)).toMap
    values.foreach { case (key, value) => spark.conf.set(key, value) }
    try {
      body
    } finally {
      old.foreach {
        case (key, Some(value)) => spark.conf.set(key, value)
        case (key, None) => spark.conf.unset(key)
      }
    }
  }

  private def makeInput(): DataFrame = {
    val input = spark.range(0, rows, 1, partitions).select(
      col("id").as("a"),
      (col("id") * lit(3L) + lit(1L)).as("b"),
      (col("id") * lit(7L) + lit(3L)).as("c"))
    input.cache()
    input.count()
    input
  }

  private def independentExpression(index: Int, depth: Int): Column = {
    var state = index.toLong + 1L
    var expression = index % 3 match {
      case 0 => col("a") + col("b")
      case 1 => col("b") * col("c")
      case _ => col("c") + col("a")
    }
    var level = 1
    while (level < depth) {
      state = state * 6364136223846793005L + 1442695040888963407L
      val rhs = ((state >>> 32) % 3).toInt match {
        case 0 => col("a")
        case 1 => col("b")
        case _ => col("c")
      }
      expression = if ((state & 1L) == 0L) expression + rhs else expression * rhs
      level += 1
    }
    expression
  }

  private def sharedPrefixExpressions(outputCount: Int, depth: Int): Seq[Column] = {
    var prefix = col("a") + col("b")
    var level = 1
    while (level < depth) {
      prefix = if ((level & 1) == 0) prefix + col("a") else prefix * col("c")
      level += 1
    }
    (0 until outputCount).map { index =>
      var output = prefix
      var suffix = 0
      while (suffix < 3) {
        val rhs = (index + suffix) % 3 match {
          case 0 => col("a")
          case 1 => col("b")
          case _ => col("c")
        }
        output = if (((index >>> suffix) & 1) == 0) output + rhs else output * rhs
        suffix += 1
      }
      output
    }
  }

  private def query(
      input: DataFrame,
      workload: String,
      outputCount: Int,
      depth: Int): DataFrame = {
    val expressions = workload match {
      case "independent" => (0 until outputCount).map(independentExpression(_, depth))
      case "shared_prefix" => sharedPrefixExpressions(outputCount, depth)
      case other => throw new IllegalArgumentException(s"Unknown BENCH_WORKLOADS entry: $other")
    }
    input.select(expressions.zipWithIndex.map { case (expr, index) =>
      expr.as(s"out_$index")
    }: _*)
  }

  private def consume(
      df: DataFrame,
      mode: String,
      outputCount: Int): (Long, Int, Boolean, String) = {
    val aggregate = df.agg(
      count(lit(1)).as("row_count"),
      sum(xxhash64(df.columns.map(col): _*)).as("checksum"))
    val plan = aggregate.queryExecution.executedPlan
    val planText = plan.toString()
    val astJitExpressions = planText.split("AST_JIT", -1).length - 1
    val legacyAstProject = planText.contains("GpuProjectAst")
    mode match {
      case "gpu_project" if astJitExpressions != 0 || legacyAstProject =>
        throw new IllegalStateException(s"Unexpected AST projection in $mode plan:\n$planText")
      case "legacy_ast" if astJitExpressions != 0 || !legacyAstProject =>
        throw new IllegalStateException(s"Expected only legacy AST projection in plan:\n$planText")
      case "ast_jit" if astJitExpressions != outputCount || legacyAstProject =>
        throw new IllegalStateException(
          s"Expected $outputCount AST JIT expressions and no legacy AST projection in plan:\n" +
            planText)
      case _ =>
    }
    val row = aggregate.collect().head
    val checksum = if (row.isNullAt(1)) 0L else row.getLong(1)
    val projectNodes = plan.collect {
      case node if node.nodeName.contains("Project") => node.nodeName
    }.distinct.mkString("+")
    (checksum, astJitExpressions, legacyAstProject, projectNodes)
  }

  private def runOne(
      input: DataFrame,
      mode: String,
      workload: String,
      outputCount: Int,
      depth: Int,
      iteration: Int,
      isWarmup: Boolean): Unit = withMode(mode) {
    val start = System.nanoTime()
    val (checksum, astJitExpressions, legacyAstProject, nodes) =
      consume(query(input, workload, outputCount, depth), mode, outputCount)
    val elapsedMs = (System.nanoTime() - start) / 1e6
    val isCold = iteration == 0 && (isWarmup || warmups == 0)
    val result = Result(nativeBackend, mode, workload, outputCount, depth, iteration,
      isWarmup, isCold, elapsedMs, checksum, astJitExpressions, legacyAstProject, nodes)
    results += result
    println(f"backend=$nativeBackend mode=$mode workload=$workload outputs=$outputCount " +
      f"depth=$depth iteration=$iteration warmup=$isWarmup cold=$isCold wallMs=$elapsedMs%.3f " +
      s"checksum=$checksum astJitExpressions=$astJitExpressions " +
      s"legacyAstProject=$legacyAstProject projects=$nodes")
  }

  private def writeResults(): Unit = {
    val path = Paths.get(resultPath).toAbsolutePath
    Option(path.getParent).foreach { parent =>
      Files.createDirectories(parent)
    }
    val writer = new PrintWriter(path.toFile)
    try {
      writer.println("backend,mode,workload,output_count,depth,iteration,warmup,cold,wall_ms," +
        "checksum,ast_jit_expressions,legacy_ast_project,project_nodes")
      results.foreach { result =>
        writer.println(Seq(result.backend, result.mode, result.workload, result.outputCount,
          result.depth, result.iteration, result.warmup, result.cold, result.wallMs,
          result.checksum, result.astJitExpressions, result.legacyAstProject,
          result.projectNodes).mkString(","))
      }
    } finally {
      writer.close()
    }
    println(s"results=$path")
  }

  def run(): Unit = {
    spark.conf.set("spark.sql.ansi.enabled", "false")
    spark.conf.set("spark.sql.adaptive.enabled", "false")
    println(s"backend=$nativeBackend rows=$rows partitions=$partitions")
    val input = makeInput()
    try {
      workloads.foreach { workload =>
        outputCounts.foreach { outputCount =>
          depths.foreach { depth =>
            modes.foreach { mode =>
              (0 until warmups).foreach { iteration =>
                runOne(input, mode, workload, outputCount, depth, iteration, isWarmup = true)
              }
              (0 until iterations).foreach { iteration =>
                runOne(input, mode, workload, outputCount, depth, iteration, isWarmup = false)
              }
            }
          }
        }
      }
      writeResults()
    } finally {
      input.unpersist(blocking = true)
    }
  }
}

ProjectAstJitUltimateBench.run()
