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

import java.util.concurrent.{ConcurrentHashMap, ConcurrentLinkedQueue}

import scala.collection.mutable

import ai.rapids.cudf.{ColumnVector, DType, Table}
import ai.rapids.cudf.ast.AstJitProgram
import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.jni.LencNative

import org.apache.spark.sql.catalyst.expressions.{Expression, ExprId}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids.{GpuEqualTo, GpuGreaterThanOrEqual, GpuLessThanOrEqual,
  GpuOr}
import org.apache.spark.sql.types.{IntegerType, StringType}
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.unsafe.types.UTF8String

/**
 * Utilities for DICTIONARY32 column flow, Dictionary-Aware AST-JIT rewriting, and
 * executor-level AstJitProgram pooling across Spark tasks.
 */
object GpuDictionaryUtils {
  val DICT32_COLS_CONF = "spark.rapids.sql.dictionary32.columns"
  val PROGRAM_POOL_CONF = "spark.rapids.sql.projectAstJit.programPoolEnabled"

  @volatile private var driverDict32Cols: Set[String] = Set.empty

  def configuredDict32Cols(conf: SQLConf): Set[String] = {
    val raw = conf.getConfString(DICT32_COLS_CONF, "").trim
    if (raw.nonEmpty) {
      val set = raw.split(",").map(_.trim).filter(_.nonEmpty).toSet
      driverDict32Cols = set
      set
    } else {
      driverDict32Cols
    }
  }

  def updateDriverDict32Cols(conf: SQLConf): Set[String] = {
    val raw = conf.getConfString(DICT32_COLS_CONF, "").trim
    val set = if (raw.isEmpty) Set.empty[String] else raw.split(",").map(_.trim).filter(_.nonEmpty).toSet
    driverDict32Cols = set
    set
  }

  def programPoolEnabled(): Boolean =
    SQLConf.get.getConfString(PROGRAM_POOL_CONF, "true").toBoolean

  def hasDict32(cb: ColumnarBatch): Boolean = {
    var i = 0
    val n = cb.numCols()
    while (i < n) {
      cb.column(i) match {
        case gcv: GpuColumnVector if gcv.getBase.getType == DType.DICTIONARY32 => return true
        case _ =>
      }
      i += 1
    }
    false
  }

  /**
   * If `cb` contains any DICTIONARY32 columns, decodes them into temporary STRING columns,
   * runs `f`, and closes all temporary decoded columns. Zero overhead when no DICTIONARY32 exists.
   */
  def withDecodedBatch[T](cb: ColumnarBatch)(f: ColumnarBatch => T): T = {
    if (!hasDict32(cb)) {
      f(cb)
    } else {
      val tempDecoded = mutable.ArrayBuffer.empty[GpuColumnVector]
      try {
        val cols = Array.tabulate[org.apache.spark.sql.vectorized.ColumnVector](cb.numCols()) { i =>
          cb.column(i) match {
            case gcv: GpuColumnVector if gcv.getBase.getType == DType.DICTIONARY32 =>
              val decoded = GpuColumnVector.from(LencNative.decodeDictionary(gcv.getBase), gcv.dataType())
              tempDecoded += decoded
              decoded
            case other => other
          }
        }
        f(new ColumnarBatch(cols, cb.numRows()))
      } finally {
        tempDecoded.safeClose()
      }
    }
  }
}

/**
 * Thread-safe executor-level pool of compiled AstJitProgram instances keyed by
 * (canonical expressions, input schema). Eliminates per-task row_ir lowering and
 * AstJitProgram.compile overhead across tasks on warm runs.
 */
object GpuAstJitProgramPool {
  final case class SchemaCol(ordinal: Int, dtype: DType, nullable: Boolean)
  final case class PoolKey(exprs: Vector[Expression], schema: Vector[SchemaCol])

  private val MAX_KEYS = 256
  private val MAX_PER_KEY = 8
  private val pools = new ConcurrentHashMap[PoolKey, ConcurrentLinkedQueue[AstJitProgram]]()

  Runtime.getRuntime.addShutdownHook(new Thread(() => {
    val it = pools.values().iterator()
    while (it.hasNext) {
      val q = it.next()
      var p = q.poll()
      while (p != null) {
        p.safeClose()
        p = q.poll()
      }
    }
    pools.clear()
  }))

  def withProgram[T](
      key: PoolKey,
      build: => AstJitProgram)(body: AstJitProgram => T): (T, Boolean) = {
    val queue = pools.computeIfAbsent(key, _ => new ConcurrentLinkedQueue[AstJitProgram]())
    val existing = queue.poll()
    val hit = existing != null
    val prog = if (hit) existing else build
    try {
      (body(prog), hit)
    } finally {
      if (pools.size() > MAX_KEYS || queue.size() >= MAX_PER_KEY) {
        prog.safeClose()
      } else {
        queue.offer(prog)
      }
    }
  }
}

/**
 * Dictionary-Aware AST-JIT rewriting:
 * When an input column in `table` is DICTIONARY32 (from Parquet #24798 or .lenc) and every
 * reference to it in an AST-JIT expression group is a string literal comparison
 * (EqualTo / InSet / IsNull / IsNotNull), remaps dictionary indices to canonical query-literal
 * INT32 codes on GPU (<0.1ms) and rewrites the AST to compare INT32 codes in registers!
 */
object GpuDictAwareAstJit {
  final case class PreparedEval(
      rewrittenChildren: Vector[GpuExpression],
      evalTable: Table,
      tempCols: Seq[ColumnVector],
      ownsTable: Boolean) extends AutoCloseable {
    override def close(): Unit = {
      if (ownsTable) {
        evalTable.safeClose()
      }
      tempCols.safeClose()
    }
  }

  private def hasDict32Referenced(
      referencedOrdinals: Seq[Int],
      table: Table): Boolean = {
    referencedOrdinals.exists(ord => table.getColumn(ord).getType == DType.DICTIONARY32)
  }

  /**
   * Checks whether every occurrence of `ordinal` in `expr` is strictly inside supported
   * dictionary-predicate forms (EqualTo literal, InSet string literals, IsNull, IsNotNull),
   * and collects all compared String literals into `literalsOut`.
   */
  private[rapids] def collectDictPredicates(
      expr: Expression,
      ordinal: Int,
      literalsOut: mutable.HashSet[String]): Boolean = {
    expr match {
      case GpuEqualTo(ref: GpuBoundReference, GpuLiteral(s: UTF8String, StringType))
          if ref.ordinal == ordinal =>
        literalsOut += s.toString
        true
      case GpuEqualTo(GpuLiteral(s: UTF8String, StringType), ref: GpuBoundReference)
          if ref.ordinal == ordinal =>
        literalsOut += s.toString
        true
      case GpuInSet(ref: GpuBoundReference, list, _) if ref.ordinal == ordinal =>
        val ok = list.forall {
          case null => true
          case _: UTF8String => true
          case _ => false
        }
        if (ok) {
          list.foreach {
            case s: UTF8String => literalsOut += s.toString
            case _ =>
          }
        }
        ok
      case GpuIsNull(ref: GpuBoundReference) if ref.ordinal == ordinal => true
      case GpuIsNotNull(ref: GpuBoundReference) if ref.ordinal == ordinal => true
      case ref: GpuBoundReference if ref.ordinal == ordinal =>
        // Unsupported direct use of the string column (e.g. pass-through, substr, concat)
        false
      case other =>
        other.children.forall(collectDictPredicates(_, ordinal, literalsOut))
    }
  }

  private[rapids] def rewriteExprForDict(
      expr: Expression,
      ordinal: Int,
      litToId: Map[String, Int],
      totalLits: Int): Expression = {
    def makeCodeRef(ref: GpuBoundReference): GpuBoundReference =
      GpuBoundReference(ref.ordinal, IntegerType, ref.nullable)(ExprId(ref.exprId.id), ref.name)

    expr match {
      case GpuEqualTo(ref: GpuBoundReference, GpuLiteral(s: UTF8String, StringType))
          if ref.ordinal == ordinal =>
        GpuEqualTo(makeCodeRef(ref), GpuLiteral(litToId(s.toString), IntegerType))
      case GpuEqualTo(GpuLiteral(s: UTF8String, StringType), ref: GpuBoundReference)
          if ref.ordinal == ordinal =>
        GpuEqualTo(makeCodeRef(ref), GpuLiteral(litToId(s.toString), IntegerType))
      case GpuInSet(ref: GpuBoundReference, list, _) if ref.ordinal == ordinal =>
        val codeRef = makeCodeRef(ref)
        val hasNull = list.contains(null)
        val ids = list.collect { case s: UTF8String => litToId(s.toString) }.distinct.sorted
        val nonNullPred: Expression = if (ids.isEmpty) {
          GpuLiteral(false, org.apache.spark.sql.types.BooleanType)
        } else if (ids.head == 0 && ids.last == ids.length - 1) {
          // Contiguous [0..K-1]: if covering all literals [0..totalLits-1], single check `code >= 0`!
          if (ids.length == totalLits) {
            GpuGreaterThanOrEqual(codeRef, GpuLiteral(0, IntegerType))
          } else {
            org.apache.spark.sql.rapids.GpuAnd(
              GpuGreaterThanOrEqual(codeRef, GpuLiteral(0, IntegerType)),
              GpuLessThanOrEqual(codeRef, GpuLiteral(ids.last, IntegerType)))
          }
        } else {
          ids.map(id => GpuEqualTo(codeRef, GpuLiteral(id, IntegerType)): Expression)
            .reduceLeft((a, b) => GpuOr(a, b))
        }
        if (hasNull) {
          GpuOr(nonNullPred, GpuEqualTo(codeRef, GpuLiteral(null, IntegerType)))
        } else {
          nonNullPred
        }
      case GpuIsNull(ref: GpuBoundReference) if ref.ordinal == ordinal =>
        GpuIsNull(makeCodeRef(ref))
      case GpuIsNotNull(ref: GpuBoundReference) if ref.ordinal == ordinal =>
        GpuIsNotNull(makeCodeRef(ref))
      case other =>
        other.mapChildren(rewriteExprForDict(_, ordinal, litToId, totalLits))
    }
  }

  def prepare(
      expressions: Seq[GpuAstJitExpression],
      referencedOrdinals: Seq[Int],
      table: Table): PreparedEval = {
    if (!hasDict32Referenced(referencedOrdinals, table)) {
      return PreparedEval(expressions.map(_.child).toVector, table, Seq.empty, ownsTable = false)
    }

    var currentChildren = expressions.map(_.child).toVector
    val replacements = mutable.HashMap.empty[Int, ColumnVector]
    val tempCols = mutable.ArrayBuffer.empty[ColumnVector]

    try {
      referencedOrdinals.foreach { ord =>
        val col = table.getColumn(ord)
        if (col.getType == DType.DICTIONARY32) {
          val litSet = mutable.HashSet.empty[String]
          val eligible = currentChildren.forall(collectDictPredicates(_, ord, litSet)) && litSet.nonEmpty
          if (eligible) {
            val sortedLits = litSet.toArray.sorted
            val litToId = sortedLits.zipWithIndex.toMap
            currentChildren = currentChildren.map { c =>
              rewriteExprForDict(c, ord, litToId, sortedLits.length).asInstanceOf[GpuExpression]
            }
            val remapped = LencNative.remapDictionaryToCanonical(col, sortedLits)
            tempCols += remapped
            replacements(ord) = remapped
          } else {
            val decoded = LencNative.decodeDictionary(col)
            tempCols += decoded
            replacements(ord) = decoded
          }
        }
      }

      val numCols = table.getNumberOfColumns
      val tableCols = Array.tabulate[ColumnVector](numCols) { i =>
        replacements.getOrElse(i, table.getColumn(i))
      }
      val newTable = new Table(tableCols: _*)
      PreparedEval(currentChildren, newTable, tempCols.toSeq, ownsTable = true)
    } catch {
      case t: Throwable =>
        tempCols.safeClose()
        throw t
    }
  }
}
