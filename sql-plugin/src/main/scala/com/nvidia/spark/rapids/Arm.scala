/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths, StandardOpenOption}
import java.util.Collections
import java.util.concurrent.ConcurrentHashMap

import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.util.control.{ControlThrowable, NonFatal}

import com.nvidia.spark.rapids.RapidsPluginImplicits._
import com.nvidia.spark.rapids.RmmRapidsRetryIterator.SizeProvider

import org.apache.spark.sql.vectorized._

/** Implementation of the automatic-resource-management pattern */
object Arm extends ArmScalaSpecificImpl {

  private[this] val retryCoverageEnabled: Boolean = true

  private[this] val retryCoverageLogPath: Path =
    Paths.get("/home/haoyangl/spark-rapids", "retry-coverage.log")

  private[this] val retryCoverageDedup: java.util.Set[String] =
    Collections.newSetFromMap(new ConcurrentHashMap[String, java.lang.Boolean]())

  private[this] def isRetryCoverageCandidate(res: Any): Boolean = res match {
    case _: SizeProvider => true
    case _: ColumnarBatch => true
    // TODO: add more cases here
    // case _: ColumnVector => true
    case _ => false
  }

  private[this] def recordRetryCoverage(callingFrame: StackTraceElement): Unit = {
    try {
      val stackKey = s"${callingFrame.getClassName}.${callingFrame.getMethodName}:${callingFrame.getLineNumber}"
      if (retryCoverageDedup.add(stackKey)) {
        val msg = s"missing withRetry at $stackKey\n"
        Files.createDirectories(retryCoverageLogPath.getParent)
        Files.write(retryCoverageLogPath, msg.getBytes(StandardCharsets.UTF_8),
          StandardOpenOption.CREATE, StandardOpenOption.APPEND, StandardOpenOption.WRITE)
      }
    } catch {
      case NonFatal(e) =>
        // Best effort; do not let coverage logging break normal behavior
        if (retryCoverageDedup.add("logging-error")) {
          System.err.println(s"Failed to record retry coverage gap: ${e.getMessage}")
        }
    }
  }

  private[this] def checkRetryCoverage(res: Any): Unit = {
    if (!retryCoverageEnabled || res == null || !isRetryCoverageCandidate(res)) {
      ()
    } else {
      val stack = Thread.currentThread().getStackTrace
      val hasRetry = stack.exists { ste =>
        val method = ste.getMethodName
        method != null && method.toLowerCase.contains("withretry")
      }
      if (!hasRetry) {
        val firstCaller = stack.dropWhile { ste =>
          val cls = ste.getClassName
          cls.startsWith("com.nvidia.spark.rapids.Arm$") || cls == "java.lang.Thread"
        }.headOption
        firstCaller.foreach(recordRetryCoverage)
      }
    }
  }

  /** Executes the provided code block and then closes the resource */
  def withResource[T <: AutoCloseable, V](r: T)(block: T => V): V = {
    checkRetryCoverage(r)
    try {
      block(r)
    } finally {
      r.safeClose()
    }
  }

  /** Executes the provided code block and then closes the Option[resource] */
  def withResource[T <: AutoCloseable, V](r: Option[T])(block: Option[T] => V): V = {
    r.foreach(checkRetryCoverage)
    try {
      block(r)
    } finally {
      r.foreach(_.safeClose())
    }
  }

  /** Executes the provided code block and then closes the sequence of resources */
  def withResource[T <: AutoCloseable, V](r: Seq[T])(block: Seq[T] => V): V = {
    r.headOption.foreach(checkRetryCoverage)
    try {
      block(r)
    } finally {
      r.safeClose()
    }
  }

  /** Executes the provided code block and then closes the array of resources */
  def withResource[T <: AutoCloseable, V](r: Array[T])(block: Array[T] => V): V = {
    r.headOption.foreach(checkRetryCoverage)
    try {
      block(r)
    } finally {
      r.safeClose()
    }
  }

  /** Executes the provided code block and then closes the array buffer of resources */
  def withResource[T <: AutoCloseable, V](r: ArrayBuffer[T])(block: ArrayBuffer[T] => V): V = {
    r.headOption.foreach(checkRetryCoverage)
    try {
      block(r)
    } finally {
      r.safeClose()
    }
  }

  /** Executes the provided code block and then closes the queue of resources */
  def withResource[T <: AutoCloseable, V](r: mutable.Queue[T])(block: mutable.Queue[T] => V): V = {
    r.headOption.foreach(checkRetryCoverage)
    try {
      block(r)
    } finally {
      r.safeClose()
    }
  }

  /** Executes the provided code block and then closes the value if it is AutoCloseable */
  def withResourceIfAllowed[T, V](r: T)(block: T => V): V = {
    r match {
      case c: AutoCloseable => checkRetryCoverage(c)
      case scala.util.Left(c: AutoCloseable) => checkRetryCoverage(c)
      case scala.util.Right(c: AutoCloseable) => checkRetryCoverage(c)
      case _ =>
    }
    try {
      block(r)
    } finally {
      r match {
        case c: AutoCloseable => c.safeClose()
        case scala.util.Left(c: AutoCloseable) => c.safeClose()
        case scala.util.Right(c: AutoCloseable) => c.safeClose()
        case _ => //NOOP
      }
    }
  }

  /** Executes the provided code block, closing the resource only if an exception occurs */
  def closeOnExcept[T <: AutoCloseable, V](r: T)(block: T => V): V = {
    try {
      block(r)
    } catch {
      case t: ControlThrowable =>
        // Don't close for these cases..
        throw t
      case t: Throwable =>
        r.safeClose(t)
        throw t
    }
  }

  /** Executes the provided code block, closing the resources only if an exception occurs */
  def closeOnExcept[T <: AutoCloseable, V](r: Array[T])(block: Array[T] => V): V = {
    try {
      block(r)
    } catch {
      case t: ControlThrowable =>
        // Don't close for these cases..
        throw t
      case t: Throwable =>
        r.safeClose(t)
        throw t
    }
  }

  /** Executes the provided code block, closing the resources only if an exception occurs */
  def closeOnExcept[T <: AutoCloseable, V](r: ArrayBuffer[T])(block: ArrayBuffer[T] => V): V = {
    try {
      block(r)
    } catch {
      case t: ControlThrowable =>
        // Don't close for these cases..
        throw t
      case t: Throwable =>
        r.safeClose(t)
        throw t
    }
  }

  /** Executes the provided code block, closing the resources only if an exception occurs */
  def closeOnExcept[T <: AutoCloseable, V](r: ListBuffer[T])(block: ListBuffer[T] => V): V = {
    try {
      block(r)
    } catch {
      case t: ControlThrowable =>
        // Don't close for these cases..
        throw t
      case t: Throwable =>
        r.safeClose(t)
        throw t
    }
  }


  /** Executes the provided code block, closing the resources only if an exception occurs */
  def closeOnExcept[T <: AutoCloseable, V](r: mutable.Queue[T])(block: mutable.Queue[T] => V): V = {
    try {
      block(r)
    } catch {
      case t: ControlThrowable =>
        // Don't close for these cases..
        throw t
      case t: Throwable =>
        r.safeClose(t)
        throw t
    }
  }

  /** Executes the provided code block, closing the resources only if an exception occurs */
  def closeOnExcept[T <: AutoCloseable, V](r: Option[T])(block: Option[T] => V): V = {
    try {
      block(r)
    } catch {
      case t: ControlThrowable =>
        // Don't close for these cases..
        throw t
      case t: Throwable =>
        r.foreach(_.safeClose(t))
        throw t
    }
  }

  /** Executes the provided code block and then closes the resource */
  def withResource[T <: AutoCloseable, V](h: CloseableHolder[T])
      (block: CloseableHolder[T] => V): V = {
    try {
      block(h)
    } finally {
      h.close()
    }
  }
}

class CloseableHolder[T <: AutoCloseable](var t: T) {
  def setAndCloseOld(newT: T): Unit = {
    val oldT = t
    t = newT
    oldT.close()
  }

  def get: T = t

  def close(): Unit = t.close()
}
