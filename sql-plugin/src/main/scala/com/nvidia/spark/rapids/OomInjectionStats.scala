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

import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong

import scala.annotation.tailrec
import scala.collection.JavaConverters._

import com.nvidia.spark.rapids.jni.{CpuRetryOOM, CpuSplitAndRetryOOM, GpuRetryOOM, GpuSplitAndRetryOOM}

/**
 * Test-only accounting of the OOMs that `RmmRapidsRetryIterator` injects, keyed by the code
 * that created the retry block. It answers whether an armed injection actually fired, which
 * the injection itself never reports.
 *
 * The integration tests enable and read it through py4j, so executor-side counts are only
 * visible when executors share the driver JVM (local mode).
 */
object OomInjectionStats {
  final class Counters {
    // injections set up on the JNI side
    val armed = new AtomicLong()
    // armed injections that a retry block caught at least once
    val fired = new AtomicLong()
    // fired injections that were caught by a different (nested) retry block than the arming one
    val firedNested = new AtomicLong()
    // injected exceptions caught by retry blocks, num_ooms > 1 can produce several per arm
    val caught = new AtomicLong()
    // armed injections cleared by the arming block without ever firing
    val expired = new AtomicLong()
    // armed injections replaced by a nested block before firing, the JNI keeps only one per thread
    val overwritten = new AtomicLong()
    // injections skipped because the thread was not associated with a task
    val unassociated = new AtomicLong()
    // attempts that did not arm because an outer block's injection was still pending
    val deferred = new AtomicLong()

    private[OomInjectionStats] def toJson: String =
      Seq("armed" -> armed, "fired" -> fired, "firedNested" -> firedNested, "caught" -> caught,
        "expired" -> expired, "overwritten" -> overwritten, "unassociated" -> unassociated,
        "deferred" -> deferred)
        .map { case (k, v) => s""""$k":${v.get()}""" }
        .mkString("{", ",", "}")
  }

  private final class Pending(val owner: AnyRef, val counters: Counters) {
    var fired = false
    var lastCaught: Throwable = _
  }

  @volatile private var enabled = false
  private val bySite = new ConcurrentHashMap[String, Counters]()
  private val pending = new ThreadLocal[Pending]()

  private val retryFrameworkClasses = Seq(
    "com.nvidia.spark.rapids.RmmRapidsRetryIterator",
    "com.nvidia.spark.rapids.OomInjectionStats")

  def setEnabled(on: Boolean): Unit = {
    enabled = on
  }

  /** The first frame outside of the retry framework, or null when stats are disabled. */
  def callSite(): String = {
    if (!enabled) {
      null
    } else {
      new Throwable().getStackTrace
        .find(e => !retryFrameworkClasses.exists(e.getClassName.startsWith))
        .map(e => s"${e.getFileName}:${e.getLineNumber} ${e.getClassName}.${e.getMethodName}")
        .getOrElse("<unknown>")
    }
  }

  private def countersFor(site: String): Counters =
    bySite.computeIfAbsent(Option(site).getOrElse("<unknown>"), _ => new Counters)

  def armed(owner: AnyRef, site: String): Unit = if (enabled) {
    val prev = pending.get()
    if (prev != null && !prev.fired) {
      prev.counters.overwritten.incrementAndGet()
    }
    val counters = countersFor(site)
    counters.armed.incrementAndGet()
    pending.set(new Pending(owner, counters))
  }

  def unassociated(site: String): Unit = if (enabled) {
    countersFor(site).unassociated.incrementAndGet()
  }

  def deferred(site: String): Unit = if (enabled) {
    countersFor(site).deferred.incrementAndGet()
  }

  def caught(catcher: AnyRef, t: Throwable): Unit = if (enabled) {
    val p = pending.get()
    val injected = findInjected(t)
    if (p != null && injected != null && (p.lastCaught ne injected)) {
      // an outer block can see the same exception again if an inner block rethrows it
      p.lastCaught = injected
      p.counters.caught.incrementAndGet()
      if (!p.fired) {
        p.fired = true
        p.counters.fired.incrementAndGet()
        if (catcher ne p.owner) {
          p.counters.firedNested.incrementAndGet()
        }
      }
    }
  }

  def cleared(owner: AnyRef): Unit = if (enabled) {
    val p = pending.get()
    // otherwise a nested block replaced this injection and already accounted for it
    if (p != null && (p.owner eq owner)) {
      if (!p.fired) {
        p.counters.expired.incrementAndGet()
      }
      pending.remove()
    }
  }

  /** The injected OOM in the cause chain of `t`, or null. Real OOMs have other messages. */
  @tailrec
  def findInjected(t: Throwable, depth: Int = 0): Throwable = t match {
    case null => null
    case _ if depth >= 32 => null
    case _: GpuRetryOOM | _: CpuRetryOOM | _: GpuSplitAndRetryOOM | _: CpuSplitAndRetryOOM
        if t.getMessage != null && t.getMessage.startsWith("injected ") => t
    case _ => findInjected(t.getCause, depth + 1)
  }

  /** JSON object of call site to counters accumulated since the last call. */
  def snapshotAndReset(): String = {
    val entries = bySite.keySet().asScala.toSeq.sorted.flatMap { site =>
      Option(bySite.remove(site)).map { c =>
        val escaped = site.replace("\\", "\\\\").replace("\"", "\\\"")
        s""""$escaped":${c.toJson}"""
      }
    }
    entries.mkString("{", ",", "}")
  }
}
