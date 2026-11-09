/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

import java.util
import java.util.Map
import java.util.concurrent.{ConcurrentHashMap, LinkedBlockingQueue, TimeUnit}
import java.util.concurrent.atomic.AtomicLong

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

import ai.rapids.cudf.{NvtxColor, NvtxUniqueRange, Rmm}
import com.nvidia.spark.rapids.ScalableTaskCompletion.onTaskCompletion
import com.nvidia.spark.rapids.jni.{RmmSpark, TaskPriority}
import com.nvidia.spark.rapids.metrics.GpuBubbleTimerManager
import com.nvidia.spark.rapids.spill.SpillFramework

import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids.GpuTaskMetrics

/**
 * The result of trying to acquire a semaphore could be
 * `SemaphoreAcquired` or `AcquireFailed`.
 */
sealed trait TryAcquireResult

/**
 * The Semaphore was successfully acquired.
 */
case object SemaphoreAcquired extends TryAcquireResult

/**
 * To acquire the semaphore this thread would have to block.
 * @param numWaitingTasks the number of tasks waiting at the time the request was made.
 *                        Note that this can change very quickly.
 */
case class AcquireFailed(numWaitingTasks: Int) extends TryAcquireResult

private object GpuTaskMemoryEstimator {
  private val TIME_WINDOW: Double = TimeUnit.MILLISECONDS.toNanos(100).toDouble
}

private object GpuSemaphoreDiagnostics extends Logging {
  def enabled: Boolean = log.isDebugEnabled

  def emit(event: String, stageId: Int, taskId: Long, lease: Long, permits: Long,
      demandBytes: Long, demandTimeNanos: Long, stage: GpuStageMemoryEstimator): Unit = {
    if (enabled) {
      val now = System.nanoTime()
      logDebug(s"GPU_SEMAPHORE_DIAG event=$event stage=$stageId task=$taskId lease=$lease " +
        s"timeNs=$now usedPermits=$permits grantDemandBytes=$demandBytes " +
        s"grantEstimateTimeNs=$demandTimeNanos ${stage.diagnosticSnapshot(taskId, now)} " +
        s"rmmBytes=${Rmm.getTotalBytesAllocated}")
    }
  }
}

class GpuTaskMemoryEstimator(val stageId: Int,
                             val taskId: Long,
                             val defaultEstimate: Long,
                             val allowDynamicUpdate: Boolean) {
  import GpuTaskMemoryEstimator._

  private val startTimeNanos: Long = System.nanoTime()
  private var totalTimeLost: Long = 0
  private var maxMemory: Long = 0
  private var diagnosticBornRawLost: Long = 0

  def recordDiagnosticBirth(): Unit = {
    diagnosticBornRawLost = RmmSpark.getTotalBlockedOrLostTime(taskId)
  }

  def diagnosticSnapshot(now: Long): String = {
    s"taskDefaultBytes=$defaultEstimate taskAgeNs=${now - startTimeNanos} " +
      s"cachedEstimatorMaxBytes=$maxMemory cachedEstimatorLostNs=$totalTimeLost " +
      s"maxFootprintBytes=${RmmSpark.getMaxGpuTaskMemory(taskId)} " +
      s"rawLostNs=${RmmSpark.getTotalBlockedOrLostTime(taskId)} " +
      s"bornRawLostNs=$diagnosticBornRawLost"
  }

  def update(timeLost: Long, memory: Long): Unit = {
    // timeLost and maxMemory are not reset when they are read
    totalTimeLost = timeLost
    // Taking a max here, just to be cautious, but it should be a noop
    maxMemory = math.max(memory, maxMemory)
  }

  def estimate(): Long = {
    if (!allowDynamicUpdate) {
      return defaultEstimate
    }
    if (maxMemory > defaultEstimate) {
      maxMemory
    } else {
      // Combine the default estimate and the current measured
      // amount proportionally based on how long we have been running
      // up to a given max time window
      val currentTime = System.nanoTime()
      val diff = currentTime - startTimeNanos - totalTimeLost
      val activePctOfWindow = math.max(math.min(diff / TIME_WINDOW, 1.0), 0.0)
      val pctForDefault = 1.0 - activePctOfWindow
      (defaultEstimate * pctForDefault).toLong + (activePctOfWindow * maxMemory).toLong
    }
  }
}

class StatEstimator(minEntries:Int, defaultValue: Double) {
  require(minEntries > 0, "Minimum entries must be positive")
  private val values = ArrayBuffer.empty[Double]

  def size: Int = values.size

  def add(value: Double): Unit = {
    values += value
    // We have a maximum of 200 entries, so we don't use too much
    // memory for very large stages
    if (values.length > 200) {
      values.remove(0, values.length - 200)
    }
  }

  def percentile(p: Double, others: ArrayBuffer[Double]): Double = {
    require(p >= 0 && p <= 1, "Percentile must be between 0-1")

    if (values.isEmpty && others.isEmpty) return defaultValue

    val combined = values ++ others
    while (combined.length < minEntries) {
      combined += defaultValue
    }
    val padded = combined.sorted

    val n = padded.size
    val pos = p * (n + 1)

    pos match {
      case _ if pos < 1 => padded.head
      case _ if pos >= n => padded.last
      case _ =>
        val k = math.floor(pos).toInt
        val lower = padded(k - 1)
        val upper = padded(k)
        val fraction = pos - k
        lower + fraction * (upper - lower)
    }
  }
}

class GpuStageMemoryEstimator(val stageId: Int, 
                              private val defaultEstimate: Long, 
                              val allowDynamicUpdate: Boolean) {
  private val stats = new StatEstimator(4, defaultEstimate.toDouble)
  private val activeTasks = new util.HashMap[Long, GpuTaskMemoryEstimator]()
  private var diagnosticCachedP80: Long = defaultEstimate

  override def toString: String = synchronized {
    s"Stage $stageId Estimator ${activeTasks.keySet()}"
  }

  def addTaskIfNeeded(taskId: Long): Unit = synchronized {
    activeTasks.computeIfAbsent(taskId: Long, _ => {
      val newDefaultEstimate = estimate()
      val estimator =
        new GpuTaskMemoryEstimator(stageId, taskId, newDefaultEstimate, allowDynamicUpdate)
      if (GpuSemaphoreDiagnostics.enabled) {
        estimator.recordDiagnosticBirth()
      }
      estimator
    })
  }

  def diagnosticSnapshot(taskId: Long, now: Long): String = synchronized {
    val task = activeTasks.get(taskId)
    val taskSnapshot = if (task == null) "taskEstimator=absent" else task.diagnosticSnapshot(now)
    s"stageCachedP80Bytes=$diagnosticCachedP80 stageActiveTasks=${activeTasks.size()} " +
      s"stageCompletedSamples=${stats.size} $taskSnapshot"
  }

  def taskDone(taskId: Long): Unit = synchronized {
    val estimator = activeTasks.remove(taskId)
    if (estimator != null) {
      val maxMemory = RmmSpark.getMaxGpuTaskMemory(taskId)
      if (maxMemory > 0) {
        stats.add(maxMemory.toDouble)
      }
    }
  }

  private def updateEstimates(): Unit = synchronized {
    activeTasks.forEach( (taskId, estimator) => {
      val maxMemory = RmmSpark.getMaxGpuTaskMemory(taskId)
      val timeLost = RmmSpark.getTotalBlockedOrLostTime(taskId)
      estimator.update(timeLost, maxMemory)
    })
  }

  def estimate(): Long = synchronized {
    if (!allowDynamicUpdate) {
      return defaultEstimate
    }

    updateEstimates()
    val active = ArrayBuffer.empty[Double]
    activeTasks.values().forEach(estimator => {
      active += estimator.estimate().toDouble
    })
    val result = stats.percentile(0.8, active).toLong
    if (GpuSemaphoreDiagnostics.enabled) {
      diagnosticCachedP80 = result
    }
    result
  }
}

/**
 * Admission by a live device memory ledger instead of a pure permit count.
 *
 * Permits are the memory a task is estimated to need while it holds the semaphore, but what is
 * really on the device also includes memory that tasks keep after they release the semaphore
 * (spillable batches, build tables, partial aggregates, ...) and holders that outgrow their
 * estimate. The ledger admits a task only if
 *
 *   allocated + sum over holders of max(0, reserved - resident) + max(0, reserve - resident)
 *     <= capacityFraction * pool size
 *
 * where allocated is every byte currently allocated through RMM, resident is what a task has
 * currently allocated (exact, from the JNI allocation ledger) and reserved is its permits. So
 * memory a task keeps across a release stays charged, and a task that comes back only needs its
 * remaining growth. A task is always admitted when nobody holds the semaphore, so memory that
 * waiting tasks keep cannot deadlock admission; the holder's allocations make it spill as usual.
 * With reclaim enabled, the head of the queue proactively spills the deficit from the device
 * spill store instead of waiting for holders to hit an OOM.
 */
class GpuMemoryLedger(val capacityFraction: Double, val pollMillis: Long, val reclaim: Boolean)
    extends SemaphoreAdmissionPolicy with Logging {
  private val PERMIT_BYTES = 32L * 1024 * 1024
  @volatile private var lastDeficit = 0L
  private val admitted = new AtomicLong()
  private val denied = new AtomicLong()
  private val unavailable = new AtomicLong()
  private val reclaimed = new AtomicLong()
  private val reclaimCalls = new AtomicLong()
  @volatile private var maxAllocated = 0L
  @volatile private var lastLog = System.nanoTime()

  private def capacity: Long = (GpuDeviceManager.getMemorySize * capacityFraction).toLong

  // Called with the semaphore lock held.
  override def admits(holders: util.HashMap[Long, Long], taskId: Long,
      numPermits: Long): Option[Boolean] = {
    if (holders.isEmpty) {
      admitted.incrementAndGet()
      return Some(true)
    }
    val n = holders.size() + 1
    val ids = new Array[Long](n)
    val residentN = new Array[Long](n)
    var i = 0
    holders.forEach { (t, _) =>
      ids(i) = t
      i += 1
    }
    ids(n - 1) = taskId
    val allocated = RmmSpark.getGpuMemoryLedger(ids, residentN)
    if (allocated < 0) {
      unavailable.incrementAndGet()
      return None
    }
    var headroom = 0L
    i = 0
    holders.forEach { (_, permits) =>
      headroom += math.max(0L, permits * PERMIT_BYTES - residentN(i))
      i += 1
    }
    val need = math.max(0L, numPermits * PERMIT_BYTES - residentN(n - 1))
    val committed = allocated + headroom + need
    if (allocated > maxAllocated) {
      maxAllocated = allocated
    }
    val ok = committed <= capacity
    if (ok) {
      admitted.incrementAndGet()
      lastDeficit = 0L
    } else {
      denied.incrementAndGet()
      lastDeficit = committed - capacity
    }
    maybeLog()
    Some(ok)
  }

  override def onStillBlocked(): Unit = {
    val deficit = lastDeficit
    if (reclaim && deficit > 0) {
      reclaimCalls.incrementAndGet()
      val spilled = SpillFramework.stores.deviceStore.spill(deficit)
      reclaimed.addAndGet(spilled)
    }
    maybeLog()
  }

  private def maybeLog(): Unit = {
    val now = System.nanoTime()
    if (now - lastLog > TimeUnit.SECONDS.toNanos(10)) {
      lastLog = now
      logInfo(s"GPU_MEMORY_LEDGER admitted=${admitted.get} denied=${denied.get} " +
        s"unavailable=${unavailable.get} reclaimCalls=${reclaimCalls.get} " +
        s"reclaimedBytes=${reclaimed.get} maxAllocatedBytes=$maxAllocated capacity=$capacity")
    }
  }
}

object GpuMemoryLedger {
  def fromConf(conf: RapidsConf): Option[GpuMemoryLedger] = {
    if (conf.get(RapidsConf.GPU_SEMAPHORE_MEMORY_LEDGER)) {
      Some(new GpuMemoryLedger(conf.get(RapidsConf.GPU_SEMAPHORE_MEMORY_LEDGER_CAPACITY_FRACTION),
        conf.get(RapidsConf.GPU_SEMAPHORE_MEMORY_LEDGER_POLL_MS),
        conf.get(RapidsConf.GPU_SEMAPHORE_MEMORY_LEDGER_RECLAIM)))
    } else {
      None
    }
  }
}

object GpuSemaphore {
  // DO NOT ACCESS DIRECTLY!  Use `getInstance` instead.
  @volatile private var instance: GpuSemaphore = _

  private def getInstance: GpuSemaphore = {
    if (instance == null) {
      GpuSemaphore.synchronized {
        // The instance is trying to be used before it is initialized.
        // Since we don't have access to a configuration object here,
        // default to only one task per GPU behavior.
        if (instance == null) {
          initialize(0)
        }
      }
    }
    instance
  }

  /**
   * Initializes the GPU task semaphore.
   */
  def initialize(maxConcurrentGpuTasksLimit: Int): Unit =
    initialize(maxConcurrentGpuTasksLimit, None)

  def initialize(maxConcurrentGpuTasksLimit: Int,
      ledger: Option[GpuMemoryLedger]): Unit = synchronized {
    if (instance != null) {
      throw new IllegalStateException("already initialized")
    }
    instance = new GpuSemaphore(maxConcurrentGpuTasksLimit, ledger)
  }

  /**
   * Get the last time this task acquired & released the semaphore, which can be used to track
   * sophisticated metrics, such as I/O time with GpuSemaphore held.
   */
  def getLastSemAcqAndRelTime(context: TaskContext): (Long, Long) = {
    getInstance.lastSemAcqAndRelTime(context)
  }

  /**
   * A thread may try to acquire the semaphore without blocking on it. NOTE: A task completion
   * listener will automatically be installed to ensure the semaphore is always released by the
   * time the task completes.
   */
  def tryAcquire(context: TaskContext): TryAcquireResult = {
    if (context != null) {
      getInstance.tryAcquire(context)
    } else {
      // For unit tests that might try with no context
      SemaphoreAcquired
    }
  }

  /**
   * Tasks must call this when they begin to use the GPU.
   * If the task has not already acquired the GPU semaphore then it is acquired,
   * blocking if necessary.
   * NOTE: A task completion listener will automatically be installed to ensure
   *       the semaphore is always released by the time the task completes.
   */
  def acquireIfNecessary(context: TaskContext): Unit = {
    if (context != null) {
      getInstance.acquireIfNecessary(context)
    }
  }

  /**
   * Tasks must call this when they are finished using the GPU.
   */
  def releaseIfNecessary(context: TaskContext): Unit = {
    if (context != null) {
      getInstance.releaseIfNecessary(context)
    }
  }

  /**
   * Dumps the stack traces for any tasks that have accessed the GPU semaphore
   * and have not completed. The output includes whether the task has the GPU semaphore
   * held at the time of the stack trace.
   */
  def dumpActiveStackTracesToLog(): Unit = {
    getInstance.dumpActiveStackTracesToLog()
  }

  /**
   * Uninitialize the GPU semaphore.
   * NOTE: This does not wait for active tasks to release!
   */
  def shutdown(): Unit = synchronized {
    if (instance != null) {
      instance.shutdown()
      instance = null
    }
  }

  // For now we are going to have one permit for each 32 MiB of GPU memory.
  private val PERMIT_MEMORY_SIZE: Long = 32L * 1024 * 1024

  def memToPermits(memory: Long): Long = math.max(1, memory / PERMIT_MEMORY_SIZE)

  def memToPermitsWithMax(memory: Long): Long = math.min(computeMaxPermits(), memToPermits(memory))

  private def computeMaxPermits(): Long = memToPermits(GpuDeviceManager.getMemorySize)

  private def isDynamicEnabled(conf: SQLConf): Boolean = {
    val dynamicStr = conf.getConfString(RapidsConf.DYNAMIC_CONCURRENT_GPU_TASKS.key, null)
    Option(dynamicStr)
      .map(ConfHelper.toBoolean(_, RapidsConf.DYNAMIC_CONCURRENT_GPU_TASKS.key))
      .getOrElse(RapidsConf.DYNAMIC_CONCURRENT_GPU_TASKS.defaultValue)
  }

  private def computeDefaultMemory(conf: SQLConf): Long = {
    val totalMemory = GpuDeviceManager.getMemorySize
    val concurrentInt: Integer = RapidsConf.CONCURRENT_GPU_TASKS.get(conf).getOrElse {
      val batchBytes = RapidsConf.GPU_BATCH_SIZE_BYTES.get(conf)
      math.max(1, math.min(4, totalMemory / (4 * batchBytes))).toInt
    }

    // Just to be cautious we are going to ask for at least 1 byte as the default
    // This should never happen in practice, but we want to be safe
    math.max(totalMemory / math.max(concurrentInt, 1), 1)
  }
}

/**
 * This represents the state associated with a given task. A task can have multiple threads
 * associated with it. That tends to happen when there is a UDF in an external language
 * a.k.a python.  In that case a writer thread is created to feed the python process and
 * the original thread is used as a reader thread that pulls data from the python process.
 * For the GPU semaphore to avoid deadlocks we either allow all threads associated with a task
 * on the GPU or none of them. But this requires coordination to block all of them or wake up
 * all of them. That is the primary job of this class.
 *
 * It should be noted that there is no special coordination when releasing the semaphore. This
 * can result in one thread running on the GPU when it thinks it has the semaphore, but does
 * not. As the semaphore is used as a first line of defense to avoid using too much GPU memory
 * this is considered to be okay as there are other mechanisms in place, and it should be rather
 * rare.
 */
private final class SemaphoreTaskInfo(val stageId: Int, val taskAttemptId: Long,
                                      memoryEstimator: GpuStageMemoryEstimator,
                                      bubbleTimerMgr: GpuBubbleTimerManager) extends Logging {
  /**
   * This holds threads that are not on the GPU yet. Most of the time they are
   * blocked waiting for the semaphore to let them on, but it may hold one
   * briefly even when this task is holding the semaphore. This is a queue
   * mostly to give us a simple way to elect one thread to block on the semaphore
   * while the others will block with a call to `wait`. There should typically be
   * very few threads in here, if any.
   */
  private val blockedThreads = new LinkedBlockingQueue[Thread]()
  /**
   * All threads that are currently active on the GPU. This is mostly used for
   * debugging. It is a `Set` to avoid duplicates, not for performance because there
   * should be very few in here at a time.
   */
  private val activeThreads = new util.LinkedHashSet[Thread]()
  private var permitsUsed: Long = 0;
  private var diagnosticLease: Long = 0
  private var diagnosticDemandBytes: Long = 0
  private var diagnosticDemandTimeNanos: Long = 0
  private lazy val trackSemaphore = RapidsConf.TRACE_TASK_GPU_OWNERSHIP.get(SQLConf.get)
  /**
   * If this task holds the GPU semaphore or not.
   */
  private var hasSemaphore = false
  @volatile private var lastAcquired: Long = 0L
  @volatile private var lastReleased: Long = 0L

  type GpuBackingSemaphore = PrioritySemaphore[Long]

  var nvtxRange: Option[NvtxUniqueRange] = None

  /**
   * Does this task have the GPU semaphore or not. Be careful because it can change at
   * any point in time. So only use it for logging.
   */
  def isHoldingSemaphore: Boolean = synchronized {
    hasSemaphore
  }

  /**
   * Get the last time this task acquired & released the semaphore for the recording of
   * sophisticated metrics, such as I/O time with GpuSemaphore held.
   */
  def lastAcquireAndReleaseTime: (Long, Long) = (lastAcquired, lastReleased)

  /**
   * Get the list of threads currently running on the GPU Semaphore for this task. Be
   * careful because these can change at any point in time. So only use it for logging.
   */
  def getActiveThreads: Seq[Thread] = synchronized {
    val ret = ArrayBuffer.empty[Thread]
    activeThreads.forEach { item =>
      ret += item
    }
    ret.toSeq
  }

  private def moveToActive(t: Thread): Unit = synchronized {
    if (!hasSemaphore) {
      throw new IllegalStateException("Should not move to active without holding the semaphore")
    }
    blockedThreads.remove(t)
    // signal the GpuBubbleTimerManager about the decrease of blockedThreads
    bubbleTimerMgr.removeWaiter()
    activeThreads.add(t)
  }

  /**
   * Block the current thread until we have the semaphore.
   * @param semaphore what we are going to wait on.
   * @return the permits newly acquired, or zero if this task already held the semaphore.
   */
  def blockUntilReady(semaphore: GpuBackingSemaphore): Long = {
    val t = Thread.currentThread()
    // All threads start out in blocked, but will move out of it inside of the while loop.
    synchronized {
      blockedThreads.add(t)
      // signal the GpuBubbleTimerManager about the increase of blockedThreads
      bubbleTimerMgr.addWaiter()
    }
    var done = false
    var shouldBlockOnSemaphore = false
    var acquiredPermits = 0L
    while (!done) {
      try {
        synchronized {
          // This thread can continue if this task owns the GPU semaphore. When that happens
          // move the state of the thread from blocked to active.
          done = hasSemaphore
          if (done) {
            moveToActive(t)
          }
          // Only one thread can block on the semaphore itself, we pick the first thread in
          // blockedThread to be that one. This is arbitrary and does not matter, it is just
          // simple to do.
          shouldBlockOnSemaphore = t == blockedThreads.peek
          if (!done && !shouldBlockOnSemaphore) {
            // If we need to block and are not blocking on the semaphore we will wait
            // on this class until the task has the semaphore and we wake up.
            wait()
            if (hasSemaphore) {
              moveToActive(t)
              done = true
            }
          }
        }
        if (!done && shouldBlockOnSemaphore) {
          // We cannot be in a synchronized block and wait on the semaphore
          // so we have to release it and grab it again afterwards.
          var demandBytes = 0L
          var demandTimeNanos = 0L
          val used = semaphore.acquire(() => {
              val estimate = memoryEstimator.estimate()
              if (GpuSemaphoreDiagnostics.enabled) {
                demandBytes = estimate
                demandTimeNanos = System.nanoTime()
              }
              GpuSemaphore.memToPermitsWithMax(estimate)
            },
            () => lastAcquired > 0,
            TaskPriority.getTaskPriority(taskAttemptId), taskAttemptId)
          synchronized {
            permitsUsed = used
            acquiredPermits = used
            diagnosticLease += 1
            diagnosticDemandBytes = demandBytes
            diagnosticDemandTimeNanos = demandTimeNanos
            // We now own the semaphore so we need to wake up all of the other tasks that are
            // waiting.
            hasSemaphore = true
            lastAcquired = System.nanoTime()
            if (trackSemaphore) {
              nvtxRange =
                Some(new NvtxUniqueRange(s"Stage ${stageId} Task ${taskAttemptId} owning GPU",
                  NvtxColor.ORANGE))
            }
            moveToActive(t)
            notifyAll()
            done = true
          }
        }
      } catch {
        case throwable: Throwable =>
          synchronized {
            // a thread is exiting because of an exception, so we want to reset things,
            // and possibly elect another thread to wait on the semaphore.
            blockedThreads.remove(t)
            activeThreads.remove(t)
            if (!hasSemaphore && shouldBlockOnSemaphore) {
              // wake up the other threads so a new thread tries to get the semaphore
              notifyAll()
            }
          }
          throw throwable
      }
    }
    acquiredPermits
  }

  // Negative means failure, zero means already held, and positive means a new grant.
  def tryAcquire(semaphore: GpuBackingSemaphore,
                 taskAttemptId: Long): Long = synchronized {
    val t = Thread.currentThread()
    if (hasSemaphore) {
      activeThreads.add(t)
      0L
    } else {
      if (blockedThreads.size() == 0) {
        // No other threads for this task are waiting, so we might be able to grab this directly
        val estimate = memoryEstimator.estimate()
        val estimateTime = if (GpuSemaphoreDiagnostics.enabled) System.nanoTime() else 0L
        val numPermits = GpuSemaphore.memToPermitsWithMax(estimate)
        val ret = semaphore.tryAcquire(numPermits,
          TaskPriority.getTaskPriority(taskAttemptId),
          () => lastAcquired > 0,
          taskAttemptId)
        if (ret) {
          hasSemaphore = true
          lastAcquired = System.nanoTime()
          activeThreads.add(t)
          permitsUsed = numPermits
          diagnosticLease += 1
          diagnosticDemandBytes = estimate
          diagnosticDemandTimeNanos = estimateTime
          // no need to notify because there are no other threads and we are holding the lock
          // to ensure that.
        }
        if (ret) numPermits else -1L
      } else {
        -1L
      }
    }
  }

  def traceGrant(usedPermits: Long): Unit = synchronized {
    if (usedPermits > 0 && GpuSemaphoreDiagnostics.enabled) {
      GpuSemaphoreDiagnostics.emit("grant", stageId, taskAttemptId, diagnosticLease,
        usedPermits, diagnosticDemandBytes, diagnosticDemandTimeNanos, memoryEstimator)
    }
  }

  def releaseSemaphore(semaphore: GpuBackingSemaphore): Unit = synchronized {
    val t = Thread.currentThread()
    activeThreads.remove(t)
    if (hasSemaphore) {
      semaphore.release(permitsUsed, taskAttemptId)
      hasSemaphore = false
      lastReleased = System.nanoTime()
      GpuTaskMetrics.get.addSemaphoreHoldingTime(lastReleased - lastAcquired)
      nvtxRange.foreach(_.close())
      nvtxRange = None
      if (GpuSemaphoreDiagnostics.enabled) {
        GpuSemaphoreDiagnostics.emit("release", stageId, taskAttemptId, diagnosticLease,
          permitsUsed, diagnosticDemandBytes, diagnosticDemandTimeNanos, memoryEstimator)
      }
    }
    // It should be impossible for the current thread to be blocked when releasing the semaphore
    // because no blocked thread should ever leave `blockUntilReady`, which is where we put it in
    // the blocked state. So this is just a sanity test that we didn't do something stupid.
    if (blockedThreads.remove(t)) {
      throw new IllegalStateException(s"$t tried to release the semaphore when it is blocked!!!")
    }
  }
}

private final class GpuSemaphore(val maxConcurrentGpuTasksLimit: Int,
    ledger: Option[GpuMemoryLedger] = None) extends Logging {

  import GpuSemaphore._

  type GpuBackingSemaphore = PrioritySemaphore[Long]
  private val semaphore = new GpuBackingSemaphore(computeMaxPermits(), maxConcurrentGpuTasksLimit,
    ledger)
  // A map of taskAttemptId => semaphoreTaskInfo.
  // This map keeps track of all tasks that are both active on the GPU and blocked waiting
  // on the GPU.
  private val tasks = new ConcurrentHashMap[Long, SemaphoreTaskInfo]

  private val stageEstimators = {
    val lru = new util.LinkedHashMap[Int, GpuStageMemoryEstimator]() {
      override def removeEldestEntry(entry: Map.Entry[Int, GpuStageMemoryEstimator]): Boolean = {
        // We don't get a callback when a stage completes, and in theory if there is a retry
        // a stage might run again. So for now we are going to keep around at most
        // 100 stages. This should be a small amount of memory because there is a limit
        // on the number of active tasks. But it should be enough most of the time.
        size > 100
      }
    }
    util.Collections.synchronizedMap(lru)
  }

  // Executor-wide manager to track GPU bubble time metrics for all non-GPU workloads.
  private val bubbleTimerMgr = GpuBubbleTimerManager.getInstance

  // Get the last time this task acquired & released the semaphore
  private def lastSemAcqAndRelTime(context: TaskContext): (Long, Long) = {
    val taskAttemptId = context.taskAttemptId()
    if (!tasks.containsKey(taskAttemptId)) {
      (0, 0)
    } else {
      tasks.get(taskAttemptId).lastAcquireAndReleaseTime
    }
  }

  def tryAcquire(context: TaskContext): TryAcquireResult = {
    // Make sure that the thread/task is registered before we try and block
    TaskRegistryTracker.registerThreadForRetry()
    val taskAttemptId = context.taskAttemptId()
    val stageId = context.stageId()
    val stageEstimate = stageEstimators.computeIfAbsent(stageId, _ => {
      new GpuStageMemoryEstimator(stageId,
        computeDefaultMemory(SQLConf.get),
        isDynamicEnabled(SQLConf.get))
    })
    val taskInfo = tasks.computeIfAbsent(taskAttemptId, _ => {
      onTaskCompletion(context, completeTask)
      new SemaphoreTaskInfo(stageId, taskAttemptId, stageEstimate, bubbleTimerMgr)
    })
    val usedPermits = taskInfo.tryAcquire(semaphore, taskAttemptId)
    if (usedPermits >= 0) {
      GpuDeviceManager.initializeFromTask()
      stageEstimate.addTaskIfNeeded(taskAttemptId)
      if (GpuSemaphoreDiagnostics.enabled) {
        taskInfo.traceGrant(usedPermits)
      }
      SemaphoreAcquired
    } else {
      // We need to get the number of tasks that are still waiting
      var numWaiting = 0
      tasks.values().forEach { ti =>
        if (ti.isHoldingSemaphore) {
          numWaiting += 1
        }
      }
      AcquireFailed(numWaiting)
    }
  }

  def acquireIfNecessary(context: TaskContext): Unit = {
    // Make sure that the thread/task is registered before we try and block
    TaskRegistryTracker.registerThreadForRetry()
    GpuTaskMetrics.get.semWaitTime {
      val stageId = context.stageId()
      val stageEstimate = stageEstimators.computeIfAbsent(stageId, _ => {
        new GpuStageMemoryEstimator(stageId,
          computeDefaultMemory(SQLConf.get),
          isDynamicEnabled(SQLConf.get))
      })
      val taskAttemptId = context.taskAttemptId()
      val taskInfo = tasks.computeIfAbsent(taskAttemptId, _ => {
        onTaskCompletion(context, completeTask)
        new SemaphoreTaskInfo(stageId, taskAttemptId, stageEstimate, bubbleTimerMgr)
      })
      val usedPermits = taskInfo.blockUntilReady(semaphore)
      GpuDeviceManager.initializeFromTask()
      stageEstimate.addTaskIfNeeded(taskAttemptId)
      if (GpuSemaphoreDiagnostics.enabled) {
        taskInfo.traceGrant(usedPermits)
      }
    }
  }

  def releaseIfNecessary(context: TaskContext): Unit = {
    NvtxRegistry.RELEASE_GPU {
      val taskAttemptId = context.taskAttemptId()
      GpuTaskMetrics.get.updateRetry(taskAttemptId)
      GpuTaskMetrics.get.updateFootprint(taskAttemptId)
      val taskInfo = tasks.get(taskAttemptId)
      if (taskInfo != null) {
        taskInfo.releaseSemaphore(semaphore)
      }
    }
  }

  def completeTask(context: TaskContext): Unit = {
    val taskAttemptId = context.taskAttemptId()
    GpuTaskMetrics.get.updateRetry(taskAttemptId)
    GpuTaskMetrics.get.updateMaxMemory(taskAttemptId)
    GpuTaskMetrics.get.updateFootprint(taskAttemptId)
    val refs = tasks.remove(taskAttemptId)
    if (refs == null) {
      throw new IllegalStateException(s"Completion of unknown task $taskAttemptId")
    }
    refs.releaseSemaphore(semaphore)
    val estimator = stageEstimators.get(refs.stageId)
    if (estimator != null) {
      estimator.taskDone(refs.taskAttemptId)
    }
  }

  def dumpActiveStackTracesToLog(): Unit = {
    try {
      val stackTracesSemaphoreHeld = new mutable.ArrayBuffer[String]()
      val otherStackTraces = new mutable.ArrayBuffer[String]()
      tasks.forEach { (taskAttemptId, taskInfo) =>
        val semaphoreHeld = taskInfo.isHoldingSemaphore
        taskInfo.getActiveThreads.foreach { thread =>
          val sb = new mutable.StringBuilder()
          thread.getStackTrace.foreach { stackTraceElement =>
            sb.append("    " + stackTraceElement + "\n")
          }
          if (semaphoreHeld) {
            stackTracesSemaphoreHeld.append(
              s"Semaphore held. " +
                  s"Stack trace for task attempt id $taskAttemptId:\n${sb.toString()}")
          } else {
            otherStackTraces.append(
              s"Semaphore not held. " +
                  s"Stack trace for task attempt id $taskAttemptId:\n${sb.toString()}")
          }
        }
      }
      logWarning(s"Dumping stack traces. The semaphore sees ${tasks.size()} tasks, " +
          s"${stackTracesSemaphoreHeld.size} threads are holding onto the semaphore. " +
          stackTracesSemaphoreHeld.mkString("\n", "\n", "\n") +
          otherStackTraces.mkString("\n", "\n", "\n"))
    } catch {
      case t: Throwable =>
        logWarning("Unable to obtain stack traces in the semaphore.", t)
    }
  }

  def shutdown(): Unit = {
    if (!tasks.isEmpty) {
      logDebug(s"shutting down with ${tasks.size} tasks still registered")
    }
  }
}
