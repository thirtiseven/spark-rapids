/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
import java.util.PriorityQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.locks.{Condition, ReentrantLock}

import scala.collection.JavaConverters.asScalaIteratorConverter

import org.apache.spark.sql.rapids.GpuTaskMetrics

/**
 * An admission policy that can replace the plain permit count of a [[PrioritySemaphore]].
 * It is consulted with the semaphore lock held.
 */
trait SemaphoreAdmissionPolicy {
  /**
   * @param holders task id -> permits of every task currently holding the semaphore
   * @param firstLeaseSince task id -> System.nanoTime() of the grant, for holders that had never
   *                        held the semaphore before this lease (their footprint is unknown)
   * @param taskId the task asking to be admitted
   * @param numPermits the permits it asks for
   * @return Some(decision), or None to fall back to the plain permit count
   */
  def admits(holders: util.HashMap[Long, Long], firstLeaseSince: util.HashMap[Long, Long],
      taskId: Long, numPermits: Long, candidateFirstLease: Boolean): Option[Boolean]

  /** How long a waiter sleeps before it re-evaluates the head of the queue, in ms. */
  def pollMillis: Long

  /** Called by a waiter, without the semaphore lock, after a poll did not admit it. */
  def onStillBlocked(): Unit = {}
}

class PrioritySemaphore[T](val maxPermits: Long, val maxConcurrentGpuTasksLimit: Int,
    val admissionPolicy: Option[SemaphoreAdmissionPolicy] = None)
  (implicit ordering: Ordering[T]) {
  // This lock is used to generate condition variables, which affords us the flexibility to notify
  // specific threads at a time. If we use the regular synchronized pattern, we have to either
  // notify randomly, or if we try creating condition variables not tied to a shared lock, they
  // won't work together properly, and we see things like deadlocks.
  private val lock = new ReentrantLock()
  private var occupiedSlots: Long = 0
  private var currentConcurrentGpuTasksNum: Long = 0
  // task id -> permits for the current holders, used by the admission policy
  private val holders = new util.HashMap[Long, Long]()
  private val firstLeaseSince = new util.HashMap[Long, Long]()

  private case class ThreadInfo(priority: T,
                                condition: Condition,
                                computeNumPermits: () => Long,
                                wasOnGpuBefore: () => Boolean,
                                taskId: Long) {
    var signaled: Boolean = false
    var permitsUsed: Long = 0
  }

  // use task id as tie breaker when priorities are equal (both are 0 because never hold lock)
  private val priorityComp = Ordering.by[ThreadInfo, T](_.priority).reverse.
    thenComparing((a, b) => a.taskId.compareTo(b.taskId))

  // We expect a relatively small number of threads to be contending for this lock at any given
  // time, therefore we are not concerned with the insertion/removal time complexity.
  private val waitingQueue: PriorityQueue[ThreadInfo] =
    new PriorityQueue[ThreadInfo](priorityComp)

  def tryAcquire(numPermits: Long,
                 priority: T,
                 wasOnGpuBefore: () => Boolean,
                 taskAttemptId: Long): Boolean = {
    lock.lock()
    try {
      if (waitingQueue.size() > 0 &&
        priorityComp.compare(
          waitingQueue.peek(),
          ThreadInfo(priority, null, () => numPermits, wasOnGpuBefore, taskAttemptId)
        ) < 0) {
        false
      } else if (!canAcquire(numPermits, taskAttemptId, !wasOnGpuBefore())) {
        false
      } else {
        commitAcquire(numPermits, taskAttemptId, wasOnGpuBefore())
        true
      }
    } finally {
      lock.unlock()
    }
  }

  def acquire(computePermits: () => Long, wasOnGpuBefore: () => Boolean,
              priority: T, taskAttemptId: Long): Long = {
    lock.lock()
    try {
      val numPermitsNow = computePermits()
      if (!tryAcquire(numPermitsNow, priority, wasOnGpuBefore, taskAttemptId)) {
        val condition = lock.newCondition()
        val info = ThreadInfo(priority, condition, computePermits, wasOnGpuBefore, taskAttemptId)
        try {
          waitingQueue.add(info)
          // only count tasks that had held semaphore before,
          // so they're very likely to have remaining data on GPU
          GpuTaskMetrics.get.recordOnGpuTasksWaitingNumber(
            waitingQueue.iterator().asScala.count(_.wasOnGpuBefore()))

          admissionPolicy match {
            case None =>
              while (!info.signaled) {
                info.condition.await()
              }
            case Some(policy) =>
              // The policy depends on memory that can be freed without a release, so the head
              // of the queue re-evaluates itself periodically.
              while (!info.signaled) {
                info.condition.await(policy.pollMillis, TimeUnit.MILLISECONDS)
                if (!info.signaled && (waitingQueue.peek() eq info)) {
                  grantWaiters()
                  if (!info.signaled) {
                    lock.unlock()
                    try {
                      policy.onStillBlocked()
                    } finally {
                      lock.lock()
                    }
                  }
                }
              }
          }
          info.permitsUsed
        } catch {
          case e: Exception =>
            waitingQueue.remove(info)
            if (info.signaled) {
              release(info.permitsUsed, taskAttemptId)
            }
            throw e
        }
      } else {
        numPermitsNow
      }
    } finally {
      lock.unlock()
    }
  }

  private def commitAcquire(numPermits: Long, taskAttemptId: Long,
      wasOnGpuBefore: Boolean): Unit = {
    occupiedSlots += numPermits
    currentConcurrentGpuTasksNum += 1
    holders.put(taskAttemptId, numPermits)
    if (!wasOnGpuBefore && admissionPolicy.isDefined) {
      firstLeaseSince.put(taskAttemptId, System.nanoTime())
    }
    // Report current concurrent tasks to GpuTaskMetrics, let it handle max tracking
    GpuTaskMetrics.get.recordConcurrentGpuTasks(currentConcurrentGpuTasksNum)
  }

  def release(numPermits: Long, taskAttemptId: Long): Unit = {
    lock.lock()
    try {
      occupiedSlots -= numPermits
      currentConcurrentGpuTasksNum -= 1
      holders.remove(taskAttemptId)
      firstLeaseSince.remove(taskAttemptId)
      grantWaiters()
    } finally {
      lock.unlock()
    }
  }

  /** Release permits that were not acquired on behalf of a known task (tests). */
  def release(numPermits: Long): Unit = release(numPermits, -1L)

  /** Acquire and wake up waiters, in priority order, while the head can be admitted. */
  private def grantWaiters(): Unit = {
    var done = false
    while (!done && waitingQueue.size() > 0) {
      val nextThread = waitingQueue.peek()
      val threadPermits = nextThread.computeNumPermits()
      if (canAcquire(threadPermits, nextThread.taskId, !nextThread.wasOnGpuBefore())) {
        val popped = waitingQueue.poll()
        assert(popped eq nextThread)
        commitAcquire(threadPermits, nextThread.taskId, nextThread.wasOnGpuBefore())
        nextThread.signaled = true
        nextThread.permitsUsed = threadPermits
        nextThread.condition.signal()
      } else {
        done = true
      }
    }
  }

  private def canAcquire(numPermits: Long, taskAttemptId: Long,
      firstLease: Boolean): Boolean = {
    val withinTaskLimit = maxConcurrentGpuTasksLimit <= 0 ||
      currentConcurrentGpuTasksNum < maxConcurrentGpuTasksLimit
    withinTaskLimit && {
      admissionPolicy.flatMap(
        _.admits(holders, firstLeaseSince, taskAttemptId, numPermits, firstLease))
        .getOrElse(occupiedSlots + numPermits <= maxPermits)
    }
  }
}
