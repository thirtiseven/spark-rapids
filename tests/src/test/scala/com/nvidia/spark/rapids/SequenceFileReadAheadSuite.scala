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

import java.util.concurrent.{Callable, CompletableFuture, CountDownLatch, ExecutionException, Executors, Future, TimeUnit}
import java.util.concurrent.atomic.AtomicInteger

import com.nvidia.spark.rapids.sequencefile.SequenceFileReadAhead
import org.scalatest.funsuite.AnyFunSuite

class SequenceFileReadAheadSuite extends AnyFunSuite {
  Seq(true, false).foreach { prepareFirst =>
    test(s"submit one continuation regardless of preparation order: $prepareFirst") {
      val pending = new SequenceFileReadAhead[Int]
      val submissions = new AtomicInteger()
      def prepare(): Unit = pending.armSuccessor(() => Some(submissions.incrementAndGet()))
      if (prepareFirst) {
        prepare()
        assert(submissions.get() == 0)
        pending.markReady(hasMore = true)
      } else {
        pending.markReady(hasMore = true)
        assert(submissions.get() == 0)
        prepare()
      }
      assert(submissions.get() == 1)
      assert(pending.takeContinuation().contains(1))
      intercept[IllegalArgumentException](pending.takeContinuation())
    }

    test(s"EOF does not submit a prepared continuation: $prepareFirst") {
      val pending = new SequenceFileReadAhead[Int]
      val submissions = new AtomicInteger()
      def prepare(): Unit = pending.armSuccessor(() => Some(submissions.incrementAndGet()))
      if (prepareFirst) {
        prepare()
        pending.markReady(hasMore = false)
      } else {
        pending.markReady(hasMore = false)
        prepare()
      }
      assert(pending.takeContinuation().isEmpty)
      assert(submissions.get() == 0)
    }
  }

  test("concurrent completion and preparation submit exactly once") {
    val executor = Executors.newFixedThreadPool(2)
    try {
      for (_ <- 0 until 32) {
        val start = new CountDownLatch(1)
        val pending = new SequenceFileReadAhead[Int]
        val submissions = new AtomicInteger()
        val prepare = executor.submit(new Callable[Unit] {
          override def call(): Unit = {
            start.await()
            pending.armSuccessor(() => Some(submissions.incrementAndGet()))
          }
        })
        val complete = executor.submit(new Callable[Unit] {
          override def call(): Unit = {
            start.await()
            pending.markReady(hasMore = true)
          }
        })
        start.countDown()
        prepare.get(10, TimeUnit.SECONDS)
        complete.get(10, TimeUnit.SECONDS)
        assert(pending.takeContinuation().contains(1))
        assert(submissions.get() == 1)
      }
    } finally {
      executor.shutdownNow()
      assert(executor.awaitTermination(10, TimeUnit.SECONDS))
    }
  }

  test("read ahead stops at the next chunk until the consumer advances") {
    val executor = Executors.newSingleThreadExecutor()
    val secondReady = new CountDownLatch(1)
    val thirdStarted = new CountDownLatch(1)
    val first = new SequenceFileReadAhead[Future[Unit]]
    val second = new SequenceFileReadAhead[Future[Unit]]
    try {
      first.armSuccessor(() => Some(executor.submit(new Callable[Unit] {
        override def call(): Unit = {
          second.markReady(hasMore = true)
          secondReady.countDown()
        }
      })))
      first.markReady(hasMore = true)
      assert(secondReady.await(10, TimeUnit.SECONDS))
      assert(thirdStarted.getCount == 1)

      first.takeContinuation().get.get(10, TimeUnit.SECONDS)
      second.armSuccessor(() => Some(executor.submit(new Callable[Unit] {
        override def call(): Unit = thirdStarted.countDown()
      })))
      assert(thirdStarted.await(10, TimeUnit.SECONDS))
      second.takeContinuation().get.get(10, TimeUnit.SECONDS)
    } finally {
      executor.shutdownNow()
      assert(executor.awaitTermination(10, TimeUnit.SECONDS))
    }
  }

  test("a closed reader can decline continuation submission") {
    val pending = new SequenceFileReadAhead[Int]
    pending.armSuccessor(() => None)
    pending.markReady(hasMore = true)
    assert(pending.takeContinuation().isEmpty)
  }

  test("a prefetched failure is observed when its future is consumed") {
    val failed = new CompletableFuture[Int]
    val failure = new IllegalStateException("later chunk")
    failed.completeExceptionally(failure)
    val pending = new SequenceFileReadAhead[Future[Int]]
    pending.armSuccessor(() => Some(failed))
    pending.markReady(hasMore = true)
    val error = intercept[ExecutionException] {
      pending.takeContinuation().get.get(10, TimeUnit.SECONDS)
    }
    assert(error.getCause eq failure)
  }
}
