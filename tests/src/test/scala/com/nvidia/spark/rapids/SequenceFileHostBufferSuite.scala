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

import java.io.IOException

import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.sequencefile.BinaryHostColumnBuilder
import com.nvidia.spark.rapids.spill.SpillFramework

class SequenceFileHostBufferSuite extends RmmSparkRetrySuiteBase {
  private val dataUnit = 64 * 1024L

  private def append(builder: BinaryHostColumnBuilder, records: Seq[Array[Byte]]): Unit = {
    records.foreach { record =>
      assert(builder.append(out => out.write(record)) == record.length.toLong)
    }
  }

  private def checkFinished(
      builder: BinaryHostColumnBuilder,
      records: Seq[Array[Byte]]): Long = {
    withResource(builder.finish()) { buffers =>
      val expected = records.flatten.toArray
      assert(buffers.length == 2)
      assert(buffers(0).length == expected.length.toLong)
      assert(buffers(1).length == (records.length + 1L) * Integer.BYTES)
      withResource(buffers(0).getDataHostBuffer()) { data =>
        val actual = new Array[Byte](expected.length)
        if (actual.nonEmpty) data.getBytes(actual, 0, 0, actual.length)
        assert(actual.toSeq == expected.toSeq)
      }
      withResource(buffers(1).getDataHostBuffer()) { offsets =>
        val expectedOffsets = records.scanLeft(0)((total, record) => total + record.length)
        expectedOffsets.zipWithIndex.foreach { case (expectedOffset, index) =>
          assert(offsets.getInt(index.toLong * Integer.BYTES) == expectedOffset)
        }
      }
      withResource(buffers(0).getHostBuffer())(_.getLength)
    }
  }

  test("reuse actual chunk sizes without repeating data and offset growth copies") {
    val records = (0 until 1025).map { row =>
      Array.tabulate[Byte](64)(index => ((row + index) % 251).toByte)
    }
    val coldReallocations = new LocalGpuMetric
    val coldCopiedBytes = new LocalGpuMetric
    val nextSize = withResource(new BinaryHostColumnBuilder(
        3 * dataUnit, 4096, dataUnit / 2, 512, coldReallocations, coldCopiedBytes)) { builder =>
      append(builder, records)
      val size = builder.nextInitialSize
      assert(size == ((2 * dataUnit, 2048)))
      assert(checkFinished(builder, records) == 2 * dataUnit)
      size
    }
    assert(coldReallocations.value == 4)
    assert(coldCopiedBytes.value == dataUnit / 2 + dataUnit + (513L + 1025L) * Integer.BYTES)

    val warmReallocations = new LocalGpuMetric
    val warmCopiedBytes = new LocalGpuMetric
    withResource(new BinaryHostColumnBuilder(
        3 * dataUnit, 4096, nextSize._1, nextSize._2, warmReallocations, warmCopiedBytes)) {
      builder =>
        append(builder, records)
        assert(checkFinished(builder, records) == nextSize._1)
    }
    assert(warmReallocations.value == 0)
    assert(warmCopiedBytes.value == 0)
  }

  test("rolled-back partial writes do not enlarge the next chunk's initial sizes") {
    val record = Array.fill[Byte](dataUnit.toInt)(17)
    val reallocations = new LocalGpuMetric
    val copiedBytes = new LocalGpuMetric
    withResource(new BinaryHostColumnBuilder(
        8 * dataUnit, 4096, dataUnit, 1024, reallocations, copiedBytes)) { builder =>
      append(builder, Seq(record))
      val mark = builder.mark()
      intercept[IOException] {
        builder.append { out =>
          out.write(new Array[Byte](dataUnit.toInt + 1))
          out.write(new Array[Byte]((8 * dataUnit).toInt))
        }
      }
      assert(builder.mark() == 2 * dataUnit + 1)
      builder.rollback(1, mark)
      assert(builder.nextInitialSize == ((dataUnit, 1024)))
      assert(checkFinished(builder, Seq(record)) == 2 * dataUnit + 1)
    }
    assert(reallocations.value == 1)
    assert(copiedBytes.value == dataUnit)
  }

  test("reject boundary writes before growing data or offsets") {
    val record = Array[Byte](1, 2, 3, 4, 5, 6, 7, 8)
    val reallocations = new LocalGpuMetric
    val copiedBytes = new LocalGpuMetric
    withResource(new BinaryHostColumnBuilder(
        32, 4, record.length, 1, reallocations, copiedBytes)) { builder =>
      assert(builder.append(out => out.write(record), maxBytes = record.length) == record.length)
      val mark = builder.mark()
      intercept[IOException] {
        builder.append(out => out.write(9), maxBytes = 0)
      }
      assert(builder.mark() == mark)
      intercept[IOException] {
        builder.append(out => out.write(Array[Byte](9)), maxBytes = 0)
      }
      assert(builder.mark() == mark)
      assert(reallocations.value == 0)
      assert(copiedBytes.value == 0)
      builder.rollback(1, mark)
      assert(checkFinished(builder, Seq(record)) == record.length)
    }
  }

  test("retry a partially written record after restoring its byte and row positions") {
    val first = Array[Byte](1, 2, 3, 4, 5, 6, 7, 8)
    val next = Array.tabulate[Byte](12)(index => (index + 20).toByte)
    val reallocations = new LocalGpuMetric
    val copiedBytes = new LocalGpuMetric
    withResource(new BinaryHostColumnBuilder(
        64, 4, 16, 1, reallocations, copiedBytes)) { builder =>
      append(builder, Seq(first))
      val mark = builder.mark()
      intercept[IOException] {
        builder.append(out => {
          out.write(next, 0, 4)
          out.write(next, 4, next.length - 4)
        }, maxBytes = 8)
      }
      assert(builder.mark() == mark + 4)
      assert(reallocations.value == 0)
      assert(copiedBytes.value == 0)
      builder.rollback(1, mark)
      assert(builder.append(out => {
        out.write(next, 0, 4)
        out.write(next, 4, next.length - 4)
      }, maxBytes = next.length) == next.length)
      assert(checkFinished(builder, Seq(first, next)) == 32)
    }
    assert(reallocations.value == 2)
    assert(copiedBytes.value == 12 + 2L * Integer.BYTES)
  }

  test("accept exact-fit and zero-width records at the byte limit") {
    val record = Array[Byte](1, 2, 3, 4, 5, 6, 7, 8)
    val reallocations = new LocalGpuMetric
    val copiedBytes = new LocalGpuMetric
    withResource(new BinaryHostColumnBuilder(
        record.length, 4, record.length, 4, reallocations, copiedBytes)) { builder =>
      assert(builder.append(out => out.write(record), maxBytes = record.length) == record.length)
      assert(builder.append(out => out.write(Array.emptyByteArray), maxBytes = 0) == 0)
      assert(builder.append(out => out.write(Array.emptyByteArray),
        maxBytes = Long.MaxValue) == 0)
      intercept[IOException] {
        builder.append(out => out.write(9), maxBytes = Long.MaxValue)
      }
      assert(builder.append(out => out.write(Array.emptyByteArray)) == 0)
      assert(checkFinished(builder,
        Seq(record, Array.emptyByteArray, Array.emptyByteArray, Array.emptyByteArray)) ==
        record.length)
    }
    assert(reallocations.value == 0)
    assert(copiedBytes.value == 0)
  }

  test("independent column sizes adapt when empty and wide columns exchange widths") {
    val empty = Seq.fill(16)(Array.emptyByteArray)
    val wide = (0 until 16).map(row => Array.fill[Byte](8192)(row.toByte))
    val firstColumns = Seq(empty, wide)
    val sizes = firstColumns.map { records =>
      withResource(new BinaryHostColumnBuilder(4 * dataUnit, 2048, dataUnit, 1024)) { builder =>
        append(builder, records)
        val size = builder.nextInitialSize
        checkFinished(builder, records)
        size
      }
    }
    assert(sizes == Seq((1L, 1024), (2 * dataUnit, 1024)))

    firstColumns.reverse.zip(sizes).foreach { case (records, (dataSize, rowSize)) =>
      withResource(new BinaryHostColumnBuilder(4 * dataUnit, 2048, dataSize, rowSize)) { builder =>
        append(builder, records)
        val expectedDataSize = if (records.head.isEmpty) 1L else 2 * dataUnit
        assert(builder.nextInitialSize == ((expectedDataSize, 1024)))
        checkFinished(builder, records)
      }
    }
  }

  test("initial sizes respect limits and a short tail releases its larger allocation") {
    val initialHandles = SpillFramework.stores.hostStore.numHandles
    val maxData = dataUnit + 123
    val full = Seq(Array.fill[Byte](maxData.toInt)(23))
    val size = withResource(new BinaryHostColumnBuilder(maxData, 3, maxData, 3)) { builder =>
      append(builder, full)
      val next = builder.nextInitialSize
      assert(next == ((maxData, 3)))
      assert(checkFinished(builder, full) == maxData)
      next
    }
    withResource(new BinaryHostColumnBuilder(maxData, 3, size._1, size._2)) { builder =>
      val tail = Seq(Array[Byte](1, 2, 3))
      append(builder, tail)
      assert(builder.nextInitialSize == ((dataUnit, 3)))
      assert(checkFinished(builder, tail) == maxData)
    }
    withResource(new BinaryHostColumnBuilder(maxData, 3, 1, 1)) { builder =>
      assert(builder.nextInitialSize == ((1L, 1)))
      assert(checkFinished(builder, Seq.empty) == 1)
    }
    assert(SpillFramework.stores.hostStore.numHandles == initialHandles)
  }
}
