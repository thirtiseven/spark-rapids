/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.
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
import java.nio.charset.StandardCharsets
import java.nio.file.Files

import scala.collection.mutable.ArrayBuffer

import ai.rapids.cudf.{CSVOptions, HostMemoryBuffer, Table}
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.jni.{GpuSplitAndRetryOOM, RmmSpark}
import com.nvidia.spark.rapids.shims.PartitionedFileUtilsShim
import org.apache.hadoop.conf.Configuration

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.csv.{CSVOptions => SparkCSVOptions}
import org.apache.spark.sql.types._

class CsvScanRetrySuite extends RmmSparkRetrySuiteBase {
  private val stringSchema = StructType(Seq(
    StructField("a", StringType), StructField("b", StringType)))

  private def withCsvTables[T](lines: Seq[String], header: Boolean = false)
      (fn: (Iterator[Table] with AutoCloseable) => T): T = {
    withResource(FilterCsvEmptyHostLineBuffererFactory.createBufferer(32,
      Array('\n'.toByte))) { buffer =>
      lines.foreach { line =>
        val bytes = line.getBytes(StandardCharsets.UTF_8)
        buffer.add(bytes, 0, bytes.length)
      }
      withResource(CSVPartitionReader.readToTables(buffer, GpuColumnVector.from(stringSchema),
        NoopMetric, hasHeader => CSVOptions.builder().hasHeader(hasHeader)
          .withComment('#').build(), header, '#'.toByte, null))(fn)
    }
  }

  private def collectStrings(tables: Iterator[Table]): (Seq[Seq[String]], Int) = {
    val rows = ArrayBuffer[Seq[String]]()
    var batches = 0
    tables.foreach { table =>
      withResource(table) { _ =>
        withResource(table.getColumn(0).copyToHost()) { a =>
          withResource(table.getColumn(1).copyToHost()) { b =>
            (0 until table.getRowCount.toInt).foreach { i =>
              rows += Seq(a.getJavaString(i), b.getJavaString(i))
            }
          }
        }
      }
      batches += 1
    }
    (rows.toSeq, batches)
  }

  for (header <- Seq(false, true)) {
    test(s"CSV recursively splits host input on OOM, header=$header") {
      val data = (0 until 16).map(i => s"""$i,"value,$i é"""")
      val lines = if (header) Seq("# before header", "a,b") ++ data else data
      withCsvTables(lines, header) { tables =>
        RmmSpark.getAndResetNumSplitRetryThrow(1)
        RmmSpark.forceSplitAndRetryOOM(RmmSpark.getCurrentThreadId, 2,
          RmmSpark.OomInjectionType.GPU.ordinal, 0)
        val (actual, batches) = collectStrings(tables)
        assert(actual == (0 until 16).map(i => Seq(i.toString, s"value,$i é")))
        assert(batches >= 3)
        assert(RmmSpark.getAndResetNumSplitRetryThrow(1) >= 2)
      }
    }
  }

  test("CSV single record split failure is terminal") {
    withCsvTables(Seq("one,two")) { tables =>
      RmmSpark.forceSplitAndRetryOOM(RmmSpark.getCurrentThreadId, 1,
        RmmSpark.OomInjectionType.GPU.ordinal, 0)
      val error = intercept[IOException](tables.next())
      assert(error.getCause.isInstanceOf[GpuSplitAndRetryOOM])
      assert(!tables.hasNext)
    }
  }

  for (bom <- Seq("", "\ufeff")) {
    test(s"CSV split preserves a header after leading comments, BOM=${bom.nonEmpty}") {
      val prefix = bom + "# comment taking up most of the buffer\n# another comment\na,b\n"
      val bytes = (prefix + "1,2\n").getBytes(StandardCharsets.UTF_8)
      val buffer = HostMemoryBuffer.allocate(bytes.length)
      buffer.setBytes(0, bytes, 0, bytes.length)
      val chunk = CSVPartitionReader.CsvReadChunk(buffer, true, '#'.toByte)
      withResource(chunk.split()) { parts =>
        assert(parts.size == 2)
        assert(parts.head.hasHeader)
        assert(!parts.last.hasHeader)
        assert(parts.head.buffer.getLength == prefix.getBytes(StandardCharsets.UTF_8).length)
      }
      assert(buffer.getRefCount == 0)
    }
  }

  test("closing CSV retry input releases pending slices without parsing them") {
    val bytes = "1,2\n3,4\n5,6\n7,8\n".getBytes(StandardCharsets.UTF_8)
    val buffer = HostMemoryBuffer.allocate(bytes.length)
    buffer.setBytes(0, bytes, 0, bytes.length)
    var attempts = 0
    val chunks = RmmRapidsRetryIterator.withRetry(
      CSVPartitionReader.CsvReadChunk(buffer, false, 0.toByte),
      (chunk: CSVPartitionReader.CsvReadChunk) => chunk.split()) { chunk =>
      attempts += 1
      if (attempts == 1) throw new GpuSplitAndRetryOOM("split for test")
      chunk.buffer.getLength
    }
    withResource(chunks) { _ =>
      assert(chunks.next() < bytes.length)
      assert(chunks.hasNext)
    }
    assert(attempts == 2)
    assert(buffer.getRefCount == 0)
  }

  private def withReader[T](text: String, readSchema: StructType, maxBytes: Long,
      memoryBudget: Long = Long.MaxValue)
      (fn: CSVPartitionReader => T): T = {
    val file = Files.createTempFile("csv-reader-retry", ".csv")
    Files.write(file, text.getBytes(StandardCharsets.UTF_8))
    try {
      withResource(new CSVPartitionReader(new Configuration(),
        PartitionedFileUtilsShim.newPartitionedFile(
          InternalRow.empty, file.toString, 0, Files.size(file)),
        stringSchema, readSchema,
        new SparkCSVOptions(Map("header" -> "true", "comment" -> "#"), false, "UTC"),
        1024, maxBytes, Map[String, GpuMetric]().withDefaultValue(NoopMetric)) {
        override protected lazy val gpuMemoryBudget: Long = memoryBudget
      })(fn)
    } finally {
      Files.deleteIfExists(file)
    }
  }

  for (readSchema <- Seq(stringSchema, StructType(Seq.empty))) {
    test(s"CSV continues after header-only and comment-only chunks: $readSchema") {
      withReader("a,b\n# comment\n\n1,2\n# another comment\n3,4\n", readSchema, 1) { reader =>
        var rows = 0
        while (reader.next()) {
          withResource(reader.get()) { batch => rows += batch.numRows() }
        }
        assert(rows == 2)
      }
    }
  }

  test("CSV reader drains all split outputs after the file is exhausted") {
    withReader("a,b\n" + (0 until 16).map(i => s"$i,$i\n").mkString,
      StructType(Seq(StructField("a", IntegerType), StructField("b", IntegerType))),
      1024 * 1024) { reader =>
      RmmSpark.forceSplitAndRetryOOM(RmmSpark.getCurrentThreadId, 2,
        RmmSpark.OomInjectionType.GPU.ordinal, 0)
      var rows = 0
      var batches = 0
      val actual = ArrayBuffer[Int]()
      while (reader.next()) {
        withResource(reader.get()) { batch =>
          rows += batch.numRows()
          withResource(GpuColumnVector.from(batch)) { table =>
            withResource(table.getColumn(0).copyToHost()) { host =>
              (0 until batch.numRows()).foreach(i => actual += host.getInt(i))
            }
          }
        }
        batches += 1
      }
      assert(rows == 16)
      assert(actual.toSeq == (0 until 16))
      assert(batches >= 3)
    }
  }

  test("CSV reader continues after a split containing only comments and header") {
    withReader("# long comment before header\n" * 8 + "a,b\n1,2\n3,4\n",
      stringSchema, 1024 * 1024) { reader =>
      RmmSpark.forceSplitAndRetryOOM(RmmSpark.getCurrentThreadId, 1,
        RmmSpark.OomInjectionType.GPU.ordinal, 0)
      var rows = 0
      while (reader.next()) {
        withResource(reader.get()) { batch => rows += batch.numRows() }
      }
      assert(rows == 2)
    }
  }

  test("CSV memory estimate accounts for projected cells before allocating GPU input") {
    val wide = CSVPartitionReader.estimatedGpuBytes(1024, 100, 46)
    val narrow = CSVPartitionReader.estimatedGpuBytes(1024, 100, 1)
    assert(wide > narrow)
    assert(CSVPartitionReader.estimatedGpuBytes(1024, 100, 0) == narrow)
    assert(CSVPartitionReader.estimatedGpuBytes(Int.MaxValue, Int.MaxValue, 46) > Int.MaxValue)
    withReader("a,b\n" + (0 until 32).map(i => s"$i,$i\n").mkString,
      stringSchema, 1024 * 1024, memoryBudget = 256) { reader =>
      var rows = 0
      var batches = 0
      while (reader.next()) {
        withResource(reader.get()) { batch =>
          assert(batch.numRows() <= 5)
          rows += batch.numRows()
        }
        batches += 1
      }
      assert(rows == 32)
      assert(batches > 1)
    }
  }

  test("CSV preserves the input byte limit when the GPU budget is ample") {
    val text = "a,b\n" + (0 until 32).map(i => s"$i,$i\n").mkString
    withReader(text, stringSchema, 256) { reader =>
      assert(reader.next())
      assert(reader.get().numRows() == 32)
      assert(!reader.next())
    }
    withReader(text, stringSchema, 16) { reader =>
      var rows = 0
      var batches = 0
      while (reader.next()) {
        withResource(reader.get()) { batch => rows += batch.numRows() }
        batches += 1
      }
      assert(rows == 32)
      assert(batches > 1)
    }
  }

  test("test simple retry") {
    val bufferer = HostLineBuffererFactory.createBufferer(100, Array('\n'.toByte))
    bufferer.add("1,2".getBytes, 0, 3)

    val cudfSchema = GpuColumnVector.from(StructType(Seq(StructField("a", IntegerType),
      StructField("b", IntegerType))))
    val opts = CSVOptions.builder().hasHeader(false)
    RmmSpark.forceRetryOOM(RmmSpark.getCurrentThreadId, 1,
      RmmSpark.OomInjectionType.GPU.ordinal, 0)
    val table = CSVPartitionReader.readToTable(bufferer, cudfSchema, NoopMetric,
      opts, "CSV", null)
    table.close()
    // We don't have any good way to verify that the retry was thrown, but we are going to trust
    // that it was.
  }

  test("cast table to desired types is retried on OOM") {
    val dataSchema = StructType(Seq(
      StructField("a", IntegerType),
      StructField("b", IntegerType)))
    val csvFile = Files.createTempFile("csv-cast-retry", ".csv")
    Files.write(csvFile, "1,2\n".getBytes(StandardCharsets.UTF_8))
    var reader: CSVPartitionReader = null
    try {
      reader = new CSVPartitionReader(
          new Configuration(),
          PartitionedFileUtilsShim.newPartitionedFile(
            InternalRow.empty, csvFile.toString, 0, Files.size(csvFile)),
          dataSchema, dataSchema,
          new SparkCSVOptions(Map.empty, false, "UTC"),
          1024, 128 * 1024,
          Map[String, GpuMetric]().withDefaultValue(NoopMetric)) {

        override def castToOutputTypesWithRetryAndClose(table: Table,
            readSchema: StructType): Table = {
          // inject a GPU OOM before running into the actual operation.
          RmmSpark.forceRetryOOM(RmmSpark.getCurrentThreadId, 1,
            RmmSpark.OomInjectionType.GPU.ordinal, 0)

          super.castToOutputTypesWithRetryAndClose(table, readSchema)
        }
      }
      assert(reader.next())
      Arm.withResource(reader.get()) { cb =>
        assert(cb.numRows() == 1)
        assert(cb.numCols() == 2)
      }
    } finally {
      if (reader != null) {
        reader.close()
        reader = null
      }
      Files.deleteIfExists(csvFile)
    }
  }

}
