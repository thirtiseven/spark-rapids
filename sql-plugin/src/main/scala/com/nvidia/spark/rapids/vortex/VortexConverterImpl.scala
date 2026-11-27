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

package com.nvidia.spark.rapids.vortex

import java.io.File
import java.net.URI
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

import scala.collection.mutable.ArrayBuffer

import ai.rapids.cudf.{ParquetOptions, Table => CudfTable}
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.jni.VortexNative

import org.apache.spark.sql.types.{DataType, StructType}

/** Converts a directory of Parquet files (flat or Hive col=val/ partitioned) to standard .vortex files. */
class VortexConverterImpl extends java.util.function.Function[Array[String], java.lang.Long] {
  private val SCHEMA_FILE = "_vortex_schema.json"

  private def stripFileScheme(p: String): String =
    if (p.startsWith("file:")) new URI(p).getPath else p

  private def collectParquetFiles(dir: File): Array[File] = {
    val buf = new ArrayBuffer[File]()
    def walk(f: File): Unit = {
      val list = f.listFiles()
      if (list != null) {
        list.foreach { child =>
          if (child.isDirectory) walk(child)
          else if (child.isFile && child.getName.endsWith(".parquet")) buf += child
        }
      }
    }
    walk(dir)
    buf.toArray.sortBy(_.getAbsolutePath)
  }

  override def apply(args: Array[String]): java.lang.Long = {
    val parquetDir = stripFileScheme(args(0))
    val vortexDir = stripFileScheme(args(1))
    val schemaJson = args(2)
    val includeCols: Array[String] =
      if (args.length >= 4 && args(3).nonEmpty) args(3).split(",") else Array.empty
    val numThreads: Int =
      if (args.length >= 5 && args(4).nonEmpty) args(4).toInt else 4

    val schema = DataType.fromJson(schemaJson).asInstanceOf[StructType]
    val colNames = if (includeCols.nonEmpty) includeCols else schema.fieldNames

    val srcRoot = new File(parquetDir).getCanonicalFile
    val outDir = new File(vortexDir)
    outDir.mkdirs()
    Files.write(
      Paths.get(vortexDir, SCHEMA_FILE),
      schemaJson.getBytes(StandardCharsets.UTF_8))

    val pqFiles = collectParquetFiles(srcRoot)
    val optsBuilder = ParquetOptions.builder()
    if (includeCols.nonEmpty) optsBuilder.includeColumn(includeCols: _*)
    val opts = optsBuilder.build()

    val hasHiveDirs = pqFiles.exists(f => f.getParentFile.getCanonicalPath != srcRoot.getPath)
    val totalBytes = new java.util.concurrent.atomic.AtomicLong(0L)
    val outRootPath = outDir.getCanonicalFile.toPath
    val pool = java.util.concurrent.Executors.newFixedThreadPool(math.max(1, numThreads))
    try {
      val futures = pqFiles.zipWithIndex.map { case (pf, idx) =>
        pool.submit(new Runnable {
          override def run(): Unit = {
            val outFile = if (hasHiveDirs) {
              val relParent = srcRoot.toPath.relativize(pf.getParentFile.getCanonicalFile.toPath)
              val leafDir = outRootPath.resolve(relParent).toFile
              leafDir.mkdirs()
              new File(leafDir, pf.getName.stripSuffix(".parquet") + ".vortex")
            } else {
              new File(outDir, f"part-$idx%05d.vortex")
            }
            if (!outFile.exists() || outFile.length() == 0L) {
              withResource(CudfTable.readParquet(opts, pf)) { tbl =>
                if (tbl.getRowCount > 0) {
                  VortexNative.writeTable(tbl, colNames, outFile.getAbsolutePath)
                }
              }
            }
            if (outFile.exists()) {
              totalBytes.addAndGet(outFile.length())
            }
          }
        })
      }
      futures.foreach(_.get())
    } finally {
      pool.shutdown()
    }
    java.lang.Long.valueOf(totalBytes.get())
  }
}
