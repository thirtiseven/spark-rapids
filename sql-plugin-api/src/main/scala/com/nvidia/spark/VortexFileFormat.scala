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

package com.nvidia.spark

import java.io.File
import java.net.URI
import java.nio.charset.StandardCharsets
import java.nio.file.Files

import org.apache.hadoop.fs.{FileStatus, Path}
import org.apache.hadoop.mapreduce.Job

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.execution.datasources.{FileFormat, OutputWriterFactory}
import org.apache.spark.sql.sources.DataSourceRegister
import org.apache.spark.sql.types.{DataType, StructType}

/**
 * Spark V1 FileFormat for Standard Vortex (.vortex) files.
 *
 * Exposing Vortex as a native V1 FileFormat (HadoopFsRelation) enables full parity with
 * Parquet across Spark's PartitioningAwareFileIndex, native 1-day Hive Dynamic Partition
 * Pruning (DPP), V1 FileSourceStrategy post-subquery column pruning, and RAPIDS logical
 * optimizer rules (JoinReorderRule, OptimizeSubquerySharedScanRule, AggPushdownRule).
 */
class VortexFileFormat extends FileFormat with DataSourceRegister with Serializable {
  override def shortName(): String = "vortex"

  override def toString: String = "Vortex"

  override def hashCode(): Int = getClass.hashCode()

  override def equals(other: Any): Boolean = other.isInstanceOf[VortexFileFormat]

  override def isSplitable(
      sparkSession: SparkSession,
      options: Map[String, String],
      path: Path): Boolean = false

  private def toLocalPath(p: Path): String = {
    val s = p.toString
    if (s.startsWith("file:")) {
      new URI(s).getPath
    } else {
      s
    }
  }

  private def findSchemaFile(startDir: File): Option[File] = {
    var cur = startDir
    var depth = 0
    while (cur != null && depth <= 5) {
      val cand = new File(cur, "_vortex_schema.json")
      if (cand.isFile) {
        return Some(cand)
      }
      cur = cur.getParentFile
      depth += 1
    }
    None
  }

  override def inferSchema(
      sparkSession: SparkSession,
      options: Map[String, String],
      files: Seq[FileStatus]): Option[StructType] = {
    val optPath = options.get("path").map { raw =>
      if (raw.startsWith("file:")) new File(new URI(raw).getPath) else new File(raw)
    }
    val schemaFileOpt = optPath.flatMap(findSchemaFile).orElse {
      files.headOption.flatMap { st =>
        val f = new File(toLocalPath(st.getPath))
        findSchemaFile(if (f.isDirectory) f else f.getParentFile)
      }
    }
    schemaFileOpt.map { f =>
      val json = new String(Files.readAllBytes(f.toPath), StandardCharsets.UTF_8)
      DataType.fromJson(json).asInstanceOf[StructType]
    }
  }

  override def prepareWrite(
      sparkSession: SparkSession,
      job: Job,
      options: Map[String, String],
      dataSchema: StructType): OutputWriterFactory = {
    throw new UnsupportedOperationException("VortexFileFormat is read-only; use VortexTools to write")
  }
}
