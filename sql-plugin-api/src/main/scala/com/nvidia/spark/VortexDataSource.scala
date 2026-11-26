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

import java.util

import com.nvidia.spark.rapids.ShimLoader

import org.apache.spark.sql.connector.catalog.{Table, TableProvider}
import org.apache.spark.sql.connector.expressions.Transform
import org.apache.spark.sql.sources.DataSourceRegister
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.util.CaseInsensitiveStringMap

/**
 * Root-classloader Spark DataSource V2 wrapper for Tier 1 Standard Vortex GPU columnar files
 * (powered by official vortex-cuda-ffi -> ArrowDeviceArray -> cudf::from_arrow_device).
 */
class VortexDataSource extends TableProvider with DataSourceRegister {
  private lazy val impl: TableProvider = ShimLoader.newVortexDataSource()

  override def shortName(): String = "vortex-gpu"

  override def supportsExternalMetadata(): Boolean = impl.supportsExternalMetadata()

  override def inferSchema(options: CaseInsensitiveStringMap): StructType =
    impl.inferSchema(options)

  override def getTable(
      schema: StructType,
      partitioning: Array[Transform],
      properties: util.Map[String, String]): Table =
    impl.getTable(schema, partitioning, properties)
}

object VortexTools {
  def convertParquetDirToVortex(
      parquetDir: String,
      vortexDir: String,
      schemaJson: String,
      includeCols: String = "",
      numThreads: Int = 4): Long = {
    ShimLoader.newVortexConverter()
      .apply(Array(parquetDir, vortexDir, schemaJson, includeCols, numThreads.toString))
      .longValue()
  }

  /**
   * Compacts a Hive date-partitioned Parquet table into 1:1 identical contiguous date-sorted
   * big Parquet files (Snappy, 128 MB row groups, ROWGROUP stats) AND Standard Vortex files
   * with a _vortex_zonemap.tsv manifest for Spark 3.5 SupportsRuntimeFiltering DPP!
   */
  def compactHiveParquetToSortedPair(
      rawHivePqDir: String,
      outPqDir: String,
      outVxDir: String,
      fullSchemaJson: String,
      partCol: String,
      targetChunkMB: Int = 512,
      numThreads: Int = 4): Long = {
    ShimLoader.newVortexConverter()
      .apply(Array(
        rawHivePqDir, outVxDir, fullSchemaJson, "", numThreads.toString,
        "COMPACT_SORTED_PAIR", outPqDir, partCol, targetChunkMB.toString))
      .longValue()
  }
}
