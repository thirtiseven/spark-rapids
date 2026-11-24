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

/** Root-classloader Spark DataSource V2 wrapper for lightweight GPU columnar (.lenc) files. */
class LencDataSource extends TableProvider with DataSourceRegister {
  private lazy val impl: TableProvider = ShimLoader.newLencDataSource()

  override def shortName(): String = "lenc"

  override def supportsExternalMetadata(): Boolean = impl.supportsExternalMetadata()

  override def inferSchema(options: CaseInsensitiveStringMap): StructType =
    impl.inferSchema(options)

  override def getTable(
      schema: StructType,
      partitioning: Array[Transform],
      properties: util.Map[String, String]): Table =
    impl.getTable(schema, partitioning, properties)
}

object LencTools {
  def convertParquetDirToLenc(
      parquetDir: String,
      lencDir: String,
      schemaJson: String): Long = {
    ShimLoader.newLencConverter().apply(Array(parquetDir, lencDir, schemaJson)).longValue()
  }
}
