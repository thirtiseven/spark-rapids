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

import com.nvidia.spark.rapids.ShimLoader

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, Expression}
import org.apache.spark.sql.columnar.CachedBatch
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.storage.StorageLevel

/**
 * Root-classloader Spark CachedBatchSerializer using GPU-Native lightweight encodings
 * (FOR256 + GPU DICT_BITPACK + dynamic 127-bigram mini-FSST + sparse-float nvCOMP LZ4):
 * both populate (encode) and read (decode) execute 100% on GPU with zero uncompressed D2H!
 */
class LightweightCachedBatchSerializer extends GpuCachedBatchSerializer {
  private lazy val realImpl: GpuCachedBatchSerializer =
    ShimLoader.newLightweightCachedBatchSerializer()

  override def supportsColumnarInput(schema: Seq[Attribute]): Boolean =
    realImpl.supportsColumnarInput(schema)

  override def convertInternalRowToCachedBatch(
      input: RDD[InternalRow],
      schema: Seq[Attribute],
      storageLevel: StorageLevel,
      conf: SQLConf): RDD[CachedBatch] =
    realImpl.convertInternalRowToCachedBatch(input, schema, storageLevel, conf)

  override def convertColumnarBatchToCachedBatch(
      input: RDD[ColumnarBatch],
      schema: Seq[Attribute],
      storageLevel: StorageLevel,
      conf: SQLConf): RDD[CachedBatch] =
    realImpl.convertColumnarBatchToCachedBatch(input, schema, storageLevel, conf)

  override def buildFilter(
      predicates: Seq[Expression],
      cachedAttributes: Seq[Attribute]): (Int, Iterator[CachedBatch]) => Iterator[CachedBatch] =
    realImpl.buildFilter(predicates, cachedAttributes)

  override def supportsColumnarOutput(schema: StructType): Boolean =
    realImpl.supportsColumnarOutput(schema)

  override def vectorTypes(attributes: Seq[Attribute], conf: SQLConf): Option[Seq[String]] =
    realImpl.vectorTypes(attributes, conf)

  override def convertCachedBatchToColumnarBatch(
      input: RDD[CachedBatch],
      cacheAttributes: Seq[Attribute],
      selectedAttributes: Seq[Attribute],
      conf: SQLConf): RDD[ColumnarBatch] =
    realImpl.convertCachedBatchToColumnarBatch(input, cacheAttributes, selectedAttributes, conf)

  override def convertCachedBatchToInternalRow(
      input: RDD[CachedBatch],
      cacheAttributes: Seq[Attribute],
      selectedAttributes: Seq[Attribute],
      conf: SQLConf): RDD[InternalRow] =
    realImpl.convertCachedBatchToInternalRow(input, cacheAttributes, selectedAttributes, conf)

  override def gpuConvertCachedBatchToColumnarBatch(
      input: RDD[CachedBatch],
      cacheAttributes: Seq[Attribute],
      selectedAttributes: Seq[Attribute],
      conf: SQLConf): RDD[ColumnarBatch] =
    realImpl.gpuConvertCachedBatchToColumnarBatch(input, cacheAttributes, selectedAttributes, conf)
}
