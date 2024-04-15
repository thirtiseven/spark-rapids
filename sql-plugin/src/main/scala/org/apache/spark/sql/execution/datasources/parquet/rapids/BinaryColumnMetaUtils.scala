/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

package org.apache.spark.sql.execution.datasources.parquet.rapids

import java.util
import java.util.Optional

import scala.collection.JavaConverters.asScalaIteratorConverter
import scala.collection.mutable

import ai.rapids.cudf.{DType, HostColumnVector, HostColumnVectorCore, HostMemoryBuffer}
import org.apache.parquet.column.ColumnDescriptor
import org.apache.parquet.column.page.{DataPage, DataPageV1, DataPageV2, DictionaryPage, PageReader, PageReadStore}
import org.apache.parquet.column.values.dictionary.PlainValuesDictionary.PlainBinaryDictionary
import org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName

import org.apache.spark.internal.Logging


case class DictLatMatPatch(dictVector: HostColumnVector, dictPageOffsets: Array[Int])

case class BinaryColumnMetaSummary(isAllDictEncoded: Boolean,
                                   sizeInBytes: Long,
                                   sizeInRows: Long,
                                   dictPages: Option[Array[DictionaryPage]])

object BinaryColumnMetaUtils extends Logging {

  // Go through each RowGroup and each page inside them to collect 4 items:
  // * whether all pages are dictionary-encoded or not
  // * total decompressed byte size
  // * total row size
  // * all decompressed dictionary pages
  def inspectColumn(rowGroups: Seq[PageReadStore],
                    descriptor: ColumnDescriptor): BinaryColumnMetaSummary = {
    require(descriptor.getPrimitiveType.getPrimitiveTypeName == PrimitiveTypeName.BINARY)

    val dictPages = mutable.ArrayBuffer[DictionaryPage]()
    var isAllDictEncoded = true
    var sizeInBytes = 0L
    var sizeInRows = 0L
    rowGroups.foreach { rowGroup =>
      val pageReader = rowGroup.getPageReader(descriptor)
      val (encoded, bs, rs) = extractPageMeta(pageReader)
      isAllDictEncoded = isAllDictEncoded && encoded
      sizeInBytes = sizeInBytes + bs
      sizeInRows = sizeInRows + rs
      if (isAllDictEncoded) {
        pageReader.readDictionaryPage() match {
          case page if page == null =>
            isAllDictEncoded = false
          case page =>
            dictPages += page
        }
      }
    }

    BinaryColumnMetaSummary(isAllDictEncoded, sizeInBytes, sizeInRows,
      if (!isAllDictEncoded) None else Some(dictPages.toArray))
  }

  private def extractPageMeta(pageReader: PageReader): (Boolean, Long, Long) = {
    require(ccPageReader.isInstance(pageReader),
      "Only supports org.apache.parquet.hadoop.ColumnChunkPageReadStore.ColumnChunkPageReader")

    val rawPagesField = ccPageReader.getDeclaredField("compressedPages")
    rawPagesField.setAccessible(true)

    // For parquet-hadoop <= 1.10.X, compressedPages is stored as LinkedList
    // For parquet-hadoop >= 1.11.X, compressedPages is stored as ArrayDeque
    val pages = rawPagesField.get(pageReader)
    val pageIterator = pages match {
      case _: util.ArrayDeque[_] =>
        pages.asInstanceOf[util.ArrayDeque[DataPage]].iterator()
      case _: util.LinkedList[_] =>
        pages.asInstanceOf[util.LinkedList[DataPage]].iterator()
      case _ =>
        throw new IllegalArgumentException(
          s"Get unknown type ${pages.getClass} from ColumnChunkPageReader::compressedPages")
    }

    val (dctEncoded, bs, rs) = pageIterator.asScala.foldLeft((true, 0L, 0L)) {
      case ((encoded, byteSize, rowSize), p: DataPage) => p match {
        case p: DataPageV1 =>
          (encoded && p.getValueEncoding.usesDictionary(),
            byteSize + p.getUncompressedSize,
            rowSize + p.getValueCount)
        case p: DataPageV2 =>
          (encoded && p.getDataEncoding.usesDictionary(),
            byteSize + p.getUncompressedSize,
            rowSize + p.getRowCount)
      }
    }

    (dctEncoded, bs, rs)
  }

  def buildDictLateMatPatch(dictPages: Seq[DictionaryPage],
                            descriptor: ColumnDescriptor): DictLatMatPatch = {
    val pageOffsets = mutable.ArrayBuffer[Int](0)
    var rowNum: Int = 0
    val dictionaries = dictPages.map { dictPage =>
      val dictionary = dictPage.getEncoding.initDictionary(descriptor, dictPage)
        .asInstanceOf[PlainBinaryDictionary]
      rowNum += dictionary.getMaxId + 1
      pageOffsets += rowNum
      dictionary
    }

    var charNum: Int = 0
    val offsetBuf = HostMemoryBuffer.allocate((rowNum + 1) * 4L)
    offsetBuf.setInt(0, 0)
    var i = 1
    dictionaries.foreach { dict =>
      (0 to dict.getMaxId).foreach { j =>
        charNum += dict.decodeToBinary(j).length()
        offsetBuf.setInt(i * 4L, charNum)
        i += 1
      }
    }

    val charBuf = HostMemoryBuffer.allocate(charNum)
    i = 0
    dictionaries.foreach { dict =>
      (0 to dict.getMaxId).foreach { j =>
        val ba = dict.decodeToBinary(j).getBytes
        charBuf.setBytes(offsetBuf.getInt(i * 4L), ba, 0, ba.length)
        i += 1
      }
    }

    val dictVector = new HostColumnVector(DType.STRING, rowNum, Optional.of(0L),
      charBuf, null, offsetBuf, new util.ArrayList[HostColumnVectorCore]())
    DictLatMatPatch(dictVector, pageOffsets.toArray)
  }

  private val ccPageReader: Class[_] = {
    val ccPageReadStore = Class.forName("org.apache.parquet.hadoop.ColumnChunkPageReadStore")
    val ccPageReader = ccPageReadStore.getDeclaredClasses.find { memberClz =>
      memberClz.getSimpleName.equals("ColumnChunkPageReader")
    }
    require(ccPageReader.nonEmpty, "can NOT find the Class definition of ColumnChunkPageReader")
    ccPageReader.get
  }

}
