/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

package org.apache.spark.sql.rapids

import ai.rapids.cudf
import ai.rapids.cudf.{BinaryOp, CudfException, DType}
import com.nvidia.spark.rapids.{GpuColumnVector, GpuUnaryExpression}
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.jni.Protobuf
import com.nvidia.spark.rapids.shims.NullIntolerantShim

import org.apache.spark.sql.catalyst.expressions.{ExpectsInputTypes, Expression}
import org.apache.spark.sql.types._

/**
 * GPU implementation for Spark's `from_protobuf` decode path.
 *
 * This is designed to replace `org.apache.spark.sql.protobuf.ProtobufDataToCatalyst` when
 * supported.
 *
 * The implementation uses a two-pass approach in the CUDA kernel:
 * - Pass 1: Scan all messages once, recording (offset, length) for each requested field
 * - Pass 2: Extract data in parallel using the recorded locations
 *
 * This is significantly faster than per-field parsing when decoding multiple fields,
 * as each message is only parsed once regardless of the number of fields.
 *
 * @param fullSchema The complete output schema (must match the original expression's dataType)
 * @param decodedFieldIndices Indices into fullSchema for fields that will be decoded by GPU.
 *                            Fields not in this array will be null columns.
 * @param fieldNumbers Protobuf field numbers for decoded fields (parallel to decodedFieldIndices)
 * @param cudfTypeIds cuDF type IDs for ALL fields in fullSchema
 * @param cudfTypeScales Encodings for decoded fields (parallel to decodedFieldIndices)
 * @param isRequired Whether each decoded field is required (parallel to decodedFieldIndices).
 *                   Required fields missing in failOnErrors mode will cause an exception.
 * @param hasDefaultValue Whether each decoded field has a default value
 *                        (parallel to decodedFieldIndices)
 * @param defaultInts Default values for int/long/enum fields
 *                    (parallel to decodedFieldIndices)
 * @param defaultFloats Default values for float/double fields
 *                      (parallel to decodedFieldIndices)
 * @param defaultBools Default values for bool fields
 *                     (parallel to decodedFieldIndices)
 * @param defaultStrings Default values for string/bytes fields as UTF-8 bytes
 *                       (parallel to decodedFieldIndices)
 * @param enumValidValues Valid enum values for each field (null if not an enum). Unknown values
 *                        will be set to null to match Spark CPU PERMISSIVE mode behavior.
 * @param failOnErrors If true, throw exception on malformed data; if false, return null
 */
case class GpuFromProtobuf(
    fullSchema: StructType,
    decodedFieldIndices: Array[Int],
    fieldNumbers: Array[Int],
    cudfTypeIds: Array[Int],
    cudfTypeScales: Array[Int],
    isRequired: Array[Boolean],
    hasDefaultValue: Array[Boolean],
    defaultInts: Array[Long],
    defaultFloats: Array[Double],
    defaultBools: Array[Boolean],
    defaultStrings: Array[Array[Byte]],
    enumValidValues: Array[Array[Int]],
    failOnErrors: Boolean,
    child: Expression)
  extends GpuUnaryExpression with ExpectsInputTypes with NullIntolerantShim {

  override def inputTypes: Seq[AbstractDataType] = Seq(BinaryType)

  override def dataType: DataType = fullSchema.asNullable

  override def nullable: Boolean = true

  // Lazy computation of unsupported field indices (complex types like StructType)
  @transient
  private lazy val unsupportedFieldIndices: Set[Int] = {
    fullSchema.fields.zipWithIndex.collect {
      case (sf, idx) if !GpuFromProtobuf.isTypeSupported(sf.dataType) => idx
    }.toSet
  }

  override protected def doColumnar(input: GpuColumnVector): cudf.ColumnVector = {
    val numRows = input.getRowCount.toInt

    // Call the optimized JNI API that:
    // 1. Uses fused kernel to scan all fields in one pass
    // 2. Creates LIST<INT8> directly for bytes fields (no intermediate strings column)
    // 3. Returns struct with decoded fields + null columns for supported types
    // 4. Validates required fields are present (in failOnErrors mode)
    // 5. Fills default values for missing fields with hasDefaultValue=true
    val jniResult = try {
      Protobuf.decodeToStruct(
        input.getBase,
        fullSchema.fields.length,  // total number of fields in output
        decodedFieldIndices,       // which fields to decode
        fieldNumbers,              // protobuf field numbers
        cudfTypeIds,               // types for ALL fields (INT8 placeholder for unsupported)
        cudfTypeScales,            // encodings for decoded fields
        isRequired,                // whether each decoded field is required
        hasDefaultValue,           // whether each decoded field has a default value
        defaultInts,               // default values for int/long/enum fields
        defaultFloats,             // default values for float/double fields
        defaultBools,              // default values for bool fields
        defaultStrings,            // default values for string/bytes fields
        enumValidValues,           // valid enum values for each field (null if not enum)
        failOnErrors)
    } catch {
      case e: CudfException if failOnErrors =>
        // Re-throw as a SparkException for consistent error handling
        throw new org.apache.spark.SparkException("Malformed protobuf message", e)
    }

    // If there are fields with unsupported types, we need to replace placeholder columns
    // with properly typed null columns
    val result = if (unsupportedFieldIndices.isEmpty) {
      jniResult
    } else {
      withResource(jniResult) { struct =>
        // Build children array, replacing placeholders with properly typed null columns
        val children = new Array[cudf.ColumnVector](fullSchema.fields.length)
        try {
          for (i <- fullSchema.fields.indices) {
            if (unsupportedFieldIndices.contains(i)) {
              // Create properly typed null column for unsupported types
              children(i) = GpuFromProtobuf.createNullColumn(fullSchema.fields(i).dataType, numRows)
            } else {
              // Copy the column from JNI result (incRefCount to share ownership)
              children(i) = struct.getChildColumnView(i).copyToColumnVector()
            }
          }
          cudf.ColumnVector.makeStruct(numRows, children: _*)
        } finally {
          children.foreach(c => if (c != null) c.close())
        }
      }
    }

    // Apply input nulls to output
    if (input.getBase.hasNulls) {
      withResource(result) { _ =>
        result.mergeAndSetValidity(BinaryOp.BITWISE_AND, input.getBase)
      }
    } else {
      result
    }
  }
}

/**
 * GPU implementation for Spark's `from_protobuf` with nested and repeated field support.
 *
 * This variant uses a flattened schema representation where nested fields have parent
 * indices pointing to their containing message field.
 *
 * @param fullSchema The complete output schema
 * @param fieldNumbers Protobuf field numbers for all fields in the flattened schema
 * @param parentIndices Parent field index for each field (-1 for top-level)
 * @param depthLevels Nesting depth for each field (0 for top-level)
 * @param wireTypes Expected wire type for each field
 * @param outputTypeIds cudf type ids for output columns
 * @param encodings Encoding info for each field
 * @param isRepeated Whether each field is a repeated field (array)
 * @param isRequired Whether each field is required (proto2)
 * @param hasDefaultValue Whether each field has a default value
 * @param defaultInts Default values for int/long/enum fields
 * @param defaultFloats Default values for float/double fields
 * @param defaultBools Default values for bool fields
 * @param defaultStrings Default values for string/bytes fields
 * @param enumValidValues Valid enum values for each field
 * @param failOnErrors If true, throw exception on malformed data
 */
/**
 * GPU implementation for Spark's `from_protobuf` decode path for nested/repeated types.
 *
 * This implementation supports schema projection: only fields in `decodedTopLevelIndices`
 * are decoded by the GPU. Fields not in this array will be filled with null columns in
 * post-processing to ensure the output matches `fullSchema`.
 *
 * @param fullSchema The complete output schema (must match the original expression's dataType)
 * @param decodedTopLevelIndices Indices in fullSchema for top-level fields decoded by GPU.
 *                               Must be sorted in ascending order.
 * @param fieldNumbers Protobuf field numbers for all fields in flattened schema
 * @param parentIndices Parent indices for all fields (-1 for top-level)
 * @param depthLevels Nesting depth for all fields (0 for top-level)
 * @param wireTypes Wire types for all fields
 * @param outputTypeIds cuDF type IDs for all fields
 * @param encodings Encodings for all fields
 * @param isRepeated Whether each field is repeated
 * @param isRequired Whether each field is required
 * @param hasDefaultValue Whether each field has a default value
 * @param defaultInts Default int/long values
 * @param defaultFloats Default float/double values
 * @param defaultBools Default bool values
 * @param defaultStrings Default string/bytes values
 * @param enumValidValues Valid enum values for each field
 * @param failOnErrors If true, throw exception on malformed data
 */
/**
 * @param nestedPrunedFields For nested schema projection: maps top-level field name to the
 *                           ordered list of decoded child field names. Only present for fields
 *                           where children were pruned. If empty map, no nested pruning.
 */
case class GpuFromProtobufNested(
    fullSchema: StructType,
    decodedTopLevelIndices: Array[Int],
    fieldNumbers: Array[Int],
    parentIndices: Array[Int],
    depthLevels: Array[Int],
    wireTypes: Array[Int],
    outputTypeIds: Array[Int],
    encodings: Array[Int],
    isRepeated: Array[Boolean],
    isRequired: Array[Boolean],
    hasDefaultValue: Array[Boolean],
    defaultInts: Array[Long],
    defaultFloats: Array[Double],
    defaultBools: Array[Boolean],
    defaultStrings: Array[Array[Byte]],
    enumValidValues: Array[Array[Int]],
    nestedPrunedFields: Map[String, Seq[String]],
    failOnErrors: Boolean,
    child: Expression)
  extends GpuUnaryExpression with ExpectsInputTypes with NullIntolerantShim {

  override def inputTypes: Seq[AbstractDataType] = Seq(BinaryType)

  // Output full schema so downstream GetStructField ordinals remain valid.
  // The GPU decoder only decodes needed fields (schema projection), but the
  // output is expanded to match fullSchema by inserting null columns.
  override def dataType: DataType = fullSchema.asNullable

  override def nullable: Boolean = true

  // Schema projection is active when not all top-level fields are decoded,
  // or when nested children are pruned.
  private val needsSchemaExpansion: Boolean =
    decodedTopLevelIndices.length != fullSchema.fields.length ||
    nestedPrunedFields.nonEmpty

  override protected def doColumnar(input: GpuColumnVector): cudf.ColumnVector = {
    val numRows = input.getRowCount.toInt
    val jniResult = try {
      Protobuf.decodeNestedToStruct(
        input.getBase,
        fieldNumbers,
        parentIndices,
        depthLevels,
        wireTypes,
        outputTypeIds,
        encodings,
        isRepeated,
        isRequired,
        hasDefaultValue,
        defaultInts,
        defaultFloats,
        defaultBools,
        defaultStrings,
        enumValidValues,
        failOnErrors)
    } catch {
      case e: CudfException if failOnErrors =>
        throw new org.apache.spark.SparkException("Malformed protobuf message", e)
    }

    // Expand to full schema if needed (insert null columns for non-decoded fields)
    val expanded = if (needsSchemaExpansion) {
      withResource(jniResult) { decoded =>
        expandToFullSchema(decoded, numRows)
      }
    } else {
      jniResult
    }

    // Apply input nulls to output
    if (input.getBase.hasNulls) {
      withResource(expanded) { _ =>
        expanded.mergeAndSetValidity(BinaryOp.BITWISE_AND, input.getBase)
      }
    } else {
      expanded
    }
  }

  /**
   * Expand a decoded struct (with only decoded fields) to match fullSchema
   * by inserting null columns for non-decoded fields. For fields with nested
   * schema pruning, inserts null columns for pruned nested children.
   */
  private def expandToFullSchema(decoded: cudf.ColumnVector, numRows: Int): cudf.ColumnVector = {
    val children = new Array[cudf.ColumnVector](fullSchema.fields.length)
    var decodedIdx = 0

    try {
      for (i <- fullSchema.fields.indices) {
        if (decodedIdx < decodedTopLevelIndices.length &&
            decodedTopLevelIndices(decodedIdx) == i) {
          val fieldName = fullSchema.fields(i).name
          nestedPrunedFields.get(fieldName) match {
            case Some(decodedNames) =>
              // Nested pruning: expand inner struct by inserting null children
              withResource(decoded.getChildColumnView(decodedIdx)) { childView =>
                children(i) = expandNestedField(
                  childView, fullSchema.fields(i).dataType, decodedNames, numRows)
              }
            case None =>
              withResource(decoded.getChildColumnView(decodedIdx)) { childView =>
                children(i) = childView.copyToColumnVector()
              }
          }
          decodedIdx += 1
        } else {
          children(i) = GpuColumnVector.columnVectorFromNull(numRows, fullSchema.fields(i).dataType)
        }
      }

      cudf.ColumnVector.makeStruct(numRows, children: _*)
    } finally {
      children.foreach(col => if (col != null) col.close())
    }
  }

  /**
   * Expand a decoded field with pruned nested children to match the full type.
   */
  private def expandNestedField(
      decoded: cudf.ColumnView,
      fullType: DataType,
      decodedChildNames: Seq[String],
      numRows: Int): cudf.ColumnVector = {
    fullType match {
      case st: StructType =>
        expandStructChildren(decoded, st, decodedChildNames, decoded.getRowCount.toInt)

      case ArrayType(st: StructType, _) =>
        // LIST<STRUCT>: expand struct child, rebuild list
        withResource(decoded.getChildColumnView(1)) { structChild =>
          val structRows = structChild.getRowCount.toInt
          withResource(expandStructChildren(structChild, st, decodedChildNames, structRows)) {
            expandedStruct =>
              withResource(decoded.replaceListChild(expandedStruct)) { expandedList =>
                expandedList.copyToColumnVector()
              }
          }
        }

      case _ =>
        decoded.copyToColumnVector()
    }
  }

  /**
   * Expand a STRUCT by inserting null columns for pruned children.
   */
  private def expandStructChildren(
      decoded: cudf.ColumnView,
      targetSchema: StructType,
      decodedChildNames: Seq[String],
      numRows: Int): cudf.ColumnVector = {
    val children = new Array[cudf.ColumnVector](targetSchema.fields.length)
    var decodedChildIdx = 0

    try {
      for (i <- targetSchema.fields.indices) {
        if (decodedChildIdx < decodedChildNames.length &&
            decodedChildNames(decodedChildIdx) == targetSchema.fields(i).name) {
          withResource(decoded.getChildColumnView(decodedChildIdx)) { childView =>
            children(i) = childView.copyToColumnVector()
          }
          decodedChildIdx += 1
        } else {
          children(i) = GpuColumnVector.columnVectorFromNull(
            numRows, targetSchema.fields(i).dataType)
        }
      }

      cudf.ColumnVector.makeStruct(numRows, children: _*)
    } finally {
      children.foreach(col => if (col != null) col.close())
    }
  }
}

object GpuFromProtobuf {
  // Encodings from com.nvidia.spark.rapids.jni.Protobuf
  val ENC_DEFAULT = 0
  val ENC_FIXED   = 1
  val ENC_ZIGZAG  = 2

  /**
   * Maps a Spark DataType to the corresponding cuDF native type ID.
   * Note: The encoding (varint/zigzag/fixed) is determined by the protobuf field type,
   * not the Spark data type, so it must be set separately based on the protobuf schema.
   *
   * @return Some(typeId) for supported types, None for unsupported types
   */
  def sparkTypeToCudfIdOpt(dt: DataType): Option[Int] = dt match {
    case BooleanType => Some(DType.BOOL8.getTypeId.getNativeId)
    case IntegerType => Some(DType.INT32.getTypeId.getNativeId)
    case LongType => Some(DType.INT64.getTypeId.getNativeId)
    case FloatType => Some(DType.FLOAT32.getTypeId.getNativeId)
    case DoubleType => Some(DType.FLOAT64.getTypeId.getNativeId)
    case StringType => Some(DType.STRING.getTypeId.getNativeId)
    case BinaryType => Some(DType.LIST.getTypeId.getNativeId)
    case _ => None
  }

  /**
   * Check if a Spark DataType is supported by the GPU protobuf decoder.
   */
  def isTypeSupported(dt: DataType): Boolean = sparkTypeToCudfIdOpt(dt).isDefined

  /**
   * Create an all-null column of the specified Spark DataType.
   * This is used for fields with unsupported types (nested structs, arrays, etc.)
   * that are not decoded but need to be present in the output struct.
   */
  def createNullColumn(dt: DataType, numRows: Int): cudf.ColumnVector = {
    // Helper to create null arrays for boxed types
    def nullBools = Array.fill[java.lang.Boolean](numRows)(null)
    def nullInts = Array.fill[java.lang.Integer](numRows)(null)
    def nullLongs = Array.fill[java.lang.Long](numRows)(null)
    def nullFloats = Array.fill[java.lang.Float](numRows)(null)
    def nullDoubles = Array.fill[java.lang.Double](numRows)(null)

    dt match {
      case BooleanType => cudf.ColumnVector.fromBoxedBooleans(nullBools: _*)
      case IntegerType => cudf.ColumnVector.fromBoxedInts(nullInts: _*)
      case LongType => cudf.ColumnVector.fromBoxedLongs(nullLongs: _*)
      case FloatType => cudf.ColumnVector.fromBoxedFloats(nullFloats: _*)
      case DoubleType => cudf.ColumnVector.fromBoxedDoubles(nullDoubles: _*)
      case StringType => cudf.ColumnVector.fromStrings(Array.fill[String](numRows)(null): _*)
      case BinaryType =>
        // Binary is LIST<INT8> - create all-null list column using Scalar API
        val elementType = new cudf.HostColumnVector.BasicType(true, DType.INT8)
        withResource(cudf.Scalar.listFromNull(elementType)) { nullScalar =>
          cudf.ColumnVector.fromScalar(nullScalar, numRows)
        }
      case st: StructType =>
        // Recursively create null columns for struct fields
        val children = st.fields.map(f => createNullColumn(f.dataType, numRows))
        try {
          withResource(cudf.ColumnVector.makeStruct(numRows, children: _*)) { structCol =>
            // Set all rows to null - mergeAndSetValidity returns a NEW column
            withResource(cudf.ColumnVector.fromBoxedBooleans(nullBools: _*)) { nullMask =>
              structCol.mergeAndSetValidity(BinaryOp.BITWISE_AND, nullMask)
            }
          }
        } finally {
          children.foreach(_.close())
        }
      case ArrayType(elementType, _) =>
        // Create empty arrays with all nulls using Scalar API
        val cudfElementDType = sparkTypeToCudfIdOpt(elementType)
          .map(id => DType.fromNative(id, 0))
          .getOrElse(DType.INT8)  // fallback for nested complex types
        val elemType = new cudf.HostColumnVector.BasicType(true, cudfElementDType)
        withResource(cudf.Scalar.listFromNull(elemType)) { nullScalar =>
          cudf.ColumnVector.fromScalar(nullScalar, numRows)
        }
      case MapType(keyType, valueType, _) =>
        // Maps are represented as LIST<STRUCT<key, value>> in cuDF
        // For all-null maps, we create a list column with STRUCT<key, value> element type
        val cudfKeyDType = sparkTypeToCudfIdOpt(keyType)
          .map(id => DType.fromNative(id, 0))
          .getOrElse(DType.INT8)
        val cudfValueDType = sparkTypeToCudfIdOpt(valueType)
          .map(id => DType.fromNative(id, 0))
          .getOrElse(DType.INT8)
        // Create the struct type for map entries (key, value)
        val keyFieldType = new cudf.HostColumnVector.BasicType(true, cudfKeyDType)
        val valueFieldType = new cudf.HostColumnVector.BasicType(true, cudfValueDType)
        val structType = new cudf.HostColumnVector.StructType(true, keyFieldType, valueFieldType)
        // Create an all-null map column (list of structs)
        withResource(cudf.Scalar.listFromNull(structType)) { nullScalar =>
          cudf.ColumnVector.fromScalar(nullScalar, numRows)
        }
      case _ =>
        // Fallback for any other types - create INT8 nulls as placeholder
        // This should not happen in practice since unsupported types should be caught earlier
        cudf.ColumnVector.fromBoxedBytes(Array.fill[java.lang.Byte](numRows)(null): _*)
    }
  }
}
