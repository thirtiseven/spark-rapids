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

/*** spark-rapids-shim-json-lines
{"spark": "340"}
{"spark": "341"}
{"spark": "342"}
{"spark": "343"}
{"spark": "344"}
{"spark": "350"}
{"spark": "351"}
{"spark": "352"}
{"spark": "353"}
{"spark": "354"}
{"spark": "355"}
{"spark": "356"}
{"spark": "357"}
{"spark": "400"}
{"spark": "401"}
spark-rapids-shim-json-lines ***/

package com.nvidia.spark.rapids.shims

import scala.collection.mutable
import scala.util.Try

import ai.rapids.cudf.DType
import com.nvidia.spark.rapids._
import com.nvidia.spark.rapids.jni.Protobuf.{WT_32BIT, WT_64BIT, WT_LEN, WT_VARINT}

import org.apache.spark.sql.catalyst.expressions.{
  AttributeReference, Expression, GetArrayStructFields, GetStructField, UnaryExpression
}
import org.apache.spark.sql.execution.ProjectExec
import org.apache.spark.sql.rapids.{GpuFromProtobuf, GpuFromProtobufNested}
import org.apache.spark.sql.types._

/**
 * Information about a protobuf field for schema projection support.
 */
private[shims] case class ProtobufFieldInfo(
    fieldNumber: Int,
    protoTypeName: String,
    sparkType: DataType,
    encoding: Int,
    isSupported: Boolean,
    unsupportedReason: Option[String],
    isRequired: Boolean,
    hasDefaultValue: Boolean,
    defaultValue: Option[Any],  // Stored as protobuf-java type, will be converted for JNI
    // Valid enum values (only for ENUM fields with enumsAsInts)
    enumValues: Option[Set[Int]] = None,
    isRepeated: Boolean = false  // Whether this is a repeated field
)

/**
 * Flattened field descriptor for nested protobuf schemas.
 * Used to represent a hierarchical schema as a linear array for GPU processing.
 */
private[shims] case class FlattenedFieldDescriptor(
    fieldNumber: Int,
    parentIdx: Int,          // Index of parent field in flattened array (-1 for top-level)
    depth: Int,              // Nesting depth (0 for top-level)
    wireType: Int,           // Protobuf wire type
    outputTypeId: Int,       // cudf type id for the output (element type for repeated)
    encoding: Int,           // Encoding (default/fixed/zigzag)
    isRepeated: Boolean,     // Whether this is a repeated field
    isRequired: Boolean,     // Whether this field is required (proto2)
    hasDefaultValue: Boolean,
    defaultInt: Long,
    defaultFloat: Double,
    defaultBool: Boolean,
    defaultString: Array[Byte],
    enumValidValues: Array[Int]
)

/**
 * Spark 3.4+ optional integration for spark-protobuf expressions.
 *
 * spark-protobuf is an external module, so these rules must be registered by reflection.
 */
object ProtobufExprShims {
  private[this] val protobufDataToCatalystClassName =
    "org.apache.spark.sql.protobuf.ProtobufDataToCatalyst"

  private[this] val sparkProtobufUtilsObjectClassName =
    "org.apache.spark.sql.protobuf.utils.ProtobufUtils$"

  def exprs: Map[Class[_ <: Expression], ExprRule[_ <: Expression]] = {
    try {
      val clazz = ShimReflectionUtils.loadClass(protobufDataToCatalystClassName)
        .asInstanceOf[Class[_ <: UnaryExpression]]
      Map(clazz.asInstanceOf[Class[_ <: Expression]] -> fromProtobufRule)
    } catch {
      case _: ClassNotFoundException => Map.empty
    }
  }

  private def fromProtobufRule: ExprRule[_ <: Expression] = {
    GpuOverrides.expr[UnaryExpression](
      "Decode a BinaryType column (protobuf) into a Spark SQL struct",
      ExprChecks.unaryProject(
        // Use TypeSig.all here because schema projection determines which fields
        // actually need GPU support. Detailed type checking is done in tagExprForGpu.
        TypeSig.all,
        TypeSig.all,
        TypeSig.BINARY,
        TypeSig.BINARY),
      (e, conf, p, r) => new UnaryExprMeta[UnaryExpression](e, conf, p, r) {

        // Full schema from the expression (must match original dataType for compatibility)
        private var fullSchema: StructType = _
        // Indices into fullSchema for fields that will be decoded by GPU
        private var decodedFieldIndices: Array[Int] = _
        private var fieldNumbers: Array[Int] = _
        // cudfTypeIds contains type IDs for ALL fields in fullSchema (for the new optimized API)
        private var cudfTypeIds: Array[Int] = _
        // cudfTypeScales: encodings for decoded fields (parallel to decodedFieldIndices)
        private var cudfTypeScales: Array[Int] = _
        // isRequired: whether each decoded field is required (parallel to decodedFieldIndices)
        private var isRequired: Array[Boolean] = _
        // hasDefaultValue: whether each decoded field has a default value
        private var hasDefaultValue: Array[Boolean] = _
        // defaultInts: default values for int/long/enum fields (0 if no default)
        private var defaultInts: Array[Long] = _
        // defaultFloats: default values for float/double fields (0.0 if no default)
        private var defaultFloats: Array[Double] = _
        // defaultBools: default values for bool fields (false if no default)
        private var defaultBools: Array[Boolean] = _
        // defaultStrings: default values for string/bytes fields (null if no default)
        private var defaultStrings: Array[Array[Byte]] = _
        // enumValidValues: valid enum values for each decoded field (null if not an enum)
        private var enumValidValues: Array[Array[Int]] = _
        private var failOnErrors: Boolean = _

        // Variables for nested API (used when schema has nested or repeated fields)
        private var useNestedApi: Boolean = false
        private var nestedFieldNumbers: Array[Int] = _
        private var nestedParentIndices: Array[Int] = _
        private var nestedDepthLevels: Array[Int] = _
        private var nestedWireTypes: Array[Int] = _
        private var nestedOutputTypeIds: Array[Int] = _
        private var nestedEncodings: Array[Int] = _
        private var nestedIsRepeated: Array[Boolean] = _
        private var nestedIsRequired: Array[Boolean] = _
        private var nestedHasDefaultValue: Array[Boolean] = _
        private var nestedDefaultInts: Array[Long] = _
        private var nestedDefaultFloats: Array[Double] = _
        private var nestedDefaultBools: Array[Boolean] = _
        private var nestedDefaultStrings: Array[Array[Byte]] = _
        private var nestedEnumValidValues: Array[Array[Int]] = _
        // Indices in fullSchema for top-level fields that were decoded (for schema projection)
        private var nestedDecodedTopLevelIndices: Array[Int] = _

        override def tagExprForGpu(): Unit = {
          fullSchema = e.dataType match {
            case st: StructType => st
            case other =>
              willNotWorkOnGpu(
                s"Only StructType output is supported for from_protobuf, got $other")
              return
          }

          val options = getOptionsMap(e)
          val supportedOptions = Set("enums.as.ints", "mode")
          val unsupportedOptions = options.keys.filterNot(supportedOptions.contains)
          if (unsupportedOptions.nonEmpty) {
            val keys = unsupportedOptions.mkString(",")
            willNotWorkOnGpu(
              s"from_protobuf options are not supported yet on GPU: $keys")
            return
          }

          val enumsAsInts = options.getOrElse("enums.as.ints", "false").toBoolean
          failOnErrors = options.getOrElse("mode", "PERMISSIVE").equalsIgnoreCase("FAILFAST")
          val messageName = getMessageName(e)

          // Try to get descriptor: either file path (Spark 3.4.x) or binary bytes (Spark 3.5+)
          val descFilePathOrBytes: Option[Either[String, Array[Byte]]] =
            getDescFilePath(e).map(Left(_)).orElse {
              getDescriptorBytes(e).map(Right(_))
            }

          if (descFilePathOrBytes.isEmpty) {
            willNotWorkOnGpu(
              "from_protobuf requires a descriptor set " +
                "(descFilePath or binaryFileDescriptorSet)")
            return
          }

          val msgDesc = try {
            // Spark 3.4.x: buildDescriptor(messageName, descFilePath: Option[String])
            // Spark 3.5+:  buildDescriptor(messageName, binaryFileDescriptorSet)
            buildMessageDescriptorWithSparkProtobuf(messageName, descFilePathOrBytes.get)
          } catch {
            case t: Throwable =>
              willNotWorkOnGpu(
                s"Failed to resolve protobuf descriptor for message '$messageName': " +
                  s"${t.getMessage}")
              return
          }

          // Step 1: Analyze all fields and build field info map
          val allFieldsInfo = analyzeAllFields(fullSchema, msgDesc, enumsAsInts, messageName)
          if (allFieldsInfo.isEmpty) {
            // Error was already reported in analyzeAllFields
            return
          }
          val fieldsInfoMap = allFieldsInfo.get

          // Step 2: Determine which fields are actually required by downstream operations
          val requiredFieldNames = analyzeRequiredFields(fieldsInfoMap.keySet)

          // Step 3: Check if all required fields are supported
          val unsupportedRequired = requiredFieldNames.filter { name =>
            fieldsInfoMap.get(name).exists(!_.isSupported)
          }

          if (unsupportedRequired.nonEmpty) {
            val reasons = unsupportedRequired.map { name =>
              val info = fieldsInfoMap(name)
              s"${name}: ${info.unsupportedReason.getOrElse("unknown reason")}"
            }
            willNotWorkOnGpu(
              s"Required fields not supported for from_protobuf: ${reasons.mkString(", ")}")
            return
          }

          // Step 4: Identify which fields in fullSchema need to be decoded
          // These are fields that are required AND supported
          val indicesToDecode = fullSchema.fields.zipWithIndex.collect {
            case (sf, idx) if requiredFieldNames.contains(sf.name) => idx
          }
          decodedFieldIndices = indicesToDecode

          // Step 5: Build cudfTypeIds for ALL fields in fullSchema
          // For unsupported types (nested struct, array, etc.), use INT8 as placeholder.
          // These placeholder columns will be replaced with properly typed null columns in Scala.
          cudfTypeIds = fullSchema.fields.map { sf =>
            GpuFromProtobuf.sparkTypeToCudfIdOpt(sf.dataType)
              .getOrElse(DType.INT8.getTypeId.getNativeId)  // placeholder for unsupported types
          }

          // Step 6: Build arrays for decoded fields only (parallel to decodedFieldIndices)
          val fnums = new Array[Int](indicesToDecode.length)
          val scales = new Array[Int](indicesToDecode.length)
          val required = new Array[Boolean](indicesToDecode.length)
          val hasDefaults = new Array[Boolean](indicesToDecode.length)
          val defInts = new Array[Long](indicesToDecode.length)
          val defFloats = new Array[Double](indicesToDecode.length)
          val defBools = new Array[Boolean](indicesToDecode.length)
          val defStrings = new Array[Array[Byte]](indicesToDecode.length)
          val enumVals = new Array[Array[Int]](indicesToDecode.length)

          indicesToDecode.zipWithIndex.foreach { case (schemaIdx, arrIdx) =>
            val sf = fullSchema.fields(schemaIdx)
            val info = fieldsInfoMap(sf.name)
            fnums(arrIdx) = info.fieldNumber
            scales(arrIdx) = info.encoding
            required(arrIdx) = info.isRequired
            hasDefaults(arrIdx) = info.hasDefaultValue

            // Convert default value to the appropriate JNI type
            if (info.hasDefaultValue && info.defaultValue.isDefined) {
              val defVal = info.defaultValue.get
              sf.dataType match {
                case BooleanType =>
                  defBools(arrIdx) = defVal.asInstanceOf[java.lang.Boolean]
                case IntegerType | LongType =>
                  // Protobuf returns Integer for int32, Long for int64
                  defInts(arrIdx) = defVal match {
                    case i: java.lang.Integer => i.longValue()
                    case l: java.lang.Long => l.longValue()
                    case _ => 0L
                  }
                case FloatType =>
                  defFloats(arrIdx) = defVal.asInstanceOf[java.lang.Float].doubleValue()
                case DoubleType =>
                  defFloats(arrIdx) = defVal.asInstanceOf[java.lang.Double]
                case StringType =>
                  // Protobuf returns String, convert to UTF-8 bytes
                  val str = defVal.asInstanceOf[String]
                  defStrings(arrIdx) = if (str != null) str.getBytes("UTF-8") else null
                case BinaryType =>
                  // Protobuf returns ByteString, convert to byte array
                  defStrings(arrIdx) = Try {
                    invoke0[Array[Byte]](defVal.asInstanceOf[AnyRef], "toByteArray")
                  }.getOrElse(null)
                case _ =>
                  // For other types (enums as ints, etc.)
                  defVal match {
                    case enumVal: AnyRef if info.protoTypeName == "ENUM" =>
                      defInts(arrIdx) = Try {
                        invoke0[java.lang.Integer](enumVal, "getNumber").longValue()
                      }.getOrElse(0L)
                    case _ => // leave as default (0)
                  }
              }
            }

            // Store enum valid values if this is an enum field
            info.enumValues match {
              case Some(values) => enumVals(arrIdx) = values.toArray.sorted
              case None => enumVals(arrIdx) = null
            }
          }

          fieldNumbers = fnums
          cudfTypeScales = scales
          isRequired = required
          hasDefaultValue = hasDefaults
          defaultInts = defInts
          defaultFloats = defFloats
          defaultBools = defBools
          defaultStrings = defStrings
          enumValidValues = enumVals

          // Check if we need the nested API (for repeated fields or nested messages)
          // Only check fields that will actually be decoded, not all fields in the schema
          useNestedApi = indicesToDecode.exists { idx =>
            val sf = fullSchema.fields(idx)
            val info = fieldsInfoMap(sf.name)
            info.isRepeated || info.protoTypeName == "MESSAGE"
          }

          // Double-check: verify all fields to be decoded are actually supported
          // (This catches edge cases where field analysis might have issues)
          val unsupportedInDecode = indicesToDecode.filter { idx =>
            val sf = fullSchema.fields(idx)
            fieldsInfoMap.get(sf.name).exists(!_.isSupported)
          }
          if (unsupportedInDecode.nonEmpty) {
            val reasons = unsupportedInDecode.map { idx =>
              val sf = fullSchema.fields(idx)
              val info = fieldsInfoMap(sf.name)
              s"${sf.name}: ${info.unsupportedReason.getOrElse("unknown reason")}"
            }
            willNotWorkOnGpu(
              s"Fields not supported for from_protobuf: ${reasons.mkString(", ")}")
            return
          }

          if (useNestedApi) {
            // Build flattened schema for nested API
            val flatFields = mutable.ArrayBuffer[FlattenedFieldDescriptor]()

            // Helper to add a field and its children recursively
            def addFieldWithChildren(
                sf: StructField,
                info: ProtobufFieldInfo,
                parentIdx: Int,
                depth: Int,
                nestedMsgDesc: AnyRef): Unit = {

              val currentIdx = flatFields.size

              val outputType = sf.dataType match {
                case ArrayType(elemType, _) =>
                  elemType match {
                    case _: StructType =>
                      // Repeated message field: ArrayType(StructType) - element type is STRUCT
                      DType.STRUCT.getTypeId.getNativeId
                    case other =>
                      GpuFromProtobuf.sparkTypeToCudfIdOpt(other)
                        .getOrElse(DType.INT8.getTypeId.getNativeId)
                  }
                case _: StructType =>
                  DType.STRUCT.getTypeId.getNativeId
                case other =>
                  GpuFromProtobuf.sparkTypeToCudfIdOpt(other)
                    .getOrElse(DType.INT8.getTypeId.getNativeId)
              }

              val wireType = getWireType(info.protoTypeName, info.encoding)

              val hasDefault = info.hasDefaultValue && info.defaultValue.isDefined
              val (defInt, defFloat, defBool, defString) = if (hasDefault) {
                val defVal = info.defaultValue.get
                sf.dataType match {
                  case BooleanType =>
                    val b = defVal.asInstanceOf[java.lang.Boolean].booleanValue()
                    (0L, 0.0, b, null: Array[Byte])
                  case IntegerType | LongType =>
                    val intVal = defVal match {
                      case i: java.lang.Integer => i.longValue()
                      case l: java.lang.Long => l.longValue()
                      case _ => 0L
                    }
                    (intVal, 0.0, false, null: Array[Byte])
                  case FloatType =>
                    val f = defVal.asInstanceOf[java.lang.Float].doubleValue()
                    (0L, f, false, null: Array[Byte])
                  case DoubleType =>
                    val d = defVal.asInstanceOf[java.lang.Double].doubleValue()
                    (0L, d, false, null: Array[Byte])
                  case StringType =>
                    val str = defVal.asInstanceOf[String]
                    val bytes = if (str != null) str.getBytes("UTF-8") else null
                    (0L, 0.0, false, bytes)
                  case _ => (0L, 0.0, false, null: Array[Byte])
                }
              } else {
                (0L, 0.0, false, null: Array[Byte])
              }

              val enumValsArr = info.enumValues.map(_.toArray.sorted).orNull

              flatFields += FlattenedFieldDescriptor(
                fieldNumber = info.fieldNumber,
                parentIdx = parentIdx,
                depth = depth,
                wireType = wireType,
                outputTypeId = outputType,
                encoding = info.encoding,
                isRepeated = info.isRepeated,
                isRequired = info.isRequired,
                hasDefaultValue = info.hasDefaultValue,
                defaultInt = defInt,
                defaultFloat = defFloat,
                defaultBool = defBool,
                defaultString = defString,
                enumValidValues = enumValsArr
              )

              // For nested struct types (including repeated message = ArrayType(StructType)), 
              // add child fields
              sf.dataType match {
                case st: StructType if nestedMsgDesc != null =>
                  // Non-repeated nested message - pruning OK
                  addChildFieldsFromStruct(st, nestedMsgDesc, sf.name, currentIdx, depth,
                    isRepeatedParent = false)
                  
                case ArrayType(st: StructType, _) if nestedMsgDesc != null =>
                  // Repeated message field - no pruning (expansion too expensive for arrays)
                  addChildFieldsFromStruct(st, nestedMsgDesc, sf.name, currentIdx, depth,
                    isRepeatedParent = true)
                  
                case _ => // Not a struct, no children to add
              }
            }
            
            // Helper to add child fields from a struct type
            // isRepeatedParent: if true, this is a repeated message field (ArrayType(StructType)).
            // Nested pruning is disabled for repeated fields because expanding the inner struct
            // (inserting null columns per element) is too expensive for large arrays.
            def addChildFieldsFromStruct(
                st: StructType,
                parentMsgDesc: AnyRef,
                fieldName: String,
                parentIdx: Int,
                parentDepth: Int,
                isRepeatedParent: Boolean): Unit = {
              val fd = invoke1[AnyRef](
                parentMsgDesc, "findFieldByName", classOf[String], fieldName)
              if (fd != null) {
                Try {
                  val childMsgDesc = invoke0[AnyRef](fd, "getMessageType")
                  // Filter children based on nested schema projection requirements.
                  // ONLY for non-repeated struct fields. For repeated message fields
                  // (ArrayType(StructType)), always decode all children because
                  // post-expansion of inner LIST struct elements is too memory-expensive.
                  val filteredFields = if (!isRepeatedParent) {
                    val requiredChildren = nestedFieldRequirements.get(fieldName)
                    requiredChildren match {
                      case Some(Some(childNames)) =>
                        st.fields.filter(f => childNames.contains(f.name))
                      case _ => st.fields
                    }
                  } else {
                    st.fields  // Always decode all children for repeated messages
                  }
                  filteredFields.foreach { childSf =>
                    val childFd = invoke1[AnyRef](
                      childMsgDesc, "findFieldByName", classOf[String], childSf.name)
                    if (childFd != null) {
                      val childProtoType = invoke0[AnyRef](childFd, "getType")
                      val childProtoTypeName = typeName(childProtoType)
                      val childFieldNumber = invoke0[java.lang.Integer](
                        childFd, "getNumber").intValue()
                      val childIsRepeated = Try {
                        invoke0[java.lang.Boolean](childFd, "isRepeated").booleanValue()
                      }.getOrElse(false)
                      val childIsRequired = Try {
                        invoke0[java.lang.Boolean](childFd, "isRequired").booleanValue()
                      }.getOrElse(false)
                      val childHasDefault = Try {
                        invoke0[java.lang.Boolean](childFd, "hasDefaultValue").booleanValue()
                      }.getOrElse(false)
                      val (_, _, childEncoding) = checkFieldSupport(
                        childSf.dataType, childProtoTypeName, childIsRepeated, enumsAsInts)

                      val childInfo = ProtobufFieldInfo(
                        fieldNumber = childFieldNumber,
                        protoTypeName = childProtoTypeName,
                        sparkType = childSf.dataType,
                        encoding = childEncoding,
                        isSupported = true,
                        unsupportedReason = None,
                        isRequired = childIsRequired,
                        hasDefaultValue = childHasDefault,
                        defaultValue = None,
                        enumValues = None,
                        isRepeated = childIsRepeated
                      )

                      addFieldWithChildren(
                        childSf, childInfo, parentIdx, parentDepth + 1, childMsgDesc)
                    }
                  }
                }
              }
            }

            // Only add top-level fields that are actually required (schema projection).
            // This significantly reduces GPU memory and computation for schemas with many
            // fields when only a few are needed. The Scala layer will post-process the
            // output to insert null columns for non-decoded fields.
            nestedDecodedTopLevelIndices = indicesToDecode
            indicesToDecode.foreach { schemaIdx =>
              val sf = fullSchema.fields(schemaIdx)
              val info = fieldsInfoMap(sf.name)
              addFieldWithChildren(sf, info, -1, 0, msgDesc)
            }

            // Populate nested API variables
            val flat = flatFields.toArray
            nestedFieldNumbers = flat.map(_.fieldNumber)
            nestedParentIndices = flat.map(_.parentIdx)
            nestedDepthLevels = flat.map(_.depth)
            nestedWireTypes = flat.map(_.wireType)
            nestedOutputTypeIds = flat.map(_.outputTypeId)
            nestedEncodings = flat.map(_.encoding)
            nestedIsRepeated = flat.map(_.isRepeated)
            nestedIsRequired = flat.map(_.isRequired)
            nestedHasDefaultValue = flat.map(_.hasDefaultValue)
            nestedDefaultInts = flat.map(_.defaultInt)
            nestedDefaultFloats = flat.map(_.defaultFloat)
            nestedDefaultBools = flat.map(_.defaultBool)
            nestedDefaultStrings = flat.map(_.defaultString)
            nestedEnumValidValues = flat.map(_.enumValidValues)
          }
        }

        /**
         * Analyze all fields in the schema and build a map of field name to ProtobufFieldInfo.
         * Returns None if there's an error that should abort processing.
         */
        private def analyzeAllFields(
            schema: StructType,
            msgDesc: AnyRef,
            enumsAsInts: Boolean,
            messageName: String): Option[Map[String, ProtobufFieldInfo]] = {
          val result = mutable.Map[String, ProtobufFieldInfo]()

          for (sf <- schema.fields) {
            val fd = invoke1[AnyRef](msgDesc, "findFieldByName", classOf[String], sf.name)
            if (fd == null) {
              willNotWorkOnGpu(
                s"Protobuf field '${sf.name}' not found in message '$messageName'")
              return None
            }

            val isRepeated = Try {
              invoke0[java.lang.Boolean](fd, "isRepeated").booleanValue()
            }.getOrElse(false)

            // Check if field is required (proto2 required fields)
            val isFieldRequired = Try {
              invoke0[java.lang.Boolean](fd, "isRequired").booleanValue()
            }.getOrElse(false)

            // Check if field has a default value (proto2 [default = xxx])
            val hasDefault = Try {
              invoke0[java.lang.Boolean](fd, "hasDefaultValue").booleanValue()
            }.getOrElse(false)

            // Get the default value if it exists
            val defaultVal = if (hasDefault) {
              Try(Some(invoke0[AnyRef](fd, "getDefaultValue"))).getOrElse(None)
            } else {
              None
            }

            val protoType = invoke0[AnyRef](fd, "getType")
            val protoTypeName = typeName(protoType)
            val fieldNumber = invoke0[java.lang.Integer](fd, "getNumber").intValue()

            // Check field support and determine encoding
            val (isSupported, unsupportedReason, encoding) =
              checkFieldSupport(sf.dataType, protoTypeName, isRepeated, enumsAsInts)

            // Extract enum values if this is an enum field with enumsAsInts enabled
            val enumVals: Option[Set[Int]] = if (protoTypeName == "ENUM" && enumsAsInts) {
              Try {
                val enumType = invoke0[AnyRef](fd, "getEnumType")
                val values = invoke0[java.util.List[_]](enumType, "getValues")
                import scala.collection.JavaConverters._
                val intSet: Set[Int] = values.asScala.map { v =>
                  invoke0[java.lang.Integer](v.asInstanceOf[AnyRef], "getNumber").intValue()
                }.toSet
                intSet
              }.toOption
            } else {
              None
            }

            result(sf.name) = ProtobufFieldInfo(
              fieldNumber = fieldNumber,
              protoTypeName = protoTypeName,
              sparkType = sf.dataType,
              encoding = encoding,
              isSupported = isSupported,
              unsupportedReason = unsupportedReason,
              isRequired = isFieldRequired,
              hasDefaultValue = hasDefault,
              defaultValue = defaultVal,
              enumValues = enumVals,
              isRepeated = isRepeated
            )
          }

          Some(result.toMap)
        }

        /**
         * Check if a field type is supported and return encoding information.
         * @return (isSupported, unsupportedReason, encoding)
         */
        private def checkFieldSupport(
            sparkType: DataType,
            protoTypeName: String,
            isRepeated: Boolean,
            enumsAsInts: Boolean): (Boolean, Option[String], Int) = {

          // Handle repeated fields (arrays)
          if (isRepeated) {
            sparkType match {
              case ArrayType(elementType, _) =>
                // Check if element type is supported
                elementType match {
                  case BooleanType | IntegerType | LongType | FloatType | DoubleType |
                       StringType | BinaryType =>
                    // Supported repeated scalar - determine encoding from proto type
                    return checkScalarEncoding(elementType, protoTypeName, enumsAsInts)
                  case _: StructType =>
                    // Repeated nested message (array of structs) - supported on GPU
                    return (true, None, GpuFromProtobuf.ENC_DEFAULT)
                  case _ =>
                    return (false, Some(s"unsupported repeated element type: $elementType"),
                      GpuFromProtobuf.ENC_DEFAULT)
                }
              case _ =>
                return (false, Some(s"repeated field should map to ArrayType, got: $sparkType"),
                  GpuFromProtobuf.ENC_DEFAULT)
            }
          }

          // Handle nested messages (non-repeated)
          if (protoTypeName == "MESSAGE") {
            sparkType match {
              case _: StructType =>
                return (true, None, GpuFromProtobuf.ENC_DEFAULT)
              case _ =>
                return (false, Some(s"nested message should map to StructType, got: $sparkType"),
                  GpuFromProtobuf.ENC_DEFAULT)
            }
          }

          // Check Spark type is one of the supported simple types
          sparkType match {
            case BooleanType | IntegerType | LongType | FloatType | DoubleType |
                 StringType | BinaryType =>
              // Supported Spark type, continue to check encoding
            case other =>
              return (false, Some(s"unsupported Spark type: $other"), GpuFromProtobuf.ENC_DEFAULT)
          }

          checkScalarEncoding(sparkType, protoTypeName, enumsAsInts)
        }

        /**
         * Determine encoding for scalar types.
         */
        private def checkScalarEncoding(
            sparkType: DataType,
            protoTypeName: String,
            enumsAsInts: Boolean): (Boolean, Option[String], Int) = {

          // Determine encoding based on Spark type and proto type combination
          val encoding = (sparkType, protoTypeName) match {
            case (BooleanType, "BOOL") => Some(GpuFromProtobuf.ENC_DEFAULT)
            case (IntegerType, "INT32" | "UINT32") => Some(GpuFromProtobuf.ENC_DEFAULT)
            case (IntegerType, "SINT32") => Some(GpuFromProtobuf.ENC_ZIGZAG)
            case (IntegerType, "FIXED32" | "SFIXED32") => Some(GpuFromProtobuf.ENC_FIXED)
            case (LongType, "INT64" | "UINT64") => Some(GpuFromProtobuf.ENC_DEFAULT)
            case (LongType, "SINT64") => Some(GpuFromProtobuf.ENC_ZIGZAG)
            case (LongType, "FIXED64" | "SFIXED64") => Some(GpuFromProtobuf.ENC_FIXED)
            // Spark may upcast smaller integers to LongType
            case (LongType, "INT32" | "UINT32" | "SINT32" | "FIXED32" | "SFIXED32") =>
              val enc = protoTypeName match {
                case "SINT32" => GpuFromProtobuf.ENC_ZIGZAG
                case "FIXED32" | "SFIXED32" => GpuFromProtobuf.ENC_FIXED
                case _ => GpuFromProtobuf.ENC_DEFAULT
              }
              Some(enc)
            case (FloatType, "FLOAT") => Some(GpuFromProtobuf.ENC_DEFAULT)
            case (DoubleType, "DOUBLE") => Some(GpuFromProtobuf.ENC_DEFAULT)
            case (StringType, "STRING") => Some(GpuFromProtobuf.ENC_DEFAULT)
            case (BinaryType, "BYTES") => Some(GpuFromProtobuf.ENC_DEFAULT)
            case (IntegerType, "ENUM") if enumsAsInts => Some(GpuFromProtobuf.ENC_DEFAULT)
            case _ => None
          }

          encoding match {
            case Some(enc) => (true, None, enc)
            case None =>
              (false,
                Some(s"type mismatch: Spark $sparkType vs Protobuf $protoTypeName"),
                GpuFromProtobuf.ENC_DEFAULT)
          }
        }

        /**
         * Get wire type constant for a given protobuf type name and encoding.
         */
        private def getWireType(protoTypeName: String, encoding: Int): Int = {
          protoTypeName match {
            case "BOOL" | "INT32" | "UINT32" | "SINT32" | "INT64" | "UINT64" | "SINT64" | "ENUM" =>
              if (encoding == GpuFromProtobuf.ENC_FIXED) {
                if (protoTypeName.contains("64")) WT_64BIT else WT_32BIT
              } else {
                WT_VARINT
              }
            case "FIXED32" | "SFIXED32" | "FLOAT" => WT_32BIT
            case "FIXED64" | "SFIXED64" | "DOUBLE" => WT_64BIT
            case "STRING" | "BYTES" | "MESSAGE" => WT_LEN
            case _ => WT_VARINT  // default
          }
        }

        /**
         * Analyze which fields are actually required by downstream operations.
         * Currently supports analyzing parent Project expressions.
         *
         * @param allFieldNames All field names in the full schema
         * @return Set of field names that are actually required
         */
        private def analyzeRequiredFields(allFieldNames: Set[String]): Set[String] = {
          // Try to find parent SparkPlanMeta and analyze downstream Project
          val parentPlanOpt = findParentPlanMeta()

          parentPlanOpt match {
            case Some(planMeta) =>
              // First, try to analyze the immediate parent
              analyzeDownstreamProject(planMeta) match {
                case Some(fields) if fields.nonEmpty =>
                  // Successfully identified required fields via schema projection
                  fields
                case _ =>
                  // The immediate parent might be a ProjectExec that just aliases the output.
                  // Try to look at its parent (the grandparent) for GetStructField references.
                  planMeta.parent match {
                    case Some(grandParentMeta: SparkPlanMeta[_]) =>
                      analyzeDownstreamProject(grandParentMeta) match {
                        case Some(fields) if fields.nonEmpty => fields
                        case _ => allFieldNames
                      }
                    case _ => allFieldNames
                  }
              }
            case None =>
              // No parent SparkPlanMeta found in the meta tree, assume all fields are needed
              allFieldNames
          }
        }

        /**
         * Find the parent SparkPlanMeta by traversing up the parent chain.
         */
        private def findParentPlanMeta(): Option[SparkPlanMeta[_]] = {
          def traverse(meta: Option[RapidsMeta[_, _, _]]): Option[SparkPlanMeta[_]] = {
            meta match {
              case Some(p: SparkPlanMeta[_]) => Some(p)
              case Some(p: RapidsMeta[_, _, _]) => traverse(p.parent)
              case _ => None
            }
          }
          traverse(parent)
        }

        /**
         * Nested field requirements: for each top-level field, what children are needed?
         * - None means the whole field is needed (all children)
         * - Some(Set("a","b")) means only children a and b are needed
         */
        // Populated during analyzeDownstreamProject, used by addChildFieldsFromStruct
        private var nestedFieldRequirements: Map[String, Option[Set[String]]] = Map.empty

        /**
         * Analyze a Project plan to find which struct fields are actually used.
         * This looks for GetStructField expressions that reference our protobuf output.
         * Also detects nested field access (e.g., decoded.ad_info.winfoid) for nested
         * schema projection.
         */
        private def analyzeDownstreamProject(planMeta: SparkPlanMeta[_]): Option[Set[String]] = {
          planMeta.wrapped match {
            case p: ProjectExec =>
              // Collect field references with nested child tracking
              // Key = top-level field name
              // Value = None (need whole field) or Some(Set(...)) (only need specific children)
              val fieldReqs = mutable.Map[String, Option[Set[String]]]()
              var hasDirectStructRef = false

              p.projectList.foreach { expr =>
                collectStructFieldReferences(expr, fieldReqs, hasDirectStructRefHolder = () => {
                  hasDirectStructRef = true
                })
              }

              if (hasDirectStructRef) {
                None
              } else if (fieldReqs.nonEmpty) {
                nestedFieldRequirements = fieldReqs.toMap
                Some(fieldReqs.keySet.toSet)
              } else {
                None
              }
            case _ =>
              None
          }
        }

        /**
         * Get the field name from a GetStructField expression using its ordinal and schema.
         */
        private def getFieldName(ordinal: Int, nameOpt: Option[String],
            schema: StructType): String = {
          nameOpt.getOrElse {
            if (ordinal < schema.fields.length) schema.fields(ordinal).name
            else s"_$ordinal"
          }
        }

        /**
         * Recursively collect field names and nested child requirements from
         * GetStructField expressions. Detects patterns like:
         *   - decoded.field_name         -> field_name: None (whole field)
         *   - decoded.ad_info.winfoid    -> ad_info: Some({winfoid})
         *   - decoded.ad_info            -> ad_info: None (whole field)
         *
         * When both whole-field and sub-field access exist, whole-field wins (None).
         */
        /**
         * Helper: record a nested child field requirement for a parent field.
         * Merges with existing requirements (whole-field wins over sub-field).
         */
        private def addNestedFieldReq(
            fieldReqs: mutable.Map[String, Option[Set[String]]],
            parentName: String,
            childName: String): Unit = {
          fieldReqs.get(parentName) match {
            case Some(None) => // Already need whole field, keep it
            case Some(Some(existing)) =>
              fieldReqs(parentName) = Some(existing + childName)
            case None =>
              fieldReqs(parentName) = Some(Set(childName))
          }
        }

        private def collectStructFieldReferences(
            expr: Expression,
            fieldReqs: mutable.Map[String, Option[Set[String]]],
            hasDirectStructRefHolder: () => Unit): Unit = {
          expr match {
            // Pattern: decoded.parent_struct.child_field (non-array struct)
            case GetStructField(child, ordinal, nameOpt) =>
              child match {
                case GetStructField(innerChild, innerOrdinal, innerNameOpt)
                    if isProtobufStructReference(innerChild) =>
                  val parentName = getFieldName(innerOrdinal, innerNameOpt, fullSchema)
                  val parentType = fullSchema.fields(innerOrdinal).dataType
                  val childSchema = parentType match {
                    case st: StructType => st
                    case ArrayType(st: StructType, _) => st
                    case _ => null
                  }
                  if (childSchema != null) {
                    val childName = getFieldName(ordinal, nameOpt, childSchema)
                    addNestedFieldReq(fieldReqs, parentName, childName)
                  } else {
                    fieldReqs(parentName) = None
                  }

                case _ if isProtobufStructReference(child) =>
                  // Direct top-level access: decoded.field_name (whole field)
                  val fieldName = getFieldName(ordinal, nameOpt, fullSchema)
                  fieldReqs(fieldName) = None

                case _ =>
                  collectStructFieldReferences(child, fieldReqs, hasDirectStructRefHolder)
              }

            // Pattern: decoded.ad_info.winfoid where ad_info is ArrayType(StructType)
            // Spark generates: GetArrayStructFields(GetStructField(decoded, ad_info_ord), field)
            case gasf: GetArrayStructFields =>
              gasf.child match {
                case GetStructField(innerChild, innerOrdinal, innerNameOpt)
                    if isProtobufStructReference(innerChild) =>
                  // Nested array-struct access: decoded.array_field.child_field
                  val parentName = getFieldName(innerOrdinal, innerNameOpt, fullSchema)
                  val childName = gasf.field.name
                  addNestedFieldReq(fieldReqs, parentName, childName)

                case _ =>
                  // Not a direct protobuf reference, recurse into children
                  gasf.children.foreach { child =>
                    collectStructFieldReferences(child, fieldReqs, hasDirectStructRefHolder)
                  }
              }

            case _ =>
              if (isProtobufStructReference(expr)) {
                hasDirectStructRefHolder()
              }
              expr.children.foreach { child =>
                collectStructFieldReferences(child, fieldReqs, hasDirectStructRefHolder)
              }
          }
        }

        /**
         * Check if an expression references the output of a protobuf decode expression.
         * This can be either:
         * 1. The ProtobufDataToCatalyst expression itself
         * 2. An AttributeReference that references the output of ProtobufDataToCatalyst
         *    (when accessing from a downstream ProjectExec)
         */
        private def isProtobufStructReference(expr: Expression): Boolean = {
          // Check if expr is a ProtobufDataToCatalyst expression
          if (expr.getClass.getName.contains("ProtobufDataToCatalyst")) {
            return true
          }
          
          // Check if expr is an AttributeReference with the same schema as our protobuf output
          // This handles the case where GetStructField references a column from a parent Project
          expr match {
            case attr: AttributeReference =>
              // Check if the data type matches our full schema (struct type from protobuf)
              attr.dataType match {
                case st: StructType => 
                  // Compare field names and types only. We intentionally do not compare
                  // nullable flags because schema transformations (like projections or
                  // certain optimizations) may change nullability while the underlying
                  // schema structure remains the same. For schema projection detection,
                  // matching names and types is sufficient to identify protobuf output.
                  st.fields.length == fullSchema.fields.length &&
                    st.fields.zip(fullSchema.fields).forall { case (a, b) =>
                      a.name == b.name && a.dataType == b.dataType
                    }
                case _ => false
              }
            case _ => false
          }
        }

        override def convertToGpu(child: Expression): GpuExpression = {
          if (useNestedApi) {
            // Build nested pruned fields map for schema expansion after decoding.
            // Only include NON-repeated struct fields. Repeated message fields
            // (ArrayType(StructType)) are NOT pruned (all children are decoded)
            // because expanding inner LIST struct elements is too memory-expensive.
            val prunedFieldsMap: Map[String, Seq[String]] = nestedFieldRequirements.collect {
              case (fieldName, Some(childNames)) =>
                val fieldIdx = fullSchema.fieldIndex(fieldName)
                val fieldType = fullSchema.fields(fieldIdx).dataType
                fieldType match {
                  case _: StructType =>
                    // Non-repeated struct: pruning IS applied
                    val childSchema = fieldType.asInstanceOf[StructType]
                    val orderedNames = childSchema.fields
                      .map(_.name)
                      .filter(childNames.contains)
                      .toSeq
                    Some(fieldName -> orderedNames)
                  case _ =>
                    // ArrayType(StructType) or other: pruning NOT applied, skip
                    None
                }
            }.flatten.toMap

            GpuFromProtobufNested(
              fullSchema, nestedDecodedTopLevelIndices, nestedFieldNumbers, nestedParentIndices,
              nestedDepthLevels, nestedWireTypes, nestedOutputTypeIds, nestedEncodings,
              nestedIsRepeated, nestedIsRequired, nestedHasDefaultValue, nestedDefaultInts,
              nestedDefaultFloats, nestedDefaultBools, nestedDefaultStrings, nestedEnumValidValues,
              prunedFieldsMap, failOnErrors, child)
          } else {
            GpuFromProtobuf(
              fullSchema, decodedFieldIndices, fieldNumbers, cudfTypeIds, cudfTypeScales,
              isRequired, hasDefaultValue, defaultInts, defaultFloats, defaultBools,
              defaultStrings, enumValidValues, failOnErrors, child)
          }
        }
      }
    )
  }

  private def getMessageName(e: Expression): String =
    invoke0[String](e, "messageName")

  /**
   * Newer Spark versions may carry an in-expression descriptor set payload.
   * - Spark 3.5.x: binaryFileDescriptorSet: Option[Array[Byte]]
   * - Spark 4.x: binaryDescriptorSet (may be Array[Byte] or Option[Array[Byte]])
   * Spark 3.4.x does not have this, so callers should fall back to descFilePath().
   */
  private def getDescriptorBytes(e: Expression): Option[Array[Byte]] = {
    // Spark 3.5.x: binaryFileDescriptorSet (note: "File" in the name)
    val spark35Result = Try(invoke0[Option[Array[Byte]]](e, "binaryFileDescriptorSet"))
      .toOption.flatten
    spark35Result.orElse {
      // Spark 4.x: binaryDescriptorSet - may be Array[Byte] or Option[Array[Byte]]
      val direct = Try(invoke0[Array[Byte]](e, "binaryDescriptorSet")).toOption
      direct.orElse {
        Try(invoke0[Option[Array[Byte]]](e, "binaryDescriptorSet")).toOption.flatten
      }
    }
  }

  /**
   * Get descriptor file path from expression.
   * Only available in Spark 3.4.x. Spark 3.5+ uses binaryFileDescriptorSet instead.
   */
  private def getDescFilePath(e: Expression): Option[String] =
    Try(invoke0[Option[String]](e, "descFilePath")).toOption.flatten

  /**
   * Build message descriptor using Spark's ProtobufUtils.
   * Supports both Spark 3.4.x (descFilePath: Option[String]) and
   * Spark 3.5+ (binaryFileDescriptorSet: Option[Array[Byte]]).
   *
   * @param messageName The protobuf message name
   * @param descFilePathOrBytes Either a file path (String) or binary descriptor bytes (Array[Byte])
   */
  private def buildMessageDescriptorWithSparkProtobuf(
      messageName: String,
      descFilePathOrBytes: Either[String, Array[Byte]]): AnyRef = {
    val cls = ShimReflectionUtils.loadClass(sparkProtobufUtilsObjectClassName)
    val module = cls.getField("MODULE$").get(null)

    descFilePathOrBytes match {
      case Left(filePath) =>
        // Spark 3.4.x: buildDescriptor(messageName: String, descFilePath: Option[String])
        val m = cls.getMethod("buildDescriptor", classOf[String], classOf[scala.Option[_]])
        m.invoke(module, messageName, Some(filePath)).asInstanceOf[AnyRef]
      case Right(bytes) =>
        // Spark 3.5+: buildDescriptor(messageName, binaryFileDescriptorSet)
        val m = cls.getMethod("buildDescriptor", classOf[String], classOf[scala.Option[_]])
        m.invoke(module, messageName, Some(bytes)).asInstanceOf[AnyRef]
    }
  }

  private def typeName(t: AnyRef): String = {
    if (t == null) {
      "null"
    } else {
      // Prefer Enum.name() when available; fall back to toString.
      Try(invoke0[String](t, "name")).getOrElse(t.toString)
    }
  }

  private def getOptionsMap(e: Expression): Map[String, String] = {
    val opt = Try(invoke0[scala.collection.Map[String, String]](e, "options")).toOption
    opt.map(_.toMap).getOrElse(Map.empty)
  }

  private def invoke0[T](obj: AnyRef, method: String): T =
    obj.getClass.getMethod(method).invoke(obj).asInstanceOf[T]

  private def invoke1[T](obj: AnyRef, method: String, arg0Cls: Class[_], arg0: AnyRef): T =
    obj.getClass.getMethod(method, arg0Cls).invoke(obj, arg0).asInstanceOf[T]
}
