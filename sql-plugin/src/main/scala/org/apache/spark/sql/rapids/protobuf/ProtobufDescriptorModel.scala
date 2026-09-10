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

package org.apache.spark.sql.rapids.protobuf

import java.util.Arrays

sealed trait ProtobufDescriptorSource

object ProtobufDescriptorSource {
  final case class DescriptorPath(path: String) extends ProtobufDescriptorSource
  final case class DescriptorBytes(bytes: Array[Byte]) extends ProtobufDescriptorSource {
    override def equals(other: Any): Boolean = other match {
      case DescriptorBytes(otherBytes) => Arrays.equals(bytes, otherBytes)
      case _ => false
    }

    override def hashCode(): Int = Arrays.hashCode(bytes)
  }
}

final case class ProtobufExprInfo(
    messageName: String,
    descriptorSource: ProtobufDescriptorSource,
    options: Map[String, String])

sealed trait ProtobufDefaultValue

object ProtobufDefaultValue {
  final case class BoolValue(value: Boolean) extends ProtobufDefaultValue
  final case class IntValue(value: Long) extends ProtobufDefaultValue
  final case class FloatValue(value: Float) extends ProtobufDefaultValue
  final case class DoubleValue(value: Double) extends ProtobufDefaultValue
  final case class StringValue(value: String) extends ProtobufDefaultValue
  final case class BinaryValue(value: Array[Byte]) extends ProtobufDefaultValue {
    override def equals(other: Any): Boolean = other match {
      case BinaryValue(otherBytes) => Arrays.equals(value, otherBytes)
      case _ => false
    }

    override def hashCode(): Int = Arrays.hashCode(value)
  }
  final case class EnumValue(number: Int, name: String) extends ProtobufDefaultValue
}

final case class ProtobufEnumValue(number: Int, name: String)

final case class ProtobufEnumMetadata(values: Seq[ProtobufEnumValue]) {
  private lazy val namesByNumber: Map[Int, String] =
    values.map(v => v.number -> v.name).toMap

  def enumDefault(number: Int): ProtobufDefaultValue.EnumValue = {
    val name = namesByNumber.getOrElse(number, number.toString)
    ProtobufDefaultValue.EnumValue(number, name)
  }
}

trait ProtobufMessageDescriptor {
  def syntax: String
  def javaStringCheckUtf8: Boolean = false
  def findField(name: String): Option[ProtobufFieldDescriptor]
}

trait ProtobufFieldDescriptor {
  def name: String
  def fieldNumber: Int
  def protoTypeName: String
  def isRepeated: Boolean
  def isRequired: Boolean
  def isInOneof: Boolean
  def defaultValueResult: Either[String, Option[ProtobufDefaultValue]]
  def enumMetadata: Option[ProtobufEnumMetadata]
  def messageDescriptor: Option[ProtobufMessageDescriptor]
  def referencedTypeSyntax: Option[String]
}
