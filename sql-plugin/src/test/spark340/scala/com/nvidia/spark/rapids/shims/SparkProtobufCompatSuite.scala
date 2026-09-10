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
{"spark": "341db"}
{"spark": "342"}
{"spark": "343"}
{"spark": "344"}
{"spark": "350"}
{"spark": "350db143"}
{"spark": "351"}
{"spark": "352"}
{"spark": "353"}
{"spark": "354"}
{"spark": "355"}
{"spark": "356"}
{"spark": "357"}
{"spark": "358"}
{"spark": "359"}
{"spark": "400"}
{"spark": "400db173"}
{"spark": "401"}
{"spark": "402"}
{"spark": "403"}
{"spark": "404"}
{"spark": "411"}
{"spark": "412"}
{"spark": "413"}
{"spark": "420"}
spark-rapids-shim-json-lines ***/

package com.nvidia.spark.rapids.shims

import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.sql.catalyst.expressions.{Expression, UnaryExpression}
import org.apache.spark.sql.catalyst.expressions.codegen.{CodegenContext, ExprCode}
import org.apache.spark.sql.rapids.protobuf._
import org.apache.spark.sql.types._

class SparkProtobufCompatSuite extends AnyFunSuite {
  private val outputSchema = StructType(Seq(
    StructField("id", IntegerType, nullable = true),
    StructField("name", StringType, nullable = true)))

  private case class FakeExprChild() extends Expression {
    override def children: Seq[Expression] = Nil
    override def nullable: Boolean = true
    override def dataType: DataType = BinaryType
    override def eval(input: org.apache.spark.sql.catalyst.InternalRow): Any = null
    override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode =
      throw new UnsupportedOperationException("not needed")
    override protected def withNewChildrenInternal(
        newChildren: IndexedSeq[Expression]): Expression = {
      assert(newChildren.isEmpty)
      this
    }
  }

  private abstract class FakeBaseProtobufExpr(childExpr: Expression) extends UnaryExpression {
    override def child: Expression = childExpr
    override def nullable: Boolean = true
    override def dataType: DataType = outputSchema
    override protected def doGenCode(ctx: CodegenContext, ev: ExprCode): ExprCode =
      throw new UnsupportedOperationException("not needed")
    override protected def withNewChildInternal(newChild: Expression): Expression = this
  }

  private case class FakePathProtobufExpr(override val child: Expression)
      extends FakeBaseProtobufExpr(child) {
    def messageName: String = "test.Message"
    def descFilePath: Option[String] = Some("/tmp/test.desc")
    def options: scala.collection.Map[String, String] = Map("mode" -> "FAILFAST")
  }

  private case class FakeBytesProtobufExpr(override val child: Expression)
      extends FakeBaseProtobufExpr(child) {
    def messageName: String = "test.Message"
    def binaryDescriptorSet: Array[Byte] = Array[Byte](1, 2, 3)
    def options: scala.collection.Map[String, String] =
      Map("mode" -> "PERMISSIVE", "enums.as.ints" -> "true")
  }

  private case class FakeMissingOptionsExpr(override val child: Expression)
      extends FakeBaseProtobufExpr(child) {
    def messageName: String = "test.Message"
    def descFilePath: Option[String] = Some("/tmp/test.desc")
  }


  private object FakeSpark34ProtobufUtils {
    def buildDescriptor(messageName: String, descFilePath: Option[String]): String =
      s"$messageName:${descFilePath.getOrElse("none")}"
  }

  private object FakeSpark35ProtobufUtils {
    def buildDescriptor(messageName: String, binaryFileDescriptorSet: Option[Array[Byte]]): String =
      s"$messageName:${binaryFileDescriptorSet.map(_.mkString(",")).getOrElse("none")}"
  }

  private object FakeSpark35RetryFailureProtobufUtils {
    def buildDescriptor(
        messageName: String,
        binaryFileDescriptorSet: Option[Array[Byte]]): String = {
      val bytes = binaryFileDescriptorSet.getOrElse(Array.emptyByteArray)
      if (bytes.sameElements(Array[Byte](1, 2, 3))) {
        throw new IllegalArgumentException(s"Unknown message $messageName")
      }
      s"$messageName:${bytes.mkString(",")}"
    }
  }

  private final class FakeFileDescriptorProto(syntax: String) {
    def getSyntax: String = syntax
  }

  private final class FakeModernFileDescriptor(syntax: String) {
    def toProto: FakeFileDescriptorProto = new FakeFileDescriptorProto(syntax)
  }

  private final class FakeModernDescriptor(syntax: String) {
    def getFile: FakeModernFileDescriptor = new FakeModernFileDescriptor(syntax)
  }

  private final class FakeDescriptorWithExtensions(val descriptor: AnyRef)

  private final class FakeLegacyFileDescriptor(syntax: String) {
    def getSyntax: String = syntax
  }

  private final class FakeLegacyDescriptor(syntax: String) {
    def getFile: FakeLegacyFileDescriptor = new FakeLegacyFileDescriptor(syntax)
  }

  private final class FakeBrokenFileDescriptor

  private final class FakeBrokenDescriptor {
    def getFile: FakeBrokenFileDescriptor = new FakeBrokenFileDescriptor
  }

  private final class FakeDescriptorWithoutFile

  test("compat extracts descriptor path and options from legacy expression") {
    val exprInfo = SparkProtobufCompat.extractExprInfo(FakePathProtobufExpr(FakeExprChild()))
    assert(exprInfo.isRight)
    val info = exprInfo.toOption.get
    assert(info.messageName == "test.Message")
    assert(info.options == Map("mode" -> "FAILFAST"))
    assert(info.descriptorSource ==
      ProtobufDescriptorSource.DescriptorPath("/tmp/test.desc"))
  }

  test("compat extracts a binary descriptor source") {
    val exprInfo = SparkProtobufCompat.extractExprInfo(FakeBytesProtobufExpr(FakeExprChild()))
    assert(exprInfo.isRight)
    val info = exprInfo.toOption.get
    info.descriptorSource match {
      case ProtobufDescriptorSource.DescriptorBytes(bytes) =>
        assert(bytes.sameElements(Array[Byte](1, 2, 3)))
      case other =>
        fail(s"Unexpected descriptor source: $other")
    }
  }

  test("compat invokes Spark 3.4 descriptor builder with descriptor path") {
    val buildMethod = FakeSpark34ProtobufUtils.getClass.getMethod(
      "buildDescriptor", classOf[String], classOf[scala.Option[_]])

    val result = SparkProtobufCompat.invokeBuildDescriptor(
      buildMethod,
      FakeSpark34ProtobufUtils,
      "test.Message",
      ProtobufDescriptorSource.DescriptorPath("/tmp/test.desc"),
      _ => fail("path-to-bytes fallback should not be needed for Spark 3.4"))

    assert(result == "test.Message:/tmp/test.desc")
  }

  test("compat retries descriptor path as bytes for Spark 3.5 descriptor builder") {
    val buildMethod = FakeSpark35ProtobufUtils.getClass.getMethod(
      "buildDescriptor", classOf[String], classOf[scala.Option[_]])
    var readCalls = 0

    val result = SparkProtobufCompat.invokeBuildDescriptor(
      buildMethod,
      FakeSpark35ProtobufUtils,
      "test.Message",
      ProtobufDescriptorSource.DescriptorPath("/tmp/test.desc"),
      _ => {
        readCalls += 1
        Array[Byte](1, 2, 3)
      })

    assert(readCalls == 1)
    assert(result == "test.Message:1,2,3")
  }

  test("compat passes bytes directly to Spark 3.5 descriptor builder") {
    val buildMethod = FakeSpark35ProtobufUtils.getClass.getMethod(
      "buildDescriptor", classOf[String], classOf[scala.Option[_]])

    val result = SparkProtobufCompat.invokeBuildDescriptor(
      buildMethod,
      FakeSpark35ProtobufUtils,
      "test.Message",
      ProtobufDescriptorSource.DescriptorBytes(Array[Byte](4, 5, 6)),
      _ => fail("binary descriptor source should not read a file"))

    assert(result == "test.Message:4,5,6")
  }

  test("compat preserves retry context when descriptor bytes fallback also fails") {
    val buildMethod = FakeSpark35RetryFailureProtobufUtils.getClass.getMethod(
      "buildDescriptor", classOf[String], classOf[scala.Option[_]])

    val ex = intercept[RuntimeException] {
      SparkProtobufCompat.invokeBuildDescriptor(
        buildMethod,
        FakeSpark35RetryFailureProtobufUtils,
        "test.Message",
        ProtobufDescriptorSource.DescriptorPath("/tmp/test.desc"),
        _ => Array[Byte](1, 2, 3))
    }

    assert(ex.getMessage.contains("descriptor bytes retry failed"))
    assert(ex.getMessage.contains("ClassCastException"))
    assert(ex.getMessage.contains("Unknown message test.Message"))
    assert(ex.getCause.isInstanceOf[IllegalArgumentException])
    assert(ex.getSuppressed.exists(_.isInstanceOf[java.lang.reflect.InvocationTargetException]))
  }

  test("compat reports missing options accessor as cpu fallback reason") {
    val exprInfo = SparkProtobufCompat.extractExprInfo(FakeMissingOptionsExpr(FakeExprChild()))
    assert(exprInfo.left.toOption.exists(
      _.contains("Cannot read from_protobuf options via reflection")))
  }

  test("compat reads syntax through protobuf 4 FileDescriptor API") {
    assert(SparkProtobufCompat.readDescriptorSyntax(
      new FakeModernDescriptor("proto2")) == "PROTO2")
    assert(SparkProtobufCompat.readDescriptorSyntax(
      new FakeModernDescriptor("proto3")) == "PROTO3")
    assert(SparkProtobufCompat.readDescriptorSyntax(
      new FakeModernDescriptor("")) == "PROTO2")
    assert(SparkProtobufCompat.readDescriptorSyntax(
      new FakeLegacyDescriptor("PROTO2")) == "PROTO2")
    assert(SparkProtobufCompat.readDescriptorSyntax(
      new FakeBrokenDescriptor) == "")
    assert(SparkProtobufCompat.readDescriptorSyntax(
      new FakeDescriptorWithoutFile) == "")
  }

  test("compat unwraps Spark 4.2 descriptor with extensions") {
    val descriptor = new FakeModernDescriptor("proto2")

    assert(SparkProtobufCompat.unwrapMessageDescriptor(
      new FakeDescriptorWithExtensions(descriptor)) eq descriptor)
    assert(SparkProtobufCompat.unwrapMessageDescriptor(descriptor) eq descriptor)
  }

  test("descriptor bytes use content equality") {
    val left = ProtobufDescriptorSource.DescriptorBytes(Array[Byte](1, 2, 3))
    val right = ProtobufDescriptorSource.DescriptorBytes(Array[Byte](1, 2, 3))

    assert(left == right)
    assert(left.hashCode() == right.hashCode())
  }

  test("compat converts reflected defaults to neutral values") {
    val enumMetadata = ProtobufEnumMetadata(Seq(
      ProtobufEnumValue(0, "UNKNOWN"),
      ProtobufEnumValue(1, "READY")))

    assert(SparkProtobufCompat.toDefaultValue(Boolean.box(true), "BOOL", None) ==
      Right(ProtobufDefaultValue.BoolValue(true)))
    assert(SparkProtobufCompat.toDefaultValue(Int.box(7), "INT32", None) ==
      Right(ProtobufDefaultValue.IntValue(7L)))
    assert(SparkProtobufCompat.toDefaultValue(Float.box(1.5f), "FLOAT", None) ==
      Right(ProtobufDefaultValue.FloatValue(1.5f)))
    assert(SparkProtobufCompat.toDefaultValue(Double.box(2.5), "DOUBLE", None) ==
      Right(ProtobufDefaultValue.DoubleValue(2.5)))
    assert(SparkProtobufCompat.toDefaultValue("value", "STRING", None) ==
      Right(ProtobufDefaultValue.StringValue("value")))
    assert(SparkProtobufCompat.toDefaultValue(Array[Byte](4, 5), "BYTES", None) ==
      Right(ProtobufDefaultValue.BinaryValue(Array[Byte](4, 5))))
    assert(SparkProtobufCompat.toDefaultValue(Int.box(1), "ENUM", Some(enumMetadata)) ==
      Right(ProtobufDefaultValue.EnumValue(1, "READY")))
  }

  test("compat returns Left for unsupported default value types") {
    val result = SparkProtobufCompat.toDefaultValue(
      "opaque-default", "MESSAGE", None)

    assert(result.left.toOption.exists(_.contains("Unsupported protobuf default value type")))
  }
}
