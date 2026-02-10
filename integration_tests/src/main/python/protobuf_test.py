# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect

import pytest

from asserts import assert_gpu_and_cpu_are_equal_collect
from data_gen import (
    BooleanGen, IntegerGen, LongGen, FloatGen, DoubleGen, StringGen,
    ProtobufSimpleMessageRowGen, ProtobufNestedMessageRowGen, 
    ProtobufRepeatedFieldRowGen, gen_df, idfn
)
from marks import ignore_order
from spark_session import with_cpu_session, is_before_spark_340
import pyspark.sql.functions as f

pytestmark = [pytest.mark.premerge_ci_1]


# =============================================================================
# Test Data Configurations for Parametrized Tests
# =============================================================================

# Random data generation configurations for simple scalars
_random_scalar_test_configs = [
    # (test_id, data_gen_config, data_length)
    ("all_types", [
        ("b", 1, BooleanGen()),
        ("i32", 2, IntegerGen()),
        ("i64", 3, LongGen()),
        ("f32", 4, FloatGen()),
        ("f64", 5, DoubleGen()),
        ("s", 6, StringGen()),
    ], 100),
    ("integers_edge_cases", [
        ("b", 1, BooleanGen()),
        ("i32", 2, IntegerGen(
            min_val=-2147483648, max_val=2147483647,
            special_cases=[-2147483648, -1, 0, 1, 2147483647])),
        ("i64", 3, LongGen(
            min_val=-9223372036854775808, max_val=9223372036854775807,
            special_cases=[-9223372036854775808, -1, 0, 1, 9223372036854775807])),
        ("f32", 4, FloatGen()),
        ("f64", 5, DoubleGen()),
        ("s", 6, StringGen()),
    ], 200),
    ("floats_edge_cases", [
        ("b", 1, BooleanGen()),
        ("i32", 2, IntegerGen()),
        ("i64", 3, LongGen()),
        ("f32", 4, FloatGen(no_nans=True, special_cases=[-0.0, 0.0, 1.0, -1.0])),
        ("f64", 5, DoubleGen(no_nans=True, special_cases=[-0.0, 0.0, 1.0, -1.0])),
        ("s", 6, StringGen()),
    ], 100),
    ("nullable_fields", [
        ("b", 1, BooleanGen(nullable=True)),
        ("i32", 2, IntegerGen(nullable=True)),
        ("i64", 3, LongGen(nullable=True)),
        ("f32", 4, FloatGen(nullable=True)),
        ("f64", 5, DoubleGen(nullable=True)),
        ("s", 6, StringGen(nullable=True)),
    ], 100),
    ("large_dataset", [
        ("b", 1, BooleanGen()),
        ("i32", 2, IntegerGen()),
        ("i64", 3, LongGen()),
        ("f32", 4, FloatGen()),
        ("f64", 5, DoubleGen()),
        ("s", 6, StringGen(pattern="[a-z]{0,50}")),
    ], 2048),
]


def _try_import_from_protobuf():
    try:
        from pyspark.sql.protobuf.functions import from_protobuf
        return from_protobuf
    except Exception:
        return None


def _spark_protobuf_jvm_available(spark) -> bool:
    """
    `spark-protobuf` is an optional external module. PySpark may have the Python wrappers
    even when the JVM side isn't present on the classpath, which manifests as:
      TypeError: 'JavaPackage' object is not callable
    when calling into `sc._jvm.org.apache.spark.sql.protobuf.functions.from_protobuf`.
    """
    jvm = spark.sparkContext._jvm
    candidates = [
        # Scala object `functions` compiles to `functions$`
        "org.apache.spark.sql.protobuf.functions$",
        # Some environments may expose it differently
        "org.apache.spark.sql.protobuf.functions",
    ]
    for cls in candidates:
        try:
            jvm.java.lang.Class.forName(cls)
            return True
        except Exception:
            continue
    return False


def _build_simple_descriptor_set_bytes(spark):
    """
    Build a FileDescriptorSet for:
      package test;
      syntax = "proto2";
      message Simple {
        optional bool   b   = 1;
        optional int32  i32 = 2;
        optional int64  i64 = 3;
        optional float  f32 = 4;
        optional double f64 = 5;
        optional string s   = 6;
      }
    """
    jvm = spark.sparkContext._jvm
    D = jvm.com.google.protobuf.DescriptorProtos

    fd = D.FileDescriptorProto.newBuilder() \
        .setName("simple.proto") \
        .setPackage("test")
    # Some Spark distributions bring an older protobuf-java where FileDescriptorProto.Builder
    # does not expose setSyntax(String). For this test we only need proto2 semantics, and
    # leaving syntax unset is sufficient/compatible.
    try:
        fd = fd.setSyntax("proto2")
    except Exception:
        # If setSyntax is unavailable (older protobuf-java), we intentionally leave syntax unset.
        pass

    msg = D.DescriptorProto.newBuilder().setName("Simple")
    label_opt = D.FieldDescriptorProto.Label.LABEL_OPTIONAL

    def add_field(name, number, ftype):
        msg.addField(
            D.FieldDescriptorProto.newBuilder()
              .setName(name)
              .setNumber(number)
              .setLabel(label_opt)
              .setType(ftype)
              .build()
        )

    add_field("b", 1, D.FieldDescriptorProto.Type.TYPE_BOOL)
    add_field("i32", 2, D.FieldDescriptorProto.Type.TYPE_INT32)
    add_field("i64", 3, D.FieldDescriptorProto.Type.TYPE_INT64)
    add_field("f32", 4, D.FieldDescriptorProto.Type.TYPE_FLOAT)
    add_field("f64", 5, D.FieldDescriptorProto.Type.TYPE_DOUBLE)
    add_field("s", 6, D.FieldDescriptorProto.Type.TYPE_STRING)

    fd.addMessageType(msg.build())

    fds = D.FileDescriptorSet.newBuilder().addFile(fd.build()).build()
    # py4j converts Java byte[] to a Python bytes-like object
    return bytes(fds.toByteArray())


def _write_bytes_to_hadoop_path(spark, path_str, data_bytes):
    sc = spark.sparkContext
    config = sc._jsc.hadoopConfiguration()
    jpath = sc._jvm.org.apache.hadoop.fs.Path(path_str)
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(config)
    out = fs.create(jpath, True)
    try:
        out.write(bytearray(data_bytes))
    finally:
        out.close()


@pytest.mark.skipif(is_before_spark_340(), reason="from_protobuf is Spark 3.4.0+")
@ignore_order(local=True)
def test_from_protobuf_simple_parquet_binary_round_trip(spark_tmp_path):
    from_protobuf = _try_import_from_protobuf()
    if from_protobuf is None:
        pytest.skip("pyspark.sql.protobuf.functions.from_protobuf is not available")
    if not with_cpu_session(_spark_protobuf_jvm_available):
        pytest.skip("spark-protobuf JVM module is not available on the classpath")

    data_path = spark_tmp_path + "/PROTOBUF_SIMPLE_PARQUET/"
    desc_path = spark_tmp_path + "/simple.desc"
    message_name = "test.Simple"

    # Generate descriptor bytes once using the JVM (no protoc dependency)
    desc_bytes = with_cpu_session(_build_simple_descriptor_set_bytes)
    with_cpu_session(lambda spark: _write_bytes_to_hadoop_path(spark, desc_path, desc_bytes))

    # Build a DF with scalar columns + binary protobuf column and write to parquet
    row_gen = ProtobufSimpleMessageRowGen([
        ("b", 1, BooleanGen(nullable=True)),
        ("i32", 2, IntegerGen(nullable=True, min_val=0, max_val=1 << 20)),
        ("i64", 3, LongGen(nullable=True, min_val=0, max_val=1 << 40, special_cases=[])),
        ("f32", 4, FloatGen(nullable=True, no_nans=True)),
        ("f64", 5, DoubleGen(nullable=True, no_nans=True)),
        ("s", 6, StringGen(nullable=True)),
    ], binary_col_name="bin")

    def write_parquet(spark):
        df = gen_df(spark, row_gen, length=512)
        df.write.mode("overwrite").parquet(data_path)

    with_cpu_session(write_parquet)

    # Sanity check correctness on CPU (decoded struct matches the original scalar columns)
    def cpu_correctness_check(spark):
        df = spark.read.parquet(data_path)
        expected = f.struct(
            f.col("b").alias("b"),
            f.col("i32").alias("i32"),
            f.col("i64").alias("i64"),
            f.col("f32").alias("f32"),
            f.col("f64").alias("f64"),
            f.col("s").alias("s"),
        ).alias("expected")

        sig = inspect.signature(from_protobuf)
        if "binaryDescriptorSet" in sig.parameters:
            decoded = from_protobuf(f.col("bin"), message_name, binaryDescriptorSet=bytearray(desc_bytes)).alias("decoded")
        else:
            decoded = from_protobuf(f.col("bin"), message_name, desc_path).alias("decoded")

        rows = df.select(expected, decoded).collect()
        for r in rows:
            assert r["expected"] == r["decoded"]

    with_cpu_session(cpu_correctness_check)

    # Main assertion: CPU and GPU results match for from_protobuf on a binary column read from parquet
    def run_on_spark(spark):
        df = spark.read.parquet(data_path)
        sig = inspect.signature(from_protobuf)
        if "binaryDescriptorSet" in sig.parameters:
            decoded = from_protobuf(f.col("bin"), message_name, binaryDescriptorSet=bytearray(desc_bytes))
        else:
            decoded = from_protobuf(f.col("bin"), message_name, desc_path)
        return df.select(decoded.alias("decoded"))

    assert_gpu_and_cpu_are_equal_collect(run_on_spark)


@pytest.mark.skipif(is_before_spark_340(), reason="from_protobuf is Spark 3.4.0+")
@ignore_order(local=True)
def test_from_protobuf_simple_null_input_returns_null(spark_tmp_path):
    from_protobuf = _try_import_from_protobuf()
    if from_protobuf is None:
        pytest.skip("pyspark.sql.protobuf.functions.from_protobuf is not available")
    if not with_cpu_session(_spark_protobuf_jvm_available):
        pytest.skip("spark-protobuf JVM module is not available on the classpath")

    desc_path = spark_tmp_path + "/simple_null_input.desc"
    message_name = "test.Simple"

    # Generate descriptor bytes once using the JVM (no protoc dependency)
    desc_bytes = with_cpu_session(_build_simple_descriptor_set_bytes)
    with_cpu_session(lambda spark: _write_bytes_to_hadoop_path(spark, desc_path, desc_bytes))

    # Spark's ProtobufDataToCatalyst is NullIntolerant (null input -> null output).
    def run_on_spark(spark):
        df = spark.createDataFrame(
            [(None,), (bytes([0x08, 0x01, 0x10, 0x7B]),)],  # b=true, i32=123
            schema="bin binary",
        )
        sig = inspect.signature(from_protobuf)
        if "binaryDescriptorSet" in sig.parameters:
            decoded = from_protobuf(
                f.col("bin"),
                message_name,
                binaryDescriptorSet=bytearray(desc_bytes),
            )
        else:
            decoded = from_protobuf(f.col("bin"), message_name, desc_path)
        return df.select(decoded.alias("decoded"))

    assert_gpu_and_cpu_are_equal_collect(run_on_spark)


def _build_nested_descriptor_set_bytes(spark):
    """
    Build a FileDescriptorSet for a message with both simple fields and nested message:
      package test;
      syntax = "proto2";
      message Nested {
        optional int32 x = 1;
      }
      message WithNested {
        optional int32  simple_int  = 1;
        optional string simple_str  = 2;
        optional Nested nested_msg  = 3;   // nested message
        optional int64  simple_long = 4;
      }
    """
    jvm = spark.sparkContext._jvm
    D = jvm.com.google.protobuf.DescriptorProtos

    fd = D.FileDescriptorProto.newBuilder() \
        .setName("nested.proto") \
        .setPackage("test")
    try:
        fd = fd.setSyntax("proto2")
    except Exception:
        pass

    label_opt = D.FieldDescriptorProto.Label.LABEL_OPTIONAL

    # Define Nested message
    nested_msg = D.DescriptorProto.newBuilder().setName("Nested")
    nested_msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("x")
            .setNumber(1)
            .setLabel(label_opt)
            .setType(D.FieldDescriptorProto.Type.TYPE_INT32)
            .build()
    )
    fd.addMessageType(nested_msg.build())

    # Define WithNested message
    with_nested_msg = D.DescriptorProto.newBuilder().setName("WithNested")
    # simple_int
    with_nested_msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("simple_int")
            .setNumber(1)
            .setLabel(label_opt)
            .setType(D.FieldDescriptorProto.Type.TYPE_INT32)
            .build()
    )
    # simple_str
    with_nested_msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("simple_str")
            .setNumber(2)
            .setLabel(label_opt)
            .setType(D.FieldDescriptorProto.Type.TYPE_STRING)
            .build()
    )
    # nested_msg (nested message type)
    with_nested_msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("nested_msg")
            .setNumber(3)
            .setLabel(label_opt)
            .setType(D.FieldDescriptorProto.Type.TYPE_MESSAGE)
            .setTypeName(".test.Nested")
            .build()
    )
    # simple_long
    with_nested_msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("simple_long")
            .setNumber(4)
            .setLabel(label_opt)
            .setType(D.FieldDescriptorProto.Type.TYPE_INT64)
            .build()
    )
    fd.addMessageType(with_nested_msg.build())

    fds = D.FileDescriptorSet.newBuilder().addFile(fd.build()).build()
    return bytes(fds.toByteArray())


@pytest.mark.skipif(is_before_spark_340(), reason="from_protobuf is Spark 3.4.0+")
@ignore_order(local=True)
def test_from_protobuf_schema_projection_simple_fields_only(spark_tmp_path):
    """
    Test schema projection: when only simple fields are selected from a protobuf message
    that also contains unsupported types (nested message), GPU should be able to decode
    just the simple fields without falling back to CPU.
    """
    from_protobuf = _try_import_from_protobuf()
    if from_protobuf is None:
        pytest.skip("pyspark.sql.protobuf.functions.from_protobuf is not available")
    if not with_cpu_session(_spark_protobuf_jvm_available):
        pytest.skip("spark-protobuf JVM module is not available on the classpath")

    desc_path = spark_tmp_path + "/nested.desc"
    message_name = "test.WithNested"

    desc_bytes = with_cpu_session(_build_nested_descriptor_set_bytes)
    with_cpu_session(lambda spark: _write_bytes_to_hadoop_path(spark, desc_path, desc_bytes))

    # Create test data: protobuf binary with simple fields set
    # Field 1 (simple_int): varint 42 -> 0x08 0x2A
    # Field 2 (simple_str): length-delimited "hello" -> 0x12 0x05 h e l l o
    # Field 4 (simple_long): varint 12345 -> 0x20 0xB9 0x60
    test_data = bytes([
        0x08, 0x2A,  # simple_int = 42
        0x12, 0x05, 0x68, 0x65, 0x6C, 0x6C, 0x6F,  # simple_str = "hello"
        0x20, 0xB9, 0x60,  # simple_long = 12345
    ])

    def run_on_spark(spark):
        df = spark.createDataFrame(
            [(test_data,), (None,)],
            schema="bin binary",
        )
        sig = inspect.signature(from_protobuf)
        if "binaryDescriptorSet" in sig.parameters:
            decoded = from_protobuf(
                f.col("bin"),
                message_name,
                binaryDescriptorSet=bytearray(desc_bytes),
            )
        else:
            decoded = from_protobuf(f.col("bin"), message_name, desc_path)
        # Only select simple fields, not the nested_msg field
        return df.select(
            decoded.getField("simple_int").alias("simple_int"),
            decoded.getField("simple_str").alias("simple_str"),
            decoded.getField("simple_long").alias("simple_long")
        )

    assert_gpu_and_cpu_are_equal_collect(run_on_spark)


def _build_enum_descriptor_set_bytes(spark):
    """
    Build a FileDescriptorSet for a message with enum field:
      package test;
      syntax = "proto2";
      message WithEnum {
        enum Color {
          RED = 0;
          GREEN = 1;
          BLUE = 2;
        }
        optional Color color = 1;
        optional int32 count = 2;
        optional string name = 3;
      }
    """
    jvm = spark.sparkContext._jvm
    D = jvm.com.google.protobuf.DescriptorProtos

    fd = D.FileDescriptorProto.newBuilder() \
        .setName("enum.proto") \
        .setPackage("test")
    try:
        fd = fd.setSyntax("proto2")
    except Exception:
        pass

    label_opt = D.FieldDescriptorProto.Label.LABEL_OPTIONAL

    # Define WithEnum message with nested enum
    msg = D.DescriptorProto.newBuilder().setName("WithEnum")

    # Add enum type definition
    enum_type = D.EnumDescriptorProto.newBuilder().setName("Color")
    enum_type.addValue(D.EnumValueDescriptorProto.newBuilder().setName("RED").setNumber(0).build())
    enum_type.addValue(D.EnumValueDescriptorProto.newBuilder().setName("GREEN").setNumber(1).build())
    enum_type.addValue(D.EnumValueDescriptorProto.newBuilder().setName("BLUE").setNumber(2).build())
    msg.addEnumType(enum_type.build())

    # Add color field (enum type)
    msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("color")
            .setNumber(1)
            .setLabel(label_opt)
            .setType(D.FieldDescriptorProto.Type.TYPE_ENUM)
            .setTypeName(".test.WithEnum.Color")
            .build()
    )
    # Add count field (int32)
    msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("count")
            .setNumber(2)
            .setLabel(label_opt)
            .setType(D.FieldDescriptorProto.Type.TYPE_INT32)
            .build()
    )
    # Add name field (string)
    msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("name")
            .setNumber(3)
            .setLabel(label_opt)
            .setType(D.FieldDescriptorProto.Type.TYPE_STRING)
            .build()
    )

    fd.addMessageType(msg.build())
    fds = D.FileDescriptorSet.newBuilder().addFile(fd.build()).build()
    return bytes(fds.toByteArray())


@pytest.mark.skipif(is_before_spark_340(), reason="from_protobuf is Spark 3.4.0+")
@ignore_order(local=True)
def test_from_protobuf_enum_as_int(spark_tmp_path):
    """
    Test enum field decoded as integer with enums.as.ints=true option.
    GPU should decode enum fields as INT32 values matching CPU behavior.
    """
    from_protobuf = _try_import_from_protobuf()
    if from_protobuf is None:
        pytest.skip("pyspark.sql.protobuf.functions.from_protobuf is not available")
    if not with_cpu_session(_spark_protobuf_jvm_available):
        pytest.skip("spark-protobuf JVM module is not available on the classpath")

    desc_path = spark_tmp_path + "/enum.desc"
    message_name = "test.WithEnum"

    desc_bytes = with_cpu_session(_build_enum_descriptor_set_bytes)
    with_cpu_session(lambda spark: _write_bytes_to_hadoop_path(spark, desc_path, desc_bytes))

    # Create test data with various enum values:
    # Row 0: color=GREEN(1), count=42, name="test"
    # Row 1: color=RED(0), count=100, name missing
    # Row 2: color missing, count=200, name="hello"
    # Row 3: null input
    test_data_row0 = bytes([
        0x08, 0x01,  # color = GREEN (1)
        0x10, 0x2A,  # count = 42
        0x1A, 0x04, 0x74, 0x65, 0x73, 0x74,  # name = "test"
    ])
    test_data_row1 = bytes([
        0x08, 0x00,  # color = RED (0)
        0x10, 0x64,  # count = 100
    ])
    test_data_row2 = bytes([
        0x10, 0xC8, 0x01,  # count = 200
        0x1A, 0x05, 0x68, 0x65, 0x6C, 0x6C, 0x6F,  # name = "hello"
    ])

    def run_on_spark(spark):
        df = spark.createDataFrame(
            [(test_data_row0,), (test_data_row1,), (test_data_row2,), (None,)],
            schema="bin binary",
        )
        sig = inspect.signature(from_protobuf)
        options = {"enums.as.ints": "true"}
        if "binaryDescriptorSet" in sig.parameters:
            decoded = from_protobuf(
                f.col("bin"),
                message_name,
                binaryDescriptorSet=bytearray(desc_bytes),
                options=options
            )
        else:
            decoded = from_protobuf(f.col("bin"), message_name, desc_path, options)
        return df.select(
            decoded.getField("color").alias("color"),
            decoded.getField("count").alias("count"),
            decoded.getField("name").alias("name")
        )

    assert_gpu_and_cpu_are_equal_collect(run_on_spark)


@pytest.mark.skipif(is_before_spark_340(), reason="from_protobuf is Spark 3.4.0+")
@ignore_order(local=True)
def test_from_protobuf_enum_unknown_value(spark_tmp_path):
    """
    Test that unknown enum values (not defined in enum) null the entire struct row.
    Both GPU and CPU implementations (PERMISSIVE mode) null the entire row when
    an unknown enum value is encountered.
    """
    from_protobuf = _try_import_from_protobuf()
    if from_protobuf is None:
        pytest.skip("pyspark.sql.protobuf.functions.from_protobuf is not available")
    if not with_cpu_session(_spark_protobuf_jvm_available):
        pytest.skip("spark-protobuf JVM module is not available on the classpath")

    desc_path = spark_tmp_path + "/enum_unknown.desc"
    message_name = "test.WithEnum"

    desc_bytes = with_cpu_session(_build_enum_descriptor_set_bytes)
    with_cpu_session(lambda spark: _write_bytes_to_hadoop_path(spark, desc_path, desc_bytes))

    # Create test data with unknown enum value (999 is not defined in Color enum)
    # 999 encoded as varint: 0xE7 0x07
    test_data = bytes([
        0x08, 0xE7, 0x07,  # color = 999 (unknown value)
        0x10, 0x2A,  # count = 42
    ])

    def run_on_spark(spark):
        df = spark.createDataFrame(
            [(test_data,)],
            schema="bin binary",
        )
        sig = inspect.signature(from_protobuf)
        # Use PERMISSIVE mode to allow unknown enum values to pass through
        options = {"enums.as.ints": "true", "mode": "PERMISSIVE"}
        if "binaryDescriptorSet" in sig.parameters:
            decoded = from_protobuf(
                f.col("bin"),
                message_name,
                binaryDescriptorSet=bytearray(desc_bytes),
                options=options
            )
        else:
            decoded = from_protobuf(f.col("bin"), message_name, desc_path, options)
        return df.select(
            decoded.getField("color").alias("color"),
            decoded.getField("count").alias("count")
        )

    assert_gpu_and_cpu_are_equal_collect(run_on_spark)


def _build_required_field_descriptor_set_bytes(spark):
    """
    Build a FileDescriptorSet for a message with required fields (proto2):
      package test;
      syntax = "proto2";
      message WithRequired {
        required int64 id = 1;
        optional string name = 2;
        optional int32 count = 3;
      }
    """
    jvm = spark.sparkContext._jvm
    D = jvm.com.google.protobuf.DescriptorProtos

    fd = D.FileDescriptorProto.newBuilder() \
        .setName("required.proto") \
        .setPackage("test")
    try:
        fd = fd.setSyntax("proto2")
    except Exception:
        pass

    label_required = D.FieldDescriptorProto.Label.LABEL_REQUIRED
    label_optional = D.FieldDescriptorProto.Label.LABEL_OPTIONAL

    msg = D.DescriptorProto.newBuilder().setName("WithRequired")
    
    # id field (required)
    msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("id")
            .setNumber(1)
            .setLabel(label_required)
            .setType(D.FieldDescriptorProto.Type.TYPE_INT64)
            .build()
    )
    # name field (optional)
    msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("name")
            .setNumber(2)
            .setLabel(label_optional)
            .setType(D.FieldDescriptorProto.Type.TYPE_STRING)
            .build()
    )
    # count field (optional)
    msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("count")
            .setNumber(3)
            .setLabel(label_optional)
            .setType(D.FieldDescriptorProto.Type.TYPE_INT32)
            .build()
    )

    fd.addMessageType(msg.build())
    fds = D.FileDescriptorSet.newBuilder().addFile(fd.build()).build()
    return bytes(fds.toByteArray())


@pytest.mark.skipif(is_before_spark_340(), reason="from_protobuf is Spark 3.4.0+")
@ignore_order(local=True)
def test_from_protobuf_required_field_present(spark_tmp_path):
    """
    Test that required fields decode correctly when present.
    GPU should produce same results as CPU.
    """
    from_protobuf = _try_import_from_protobuf()
    if from_protobuf is None:
        pytest.skip("pyspark.sql.protobuf.functions.from_protobuf is not available")
    if not with_cpu_session(_spark_protobuf_jvm_available):
        pytest.skip("spark-protobuf JVM module is not available on the classpath")

    desc_path = spark_tmp_path + "/required.desc"
    message_name = "test.WithRequired"

    desc_bytes = with_cpu_session(_build_required_field_descriptor_set_bytes)
    with_cpu_session(lambda spark: _write_bytes_to_hadoop_path(spark, desc_path, desc_bytes))

    # Create test data with required field present
    # Row 0: id=100, name="test", count=42
    # Row 1: id=200, name missing, count missing
    test_data_row0 = bytes([
        0x08, 0x64,  # id = 100
        0x12, 0x04, 0x74, 0x65, 0x73, 0x74,  # name = "test"
        0x18, 0x2A,  # count = 42
    ])
    test_data_row1 = bytes([
        0x08, 0xC8, 0x01,  # id = 200
    ])

    def run_on_spark(spark):
        df = spark.createDataFrame(
            [(test_data_row0,), (test_data_row1,)],
            schema="bin binary",
        )
        sig = inspect.signature(from_protobuf)
        if "binaryDescriptorSet" in sig.parameters:
            decoded = from_protobuf(
                f.col("bin"),
                message_name,
                binaryDescriptorSet=bytearray(desc_bytes)
            )
        else:
            decoded = from_protobuf(f.col("bin"), message_name, desc_path)
        return df.select(
            decoded.getField("id").alias("id"),
            decoded.getField("name").alias("name"),
            decoded.getField("count").alias("count")
        )

    assert_gpu_and_cpu_are_equal_collect(run_on_spark)


def _build_default_value_descriptor_set_bytes(spark):
    """
    Build a FileDescriptorSet for a message with default values (proto2):
      package test;
      syntax = "proto2";
      message WithDefaults {
        optional int32 count = 1 [default = 42];
        optional string name = 2 [default = "unknown"];
        optional bool flag = 3 [default = true];
      }
    Note: Setting explicit defaults requires using FieldOptions which may not be
    available via the simple DescriptorProtos API. For testing, we rely on proto2
    implicit behavior where hasDefaultValue() returns true for optional fields.
    """
    jvm = spark.sparkContext._jvm
    D = jvm.com.google.protobuf.DescriptorProtos

    fd = D.FileDescriptorProto.newBuilder() \
        .setName("defaults.proto") \
        .setPackage("test")
    try:
        fd = fd.setSyntax("proto2")
    except Exception:
        pass

    label_optional = D.FieldDescriptorProto.Label.LABEL_OPTIONAL

    msg = D.DescriptorProto.newBuilder().setName("WithDefaults")
    
    # count field with default
    count_field = D.FieldDescriptorProto.newBuilder() \
        .setName("count") \
        .setNumber(1) \
        .setLabel(label_optional) \
        .setType(D.FieldDescriptorProto.Type.TYPE_INT32) \
        .setDefaultValue("42")
    msg.addField(count_field.build())
    
    # name field with default
    name_field = D.FieldDescriptorProto.newBuilder() \
        .setName("name") \
        .setNumber(2) \
        .setLabel(label_optional) \
        .setType(D.FieldDescriptorProto.Type.TYPE_STRING) \
        .setDefaultValue("unknown")
    msg.addField(name_field.build())
    
    # flag field with default
    flag_field = D.FieldDescriptorProto.newBuilder() \
        .setName("flag") \
        .setNumber(3) \
        .setLabel(label_optional) \
        .setType(D.FieldDescriptorProto.Type.TYPE_BOOL) \
        .setDefaultValue("true")
    msg.addField(flag_field.build())

    fd.addMessageType(msg.build())
    fds = D.FileDescriptorSet.newBuilder().addFile(fd.build()).build()
    return bytes(fds.toByteArray())


@pytest.mark.skipif(is_before_spark_340(), reason="from_protobuf is Spark 3.4.0+")
@ignore_order(local=True)
def test_from_protobuf_default_values_field_present(spark_tmp_path):
    """
    Test that when fields with defaults are present, the actual values are used.
    This validates the GPU correctly decodes present values (not using defaults).
    """
    from_protobuf = _try_import_from_protobuf()
    if from_protobuf is None:
        pytest.skip("pyspark.sql.protobuf.functions.from_protobuf is not available")
    if not with_cpu_session(_spark_protobuf_jvm_available):
        pytest.skip("spark-protobuf JVM module is not available on the classpath")

    desc_path = spark_tmp_path + "/defaults.desc"
    message_name = "test.WithDefaults"

    desc_bytes = with_cpu_session(_build_default_value_descriptor_set_bytes)
    with_cpu_session(lambda spark: _write_bytes_to_hadoop_path(spark, desc_path, desc_bytes))

    # Create test data where all fields are present (actual values, not defaults)
    test_data = bytes([
        0x08, 0x64,  # count = 100 (not the default 42)
        0x12, 0x04, 0x74, 0x65, 0x73, 0x74,  # name = "test" (not "unknown")
        0x18, 0x00,  # flag = false (not true)
    ])

    def run_on_spark(spark):
        df = spark.createDataFrame(
            [(test_data,)],
            schema="bin binary",
        )
        sig = inspect.signature(from_protobuf)
        if "binaryDescriptorSet" in sig.parameters:
            decoded = from_protobuf(
                f.col("bin"),
                message_name,
                binaryDescriptorSet=bytearray(desc_bytes)
            )
        else:
            decoded = from_protobuf(f.col("bin"), message_name, desc_path)
        return df.select(
            decoded.getField("count").alias("count"),
            decoded.getField("name").alias("name"),
            decoded.getField("flag").alias("flag")
        )

    assert_gpu_and_cpu_are_equal_collect(run_on_spark)


@pytest.mark.skipif(is_before_spark_340(), reason="from_protobuf is Spark 3.4.0+")
@ignore_order(local=True)
def test_from_protobuf_default_values_missing_fields(spark_tmp_path):
    """
    Test that when fields with defaults are missing, the default values are filled in.
    This validates the GPU correctly fills default values for missing fields.
    """
    from_protobuf = _try_import_from_protobuf()
    if from_protobuf is None:
        pytest.skip("pyspark.sql.protobuf.functions.from_protobuf is not available")
    if not with_cpu_session(_spark_protobuf_jvm_available):
        pytest.skip("spark-protobuf JVM module is not available on the classpath")

    desc_path = spark_tmp_path + "/defaults.desc"
    message_name = "test.WithDefaults"

    desc_bytes = with_cpu_session(_build_default_value_descriptor_set_bytes)
    with_cpu_session(lambda spark: _write_bytes_to_hadoop_path(spark, desc_path, desc_bytes))

    # Create test data where all fields are MISSING (should use defaults)
    # Empty protobuf message
    test_data_empty = bytes([])
    
    # Partial message: only count field is present
    test_data_partial = bytes([
        0x08, 0x64,  # count = 100
    ])

    def run_on_spark(spark):
        df = spark.createDataFrame(
            [(test_data_empty,), (test_data_partial,)],
            schema="bin binary",
        )
        sig = inspect.signature(from_protobuf)
        if "binaryDescriptorSet" in sig.parameters:
            decoded = from_protobuf(
                f.col("bin"),
                message_name,
                binaryDescriptorSet=bytearray(desc_bytes)
            )
        else:
            decoded = from_protobuf(f.col("bin"), message_name, desc_path)
        return df.select(
            decoded.getField("count").alias("count"),
            decoded.getField("name").alias("name"),
            decoded.getField("flag").alias("flag")
        )

    assert_gpu_and_cpu_are_equal_collect(run_on_spark)


@pytest.mark.skipif(is_before_spark_340(), reason="from_protobuf is Spark 3.4.0+")
@ignore_order(local=True)
def test_from_protobuf_string_default_value(spark_tmp_path):
    """
    Test string default value specifically.
    """
    from_protobuf = _try_import_from_protobuf()
    if from_protobuf is None:
        pytest.skip("pyspark.sql.protobuf.functions.from_protobuf is not available")
    if not with_cpu_session(_spark_protobuf_jvm_available):
        pytest.skip("spark-protobuf JVM module is not available on the classpath")

    desc_path = spark_tmp_path + "/defaults.desc"
    message_name = "test.WithDefaults"

    desc_bytes = with_cpu_session(_build_default_value_descriptor_set_bytes)
    with_cpu_session(lambda spark: _write_bytes_to_hadoop_path(spark, desc_path, desc_bytes))

    # Create test data where only non-string fields are present
    # name field is missing, should use default "unknown"
    test_data = bytes([
        0x08, 0x2A,  # count = 42
        0x18, 0x01,  # flag = true
    ])

    def run_on_spark(spark):
        df = spark.createDataFrame(
            [(test_data,)],
            schema="bin binary",
        )
        sig = inspect.signature(from_protobuf)
        if "binaryDescriptorSet" in sig.parameters:
            decoded = from_protobuf(
                f.col("bin"),
                message_name,
                binaryDescriptorSet=bytearray(desc_bytes)
            )
        else:
            decoded = from_protobuf(f.col("bin"), message_name, desc_path)
        return df.select(
            decoded.getField("name").alias("name")
        )

    assert_gpu_and_cpu_are_equal_collect(run_on_spark)


# =============================================================================
# Enhanced Protobuf Tests with Random Data Generation
# =============================================================================

def _build_all_scalars_descriptor_set_bytes(spark):
    """
    Build a FileDescriptorSet for message with all scalar types:
      message AllScalars {
        optional bool b = 1;
        optional int32 i32 = 2;
        optional int64 i64 = 3;
        optional float f32 = 4;
        optional double f64 = 5;
        optional string s = 6;
        optional sint32 si32 = 7;  // zigzag encoding
        optional sint64 si64 = 8;  // zigzag encoding
        optional fixed32 fx32 = 9;
        optional fixed64 fx64 = 10;
      }
    """
    jvm = spark.sparkContext._jvm
    D = jvm.com.google.protobuf.DescriptorProtos

    fd = D.FileDescriptorProto.newBuilder() \
        .setName("all_scalars.proto") \
        .setPackage("test")
    try:
        fd = fd.setSyntax("proto2")
    except Exception:
        pass

    msg = D.DescriptorProto.newBuilder().setName("AllScalars")
    label_opt = D.FieldDescriptorProto.Label.LABEL_OPTIONAL

    def add_field(name, number, ftype):
        msg.addField(
            D.FieldDescriptorProto.newBuilder()
              .setName(name)
              .setNumber(number)
              .setLabel(label_opt)
              .setType(ftype)
              .build()
        )

    T = D.FieldDescriptorProto.Type
    add_field("b", 1, T.TYPE_BOOL)
    add_field("i32", 2, T.TYPE_INT32)
    add_field("i64", 3, T.TYPE_INT64)
    add_field("f32", 4, T.TYPE_FLOAT)
    add_field("f64", 5, T.TYPE_DOUBLE)
    add_field("s", 6, T.TYPE_STRING)
    add_field("si32", 7, T.TYPE_SINT32)
    add_field("si64", 8, T.TYPE_SINT64)
    add_field("fx32", 9, T.TYPE_FIXED32)
    add_field("fx64", 10, T.TYPE_FIXED64)

    fd.addMessageType(msg.build())
    fds = D.FileDescriptorSet.newBuilder().addFile(fd.build()).build()
    return bytes(fds.toByteArray())


def _scalar_test_id(config):
    """Generate stable test ID using only the first element (test name)."""
    return config[0] if isinstance(config, tuple) else str(config)


@pytest.mark.skipif(is_before_spark_340(), reason="from_protobuf is Spark 3.4.0+")
@pytest.mark.parametrize("test_config", _random_scalar_test_configs, ids=_scalar_test_id)
@ignore_order(local=True)
def test_from_protobuf_random_scalars(spark_tmp_path, test_config):
    """
    Parametrized test for from_protobuf with randomly generated scalar data.
    Covers: all types, integer edge cases, float edge cases, nullable fields, large datasets.
    """
    test_id, field_configs, data_length = test_config
    
    from_protobuf = _try_import_from_protobuf()
    if from_protobuf is None:
        pytest.skip("from_protobuf not available")
    if not with_cpu_session(_spark_protobuf_jvm_available):
        pytest.skip("spark-protobuf JVM not available")

    desc_path = spark_tmp_path + "/simple.desc"
    message_name = "test.Simple"

    desc_bytes = with_cpu_session(_build_simple_descriptor_set_bytes)
    with_cpu_session(lambda spark: _write_bytes_to_hadoop_path(spark, desc_path, desc_bytes))

    data_gen = ProtobufSimpleMessageRowGen(field_configs)

    def run_on_spark(spark):
        df = gen_df(spark, data_gen, length=data_length)
        
        sig = inspect.signature(from_protobuf)
        if "binaryDescriptorSet" in sig.parameters:
            decoded = from_protobuf(
                f.col("bin"), message_name,
                binaryDescriptorSet=bytearray(desc_bytes))
        else:
            decoded = from_protobuf(f.col("bin"), message_name, desc_path)
        
        # Select all decoded fields
        return df.select(
            decoded.getField("b").alias("b"),
            decoded.getField("i32").alias("i32"),
            decoded.getField("i64").alias("i64"),
            decoded.getField("f32").alias("f32"),
            decoded.getField("f64").alias("f64"),
            decoded.getField("s").alias("s"),
        )

    assert_gpu_and_cpu_are_equal_collect(run_on_spark)


@pytest.mark.skipif(is_before_spark_340(), reason="from_protobuf is Spark 3.4.0+")
@ignore_order(local=True)  
def test_from_protobuf_nullable_fields(spark_tmp_path):
    """
    Test from_protobuf with nullable fields - some rows have missing fields.
    """
    from_protobuf = _try_import_from_protobuf()
    if from_protobuf is None:
        pytest.skip("pyspark.sql.protobuf.functions.from_protobuf is not available")
    if not with_cpu_session(_spark_protobuf_jvm_available):
        pytest.skip("spark-protobuf JVM module is not available on the classpath")

    desc_path = spark_tmp_path + "/simple.desc"
    message_name = "test.Simple"

    desc_bytes = with_cpu_session(_build_simple_descriptor_set_bytes)
    with_cpu_session(lambda spark: _write_bytes_to_hadoop_path(spark, desc_path, desc_bytes))

    # Generate data with high null probability
    data_gen = ProtobufSimpleMessageRowGen([
        ("b", 1, BooleanGen(nullable=True)),
        ("i32", 2, IntegerGen(nullable=True)),
        ("i64", 3, LongGen(nullable=True)),
        ("f32", 4, FloatGen(nullable=True)),
        ("f64", 5, DoubleGen(nullable=True)),
        ("s", 6, StringGen(nullable=True)),
    ])

    def run_on_spark(spark):
        df = gen_df(spark, data_gen, length=100)
        
        sig = inspect.signature(from_protobuf)
        if "binaryDescriptorSet" in sig.parameters:
            decoded = from_protobuf(
                f.col("bin"),
                message_name,
                binaryDescriptorSet=bytearray(desc_bytes),
            )
        else:
            decoded = from_protobuf(f.col("bin"), message_name, desc_path)
        
        return df.select(
            decoded.getField("b").alias("b"),
            decoded.getField("i32").alias("i32"),
            decoded.getField("i64").alias("i64"),
            decoded.getField("f32").alias("f32"),
            decoded.getField("f64").alias("f64"),
            decoded.getField("s").alias("s"),
        )

    assert_gpu_and_cpu_are_equal_collect(run_on_spark)


# =============================================================================
# Nested and Repeated Field Integration Tests
# =============================================================================

def _build_repeated_int_descriptor_set_bytes(spark):
    """
    Build a FileDescriptorSet for a message with repeated int32 field:
      message WithRepeatedInt {
        optional int32 id = 1;
        repeated int32 values = 2;
      }
    """
    jvm = spark.sparkContext._jvm
    D = jvm.com.google.protobuf.DescriptorProtos

    fd = D.FileDescriptorProto.newBuilder() \
        .setName("repeated_int.proto") \
        .setPackage("test")
    try:
        fd = fd.setSyntax("proto2")
    except Exception:
        pass

    label_opt = D.FieldDescriptorProto.Label.LABEL_OPTIONAL
    label_rep = D.FieldDescriptorProto.Label.LABEL_REPEATED

    msg = D.DescriptorProto.newBuilder().setName("WithRepeatedInt")
    msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("id")
            .setNumber(1)
            .setLabel(label_opt)
            .setType(D.FieldDescriptorProto.Type.TYPE_INT32)
            .build()
    )
    msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("values")
            .setNumber(2)
            .setLabel(label_rep)
            .setType(D.FieldDescriptorProto.Type.TYPE_INT32)
            .build()
    )
    fd.addMessageType(msg.build())

    fds = D.FileDescriptorSet.newBuilder().addFile(fd.build()).build()
    return bytes(fds.toByteArray())


@pytest.mark.skipif(is_before_spark_340(), reason="from_protobuf is Spark 3.4.0+")
@ignore_order(local=True)
def test_from_protobuf_repeated_int32(spark_tmp_path):
    """
    Test decoding repeated int32 field (ArrayType of integers).
    """
    from_protobuf = _try_import_from_protobuf()
    if from_protobuf is None:
        pytest.skip("from_protobuf not available")
    if not with_cpu_session(_spark_protobuf_jvm_available):
        pytest.skip("spark-protobuf JVM not available")

    desc_path = spark_tmp_path + "/repeated_int.desc"
    message_name = "test.WithRepeatedInt"

    desc_bytes = with_cpu_session(_build_repeated_int_descriptor_set_bytes)
    with_cpu_session(lambda spark: _write_bytes_to_hadoop_path(
        spark, desc_path, desc_bytes))

    # Create test data:
    # id = 1, values = [10, 20, 30]
    # Field 1 (id): varint 1 -> 0x08 0x01
    # Field 2 (values): repeated varint 10, 20, 30 -> 0x10 0x0A, 0x10 0x14, 0x10 0x1E
    test_data_1 = bytes([
        0x08, 0x01,  # id = 1
        0x10, 0x0A,  # values[0] = 10
        0x10, 0x14,  # values[1] = 20
        0x10, 0x1E,  # values[2] = 30
    ])
    
    # id = 2, values = [] (empty array)
    test_data_2 = bytes([
        0x08, 0x02,  # id = 2
    ])
    
    # id = 3, values = [100]
    test_data_3 = bytes([
        0x08, 0x03,  # id = 3
        0x10, 0x64,  # values[0] = 100
    ])

    def run_on_spark(spark):
        df = spark.createDataFrame(
            [(test_data_1,), (test_data_2,), (test_data_3,), (None,)],
            schema="bin binary",
        )
        sig = inspect.signature(from_protobuf)
        if "binaryDescriptorSet" in sig.parameters:
            decoded = from_protobuf(
                f.col("bin"), message_name,
                binaryDescriptorSet=bytearray(desc_bytes))
        else:
            decoded = from_protobuf(f.col("bin"), message_name, desc_path)
        
        return df.select(
            decoded.getField("id").alias("id"),
            decoded.getField("values").alias("values"),
        )

    assert_gpu_and_cpu_are_equal_collect(run_on_spark)


def _build_repeated_string_descriptor_set_bytes(spark):
    """
    Build a FileDescriptorSet for a message with repeated string field:
      message WithRepeatedString {
        optional string name = 1;
        repeated string tags = 2;
      }
    """
    jvm = spark.sparkContext._jvm
    D = jvm.com.google.protobuf.DescriptorProtos

    fd = D.FileDescriptorProto.newBuilder() \
        .setName("repeated_string.proto") \
        .setPackage("test")
    try:
        fd = fd.setSyntax("proto2")
    except Exception:
        pass

    label_opt = D.FieldDescriptorProto.Label.LABEL_OPTIONAL
    label_rep = D.FieldDescriptorProto.Label.LABEL_REPEATED

    msg = D.DescriptorProto.newBuilder().setName("WithRepeatedString")
    msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("name")
            .setNumber(1)
            .setLabel(label_opt)
            .setType(D.FieldDescriptorProto.Type.TYPE_STRING)
            .build()
    )
    msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("tags")
            .setNumber(2)
            .setLabel(label_rep)
            .setType(D.FieldDescriptorProto.Type.TYPE_STRING)
            .build()
    )
    fd.addMessageType(msg.build())

    fds = D.FileDescriptorSet.newBuilder().addFile(fd.build()).build()
    return bytes(fds.toByteArray())


@pytest.mark.skipif(is_before_spark_340(), reason="from_protobuf is Spark 3.4.0+")
@ignore_order(local=True)
def test_from_protobuf_repeated_string(spark_tmp_path):
    """
    Test decoding repeated string field (ArrayType of strings).
    """
    from_protobuf = _try_import_from_protobuf()
    if from_protobuf is None:
        pytest.skip("from_protobuf not available")
    if not with_cpu_session(_spark_protobuf_jvm_available):
        pytest.skip("spark-protobuf JVM not available")

    desc_path = spark_tmp_path + "/repeated_string.desc"
    message_name = "test.WithRepeatedString"

    desc_bytes = with_cpu_session(_build_repeated_string_descriptor_set_bytes)
    with_cpu_session(lambda spark: _write_bytes_to_hadoop_path(
        spark, desc_path, desc_bytes))

    # Create test data:
    # name = "item1", tags = ["a", "b", "c"]
    test_data_1 = bytes([
        0x0A, 0x05, 0x69, 0x74, 0x65, 0x6D, 0x31,  # name = "item1"
        0x12, 0x01, 0x61,  # tags[0] = "a"
        0x12, 0x01, 0x62,  # tags[1] = "b"
        0x12, 0x01, 0x63,  # tags[2] = "c"
    ])
    
    # name = "item2", tags = []
    test_data_2 = bytes([
        0x0A, 0x05, 0x69, 0x74, 0x65, 0x6D, 0x32,  # name = "item2"
    ])

    def run_on_spark(spark):
        df = spark.createDataFrame(
            [(test_data_1,), (test_data_2,), (None,)],
            schema="bin binary",
        )
        sig = inspect.signature(from_protobuf)
        if "binaryDescriptorSet" in sig.parameters:
            decoded = from_protobuf(
                f.col("bin"), message_name,
                binaryDescriptorSet=bytearray(desc_bytes))
        else:
            decoded = from_protobuf(f.col("bin"), message_name, desc_path)
        
        return df.select(
            decoded.getField("name").alias("name"),
            decoded.getField("tags").alias("tags"),
        )

    assert_gpu_and_cpu_are_equal_collect(run_on_spark)


@pytest.mark.skipif(is_before_spark_340(), reason="from_protobuf is Spark 3.4.0+")
@ignore_order(local=True)
def test_from_protobuf_nested_message(spark_tmp_path):
    """
    Test decoding nested message field (StructType).
    """
    from_protobuf = _try_import_from_protobuf()
    if from_protobuf is None:
        pytest.skip("from_protobuf not available")
    if not with_cpu_session(_spark_protobuf_jvm_available):
        pytest.skip("spark-protobuf JVM not available")

    desc_path = spark_tmp_path + "/nested.desc"
    message_name = "test.WithNested"

    desc_bytes = with_cpu_session(_build_nested_descriptor_set_bytes)
    with_cpu_session(lambda spark: _write_bytes_to_hadoop_path(
        spark, desc_path, desc_bytes))

    # Create test data with nested message:
    # simple_int = 42, simple_str = "hello", nested_msg = {x: 100}, simple_long = 999
    # Field 1: varint 42 -> 0x08 0x2A
    # Field 2: len-delim "hello" -> 0x12 0x05 h e l l o
    # Field 3: len-delim nested {x=100} -> 0x1A 0x02 0x08 0x64
    # Field 4: varint 999 -> 0x20 0xE7 0x07
    test_data_with_nested = bytes([
        0x08, 0x2A,  # simple_int = 42
        0x12, 0x05, 0x68, 0x65, 0x6C, 0x6C, 0x6F,  # simple_str = "hello"
        0x1A, 0x02, 0x08, 0x64,  # nested_msg = {x: 100}
        0x20, 0xE7, 0x07,  # simple_long = 999
    ])
    
    # Test with empty nested message
    test_data_empty_nested = bytes([
        0x08, 0x01,  # simple_int = 1
        0x1A, 0x00,  # nested_msg = {} (empty)
    ])
    
    # Test without nested message
    test_data_no_nested = bytes([
        0x08, 0x02,  # simple_int = 2
        0x12, 0x05, 0x77, 0x6F, 0x72, 0x6C, 0x64,  # simple_str = "world"
    ])

    def run_on_spark(spark):
        df = spark.createDataFrame(
            [(test_data_with_nested,), (test_data_empty_nested,), 
             (test_data_no_nested,), (None,)],
            schema="bin binary",
        )
        sig = inspect.signature(from_protobuf)
        if "binaryDescriptorSet" in sig.parameters:
            decoded = from_protobuf(
                f.col("bin"), message_name,
                binaryDescriptorSet=bytearray(desc_bytes))
        else:
            decoded = from_protobuf(f.col("bin"), message_name, desc_path)
        
        # Select all fields including nested
        return df.select(
            decoded.getField("simple_int").alias("simple_int"),
            decoded.getField("simple_str").alias("simple_str"),
            decoded.getField("nested_msg").alias("nested_msg"),
            decoded.getField("simple_long").alias("simple_long"),
        )

    assert_gpu_and_cpu_are_equal_collect(run_on_spark)


@pytest.mark.skipif(is_before_spark_340(), reason="from_protobuf is Spark 3.4.0+")
@ignore_order(local=True)
def test_from_protobuf_nested_message_field_access(spark_tmp_path):
    """
    Test accessing fields within nested message.
    """
    from_protobuf = _try_import_from_protobuf()
    if from_protobuf is None:
        pytest.skip("from_protobuf not available")
    if not with_cpu_session(_spark_protobuf_jvm_available):
        pytest.skip("spark-protobuf JVM not available")

    desc_path = spark_tmp_path + "/nested.desc"
    message_name = "test.WithNested"

    desc_bytes = with_cpu_session(_build_nested_descriptor_set_bytes)
    with_cpu_session(lambda spark: _write_bytes_to_hadoop_path(
        spark, desc_path, desc_bytes))

    test_data = bytes([
        0x08, 0x2A,  # simple_int = 42
        0x1A, 0x02, 0x08, 0x64,  # nested_msg = {x: 100}
    ])

    def run_on_spark(spark):
        df = spark.createDataFrame(
            [(test_data,), (None,)],
            schema="bin binary",
        )
        sig = inspect.signature(from_protobuf)
        if "binaryDescriptorSet" in sig.parameters:
            decoded = from_protobuf(
                f.col("bin"), message_name,
                binaryDescriptorSet=bytearray(desc_bytes))
        else:
            decoded = from_protobuf(f.col("bin"), message_name, desc_path)
        
        # Access nested field directly
        return df.select(
            decoded.getField("simple_int").alias("simple_int"),
            decoded.getField("nested_msg").getField("x").alias("nested_x"),
        )

    assert_gpu_and_cpu_are_equal_collect(run_on_spark)


def _build_deep_nested_descriptor_set_bytes(spark):
    """
    Build a FileDescriptorSet for deeply nested message:
      message Inner {
        optional int32 value = 1;
      }
      message Middle {
        optional string name = 1;
        optional Inner inner = 2;
      }
      message Outer {
        optional int32 id = 1;
        optional Middle middle = 2;
      }
    """
    jvm = spark.sparkContext._jvm
    D = jvm.com.google.protobuf.DescriptorProtos

    fd = D.FileDescriptorProto.newBuilder() \
        .setName("deep_nested.proto") \
        .setPackage("test")
    try:
        fd = fd.setSyntax("proto2")
    except Exception:
        pass

    label_opt = D.FieldDescriptorProto.Label.LABEL_OPTIONAL

    # Inner message
    inner_msg = D.DescriptorProto.newBuilder().setName("Inner")
    inner_msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("value")
            .setNumber(1)
            .setLabel(label_opt)
            .setType(D.FieldDescriptorProto.Type.TYPE_INT32)
            .build()
    )
    fd.addMessageType(inner_msg.build())

    # Middle message
    middle_msg = D.DescriptorProto.newBuilder().setName("Middle")
    middle_msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("name")
            .setNumber(1)
            .setLabel(label_opt)
            .setType(D.FieldDescriptorProto.Type.TYPE_STRING)
            .build()
    )
    middle_msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("inner")
            .setNumber(2)
            .setLabel(label_opt)
            .setType(D.FieldDescriptorProto.Type.TYPE_MESSAGE)
            .setTypeName(".test.Inner")
            .build()
    )
    fd.addMessageType(middle_msg.build())

    # Outer message
    outer_msg = D.DescriptorProto.newBuilder().setName("Outer")
    outer_msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("id")
            .setNumber(1)
            .setLabel(label_opt)
            .setType(D.FieldDescriptorProto.Type.TYPE_INT32)
            .build()
    )
    outer_msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("middle")
            .setNumber(2)
            .setLabel(label_opt)
            .setType(D.FieldDescriptorProto.Type.TYPE_MESSAGE)
            .setTypeName(".test.Middle")
            .build()
    )
    fd.addMessageType(outer_msg.build())

    fds = D.FileDescriptorSet.newBuilder().addFile(fd.build()).build()
    return bytes(fds.toByteArray())


@pytest.mark.skipif(is_before_spark_340(), reason="from_protobuf is Spark 3.4.0+")
@ignore_order(local=True)
def test_from_protobuf_deep_nested(spark_tmp_path):
    """
    Test decoding deeply nested messages (3 levels).
    """
    from_protobuf = _try_import_from_protobuf()
    if from_protobuf is None:
        pytest.skip("from_protobuf not available")
    if not with_cpu_session(_spark_protobuf_jvm_available):
        pytest.skip("spark-protobuf JVM not available")

    desc_path = spark_tmp_path + "/deep_nested.desc"
    message_name = "test.Outer"

    desc_bytes = with_cpu_session(_build_deep_nested_descriptor_set_bytes)
    with_cpu_session(lambda spark: _write_bytes_to_hadoop_path(
        spark, desc_path, desc_bytes))

    # Outer{id=1, middle=Middle{name="test", inner=Inner{value=42}}}
    # Inner{value=42}: 0x08 0x2A (2 bytes)
    # Middle{name="test", inner=...}: 0x0A 0x04 t e s t, 0x12 0x02 <inner> 
    # Outer{id=1, middle=...}: 0x08 0x01, 0x12 <len> <middle>
    inner_bytes = bytes([0x08, 0x2A])  # value = 42
    middle_bytes = bytes([
        0x0A, 0x04, 0x74, 0x65, 0x73, 0x74,  # name = "test"
        0x12, len(inner_bytes)
    ]) + inner_bytes
    outer_bytes = bytes([
        0x08, 0x01,  # id = 1
        0x12, len(middle_bytes)
    ]) + middle_bytes

    def run_on_spark(spark):
        df = spark.createDataFrame(
            [(outer_bytes,), (None,)],
            schema="bin binary",
        )
        sig = inspect.signature(from_protobuf)
        if "binaryDescriptorSet" in sig.parameters:
            decoded = from_protobuf(
                f.col("bin"), message_name,
                binaryDescriptorSet=bytearray(desc_bytes))
        else:
            decoded = from_protobuf(f.col("bin"), message_name, desc_path)
        
        return df.select(
            decoded.getField("id").alias("id"),
            decoded.getField("middle").alias("middle"),
        )

    assert_gpu_and_cpu_are_equal_collect(run_on_spark)


def _build_repeated_message_descriptor_set_bytes(spark):
    """
    Build a FileDescriptorSet for repeated message field (array of structs):
      message Item {
        optional int32 id = 1;
        optional string name = 2;
      }
      message Container {
        optional string title = 1;
        repeated Item items = 2;
      }
    """
    jvm = spark.sparkContext._jvm
    D = jvm.com.google.protobuf.DescriptorProtos

    fd = D.FileDescriptorProto.newBuilder() \
        .setName("repeated_message.proto") \
        .setPackage("test")
    try:
        fd = fd.setSyntax("proto2")
    except Exception:
        pass

    label_opt = D.FieldDescriptorProto.Label.LABEL_OPTIONAL
    label_rep = D.FieldDescriptorProto.Label.LABEL_REPEATED

    # Item message
    item_msg = D.DescriptorProto.newBuilder().setName("Item")
    item_msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("id")
            .setNumber(1)
            .setLabel(label_opt)
            .setType(D.FieldDescriptorProto.Type.TYPE_INT32)
            .build()
    )
    item_msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("name")
            .setNumber(2)
            .setLabel(label_opt)
            .setType(D.FieldDescriptorProto.Type.TYPE_STRING)
            .build()
    )
    fd.addMessageType(item_msg.build())

    # Container message
    container_msg = D.DescriptorProto.newBuilder().setName("Container")
    container_msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("title")
            .setNumber(1)
            .setLabel(label_opt)
            .setType(D.FieldDescriptorProto.Type.TYPE_STRING)
            .build()
    )
    container_msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("items")
            .setNumber(2)
            .setLabel(label_rep)
            .setType(D.FieldDescriptorProto.Type.TYPE_MESSAGE)
            .setTypeName(".test.Item")
            .build()
    )
    fd.addMessageType(container_msg.build())

    fds = D.FileDescriptorSet.newBuilder().addFile(fd.build()).build()
    return bytes(fds.toByteArray())


@pytest.mark.skipif(is_before_spark_340(), reason="from_protobuf is Spark 3.4.0+")
@ignore_order(local=True)
def test_from_protobuf_repeated_message(spark_tmp_path):
    """
    Test decoding repeated message field (ArrayType of StructType).
    """
    from_protobuf = _try_import_from_protobuf()
    if from_protobuf is None:
        pytest.skip("from_protobuf not available")
    if not with_cpu_session(_spark_protobuf_jvm_available):
        pytest.skip("spark-protobuf JVM not available")

    desc_path = spark_tmp_path + "/repeated_message.desc"
    message_name = "test.Container"

    desc_bytes = with_cpu_session(_build_repeated_message_descriptor_set_bytes)
    with_cpu_session(lambda spark: _write_bytes_to_hadoop_path(
        spark, desc_path, desc_bytes))

    # Container{title="list", items=[Item{id=1,name="a"}, Item{id=2,name="b"}]}
    item1 = bytes([0x08, 0x01, 0x12, 0x01, 0x61])  # id=1, name="a"
    item2 = bytes([0x08, 0x02, 0x12, 0x01, 0x62])  # id=2, name="b"
    test_data = bytes([
        0x0A, 0x04, 0x6C, 0x69, 0x73, 0x74,  # title = "list"
        0x12, len(item1)
    ]) + item1 + bytes([0x12, len(item2)]) + item2
    
    # Empty items array
    test_data_empty = bytes([
        0x0A, 0x05, 0x65, 0x6D, 0x70, 0x74, 0x79,  # title = "empty"
    ])

    def run_on_spark(spark):
        df = spark.createDataFrame(
            [(test_data,), (test_data_empty,), (None,)],
            schema="bin binary",
        )
        sig = inspect.signature(from_protobuf)
        if "binaryDescriptorSet" in sig.parameters:
            decoded = from_protobuf(
                f.col("bin"), message_name,
                binaryDescriptorSet=bytearray(desc_bytes))
        else:
            decoded = from_protobuf(f.col("bin"), message_name, desc_path)
        
        return df.select(
            decoded.getField("title").alias("title"),
            decoded.getField("items").alias("items"),
        )

    assert_gpu_and_cpu_are_equal_collect(run_on_spark)


# =============================================================================
# Complex Nested Type Tests
# =============================================================================

def _build_nested_with_repeated_descriptor_set_bytes(spark):
    """
    Build a FileDescriptorSet for nested message with repeated fields:
      message NestedWithRepeated {
        optional string name = 1;
        repeated int32 values = 2;
        optional int32 count = 3;
      }
      message OuterWithNestedRepeated {
        optional int32 id = 1;
        optional NestedWithRepeated nested = 2;
      }
    """
    jvm = spark.sparkContext._jvm
    D = jvm.com.google.protobuf.DescriptorProtos

    fd = D.FileDescriptorProto.newBuilder() \
        .setName("nested_with_repeated.proto") \
        .setPackage("test")
    try:
        fd = fd.setSyntax("proto2")
    except Exception:
        pass

    label_opt = D.FieldDescriptorProto.Label.LABEL_OPTIONAL
    label_rep = D.FieldDescriptorProto.Label.LABEL_REPEATED

    # NestedWithRepeated message
    nested_msg = D.DescriptorProto.newBuilder().setName("NestedWithRepeated")
    nested_msg.addField(D.FieldDescriptorProto.newBuilder()
        .setName("name").setNumber(1).setLabel(label_opt)
        .setType(D.FieldDescriptorProto.Type.TYPE_STRING))
    nested_msg.addField(D.FieldDescriptorProto.newBuilder()
        .setName("values").setNumber(2).setLabel(label_rep)
        .setType(D.FieldDescriptorProto.Type.TYPE_INT32))
    nested_msg.addField(D.FieldDescriptorProto.newBuilder()
        .setName("count").setNumber(3).setLabel(label_opt)
        .setType(D.FieldDescriptorProto.Type.TYPE_INT32))
    fd.addMessageType(nested_msg)

    # OuterWithNestedRepeated message
    outer_msg = D.DescriptorProto.newBuilder().setName("OuterWithNestedRepeated")
    outer_msg.addField(D.FieldDescriptorProto.newBuilder()
        .setName("id").setNumber(1).setLabel(label_opt)
        .setType(D.FieldDescriptorProto.Type.TYPE_INT32))
    outer_msg.addField(D.FieldDescriptorProto.newBuilder()
        .setName("nested").setNumber(2).setLabel(label_opt)
        .setType(D.FieldDescriptorProto.Type.TYPE_MESSAGE)
        .setTypeName(".test.NestedWithRepeated"))
    fd.addMessageType(outer_msg)

    fds = D.FileDescriptorSet.newBuilder().addFile(fd).build()
    return bytes(fds.toByteArray())


@pytest.mark.skipif(is_before_spark_340(), reason="from_protobuf is Spark 3.4.0+")
@ignore_order(local=True)
def test_from_protobuf_nested_with_repeated(spark_tmp_path):
    """
    Test decoding nested message that contains repeated fields.
    Schema: OuterWithNestedRepeated { id, nested: NestedWithRepeated { name, values[], count } }
    This tests StructType containing StructType containing ArrayType.
    """
    from_protobuf = _try_import_from_protobuf()
    if from_protobuf is None:
        pytest.skip("from_protobuf not available")
    if not with_cpu_session(_spark_protobuf_jvm_available):
        pytest.skip("spark-protobuf JVM not available")

    desc_path = spark_tmp_path + "/nested_with_repeated.desc"
    message_name = "test.OuterWithNestedRepeated"

    desc_bytes = with_cpu_session(_build_nested_with_repeated_descriptor_set_bytes)
    with_cpu_session(lambda spark: _write_bytes_to_hadoop_path(
        spark, desc_path, desc_bytes))

    # NestedWithRepeated{name="test", values=[1,2,3], count=3}
    # name: 0x0A 0x04 "test"
    # values: 0x10 0x01, 0x10 0x02, 0x10 0x03
    # count: 0x18 0x03
    nested_bytes = bytes([
        0x0A, 0x04, 0x74, 0x65, 0x73, 0x74,  # name = "test"
        0x10, 0x01,  # values = 1
        0x10, 0x02,  # values = 2
        0x10, 0x03,  # values = 3
        0x18, 0x03,  # count = 3
    ])
    
    # OuterWithNestedRepeated{id=42, nested=...}
    outer_bytes = bytes([
        0x08, 0x2A,  # id = 42
        0x12, len(nested_bytes)
    ]) + nested_bytes

    # Empty nested
    empty_nested_bytes = bytes([
        0x08, 0x01,  # id = 1
        0x12, 0x00,  # nested = {} (empty)
    ])

    def run_on_spark(spark):
        df = spark.createDataFrame(
            [(outer_bytes,), (empty_nested_bytes,), (None,)],
            schema="bin binary",
        )
        sig = inspect.signature(from_protobuf)
        if "binaryDescriptorSet" in sig.parameters:
            decoded = from_protobuf(
                f.col("bin"), message_name,
                binaryDescriptorSet=bytearray(desc_bytes))
        else:
            decoded = from_protobuf(f.col("bin"), message_name, desc_path)
        
        return df.select(
            decoded.getField("id").alias("id"),
            decoded.getField("nested").alias("nested"),
        )

    assert_gpu_and_cpu_are_equal_collect(run_on_spark)


def _build_repeated_with_nested_descriptor_set_bytes(spark):
    """
    Build a FileDescriptorSet for repeated message with nested message:
      message Inner {
        optional int32 value = 1;
      }
      message ItemWithNested {
        optional int32 id = 1;
        optional Inner inner = 2;
        optional string name = 3;
      }
      message ContainerWithNestedItems {
        optional string title = 1;
        repeated ItemWithNested items = 2;
      }
    """
    jvm = spark.sparkContext._jvm
    D = jvm.com.google.protobuf.DescriptorProtos

    fd = D.FileDescriptorProto.newBuilder() \
        .setName("repeated_with_nested.proto") \
        .setPackage("test")
    try:
        fd = fd.setSyntax("proto2")
    except Exception:
        pass

    label_opt = D.FieldDescriptorProto.Label.LABEL_OPTIONAL
    label_rep = D.FieldDescriptorProto.Label.LABEL_REPEATED

    # Inner message
    inner_msg = D.DescriptorProto.newBuilder().setName("Inner")
    inner_msg.addField(D.FieldDescriptorProto.newBuilder()
        .setName("value").setNumber(1).setLabel(label_opt)
        .setType(D.FieldDescriptorProto.Type.TYPE_INT32))
    fd.addMessageType(inner_msg)

    # ItemWithNested message
    item_msg = D.DescriptorProto.newBuilder().setName("ItemWithNested")
    item_msg.addField(D.FieldDescriptorProto.newBuilder()
        .setName("id").setNumber(1).setLabel(label_opt)
        .setType(D.FieldDescriptorProto.Type.TYPE_INT32))
    item_msg.addField(D.FieldDescriptorProto.newBuilder()
        .setName("inner").setNumber(2).setLabel(label_opt)
        .setType(D.FieldDescriptorProto.Type.TYPE_MESSAGE)
        .setTypeName(".test.Inner"))
    item_msg.addField(D.FieldDescriptorProto.newBuilder()
        .setName("name").setNumber(3).setLabel(label_opt)
        .setType(D.FieldDescriptorProto.Type.TYPE_STRING))
    fd.addMessageType(item_msg)

    # ContainerWithNestedItems message
    container_msg = D.DescriptorProto.newBuilder().setName("ContainerWithNestedItems")
    container_msg.addField(D.FieldDescriptorProto.newBuilder()
        .setName("title").setNumber(1).setLabel(label_opt)
        .setType(D.FieldDescriptorProto.Type.TYPE_STRING))
    container_msg.addField(D.FieldDescriptorProto.newBuilder()
        .setName("items").setNumber(2).setLabel(label_rep)
        .setType(D.FieldDescriptorProto.Type.TYPE_MESSAGE)
        .setTypeName(".test.ItemWithNested"))
    fd.addMessageType(container_msg)

    fds = D.FileDescriptorSet.newBuilder().addFile(fd).build()
    return bytes(fds.toByteArray())


@pytest.mark.skipif(is_before_spark_340(), reason="from_protobuf is Spark 3.4.0+")
@ignore_order(local=True)
def test_from_protobuf_repeated_with_nested(spark_tmp_path):
    """
    Test decoding repeated message that contains nested message.
    Schema: ContainerWithNestedItems { title, items[]: ItemWithNested { id, inner: Inner { value }, name } }
    This tests ArrayType(StructType(StructType)) - nested struct inside repeated message.
    """
    from_protobuf = _try_import_from_protobuf()
    if from_protobuf is None:
        pytest.skip("from_protobuf not available")
    if not with_cpu_session(_spark_protobuf_jvm_available):
        pytest.skip("spark-protobuf JVM not available")

    desc_path = spark_tmp_path + "/repeated_with_nested.desc"
    message_name = "test.ContainerWithNestedItems"

    desc_bytes = with_cpu_session(_build_repeated_with_nested_descriptor_set_bytes)
    with_cpu_session(lambda spark: _write_bytes_to_hadoop_path(
        spark, desc_path, desc_bytes))

    # Inner{value=100}: 0x08 0x64
    inner1 = bytes([0x08, 0x64])  # value = 100
    inner2 = bytes([0x08, 0xC8, 0x01])  # value = 200
    
    # ItemWithNested{id=1, inner={value=100}, name="first"}
    item1 = bytes([
        0x08, 0x01,  # id = 1
        0x12, len(inner1)
    ]) + inner1 + bytes([
        0x1A, 0x05, 0x66, 0x69, 0x72, 0x73, 0x74,  # name = "first"
    ])
    
    # ItemWithNested{id=2, inner={value=200}, name="second"}
    item2 = bytes([
        0x08, 0x02,  # id = 2
        0x12, len(inner2)
    ]) + inner2 + bytes([
        0x1A, 0x06, 0x73, 0x65, 0x63, 0x6F, 0x6E, 0x64,  # name = "second"
    ])

    # ContainerWithNestedItems{title="container", items=[item1, item2]}
    test_data = bytes([
        0x0A, 0x09, 0x63, 0x6F, 0x6E, 0x74, 0x61, 0x69, 0x6E, 0x65, 0x72,  # title = "container"
        0x12, len(item1)
    ]) + item1 + bytes([0x12, len(item2)]) + item2

    # Empty items
    empty_data = bytes([
        0x0A, 0x05, 0x65, 0x6D, 0x70, 0x74, 0x79,  # title = "empty"
    ])

    def run_on_spark(spark):
        df = spark.createDataFrame(
            [(test_data,), (empty_data,), (None,)],
            schema="bin binary",
        )
        sig = inspect.signature(from_protobuf)
        if "binaryDescriptorSet" in sig.parameters:
            decoded = from_protobuf(
                f.col("bin"), message_name,
                binaryDescriptorSet=bytearray(desc_bytes))
        else:
            decoded = from_protobuf(f.col("bin"), message_name, desc_path)
        
        return df.select(
            decoded.getField("title").alias("title"),
            decoded.getField("items").alias("items"),
        )

    assert_gpu_and_cpu_are_equal_collect(run_on_spark)


# =============================================================================
# Packed Repeated Fields Tests
# =============================================================================

def _build_packed_repeated_descriptor_set_bytes(spark):
    """
    Build a FileDescriptorSet for message with packed repeated fields:
      message WithPackedRepeated {
        optional int32 id = 1;
        repeated int32 int_values = 2 [packed=true];
        repeated double double_values = 3 [packed=true];
        repeated bool bool_values = 4 [packed=true];
      }
    Note: In proto3, repeated numeric fields are packed by default.
    """
    jvm = spark.sparkContext._jvm
    D = jvm.com.google.protobuf.DescriptorProtos

    fd = D.FileDescriptorProto.newBuilder() \
        .setName("packed_repeated.proto") \
        .setPackage("test")
    try:
        fd = fd.setSyntax("proto3")  # proto3 has packed by default
    except Exception:
        pass

    label_rep = D.FieldDescriptorProto.Label.LABEL_REPEATED
    label_opt = D.FieldDescriptorProto.Label.LABEL_OPTIONAL

    msg = D.DescriptorProto.newBuilder().setName("WithPackedRepeated")
    msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("id")
            .setNumber(1)
            .setLabel(label_opt)
            .setType(D.FieldDescriptorProto.Type.TYPE_INT32)
            .build()
    )
    msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("int_values")
            .setNumber(2)
            .setLabel(label_rep)
            .setType(D.FieldDescriptorProto.Type.TYPE_INT32)
            .build()
    )
    msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("double_values")
            .setNumber(3)
            .setLabel(label_rep)
            .setType(D.FieldDescriptorProto.Type.TYPE_DOUBLE)
            .build()
    )
    msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("bool_values")
            .setNumber(4)
            .setLabel(label_rep)
            .setType(D.FieldDescriptorProto.Type.TYPE_BOOL)
            .build()
    )
    fd.addMessageType(msg.build())

    fds = D.FileDescriptorSet.newBuilder().addFile(fd.build()).build()
    return bytes(fds.toByteArray())


import struct

# Packed repeated field test configurations: (field_name, field_key, packed_data, id_value)
_packed_repeated_test_configs = [
    ("int_values", 0x12, bytes([0x01, 0x02, 0x03, 0x7F, 0x80, 0x01]), 1),  # [1,2,3,127,128]
    ("double_values", 0x1A, struct.pack("<ddd", 1.5, 2.5, 3.5), 2),  # [1.5, 2.5, 3.5]
    ("bool_values", 0x22, bytes([0x01, 0x00, 0x01, 0x01, 0x00]), 3),  # [T,F,T,T,F]
]


@pytest.mark.skipif(is_before_spark_340(), reason="from_protobuf is Spark 3.4.0+")
@pytest.mark.parametrize("field_name,field_key,packed_data,id_val", 
                         _packed_repeated_test_configs, ids=idfn)
@ignore_order(local=True)
def test_from_protobuf_packed_repeated(spark_tmp_path, field_name, field_key, 
                                        packed_data, id_val):
    """
    Parametrized test for packed repeated fields (int, double, bool).
    """
    from_protobuf = _try_import_from_protobuf()
    if from_protobuf is None:
        pytest.skip("from_protobuf not available")
    if not with_cpu_session(_spark_protobuf_jvm_available):
        pytest.skip("spark-protobuf JVM not available")

    desc_path = spark_tmp_path + "/packed.desc"
    message_name = "test.WithPackedRepeated"

    desc_bytes = with_cpu_session(_build_packed_repeated_descriptor_set_bytes)
    with_cpu_session(lambda spark: _write_bytes_to_hadoop_path(
        spark, desc_path, desc_bytes))

    # Build test data: id + packed field
    test_data = bytes([0x08, id_val, field_key, len(packed_data)]) + packed_data

    def run_on_spark(spark):
        df = spark.createDataFrame(
            [(test_data,), (None,)],
            schema="bin binary",
        )
        sig = inspect.signature(from_protobuf)
        if "binaryDescriptorSet" in sig.parameters:
            decoded = from_protobuf(
                f.col("bin"), message_name,
                binaryDescriptorSet=bytearray(desc_bytes))
        else:
            decoded = from_protobuf(f.col("bin"), message_name, desc_path)
        
        return df.select(
            decoded.getField("id").alias("id"),
            decoded.getField(field_name).alias(field_name),
        )

    assert_gpu_and_cpu_are_equal_collect(run_on_spark)


# =============================================================================
# More Repeated Field Data Types Tests
# =============================================================================

def _build_repeated_all_types_descriptor_set_bytes(spark):
    """
    Build a FileDescriptorSet for message with various repeated field types:
      message WithRepeatedAllTypes {
        optional int32 id = 1;
        repeated int64 long_values = 2;
        repeated float float_values = 3;
        repeated double double_values = 4;
        repeated bool bool_values = 5;
        repeated bytes bytes_values = 6;
      }
    """
    jvm = spark.sparkContext._jvm
    D = jvm.com.google.protobuf.DescriptorProtos

    fd = D.FileDescriptorProto.newBuilder() \
        .setName("repeated_all.proto") \
        .setPackage("test")
    try:
        fd = fd.setSyntax("proto2")
    except Exception:
        pass

    label_rep = D.FieldDescriptorProto.Label.LABEL_REPEATED
    label_opt = D.FieldDescriptorProto.Label.LABEL_OPTIONAL

    msg = D.DescriptorProto.newBuilder().setName("WithRepeatedAllTypes")
    msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("id")
            .setNumber(1)
            .setLabel(label_opt)
            .setType(D.FieldDescriptorProto.Type.TYPE_INT32)
            .build()
    )
    msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("long_values")
            .setNumber(2)
            .setLabel(label_rep)
            .setType(D.FieldDescriptorProto.Type.TYPE_INT64)
            .build()
    )
    msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("float_values")
            .setNumber(3)
            .setLabel(label_rep)
            .setType(D.FieldDescriptorProto.Type.TYPE_FLOAT)
            .build()
    )
    msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("double_values")
            .setNumber(4)
            .setLabel(label_rep)
            .setType(D.FieldDescriptorProto.Type.TYPE_DOUBLE)
            .build()
    )
    msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("bool_values")
            .setNumber(5)
            .setLabel(label_rep)
            .setType(D.FieldDescriptorProto.Type.TYPE_BOOL)
            .build()
    )
    msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("bytes_values")
            .setNumber(6)
            .setLabel(label_rep)
            .setType(D.FieldDescriptorProto.Type.TYPE_BYTES)
            .build()
    )
    fd.addMessageType(msg.build())

    fds = D.FileDescriptorSet.newBuilder().addFile(fd.build()).build()
    return bytes(fds.toByteArray())


# Repeated all types test configurations: (field_name, test_data_bytes, id_value)
_repeated_all_types_test_configs = [
    ("long_values", bytes([
        0x08, 0x01,  # id = 1
        0x10, 0x64,  # long_values[0] = 100
        0x10, 0xC8, 0x01,  # long_values[1] = 200
        0x10, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F,  # max int64
    ])),
    ("float_values", bytes([0x08, 0x02]) +  # id = 2
        bytes([0x1D]) + struct.pack("<f", 1.5) +
        bytes([0x1D]) + struct.pack("<f", -2.5) +
        bytes([0x1D]) + struct.pack("<f", 0.0)),
    ("bytes_values", bytes([
        0x08, 0x03,  # id = 3
        0x32, 0x02, 0x01, 0x02,  # bytes_values[0] = b"\x01\x02"
        0x32, 0x03, 0x03, 0x04, 0x05,  # bytes_values[1]
        0x32, 0x00,  # bytes_values[2] = b"" (empty)
    ])),
]


@pytest.mark.skipif(is_before_spark_340(), reason="from_protobuf is Spark 3.4.0+")
@pytest.mark.parametrize("field_name,test_data", _repeated_all_types_test_configs, ids=idfn)
@ignore_order(local=True)
def test_from_protobuf_repeated_all_types(spark_tmp_path, field_name, test_data):
    """Parametrized test for repeated fields of various types (int64, float, bytes)."""
    from_protobuf = _try_import_from_protobuf()
    if from_protobuf is None:
        pytest.skip("from_protobuf not available")
    if not with_cpu_session(_spark_protobuf_jvm_available):
        pytest.skip("spark-protobuf JVM not available")

    desc_path = spark_tmp_path + "/repeated_all.desc"
    message_name = "test.WithRepeatedAllTypes"

    desc_bytes = with_cpu_session(_build_repeated_all_types_descriptor_set_bytes)
    with_cpu_session(lambda spark: _write_bytes_to_hadoop_path(
        spark, desc_path, desc_bytes))

    def run_on_spark(spark):
        df = spark.createDataFrame(
            [(test_data,), (None,)],
            schema="bin binary",
        )
        sig = inspect.signature(from_protobuf)
        if "binaryDescriptorSet" in sig.parameters:
            decoded = from_protobuf(
                f.col("bin"), message_name,
                binaryDescriptorSet=bytearray(desc_bytes))
        else:
            decoded = from_protobuf(f.col("bin"), message_name, desc_path)
        
        return df.select(
            decoded.getField("id").alias("id"),
            decoded.getField(field_name).alias(field_name),
        )

    assert_gpu_and_cpu_are_equal_collect(run_on_spark)


# =============================================================================
# Large Array Tests
# =============================================================================

@pytest.mark.skipif(is_before_spark_340(), reason="from_protobuf is Spark 3.4.0+")
@ignore_order(local=True)
def test_from_protobuf_large_repeated_array(spark_tmp_path):
    """
    Test decoding large repeated field (1000+ elements).
    Stress test for array handling.
    """
    from_protobuf = _try_import_from_protobuf()
    if from_protobuf is None:
        pytest.skip("from_protobuf not available")
    if not with_cpu_session(_spark_protobuf_jvm_available):
        pytest.skip("spark-protobuf JVM not available")

    desc_path = spark_tmp_path + "/repeated_int.desc"
    message_name = "test.WithRepeatedInt"

    desc_bytes = with_cpu_session(_build_repeated_int_descriptor_set_bytes)
    with_cpu_session(lambda spark: _write_bytes_to_hadoop_path(
        spark, desc_path, desc_bytes))

    # Build protobuf with 1000 elements in repeated field
    def build_large_array_protobuf():
        data = bytearray([0x08, 0x01])  # id = 1
        for i in range(1000):
            # Encode each value as unpacked repeated field
            # Field 2, wire type 0 (varint) = 0x10
            data.append(0x10)
            # Encode varint for value i
            val = i
            while val >= 128:
                data.append((val & 0x7F) | 0x80)
                val >>= 7
            data.append(val)
        return bytes(data)
    
    large_data = build_large_array_protobuf()

    def run_on_spark(spark):
        df = spark.createDataFrame(
            [(large_data,), (None,)],
            schema="bin binary",
        )
        sig = inspect.signature(from_protobuf)
        if "binaryDescriptorSet" in sig.parameters:
            decoded = from_protobuf(
                f.col("bin"), message_name,
                binaryDescriptorSet=bytearray(desc_bytes))
        else:
            decoded = from_protobuf(f.col("bin"), message_name, desc_path)
        
        return df.select(
            decoded.getField("id").alias("id"),
            f.size(decoded.getField("values")).alias("array_size"),
        )

    assert_gpu_and_cpu_are_equal_collect(run_on_spark)


# =============================================================================
# Signed Integer Encoding Tests (sint32/sint64 with zigzag)
# =============================================================================

def _build_signed_int_descriptor_set_bytes(spark):
    """
    Build a FileDescriptorSet for message with signed integer types:
      message WithSignedInts {
        optional sint32 si32 = 1;  // zigzag encoding
        optional sint64 si64 = 2;  // zigzag encoding
        optional sfixed32 sf32 = 3;  // fixed 4-byte
        optional sfixed64 sf64 = 4;  // fixed 8-byte
      }
    """
    jvm = spark.sparkContext._jvm
    D = jvm.com.google.protobuf.DescriptorProtos

    fd = D.FileDescriptorProto.newBuilder() \
        .setName("signed_int.proto") \
        .setPackage("test")
    try:
        fd = fd.setSyntax("proto2")
    except Exception:
        pass

    label_opt = D.FieldDescriptorProto.Label.LABEL_OPTIONAL

    msg = D.DescriptorProto.newBuilder().setName("WithSignedInts")
    msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("si32")
            .setNumber(1)
            .setLabel(label_opt)
            .setType(D.FieldDescriptorProto.Type.TYPE_SINT32)
            .build()
    )
    msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("si64")
            .setNumber(2)
            .setLabel(label_opt)
            .setType(D.FieldDescriptorProto.Type.TYPE_SINT64)
            .build()
    )
    msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("sf32")
            .setNumber(3)
            .setLabel(label_opt)
            .setType(D.FieldDescriptorProto.Type.TYPE_SFIXED32)
            .build()
    )
    msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("sf64")
            .setNumber(4)
            .setLabel(label_opt)
            .setType(D.FieldDescriptorProto.Type.TYPE_SFIXED64)
            .build()
    )
    fd.addMessageType(msg.build())

    fds = D.FileDescriptorSet.newBuilder().addFile(fd.build()).build()
    return bytes(fds.toByteArray())


@pytest.mark.skipif(is_before_spark_340(), reason="from_protobuf is Spark 3.4.0+")
@ignore_order(local=True)
def test_from_protobuf_signed_integers(spark_tmp_path):
    """
    Test decoding signed integer types with zigzag encoding.
    Zigzag: -1 -> 1, 1 -> 2, -2 -> 3, 2 -> 4, etc.
    """
    from_protobuf = _try_import_from_protobuf()
    if from_protobuf is None:
        pytest.skip("from_protobuf not available")
    if not with_cpu_session(_spark_protobuf_jvm_available):
        pytest.skip("spark-protobuf JVM not available")

    desc_path = spark_tmp_path + "/signed.desc"
    message_name = "test.WithSignedInts"

    desc_bytes = with_cpu_session(_build_signed_int_descriptor_set_bytes)
    with_cpu_session(lambda spark: _write_bytes_to_hadoop_path(
        spark, desc_path, desc_bytes))

    import struct
    # si32 = -1 (zigzag: 1), si64 = -100 (zigzag: 199)
    # sf32 = -12345, sf64 = -9876543210
    test_data_negative = bytes([
        0x08, 0x01,  # si32 = -1 (zigzag encoded as 1)
        0x10, 0xC7, 0x01,  # si64 = -100 (zigzag encoded as 199)
        0x1D
    ]) + struct.pack("<i", -12345) + bytes([0x21]) + struct.pack("<q", -9876543210)
    
    # Positive values
    test_data_positive = bytes([
        0x08, 0x14,  # si32 = 10 (zigzag encoded as 20)
        0x10, 0xC8, 0x01,  # si64 = 100 (zigzag encoded as 200)
        0x1D
    ]) + struct.pack("<i", 12345) + bytes([0x21]) + struct.pack("<q", 9876543210)

    def run_on_spark(spark):
        df = spark.createDataFrame(
            [(test_data_negative,), (test_data_positive,), (None,)],
            schema="bin binary",
        )
        sig = inspect.signature(from_protobuf)
        if "binaryDescriptorSet" in sig.parameters:
            decoded = from_protobuf(
                f.col("bin"), message_name,
                binaryDescriptorSet=bytearray(desc_bytes))
        else:
            decoded = from_protobuf(f.col("bin"), message_name, desc_path)
        
        return df.select(
            decoded.getField("si32").alias("si32"),
            decoded.getField("si64").alias("si64"),
            decoded.getField("sf32").alias("sf32"),
            decoded.getField("sf64").alias("sf64"),
        )

    assert_gpu_and_cpu_are_equal_collect(run_on_spark)


# =============================================================================
# Fixed Width Integer Tests
# =============================================================================

def _build_fixed_int_descriptor_set_bytes(spark):
    """
    Build a FileDescriptorSet for message with fixed-width integer types:
      message WithFixedInts {
        optional fixed32 fx32 = 1;
        optional fixed64 fx64 = 2;
      }
    """
    jvm = spark.sparkContext._jvm
    D = jvm.com.google.protobuf.DescriptorProtos

    fd = D.FileDescriptorProto.newBuilder() \
        .setName("fixed_int.proto") \
        .setPackage("test")
    try:
        fd = fd.setSyntax("proto2")
    except Exception:
        pass

    label_opt = D.FieldDescriptorProto.Label.LABEL_OPTIONAL

    msg = D.DescriptorProto.newBuilder().setName("WithFixedInts")
    msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("fx32")
            .setNumber(1)
            .setLabel(label_opt)
            .setType(D.FieldDescriptorProto.Type.TYPE_FIXED32)
            .build()
    )
    msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("fx64")
            .setNumber(2)
            .setLabel(label_opt)
            .setType(D.FieldDescriptorProto.Type.TYPE_FIXED64)
            .build()
    )
    fd.addMessageType(msg.build())

    fds = D.FileDescriptorSet.newBuilder().addFile(fd.build()).build()
    return bytes(fds.toByteArray())


@pytest.mark.skipif(is_before_spark_340(), reason="from_protobuf is Spark 3.4.0+")
@ignore_order(local=True)
def test_from_protobuf_fixed_integers(spark_tmp_path):
    """
    Test decoding fixed-width unsigned integer types.
    """
    from_protobuf = _try_import_from_protobuf()
    if from_protobuf is None:
        pytest.skip("from_protobuf not available")
    if not with_cpu_session(_spark_protobuf_jvm_available):
        pytest.skip("spark-protobuf JVM not available")

    desc_path = spark_tmp_path + "/fixed.desc"
    message_name = "test.WithFixedInts"

    desc_bytes = with_cpu_session(_build_fixed_int_descriptor_set_bytes)
    with_cpu_session(lambda spark: _write_bytes_to_hadoop_path(
        spark, desc_path, desc_bytes))

    import struct
    # fx32 = 0xDEADBEEF, fx64 = 0x123456789ABCDEF0
    test_data = bytes([0x0D]) + struct.pack("<I", 0xDEADBEEF) + \
                bytes([0x11]) + struct.pack("<Q", 0x123456789ABCDEF0)

    def run_on_spark(spark):
        df = spark.createDataFrame(
            [(test_data,), (None,)],
            schema="bin binary",
        )
        sig = inspect.signature(from_protobuf)
        if "binaryDescriptorSet" in sig.parameters:
            decoded = from_protobuf(
                f.col("bin"), message_name,
                binaryDescriptorSet=bytearray(desc_bytes))
        else:
            decoded = from_protobuf(f.col("bin"), message_name, desc_path)
        
        return df.select(
            decoded.getField("fx32").alias("fx32"),
            decoded.getField("fx64").alias("fx64"),
        )

    assert_gpu_and_cpu_are_equal_collect(run_on_spark)


# =============================================================================
# Random Data Generation Tests for Nested/Repeated Fields
# =============================================================================

from data_gen import ProtobufNestedMessageRowGen, ProtobufRepeatedFieldRowGen


@pytest.mark.skipif(is_before_spark_340(), reason="from_protobuf is Spark 3.4.0+")
@ignore_order(local=True)
def test_from_protobuf_random_nested_message(spark_tmp_path):
    """
    Test from_protobuf with randomly generated nested message data.
    Uses ProtobufNestedMessageRowGen for random data generation.
    """
    from_protobuf = _try_import_from_protobuf()
    if from_protobuf is None:
        pytest.skip("from_protobuf not available")
    if not with_cpu_session(_spark_protobuf_jvm_available):
        pytest.skip("spark-protobuf JVM not available")

    desc_path = spark_tmp_path + "/nested.desc"
    message_name = "test.WithNested"

    desc_bytes = with_cpu_session(_build_nested_descriptor_set_bytes)
    with_cpu_session(lambda spark: _write_bytes_to_hadoop_path(
        spark, desc_path, desc_bytes))

    # Generate random nested message data
    data_gen = ProtobufNestedMessageRowGen(
        scalar_fields=[
            ("simple_int", 1, IntegerGen()),
            ("simple_str", 2, StringGen(pattern="[a-z]{0,20}")),
            ("simple_long", 4, LongGen()),
        ],
        nested_fields=[
            ("nested_msg", 3, [
                ("x", 1, IntegerGen()),
            ])
        ]
    )

    def run_on_spark(spark):
        df = gen_df(spark, data_gen, length=100)
        
        sig = inspect.signature(from_protobuf)
        if "binaryDescriptorSet" in sig.parameters:
            decoded = from_protobuf(
                f.col("bin"), message_name,
                binaryDescriptorSet=bytearray(desc_bytes))
        else:
            decoded = from_protobuf(f.col("bin"), message_name, desc_path)
        
        return df.select(
            decoded.getField("simple_int").alias("simple_int"),
            decoded.getField("simple_str").alias("simple_str"),
            decoded.getField("nested_msg").alias("nested_msg"),
            decoded.getField("simple_long").alias("simple_long"),
        )

    assert_gpu_and_cpu_are_equal_collect(run_on_spark)


@pytest.mark.skipif(is_before_spark_340(), reason="from_protobuf is Spark 3.4.0+")
@ignore_order(local=True)
def test_from_protobuf_random_repeated_int(spark_tmp_path):
    """Test from_protobuf with randomly generated repeated int field data."""
    from_protobuf = _try_import_from_protobuf()
    if from_protobuf is None:
        pytest.skip("from_protobuf not available")
    if not with_cpu_session(_spark_protobuf_jvm_available):
        pytest.skip("spark-protobuf JVM not available")

    desc_path = spark_tmp_path + "/repeated_int.desc"
    message_name = "test.WithRepeatedInt"

    desc_bytes = with_cpu_session(_build_repeated_int_descriptor_set_bytes)
    with_cpu_session(lambda spark: _write_bytes_to_hadoop_path(
        spark, desc_path, desc_bytes))

    data_gen = ProtobufRepeatedFieldRowGen(
        scalar_fields=[("id", 1, IntegerGen())],
        repeated_fields=[("values", 2, IntegerGen(), False)],
        min_array_len=0,
        max_array_len=10
    )

    def run_on_spark(spark):
        df = gen_df(spark, data_gen, length=100)
        
        sig = inspect.signature(from_protobuf)
        if "binaryDescriptorSet" in sig.parameters:
            decoded = from_protobuf(
                f.col("bin"), message_name,
                binaryDescriptorSet=bytearray(desc_bytes))
        else:
            decoded = from_protobuf(f.col("bin"), message_name, desc_path)
        
        return df.select(
            decoded.getField("id").alias("id"),
            decoded.getField("values").alias("values"),
        )

    assert_gpu_and_cpu_are_equal_collect(run_on_spark)


@pytest.mark.skipif(is_before_spark_340(), reason="from_protobuf is Spark 3.4.0+")
@ignore_order(local=True)
def test_from_protobuf_random_repeated_string(spark_tmp_path):
    """Test from_protobuf with randomly generated repeated string field data."""
    from_protobuf = _try_import_from_protobuf()
    if from_protobuf is None:
        pytest.skip("from_protobuf not available")
    if not with_cpu_session(_spark_protobuf_jvm_available):
        pytest.skip("spark-protobuf JVM not available")

    desc_path = spark_tmp_path + "/repeated_string.desc"
    message_name = "test.WithRepeatedString"

    desc_bytes = with_cpu_session(_build_repeated_string_descriptor_set_bytes)
    with_cpu_session(lambda spark: _write_bytes_to_hadoop_path(
        spark, desc_path, desc_bytes))

    data_gen = ProtobufRepeatedFieldRowGen(
        scalar_fields=[("name", 1, StringGen(pattern="[a-z]{1,10}"))],
        repeated_fields=[("tags", 2, StringGen(pattern="[a-z]{0,5}"), False)],
        min_array_len=0,
        max_array_len=5
    )

    def run_on_spark(spark):
        df = gen_df(spark, data_gen, length=100)
        
        sig = inspect.signature(from_protobuf)
        if "binaryDescriptorSet" in sig.parameters:
            decoded = from_protobuf(
                f.col("bin"), message_name,
                binaryDescriptorSet=bytearray(desc_bytes))
        else:
            decoded = from_protobuf(f.col("bin"), message_name, desc_path)
        
        return df.select(
            decoded.getField("name").alias("name"),
            decoded.getField("tags").alias("tags"),
        )

    assert_gpu_and_cpu_are_equal_collect(run_on_spark)


# =============================================================================
# Nested Schema Projection Tests
# =============================================================================
# These tests verify that when only specific sub-fields of a nested message
# are accessed (e.g. decoded.detail.a instead of decoded.detail), the GPU decoder
# correctly prunes unneeded children while still producing correct results.

def _build_schema_projection_descriptor_set_bytes(spark):
    """
    Build a FileDescriptorSet for nested schema projection testing:
      message Detail {
        optional int32  a = 1;
        optional int32  b = 2;
        optional string c = 3;
      }
      message SchemaProj {
        optional int32  id     = 1;
        optional string name   = 2;
        optional Detail detail = 3;
        repeated Detail items  = 4;
      }
    The Detail message has 3 fields so we can test pruning subsets.
    """
    jvm = spark.sparkContext._jvm
    D = jvm.com.google.protobuf.DescriptorProtos

    fd = D.FileDescriptorProto.newBuilder() \
        .setName("schema_proj.proto") \
        .setPackage("test")
    try:
        fd = fd.setSyntax("proto2")
    except Exception:
        pass

    label_opt = D.FieldDescriptorProto.Label.LABEL_OPTIONAL
    label_rep = D.FieldDescriptorProto.Label.LABEL_REPEATED

    # Detail message: { a: int32, b: int32, c: string }
    detail_msg = D.DescriptorProto.newBuilder().setName("Detail")
    detail_msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("a").setNumber(1).setLabel(label_opt)
            .setType(D.FieldDescriptorProto.Type.TYPE_INT32).build())
    detail_msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("b").setNumber(2).setLabel(label_opt)
            .setType(D.FieldDescriptorProto.Type.TYPE_INT32).build())
    detail_msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("c").setNumber(3).setLabel(label_opt)
            .setType(D.FieldDescriptorProto.Type.TYPE_STRING).build())
    fd.addMessageType(detail_msg.build())

    # SchemaProj message: { id, name, detail: Detail, items: repeated Detail }
    main_msg = D.DescriptorProto.newBuilder().setName("SchemaProj")
    main_msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("id").setNumber(1).setLabel(label_opt)
            .setType(D.FieldDescriptorProto.Type.TYPE_INT32).build())
    main_msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("name").setNumber(2).setLabel(label_opt)
            .setType(D.FieldDescriptorProto.Type.TYPE_STRING).build())
    main_msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("detail").setNumber(3).setLabel(label_opt)
            .setType(D.FieldDescriptorProto.Type.TYPE_MESSAGE)
            .setTypeName(".test.Detail").build())
    main_msg.addField(
        D.FieldDescriptorProto.newBuilder()
            .setName("items").setNumber(4).setLabel(label_rep)
            .setType(D.FieldDescriptorProto.Type.TYPE_MESSAGE)
            .setTypeName(".test.Detail").build())
    fd.addMessageType(main_msg.build())

    fds = D.FileDescriptorSet.newBuilder().addFile(fd.build()).build()
    return bytes(fds.toByteArray())


def _encode_varint_proj(value):
    """Encode an unsigned integer as a protobuf varint."""
    result = []
    v = value & 0xFFFFFFFF
    while v > 0x7F:
        result.append((v & 0x7F) | 0x80)
        v >>= 7
    result.append(v & 0x7F)
    return bytes(result)


def _schema_proj_detail_bytes(a, b, c):
    """Encode a Detail message: {a: int32, b: int32, c: string}."""
    parts = []
    parts.append(bytes([0x08]) + _encode_varint_proj(a))
    parts.append(bytes([0x10]) + _encode_varint_proj(b))
    c_bytes = c.encode("utf-8")
    parts.append(bytes([0x1A]) + _encode_varint_proj(len(c_bytes)) + c_bytes)
    return b"".join(parts)


def _schema_proj_message_bytes(id_val, name_val, detail_a, detail_b, detail_c, items):
    """Encode a SchemaProj message."""
    parts = []
    parts.append(bytes([0x08]) + _encode_varint_proj(id_val))
    name_bytes = name_val.encode("utf-8")
    parts.append(bytes([0x12]) + _encode_varint_proj(len(name_bytes)) + name_bytes)
    detail_bytes = _schema_proj_detail_bytes(detail_a, detail_b, detail_c)
    parts.append(bytes([0x1A]) + _encode_varint_proj(len(detail_bytes)) + detail_bytes)
    for (a, b, c) in items:
        item_bytes = _schema_proj_detail_bytes(a, b, c)
        parts.append(bytes([0x22]) + _encode_varint_proj(len(item_bytes)) + item_bytes)
    return b"".join(parts)


_schema_proj_test_data = [
    _schema_proj_message_bytes(1, "alice", 10, 20, "d1",
                               [(100, 200, "i1"), (101, 201, "i2")]),
    _schema_proj_message_bytes(2, "bob", 30, 40, "d2",
                               [(300, 400, "i3")]),
    _schema_proj_message_bytes(3, "carol", 50, 60, "d3", []),
]


def _setup_schema_proj(spark_tmp_path):
    """Common setup: build descriptor and return (desc_path, message_name, desc_bytes)."""
    desc_path = spark_tmp_path + "/schema_proj.desc"
    message_name = "test.SchemaProj"
    desc_bytes = with_cpu_session(_build_schema_projection_descriptor_set_bytes)
    with_cpu_session(lambda spark: _write_bytes_to_hadoop_path(
        spark, desc_path, desc_bytes))
    return desc_path, message_name, desc_bytes


def _decode_schema_proj(df, from_protobuf_fn, desc_path, message_name, desc_bytes):
    """Apply from_protobuf to a binary DataFrame."""
    sig = inspect.signature(from_protobuf_fn)
    if "binaryDescriptorSet" in sig.parameters:
        return from_protobuf_fn(
            f.col("bin"), message_name,
            binaryDescriptorSet=bytearray(desc_bytes))
    else:
        return from_protobuf_fn(f.col("bin"), message_name, desc_path)


@pytest.mark.skipif(is_before_spark_340(), reason="from_protobuf is Spark 3.4.0+")
@ignore_order(local=True)
def test_from_protobuf_schema_proj_nested_single_field(spark_tmp_path):
    """Select only detail.a from nested struct (prune b, c)."""
    from_protobuf_fn = _try_import_from_protobuf()
    if from_protobuf_fn is None:
        pytest.skip("from_protobuf not available")
    if not with_cpu_session(_spark_protobuf_jvm_available):
        pytest.skip("spark-protobuf JVM not available")

    desc_path, message_name, desc_bytes = _setup_schema_proj(spark_tmp_path)

    def run_on_spark(spark):
        df = spark.createDataFrame(
            [(d,) for d in _schema_proj_test_data], schema="bin binary")
        decoded = _decode_schema_proj(
            df, from_protobuf_fn, desc_path, message_name, desc_bytes)
        return df.select(
            decoded.getField("id").alias("id"),
            decoded.getField("detail").getField("a").alias("detail_a"))

    assert_gpu_and_cpu_are_equal_collect(run_on_spark)


@pytest.mark.skipif(is_before_spark_340(), reason="from_protobuf is Spark 3.4.0+")
@ignore_order(local=True)
def test_from_protobuf_schema_proj_nested_two_fields(spark_tmp_path):
    """Select detail.a and detail.c (prune b)."""
    from_protobuf_fn = _try_import_from_protobuf()
    if from_protobuf_fn is None:
        pytest.skip("from_protobuf not available")
    if not with_cpu_session(_spark_protobuf_jvm_available):
        pytest.skip("spark-protobuf JVM not available")

    desc_path, message_name, desc_bytes = _setup_schema_proj(spark_tmp_path)

    def run_on_spark(spark):
        df = spark.createDataFrame(
            [(d,) for d in _schema_proj_test_data], schema="bin binary")
        decoded = _decode_schema_proj(
            df, from_protobuf_fn, desc_path, message_name, desc_bytes)
        return df.select(
            decoded.getField("detail").getField("a").alias("detail_a"),
            decoded.getField("detail").getField("c").alias("detail_c"))

    assert_gpu_and_cpu_are_equal_collect(run_on_spark)


@pytest.mark.skipif(is_before_spark_340(), reason="from_protobuf is Spark 3.4.0+")
@ignore_order(local=True)
def test_from_protobuf_schema_proj_whole_struct_no_pruning(spark_tmp_path):
    """Selecting whole nested struct should NOT prune children."""
    from_protobuf_fn = _try_import_from_protobuf()
    if from_protobuf_fn is None:
        pytest.skip("from_protobuf not available")
    if not with_cpu_session(_spark_protobuf_jvm_available):
        pytest.skip("spark-protobuf JVM not available")

    desc_path, message_name, desc_bytes = _setup_schema_proj(spark_tmp_path)

    def run_on_spark(spark):
        df = spark.createDataFrame(
            [(d,) for d in _schema_proj_test_data], schema="bin binary")
        decoded = _decode_schema_proj(
            df, from_protobuf_fn, desc_path, message_name, desc_bytes)
        return df.select(
            decoded.getField("id").alias("id"),
            decoded.getField("detail").alias("detail"))

    assert_gpu_and_cpu_are_equal_collect(run_on_spark)


@pytest.mark.skipif(is_before_spark_340(), reason="from_protobuf is Spark 3.4.0+")
@ignore_order(local=True)
def test_from_protobuf_schema_proj_whole_and_subfield(spark_tmp_path):
    """Selecting whole struct AND a sub-field: whole struct wins, no pruning."""
    from_protobuf_fn = _try_import_from_protobuf()
    if from_protobuf_fn is None:
        pytest.skip("from_protobuf not available")
    if not with_cpu_session(_spark_protobuf_jvm_available):
        pytest.skip("spark-protobuf JVM not available")

    desc_path, message_name, desc_bytes = _setup_schema_proj(spark_tmp_path)

    def run_on_spark(spark):
        df = spark.createDataFrame(
            [(d,) for d in _schema_proj_test_data], schema="bin binary")
        decoded = _decode_schema_proj(
            df, from_protobuf_fn, desc_path, message_name, desc_bytes)
        return df.select(
            decoded.getField("detail").alias("detail"),
            decoded.getField("detail").getField("a").alias("detail_a"))

    assert_gpu_and_cpu_are_equal_collect(run_on_spark)


@pytest.mark.skipif(is_before_spark_340(), reason="from_protobuf is Spark 3.4.0+")
@ignore_order(local=True)
def test_from_protobuf_schema_proj_scalar_plus_nested(spark_tmp_path):
    """Top-level scalar + nested sub-field: prune both top-level and nested."""
    from_protobuf_fn = _try_import_from_protobuf()
    if from_protobuf_fn is None:
        pytest.skip("from_protobuf not available")
    if not with_cpu_session(_spark_protobuf_jvm_available):
        pytest.skip("spark-protobuf JVM not available")

    desc_path, message_name, desc_bytes = _setup_schema_proj(spark_tmp_path)

    def run_on_spark(spark):
        df = spark.createDataFrame(
            [(d,) for d in _schema_proj_test_data], schema="bin binary")
        decoded = _decode_schema_proj(
            df, from_protobuf_fn, desc_path, message_name, desc_bytes)
        return df.select(
            decoded.getField("id").alias("id"),
            decoded.getField("name").alias("name"),
            decoded.getField("detail").getField("a").alias("detail_a"))

    assert_gpu_and_cpu_are_equal_collect(run_on_spark)
