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

from asserts import assert_gpu_fallback_collect
from marks import allow_non_gpu
from protobuf_data_gen import call_protobuf_function
from spark_session import is_before_spark_340, is_spark_protobuf_available, with_cpu_session
import pyspark.sql.functions as f

pytestmark = pytest.mark.skipif(
    not is_spark_protobuf_available(), reason="from_protobuf is unavailable")


@pytest.fixture(scope="module")
def from_protobuf_fn():
    from pyspark.sql.protobuf.functions import from_protobuf
    return from_protobuf


def _encode_varint(value):
    out = bytearray()
    value &= 0xFFFFFFFFFFFFFFFF
    while True:
        bits = value & 0x7F
        value >>= 7
        if value:
            out.append(bits | 0x80)
        else:
            out.append(bits)
            return bytes(out)


def _encode_simple_message(i32_value, s_value):
    buf = bytearray()
    buf += _encode_varint((1 << 3) | 0)  # field 1, VARINT
    buf += _encode_varint(i32_value)
    s_bytes = s_value.encode("utf-8")
    buf += _encode_varint((2 << 3) | 2)  # field 2, LENGTH-DELIMITED
    buf += _encode_varint(len(s_bytes))
    buf += s_bytes
    return bytes(buf)


# Avoid depending on whichever unshaded protobuf runtime the Spark driver provides.
_simple_desc_bytes = bytes.fromhex(
    "0a360a0c73696d706c652e70726f746f12047465737422200a0653696d706c65"
    "120b0a0369333218012001280512090a0173180220012809")


@pytest.fixture
def simple_desc(local_tmp_path):
    # from_protobuf reads descFilePath via java.io.File on the driver.
    desc_path = local_tmp_path + "/simple.desc"
    with open(desc_path, "wb") as fp:
        fp.write(_simple_desc_bytes)
    return desc_path, _simple_desc_bytes


@pytest.mark.skipif(is_before_spark_340(), reason="descriptor compatibility shims start at Spark 3.4")
def test_protobuf_descriptor_compat(local_tmp_path, from_protobuf_fn):
    # proto2 test.Compat: optional int32 id = 1 [default = 7]; Status FIRST = ALIAS = 0;
    # optional Status first = 2 [default = FIRST]; optional Status alias = 3 [default = ALIAS].
    desc_bytes = bytes.fromhex(
        "0ab0010a0c636f6d7061742e70726f746f1204746573742291010a06436f6d706174"
        "120d0a0269641801200128053a013712290a05666972737418022001280e32132e74"
        "6573742e436f6d7061742e5374617475733a05464952535412290a05616c696173"
        "18032001280e32132e746573742e436f6d7061742e5374617475733a05414c494153"
        "22220a0653746174757312090a054649525354100012090a05414c49415310001a02"
        "1001620670726f746f32")
    desc_path = local_tmp_path + "/compat.desc"
    with open(desc_path, "wb") as fp:
        fp.write(desc_bytes)

    def check(spark):
        decoded = call_protobuf_function(
            from_protobuf_fn, f.lit(bytearray()), "test.Compat", desc_path, desc_bytes,
            options={"mode": "FAILFAST"})
        expr = spark.range(1).select(decoded.alias("d"))._jdf.queryExecution() \
            .analyzed().projectList().apply(0).child()
        cls = spark._jvm.com.nvidia.spark.rapids.ShimReflectionUtils.loadClass(
            "com.nvidia.spark.rapids.shims.SparkProtobufCompat$")
        compat = cls.getField("MODULE$").get(None)
        extracted = compat.extractExprInfo(expr)
        assert extracted.isRight(), str(extracted)
        info = extracted.toOption().get()
        assert info.messageName() == "test.Compat"
        assert info.options().apply("mode") == "FAILFAST"
        resolved = compat.resolveMessageDescriptor(info)
        assert resolved.isRight(), str(resolved)
        descriptor = resolved.toOption().get()
        assert descriptor.syntax() == "PROTO2"
        assert descriptor.findField("missing").isEmpty()
        for name, number, proto_type, default in [
                ("id", 1, "INT32", 7), ("first", 2, "ENUM", "FIRST"),
                ("alias", 3, "ENUM", "ALIAS")]:
            field = descriptor.findField(name).get()
            assert (field.name(), field.fieldNumber(), field.protoTypeName()) == \
                (name, number, proto_type)
            assert not field.isRepeated() and not field.isRequired() and not field.isInOneof()
            result = field.defaultValueResult()
            assert result.isRight(), str(result)
            value = result.toOption().get().get()
            if proto_type == "ENUM":
                assert (value.number(), value.name()) == (0, default)
                values = field.enumMetadata().get().values()
                assert [(values.apply(i).number(), values.apply(i).name())
                        for i in range(values.size())] == [(0, "FIRST"), (0, "ALIAS")]
            else:
                assert value.value() == default
        missing = info.copy("test.Missing", info.descriptorSource(), info.options())
        assert compat.resolveMessageDescriptor(missing).isLeft()

    # This exercises the CPU-side metadata API, not execution of a GPU expression.
    with_cpu_session(check)


_smoke_rows = [(1, "a"), (-2, "bb"), (0, ""), (12345, "hello")]


def _make_smoke_df(spark):
    encoded = [(_encode_simple_message(i, s),) for (i, s) in _smoke_rows]
    return spark.createDataFrame(encoded, ["bin"])


@allow_non_gpu("ProjectExec", "ProtobufDataToCatalyst")
def test_from_protobuf_smoke_path_api(simple_desc, from_protobuf_fn):
    desc_path, _ = simple_desc

    def run(spark):
        return _make_smoke_df(spark).select(
            from_protobuf_fn(f.col("bin"), "test.Simple", desc_path).alias("d"))

    assert_gpu_fallback_collect(run, "ProtobufDataToCatalyst")


@allow_non_gpu("ProjectExec", "ProtobufDataToCatalyst")
def test_from_protobuf_smoke_binary_descriptor_api(simple_desc, from_protobuf_fn):
    if "binaryDescriptorSet" not in inspect.signature(from_protobuf_fn).parameters:
        pytest.skip("binaryDescriptorSet kwarg is Spark 3.5+ only")
    _, desc_bytes = simple_desc

    def run(spark):
        return _make_smoke_df(spark).select(
            from_protobuf_fn(f.col("bin"), "test.Simple",
                             binaryDescriptorSet=bytearray(desc_bytes)).alias("d"))

    assert_gpu_fallback_collect(run, "ProtobufDataToCatalyst")
