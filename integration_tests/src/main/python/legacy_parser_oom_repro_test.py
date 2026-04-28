# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Reproducer for customer GPU OOM on legacy timestamp parser.
#
# The interesting GPU work is the projection of
# `from_unixtime(... + unix_timestamp(ts_str, fmt))` under
# `spark.sql.legacy.timeParserPolicy=LEGACY`, which routes to
# GpuToTimestamp.parseStringAsTimestampWithLegacyParserPolicy and into cuDF
# stringReplaceWithBackrefs / matches_re. cuDF allocates a per-row scratch
# buffer (~600 B/row in our test) that the spark-rapids estimator cannot see,
# so PreProjectSplitIterator does not split the batch and the cuDF call OOMs
# the pool.
#
# Input is written on the CPU into Parquet so the GPU stage only contains the
# read + project + aggregate, all of which are genuinely on the GPU. No
# @allow_non_gpu marker is required.
#
# Expected signals when the OOM hits:
#   - INFO DeviceMemoryEventHandler: Device allocation of <N> bytes failed ...
#   - Stack frames: ColumnView.matchesRe(Native Method)
#                   GpuToTimestamp$.parseStringAsTimestampWithLegacyParserPolicy
#                   GpuTieredProject.projectWithRetrySingleBatchInternal

import pytest
import pyspark.sql.functions as f

from marks import large_data_test
from spark_session import with_cpu_session, with_gpu_session


@large_data_test
@pytest.mark.parametrize("rows", [100_000_000, 200_000_000], ids=["100M", "200M"])
def test_legacy_parser_oom_repro(spark_tmp_path, rows):
    data_path = spark_tmp_path + '/LEGACY_PARSER_OOM'

    # CASE over `id` keeps Catalyst from folding ts_str into a constant on the
    # write side; without it the parquet writer can collapse the column into
    # table metadata and the read side gets a literal back, which Catalyst
    # then folds the whole unix_timestamp(...) call away from the plan.
    # Force a single huge parquet row group so the GPU reader cannot break the
    # batch up before it reaches the project. Without this, the chunked
    # reader (and per-row-group batching) split the input into ~5 M-row
    # batches and cuDF's per-row scratch never accumulates enough to OOM.
    write_conf = {
        'spark.sql.parquet.block.size': str(4 * 1024 * 1024 * 1024),
        'parquet.block.size': str(4 * 1024 * 1024 * 1024),
    }
    with_cpu_session(lambda spark: spark.range(0, rows, numPartitions=1)
        .selectExpr(
            "id as offset_long",
            "case when id >= 0 then '2024-06-15 12:34:56' "
            "else '2024-06-15 12:34:55' end as ts_str")
        .write.mode('overwrite').parquet(data_path), conf=write_conf)

    def run(spark):
        # Aggregate to a single scalar so the projected `t` column is fully
        # materialized on the GPU but we don't ship a billion rows back to
        # the driver.
        spark.read.parquet(data_path).selectExpr(
            "from_unixtime(offset_long + "
            "unix_timestamp(ts_str, 'yyyy-MM-dd HH:mm:ss')) as t"
        ).agg(f.sum(f.length('t'))).collect()

    conf = {
        "spark.sql.legacy.timeParserPolicy": "LEGACY",
        # Required to let UnixTimestamp/FromUnixTime with LEGACY format reach
        # the GPU path (parseStringAsTimestampWithLegacyParserPolicy). This is
        # the knob the customer has turned on in production.
        "spark.rapids.sql.incompatibleDateFormats.enabled": "true",
    }
    with_gpu_session(run, conf=conf)
