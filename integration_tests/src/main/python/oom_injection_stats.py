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

"""Pytest plugin that reports whether injected retry OOMs actually fired.

Enabled with --oom_injection_stats=<json path>. Counts come from the JVM-static
com.nvidia.spark.rapids.OomInjectionStats, so they only cover executors that share the
driver JVM (local mode).
"""

import json
import warnings

import pytest

_PROPERTY = 'oom_injection_stats'
_COUNTERS = ['armed', 'fired', 'firedNested', 'caught', 'expired', 'overwritten', 'unassociated',
             'deferred']
_per_test = {}


def _enabled(config):
    return config.getoption(_PROPERTY, default=None) is not None


def _stats_class():
    from spark_init_internal import get_spark_i_know_what_i_am_doing
    spark = get_spark_i_know_what_i_am_doing()
    if spark is None:
        return None
    master = spark.sparkContext.master
    if not (master == 'local' or master.startswith('local[')):
        warnings.warn(f'--{_PROPERTY} only sees executors in the driver JVM, master is {master}')
    return spark._jvm.com.nvidia.spark.rapids.OomInjectionStats


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    stats = _stats_class() if _enabled(item.config) else None
    if stats is not None:
        stats.setEnabled(True)
        # drop whatever earlier tests left behind, e.g. jobs still finishing asynchronously
        stats.snapshotAndReset()
    yield
    if stats is not None:
        item.user_properties.append((_PROPERTY, stats.snapshotAndReset()))


def pytest_runtest_logreport(report):
    if report.when != 'call':
        return
    for name, value in report.user_properties:
        if name == _PROPERTY:
            sites = json.loads(value)
            if sites:
                _per_test[report.nodeid] = sites


def _aggregate():
    by_site = {}
    for sites in _per_test.values():
        for site, counters in sites.items():
            agg = by_site.setdefault(site, dict.fromkeys(_COUNTERS, 0))
            for k in _COUNTERS:
                agg[k] += counters.get(k, 0)
    return by_site


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    # xdist workers forward their reports to the coordinator, which writes the summary
    if not _enabled(config) or hasattr(config, 'workerinput'):
        return
    by_site = _aggregate()
    totals = {k: sum(c[k] for c in by_site.values()) for k in _COUNTERS}
    with open(config.getoption(_PROPERTY), 'w') as f:
        json.dump({'totals': totals, 'sites': by_site, 'tests': _per_test}, f, indent=1,
                  sort_keys=True)
    terminalreporter.write_sep('=', 'OOM injection stats')
    terminalreporter.write_line(f'tests with retry blocks armed: {len(_per_test)}')
    terminalreporter.write_line(
        f'call sites armed: {sum(1 for c in by_site.values() if c["armed"])}, '
        f'fired: {sum(1 for c in by_site.values() if c["fired"])}')
    terminalreporter.write_line(', '.join(f'{k}={v}' for k, v in totals.items()))
    terminalreporter.write_line(f'details: {config.getoption(_PROPERTY)}')
