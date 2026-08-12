# Copyright 2025 The RLinf Authors.
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

import json

from rlinf.utils.resource_monitor import (
    ResourceTraceWriter,
    collect_worker_resource_snapshot,
    summarize_resource_snapshots,
)


class _FakePlatform:
    @staticmethod
    def memory_allocated():
        return 2**30

    @staticmethod
    def memory_reserved():
        return 2 * 2**30

    @staticmethod
    def mem_get_info():
        return 3 * 2**30, 4 * 2**30


def test_collect_worker_resource_snapshot():
    snapshot = collect_worker_resource_snapshot(
        worker_group="ActorGroup",
        rank=2,
        device_type="npu",
        platform=_FakePlatform(),
    )

    assert snapshot["worker_group"] == "ActorGroup"
    assert snapshot["rank"] == 2
    assert snapshot["accelerator_memory_allocated_gib"] == 1.0
    assert snapshot["accelerator_memory_reserved_gib"] == 2.0
    assert snapshot["accelerator_memory_free_gib"] == 3.0
    assert snapshot["accelerator_memory_total_gib"] == 4.0


def test_summarize_and_write_resource_snapshots(tmp_path):
    snapshots = [
        {
            "worker_group": "ActorGroup",
            "process_rss_gib": 1.0,
            "accelerator_memory_allocated_gib": 2.0,
            "accelerator_memory_reserved_gib": 3.0,
            "accelerator_memory_free_gib": None,
        },
        {
            "worker_group": "ActorGroup",
            "process_rss_gib": 3.0,
            "accelerator_memory_allocated_gib": 4.0,
            "accelerator_memory_reserved_gib": 5.0,
            "accelerator_memory_free_gib": 6.0,
        },
    ]
    metrics = summarize_resource_snapshots(snapshots)
    assert metrics["resource/ActorGroup/process_rss_gib_mean"] == 2.0
    assert metrics["resource/ActorGroup/accelerator_memory_allocated_gib_max"] == 4.0
    assert metrics["resource/ActorGroup/accelerator_memory_free_gib_mean"] == 6.0

    trace_path = tmp_path / "resource_metrics.jsonl"
    ResourceTraceWriter(str(trace_path)).write({"workers": snapshots})
    assert json.loads(trace_path.read_text(encoding="utf-8"))["workers"] == snapshots
