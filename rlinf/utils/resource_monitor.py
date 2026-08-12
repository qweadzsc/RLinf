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

"""Lightweight, backend-agnostic resource sampling for distributed workers."""

import json
import os
import time
from pathlib import Path
from typing import Any

_GIB = float(2**30)


def _read_process_rss_bytes() -> int | None:
    """Return the current process RSS from procfs, when it is available."""
    try:
        with open("/proc/self/status") as status_file:
            for line in status_file:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _platform_call(platform: Any, method_name: str, default: int | None = None):
    """Call a torch platform memory API without assuming a CUDA-only signature."""
    method = getattr(platform, method_name, None)
    if method is None:
        return default
    try:
        return method()
    except (RuntimeError, TypeError):
        return default


def collect_worker_resource_snapshot(
    *,
    worker_group: str,
    rank: int,
    device_type: str,
    platform: Any,
) -> dict[str, Any]:
    """Collect a serializable resource snapshot from one worker process.

    PyTorch allocator counters work for CUDA and torch_npu. CPU-only workers
    are also supported, with accelerator fields reported as ``None``.
    """
    allocated = _platform_call(platform, "memory_allocated")
    reserved = _platform_call(platform, "memory_reserved")
    mem_info = _platform_call(platform, "mem_get_info")
    free_bytes = total_bytes = None
    if isinstance(mem_info, tuple) and len(mem_info) == 2:
        free_bytes, total_bytes = mem_info

    rss_bytes = _read_process_rss_bytes()
    return {
        "timestamp_s": time.time(),
        "pid": os.getpid(),
        "worker_group": worker_group,
        "rank": rank,
        "device_type": device_type,
        "process_rss_gib": None if rss_bytes is None else rss_bytes / _GIB,
        "accelerator_memory_allocated_gib": (
            None if allocated is None else float(allocated) / _GIB
        ),
        "accelerator_memory_reserved_gib": (
            None if reserved is None else float(reserved) / _GIB
        ),
        "accelerator_memory_free_gib": (
            None if free_bytes is None else float(free_bytes) / _GIB
        ),
        "accelerator_memory_total_gib": (
            None if total_bytes is None else float(total_bytes) / _GIB
        ),
    }


class ResourceTraceWriter:
    """Append per-step resource samples to a JSONL trace."""

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        with self.output_path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(json.dumps(record, allow_nan=False) + "\n")


def summarize_resource_snapshots(
    snapshots: list[dict[str, Any]], prefix: str = "resource"
) -> dict[str, float]:
    """Return TensorBoard-friendly mean/max metrics for worker snapshots."""
    metrics: dict[str, float] = {}
    groups = sorted({snapshot["worker_group"] for snapshot in snapshots})
    fields = (
        "process_rss_gib",
        "accelerator_memory_allocated_gib",
        "accelerator_memory_reserved_gib",
        "accelerator_memory_free_gib",
    )
    for group in groups:
        group_snapshots = [item for item in snapshots if item["worker_group"] == group]
        group_prefix = f"{prefix}/{group}"
        for field in fields:
            values = [
                item[field] for item in group_snapshots if item[field] is not None
            ]
            if values:
                metrics[f"{group_prefix}/{field}_mean"] = sum(values) / len(values)
                metrics[f"{group_prefix}/{field}_max"] = max(values)
    return metrics
