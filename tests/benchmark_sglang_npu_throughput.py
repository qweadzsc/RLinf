#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
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

"""Measure standalone SGLang NPU generation throughput.

This benchmark intentionally bypasses RLinf's AgentLoop, retriever, Ray
channels, and actor training. It is useful for comparing SGLang releases with
the same model, tensor parallelism, request concurrency, and generation length.
"""

import argparse
import asyncio
import json
import os
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

DEFAULT_PROMPT = (
    "Explain in one paragraph why batching improves autoregressive inference "
    "throughput."
)


def parse_args() -> argparse.Namespace:
    """Parse fixed-shape SGLang throughput benchmark options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", choices=("npu", "cuda"), default="npu")
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--num-requests", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup-requests", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--attention-backend", default="ascend")
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--max-running-requests", type=int, default=128)
    parser.add_argument(
        "--cuda-graph-max-bs",
        type=int,
        default=None,
        help=(
            "Largest batch size prepared for CUDA-graph capture. Defaults to "
            "--concurrency so startup work matches the measured workload."
        ),
    )
    parser.add_argument(
        "--disable-cuda-graph",
        action="store_true",
        help="Disable graph capture for a startup/debugging comparison.",
    )
    parser.add_argument(
        "--enable-torch-compile",
        action="store_true",
        help="Enable SGLang torch.compile; disabled by default for reproducible version tests.",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for the machine-readable benchmark result.",
    )
    args = parser.parse_args()
    if args.num_requests <= 0 or args.concurrency <= 0:
        parser.error("--num-requests and --concurrency must be positive")
    if args.warmup_requests < 0 or args.max_new_tokens <= 0:
        parser.error("--warmup-requests must be nonnegative and token count positive")
    if args.cuda_graph_max_bs is not None and args.cuda_graph_max_bs <= 0:
        parser.error("--cuda-graph-max-bs must be positive")
    if args.cuda_graph_max_bs is None:
        args.cuda_graph_max_bs = args.concurrency
    return args


def output_token_count(result: Any) -> int:
    """Return the number of generated token IDs from an SGLang response."""
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes, dict)):
        return sum(output_token_count(item) for item in result)
    if not isinstance(result, dict):
        raise TypeError(f"Unexpected SGLang result type: {type(result)!r}")
    output_ids = result.get("output_ids")
    if isinstance(output_ids, list):
        return len(output_ids)
    meta_info = result.get("meta_info")
    completion_tokens = (
        meta_info.get("completion_tokens") if isinstance(meta_info, dict) else None
    )
    if isinstance(completion_tokens, int):
        return completion_tokens
    raise KeyError(
        "SGLang response contains neither output_ids nor meta_info.completion_tokens"
    )


async def run_requests(
    engine: Any,
    *,
    prompt: str,
    request_count: int,
    concurrency: int,
    sampling_params: dict[str, Any],
) -> tuple[int, list[float]]:
    """Submit concurrent requests and return generated tokens and latencies."""
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one() -> tuple[int, float]:
        async with semaphore:
            start = time.perf_counter()
            result = await engine.async_generate(
                prompt=prompt,
                sampling_params=sampling_params,
            )
            return output_token_count(result), time.perf_counter() - start

    results = await asyncio.gather(*(run_one() for _ in range(request_count)))
    return sum(tokens for tokens, _ in results), [latency for _, latency in results]


async def run_benchmark(
    engine: Any,
    *,
    args: argparse.Namespace,
    sampling_params: dict[str, Any],
) -> tuple[int, list[float], float]:
    """Run warmup and measurement in one event loop."""
    if args.warmup_requests:
        await run_requests(
            engine,
            prompt=args.prompt,
            request_count=args.warmup_requests,
            concurrency=min(args.concurrency, args.warmup_requests),
            sampling_params=sampling_params,
        )
    start = time.perf_counter()
    generated_tokens, latencies = await run_requests(
        engine,
        prompt=args.prompt,
        request_count=args.num_requests,
        concurrency=args.concurrency,
        sampling_params=sampling_params,
    )
    return generated_tokens, latencies, time.perf_counter() - start


def percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile for a nonempty value list."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def main() -> None:
    """Launch an eight-way TP SGLang engine and print comparable throughput."""
    args = parse_args()
    os.environ["SGLANG_ENABLE_TORCH_COMPILE"] = (
        "1" if args.enable_torch_compile else "0"
    )
    if not args.enable_torch_compile:
        os.environ["TORCH_COMPILE_DISABLE"] = "1"
    print(
        "[benchmark] "
        f"device={args.device} tp_size={args.tp_size} "
        f"cuda_graph={not args.disable_cuda_graph} "
        f"torch_compile={args.enable_torch_compile}",
        flush=True,
    )
    import sglang as sgl

    sampling_params = {
        "temperature": 0.0,
        "max_new_tokens": args.max_new_tokens,
        "ignore_eos": True,
    }

    engine_kwargs = {
        "model_path": args.model_path,
        "device": args.device,
        "tp_size": args.tp_size,
        "dtype": args.dtype,
        "attention_backend": args.attention_backend,
        "mem_fraction_static": args.mem_fraction_static,
        "max_running_requests": args.max_running_requests,
        "cuda_graph_max_bs": args.cuda_graph_max_bs,
        "disable_cuda_graph": args.disable_cuda_graph,
        "enable_torch_compile": args.enable_torch_compile,
        "random_seed": args.seed,
    }
    engine = sgl.Engine(**engine_kwargs)
    try:
        generated_tokens, latencies, elapsed = asyncio.run(
            run_benchmark(engine, args=args, sampling_params=sampling_params)
        )
    finally:
        engine.shutdown()

    result = {
        "sglang_version": getattr(sgl, "__version__", "unknown"),
        "model_path": args.model_path,
        "device": args.device,
        "tp_size": args.tp_size,
        "attention_backend": args.attention_backend,
        "dtype": args.dtype,
        "num_requests": args.num_requests,
        "concurrency": args.concurrency,
        "max_new_tokens": args.max_new_tokens,
        "cuda_graph_max_bs": args.cuda_graph_max_bs,
        "disable_cuda_graph": args.disable_cuda_graph,
        "enable_torch_compile": args.enable_torch_compile,
        "generated_tokens": generated_tokens,
        "elapsed_s": elapsed,
        "throughput_tokens_per_s": generated_tokens / elapsed,
        "request_latency_mean_s": statistics.mean(latencies),
        "request_latency_p50_s": percentile(latencies, 0.50),
        "request_latency_p95_s": percentile(latencies, 0.95),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
