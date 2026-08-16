#!/usr/bin/env python3
"""Seconds-only byte gate for the one-pass static H3 prototype."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
STAGED = HERE.parent / "adaln_nvfp4_lab" / "native_cuda"
sys.path.insert(0, str(STAGED))

from adaln_nvfp4 import adaln_nvfp4, eager_adaln_reference  # noqa: E402
from prototype import (  # noqa: E402
    WIDTH,
    load_extension,
    rms_adaln_nvfp4_static,
    rms_adaln_nvfp4_static_debug,
)


def byte_diff(left: torch.Tensor, right: torch.Tensor) -> int:
    return int((left.view(torch.uint8) != right.view(torch.uint8)).sum().item())


def event_ms(fn, warmup: int, reps: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return float(torch.tensor(samples).median().item())


def make_inputs(rows: int, modulation_dtype: torch.dtype):
    device = torch.device("cuda")
    x = torch.randn(rows, WIDTH, dtype=torch.bfloat16, device=device)
    weight = torch.randn(WIDTH, dtype=torch.bfloat16, device=device)
    table = torch.randn(21, WIDTH * 6, dtype=modulation_dtype, device=device)
    scale = table[:, WIDTH : 2 * WIDTH]
    shift = table[:, 4 * WIDTH : 5 * WIDTH]
    row_ids = (
        torch.arange(rows, dtype=torch.int64, device=device)
        .mul_(21)
        .div_(rows, rounding_mode="floor")
        .to(torch.int32)
    )
    return x, weight, scale, shift, row_ids


def check_case(rows: int, modulation_dtype: torch.dtype, warmup: int, reps: int):
    x, weight, scale, shift, row_ids = make_inputs(rows, modulation_dtype)
    eps = 1e-5
    normed, reference_rstd = torch.ops.aten._fused_rms_norm(
        x, [WIDTH], weight, eps
    )
    materialized = eager_adaln_reference(normed, scale, shift, row_ids)
    # Reproduce TensorCoreNVFP4Layout's BF16 division and FP32 widening.  This
    # is only a convenient in-range supplied scale; both paths receive the
    # exact same tensor identity.
    global_scale = (materialized.abs().max() / 2688.0).to(torch.float32)
    reference_q, reference_s = adaln_nvfp4(
        normed, scale, shift, row_ids, global_scale
    )

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        fused_q, fused_s, fused_rstd = rms_adaln_nvfp4_static_debug(
            x, weight, scale, shift, row_ids, global_scale, eps
        )
    stream.synchronize()

    def staged():
        staged_normed = F.rms_norm(x, (WIDTH,), weight, eps)
        return adaln_nvfp4(
            staged_normed, scale, shift, row_ids, global_scale
        )

    def fused():
        return rms_adaln_nvfp4_static(
            x, weight, scale, shift, row_ids, global_scale, eps
        )

    staged_ms = event_ms(staged, warmup, reps)
    fused_ms = event_ms(fused, warmup, reps)
    result = {
        "rows": rows,
        "width": WIDTH,
        "modulation_dtype": str(modulation_dtype),
        "eps": eps,
        "nondefault_stream": True,
        "rstd_byte_mismatches": byte_diff(
            fused_rstd, reference_rstd.reshape(-1)
        ),
        "packed_byte_mismatches": byte_diff(fused_q, reference_q),
        "scale_byte_mismatches": byte_diff(fused_s, reference_s),
        "staged_static_ms": staged_ms,
        "one_pass_static_ms": fused_ms,
        "speedup": staged_ms / fused_ms,
        "activation_traffic_saved_bytes_vs_dynamic_v2":
            2 * rows * WIDTH * 2 + rows * 4,
    }
    result["byte_exact"] = (
        result["rstd_byte_mismatches"] == 0
        and result["packed_byte_mismatches"] == 0
        and result["scale_byte_mismatches"] == 0
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=129)
    parser.add_argument("--seed", type=int, default=260812)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument(
        "--modulation-dtype", choices=("fp32", "bf16", "both"), default="both"
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.rows <= 0 or args.warmup < 0 or args.reps <= 0:
        raise SystemExit("rows/reps must be positive and warmup nonnegative")

    torch.manual_seed(args.seed)
    native = load_extension()
    dtypes = {
        "fp32": [torch.float32],
        "bf16": [torch.bfloat16],
        "both": [torch.float32, torch.bfloat16],
    }[args.modulation_dtype]
    results = [
        check_case(args.rows, dtype, args.warmup, args.reps)
        for dtype in dtypes
    ]
    torch.cuda.synchronize()
    payload = {
        "native_abi": native.abi_version(),
        "native_build_id": native.build_id(),
        "results": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    if not all(result["byte_exact"] for result in results):
        raise SystemExit("one-pass static exactness gate failed")


if __name__ == "__main__":
    main()
