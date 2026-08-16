#!/usr/bin/env python3
"""Correctness and timing gate for RMSNorm + AdaLN -> dynamic NVFP4."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
STAGED = HERE.parent / "adaln_nvfp4" / "native_cuda"
sys.path.insert(0, str(STAGED))

from adaln_nvfp4 import adaln_nvfp4_dynamic, eager_adaln_reference  # noqa: E402
from rms_adaln_nvfp4 import (  # noqa: E402
    WIDTH,
    load_extension,
    rms_adaln_bf16,
    rms_adaln_nvfp4_dynamic,
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


def make_inputs(
    rows: int,
    modulation_dtype: torch.dtype,
    corpus: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    if corpus == "random":
        x = torch.randn(rows, WIDTH, dtype=torch.bfloat16, device=device)
        weight = torch.randn(WIDTH, dtype=torch.bfloat16, device=device)
        table = torch.randn(21, WIDTH * 6, dtype=modulation_dtype, device=device)
    elif corpus == "structured":
        # Finite edge corpus: both zero signs, BF16 extrema/subnormal-adjacent
        # values, and cancellation-prone modulation. NaN/Inf byte parity is
        # covered independently by check_exceptional_cuda.py.
        pattern = torch.tensor(
            [0.0, -0.0, 2.0**-126, -(2.0**-126), 2.0**-20, -(2.0**-20),
             1.0, -1.0, 3.3895313892515355e38, -3.3895313892515355e38],
            dtype=torch.float32,
            device=device,
        ).to(torch.bfloat16)
        x = pattern.repeat((rows * WIDTH + pattern.numel() - 1) // pattern.numel())[
            : rows * WIDTH
        ].reshape(rows, WIDTH).contiguous()
        weight_pattern = torch.tensor(
            [0.0, -0.0, 2.0**-10, -(2.0**-10), 0.5, -0.5, 1.0, -1.0],
            dtype=torch.float32,
            device=device,
        ).to(torch.bfloat16)
        weight = weight_pattern.repeat(
            (WIDTH + weight_pattern.numel() - 1) // weight_pattern.numel()
        )[:WIDTH].contiguous()
        mod_pattern = torch.tensor(
            [0.0, -0.0, 2.0**-20, -(2.0**-20), 0.5, -0.5, 1.0, -1.0],
            dtype=modulation_dtype,
            device=device,
        )
        table = mod_pattern.repeat(
            (21 * WIDTH * 6 + mod_pattern.numel() - 1) // mod_pattern.numel()
        )[: 21 * WIDTH * 6].reshape(21, WIDTH * 6).contiguous()
    else:
        raise ValueError(corpus)
    scale = table[:, WIDTH : 2 * WIDTH]
    shift = table[:, 4 * WIDTH : 5 * WIDTH]
    row_ids = (
        torch.arange(rows, dtype=torch.int64, device=device)
        .mul_(21)
        .div_(rows, rounding_mode="floor")
        .to(torch.int32)
    )
    return x, weight, scale, shift, row_ids


def check_case(
    rows: int,
    modulation_dtype: torch.dtype,
    corpus: str,
    warmup: int,
    reps: int,
) -> dict[str, object]:
    x, weight, scale, shift, row_ids = make_inputs(
        rows, modulation_dtype, corpus
    )
    eps = 1e-5
    normed, reference_rstd = torch.ops.aten._fused_rms_norm(
        x, [WIDTH], weight, eps
    )
    reference = eager_adaln_reference(normed, scale, shift, row_ids)
    fused_materialized, fused_rstd = rms_adaln_bf16(
        x, weight, scale, shift, row_ids, eps
    )
    reference_q, reference_s, reference_global = adaln_nvfp4_dynamic(
        normed, scale, shift, row_ids
    )
    fused_q, fused_s, fused_global = rms_adaln_nvfp4_dynamic(
        x, weight, scale, shift, row_ids, eps
    )
    global_scale_byte_mismatches = byte_diff(
        fused_global, reference_global
    )

    def staged():
        n = F.rms_norm(x, (WIDTH,), weight, eps)
        return adaln_nvfp4_dynamic(n, scale, shift, row_ids)

    def fused():
        return rms_adaln_nvfp4_dynamic(x, weight, scale, shift, row_ids, eps)

    staged_ms = event_ms(staged, warmup, reps)
    fused_ms = event_ms(fused, warmup, reps)
    result = {
        "rows": rows,
        "width": WIDTH,
        "dtype": "torch.bfloat16",
        "modulation_dtype": str(modulation_dtype),
        "modulation_stride": list(scale.stride()),
        "corpus": corpus,
        "eps": eps,
        "rstd_exact": bool(torch.equal(fused_rstd, reference_rstd.reshape(-1))),
        "rstd_mismatches": int(
            (fused_rstd.view(torch.uint8) != reference_rstd.reshape(-1).view(torch.uint8)).sum().item()
        ),
        "materialized_bf16_mismatches": byte_diff(fused_materialized, reference),
        "packed_byte_mismatches": byte_diff(fused_q, reference_q),
        "scale_byte_mismatches": byte_diff(fused_s, reference_s),
        "global_scale_exact": global_scale_byte_mismatches == 0,
        "global_scale_byte_mismatches": global_scale_byte_mismatches,
        "staged_rms_adaln_nvfp4_ms": staged_ms,
        "fused_rms_adaln_nvfp4_ms": fused_ms,
        "speedup": staged_ms / fused_ms,
        "estimated_bf16_traffic_saved_bytes": rows * WIDTH * 2 * 2,
    }
    result["byte_exact"] = all(
        (
            result["rstd_exact"],
            result["materialized_bf16_mismatches"] == 0,
            result["packed_byte_mismatches"] == 0,
            result["scale_byte_mismatches"] == 0,
            result["global_scale_byte_mismatches"] == 0,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=129)
    parser.add_argument("--seed", type=int, default=260812)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument(
        "--modulation-dtype",
        choices=("fp32", "bf16", "both"),
        default="fp32",
    )
    parser.add_argument(
        "--corpus", choices=("random", "structured", "both"), default="random"
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.rows <= 0:
        raise SystemExit("rows must be positive")

    torch.manual_seed(args.seed)
    load_extension()
    dtypes = {
        "fp32": [torch.float32],
        "bf16": [torch.bfloat16],
        "both": [torch.float32, torch.bfloat16],
    }[args.modulation_dtype]
    corpora = {
        "random": ["random"],
        "structured": ["structured"],
        "both": ["random", "structured"],
    }[args.corpus]
    results = [
        check_case(args.rows, dtype, corpus, args.warmup, args.reps)
        for dtype in dtypes
        for corpus in corpora
    ]
    torch.cuda.synchronize()
    payload: object = results[0] if len(results) == 1 else results
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    if not all(result["byte_exact"] for result in results):
        raise SystemExit("RMS+AdaLN->NVFP4 exactness gate failed")


if __name__ == "__main__":
    main()
