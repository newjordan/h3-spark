#!/usr/bin/env python3
"""Byte-exact gate: kitchen 128x128x256 single-N tile == cublasLt.

Profiler name: cutlass3x_sm120_..._128x128x256_1x1x1. Pairing both
arms doubles B smem; this tile holds one N operand. Same m16n8k64.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import comfy_kitchen as ck

from swiglu_nvfp4 import (
    fc1_nvfp4_scaled_tma128n128k4,
    fc1_paired_nvfp4_scaled_tma256k2,
    load_extension,
)


ROWS = 20423
K = 5376
N = 14336


def nvfp4_scale(x: torch.Tensor) -> torch.Tensor:
    return (x.abs().amax() / (6.0 * 448.0)).to(torch.float32).reshape(1)


def kitchen_gemm(qx, sx, qw, sw, scale_x, scale_w) -> torch.Tensor:
    alpha = (scale_x * scale_w).reshape(1)
    y = ck.scaled_mm_nvfp4(
        qx, qw, scale_x, scale_w, sx, sw, out_dtype=torch.bfloat16, alpha=alpha
    )
    return y[:ROWS]


def byte_diff(left: torch.Tensor, right: torch.Tensor) -> int:
    return int((left.view(torch.uint8) != right.view(torch.uint8)).sum().item())


def max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


def time_ms(fn) -> float:
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    fn()
    torch.cuda.synchronize()
    starter.record()
    fn()
    ender.record()
    torch.cuda.synchronize()
    return starter.elapsed_time(ender)


def main() -> int:
    load_extension(verbose=False)
    device = torch.device("cuda")
    torch.manual_seed(26081250)
    x = torch.randn(ROWS, K, device=device, dtype=torch.bfloat16)
    weight = torch.randn(2 * N, K, device=device, dtype=torch.bfloat16) * 0.02
    sx = nvfp4_scale(x)
    sw = nvfp4_scale(weight)
    qx, qxs = ck.quantize_nvfp4(x, sx, pad_16x=True, hi_first=True)
    qw, qws = ck.quantize_nvfp4(weight, sw, pad_16x=True, hi_first=True)
    qx_live = qx[:ROWS].contiguous()
    alpha = (sx * sw).reshape(1).contiguous()
    kitchen = kitchen_gemm(qx, qxs, qw, qws, sx, sw)
    full = fc1_nvfp4_scaled_tma128n128k4(qx_live, qxs, qw, qws, alpha)
    payload = {
        "identity": (
            "Kitchen 128x128x256 single-N tile == cublasLt on the "
            "m16n8k64 atom. One B operand (not paired arms). 2-stage "
            "K=256. Profiler: 128x128x256_1x1x1 s16864."
        ),
        "kitchen_kernel": (
            "cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_"
            "ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128x128x256_1x1x1"
        ),
        "rows": ROWS,
        "k": K,
        "n": 2 * N,
        "device": torch.cuda.get_device_name(device),
        "kit_vs_kitchen_mismatches": byte_diff(full, kitchen),
        "kit_vs_kitchen_max_abs": round(max_abs_diff(full, kitchen), 6),
        "full_finite": bool(torch.isfinite(full.float()).all().item()),
        "kitchen_gemm_min_ms": round(
            time_ms(lambda: kitchen_gemm(qx, qxs, qw, qws, sx, sw)), 4
        ),
        "kit_full_min_ms": round(
            time_ms(
                lambda: fc1_nvfp4_scaled_tma128n128k4(
                    qx_live, qxs, qw, qws, alpha
                )
            ),
            4,
        ),
        "k2_prod_min_ms": round(
            time_ms(
                lambda: fc1_paired_nvfp4_scaled_tma256k2(
                    qx_live, qxs, qw, qws, alpha, product=True
                )
            ),
            4,
        ),
    }
    payload["pass"] = (
        payload["kit_vs_kitchen_mismatches"] == 0
        and payload["full_finite"]
    )
    if payload["pass"]:
        payload["vs_kitchen_ms"] = round(
            payload["kitchen_gemm_min_ms"] - payload["kit_full_min_ms"], 4
        )
    text = json.dumps(payload, indent=2, sort_keys=True)
    Path(__file__).with_name(
        "gate_fc1_nvfp4_scaled_tma128n128k4_20423.json"
    ).write_text(text + "\n")
    print(text)
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
