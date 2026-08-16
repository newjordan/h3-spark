#!/usr/bin/env python3
"""Byte-exact gate: software-pipelined A on k2 == kitchen / scalar k2.

Next K=64 A/SFA overlaps m16n8k64. Same 256x64 K=128 paired-N store.
Not the B-pipe identity.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import comfy_kitchen as ck

from swiglu_nvfp4 import (
    fc1_paired_nvfp4_scaled_tma256k2,
    fc1_paired_nvfp4_scaled_tma256k2_pipea,
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
    torch.manual_seed(26081269)
    x = torch.randn(ROWS, K, device=device, dtype=torch.bfloat16)
    weight = torch.randn(2 * N, K, device=device, dtype=torch.bfloat16) * 0.02
    sx = nvfp4_scale(x)
    sw = nvfp4_scale(weight)
    qx, qxs = ck.quantize_nvfp4(x, sx, pad_16x=True, hi_first=True)
    qw, qws = ck.quantize_nvfp4(weight, sw, pad_16x=True, hi_first=True)
    qx_live = qx[:ROWS].contiguous()
    alpha = (sx * sw).reshape(1).contiguous()
    kitchen = kitchen_gemm(qx, qxs, qw, qws, sx, sw)
    k2 = fc1_paired_nvfp4_scaled_tma256k2(
        qx_live, qxs, qw, qws, alpha, product=False
    )
    pipea = fc1_paired_nvfp4_scaled_tma256k2_pipea(
        qx_live, qxs, qw, qws, alpha
    )
    payload = {
        "identity": (
            "software-pipelined A on k2 (256x64 K=128): next K=64 "
            "A/SFA overlaps m16n8k64 == kitchen == scalar k2."
        ),
        "rows": ROWS,
        "k": K,
        "n": 2 * N,
        "device": torch.cuda.get_device_name(device),
        "pipea_vs_kitchen_mismatches": byte_diff(pipea, kitchen),
        "pipea_vs_k2_mismatches": byte_diff(pipea, k2),
        "full_finite": bool(torch.isfinite(pipea.float()).all().item()),
        "kitchen_gemm_min_ms": round(
            time_ms(lambda: kitchen_gemm(qx, qxs, qw, qws, sx, sw)), 4
        ),
        "pipea_full_min_ms": round(
            time_ms(
                lambda: fc1_paired_nvfp4_scaled_tma256k2_pipea(
                    qx_live, qxs, qw, qws, alpha
                )
            ),
            4,
        ),
        "k2_full_min_ms": round(
            time_ms(
                lambda: fc1_paired_nvfp4_scaled_tma256k2(
                    qx_live, qxs, qw, qws, alpha, product=False
                )
            ),
            4,
        ),
    }
    payload["pass"] = (
        payload["pipea_vs_kitchen_mismatches"] == 0
        and payload["pipea_vs_k2_mismatches"] == 0
        and payload["full_finite"]
    )
    if payload["pass"]:
        payload["vs_kitchen_ms"] = round(
            payload["kitchen_gemm_min_ms"] - payload["pipea_full_min_ms"], 4
        )
        payload["vs_k2_ms"] = round(
            payload["k2_full_min_ms"] - payload["pipea_full_min_ms"], 4
        )
    text = json.dumps(payload, indent=2, sort_keys=True)
    Path(__file__).with_name(
        "gate_fc1_nvfp4_scaled_tma256k2_pipea_20423.json"
    ).write_text(text + "\n")
    print(text)
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
