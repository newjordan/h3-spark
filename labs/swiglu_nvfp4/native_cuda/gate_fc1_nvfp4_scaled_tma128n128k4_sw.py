#!/usr/bin/env python3
"""Byte-exact gate: SWIZZLE_128B remapped on the 128-byte K-row.

phys_col = col XOR ((row & 7) << 4). Measured on sm_121a. Same
128x128x256 single-N m16n8k64 tile as the linear kitchen clone.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import comfy_kitchen as ck

from swiglu_nvfp4 import (
    fc1_nvfp4_scaled_tma128n128k4,
    fc1_nvfp4_scaled_tma128n128k4_sw,
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
    torch.manual_seed(26081256)
    x = torch.randn(ROWS, K, device=device, dtype=torch.bfloat16)
    weight = torch.randn(2 * N, K, device=device, dtype=torch.bfloat16) * 0.02
    sx = nvfp4_scale(x)
    sw = nvfp4_scale(weight)
    qx, qxs = ck.quantize_nvfp4(x, sx, pad_16x=True, hi_first=True)
    qw, qws = ck.quantize_nvfp4(weight, sw, pad_16x=True, hi_first=True)
    qx_live = qx[:ROWS].contiguous()
    alpha = (sx * sw).reshape(1).contiguous()
    kitchen = kitchen_gemm(qx, qxs, qw, qws, sx, sw)
    linear = fc1_nvfp4_scaled_tma128n128k4(qx_live, qxs, qw, qws, alpha)
    swiz = fc1_nvfp4_scaled_tma128n128k4_sw(qx_live, qxs, qw, qws, alpha)
    payload = {
        "identity": (
            "SWIZZLE_128B TMA == linear fragment under "
            "phys_col = col XOR ((row&7)<<4) on the 128-byte K-row "
            "of the 128x128x256 single-N m16n8k64 tile. Map measured "
            "on sm_121a (probe_swizzle128, 0/16384)."
        ),
        "rows": ROWS,
        "k": K,
        "n": 2 * N,
        "device": torch.cuda.get_device_name(device),
        "xor_map": "col XOR ((row & 7) << 4)",
        "sw_vs_kitchen_mismatches": byte_diff(swiz, kitchen),
        "sw_vs_kitchen_max_abs": round(max_abs_diff(swiz, kitchen), 6),
        "sw_vs_linear_mismatches": byte_diff(swiz, linear),
        "sw_vs_linear_max_abs": round(max_abs_diff(swiz, linear), 6),
        "full_finite": bool(torch.isfinite(swiz.float()).all().item()),
        "kitchen_gemm_min_ms": round(
            time_ms(lambda: kitchen_gemm(qx, qxs, qw, qws, sx, sw)), 4
        ),
        "sw_full_min_ms": round(
            time_ms(
                lambda: fc1_nvfp4_scaled_tma128n128k4_sw(
                    qx_live, qxs, qw, qws, alpha
                )
            ),
            4,
        ),
        "linear_full_min_ms": round(
            time_ms(
                lambda: fc1_nvfp4_scaled_tma128n128k4(
                    qx_live, qxs, qw, qws, alpha
                )
            ),
            4,
        ),
    }
    payload["pass"] = (
        payload["sw_vs_kitchen_mismatches"] == 0
        and payload["sw_vs_linear_mismatches"] == 0
        and payload["full_finite"]
    )
    if payload["pass"]:
        payload["vs_kitchen_ms"] = round(
            payload["kitchen_gemm_min_ms"] - payload["sw_full_min_ms"], 4
        )
        payload["vs_linear_ms"] = round(
            payload["linear_full_min_ms"] - payload["sw_full_min_ms"], 4
        )
    text = json.dumps(payload, indent=2, sort_keys=True)
    Path(__file__).with_name(
        "gate_fc1_nvfp4_scaled_tma128n128k4_sw_20423.json"
    ).write_text(text + "\n")
    print(text)
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
