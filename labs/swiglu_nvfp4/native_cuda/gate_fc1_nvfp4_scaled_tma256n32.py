#!/usr/bin/env python3
"""Byte-exact gate: 256x32 K=128 1-stage == kitchen / k2.

Half-N of k2: 4 n-subtiles, 64-float acc. TMA B box is 64
(N=32 tile faults). 1-stage so two CTAs/SM can fit.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import comfy_kitchen as ck

from swiglu_nvfp4 import (
    fc1_paired_nvfp4_scaled_tma256k2,
    fc1_paired_nvfp4_scaled_tma256n32,
    fc1_paired_nvfp4_scaled_tma256n32_attrs,
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
    torch.manual_seed(26081272)
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
    n32 = fc1_paired_nvfp4_scaled_tma256n32(qx_live, qxs, qw, qws, alpha)
    attrs = fc1_paired_nvfp4_scaled_tma256n32_attrs()
    payload = {
        "identity": (
            "256x32 K=128 1-stage paired-N == kitchen == k2. "
            "TMA B is 64; compute is 32. 64-float acc is the occupancy cut."
        ),
        "rows": ROWS,
        "k": K,
        "n": 2 * N,
        "device": torch.cuda.get_device_name(device),
        "n32_vs_kitchen_mismatches": byte_diff(n32, kitchen),
        "n32_vs_k2_mismatches": byte_diff(n32, k2),
        "full_finite": bool(torch.isfinite(n32.float()).all().item()),
        "n32_regs": int(attrs["regs"]),
        "n32_smem": int(attrs["smem"]),
        "n32_occupancy": int(attrs["occupancy"]),
        "n32_smem_struct": int(attrs["smem_struct"]),
        "kitchen_gemm_min_ms": round(
            time_ms(lambda: kitchen_gemm(qx, qxs, qw, qws, sx, sw)), 4
        ),
        "n32_full_min_ms": round(
            time_ms(
                lambda: fc1_paired_nvfp4_scaled_tma256n32(
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
        payload["n32_vs_kitchen_mismatches"] == 0
        and payload["n32_vs_k2_mismatches"] == 0
        and payload["full_finite"]
    )
    if payload["pass"]:
        payload["vs_kitchen_ms"] = round(
            payload["kitchen_gemm_min_ms"] - payload["n32_full_min_ms"], 4
        )
        payload["vs_k2_ms"] = round(
            payload["k2_full_min_ms"] - payload["n32_full_min_ms"], 4
        )
    text = json.dumps(payload, indent=2, sort_keys=True)
    Path(__file__).with_name(
        "gate_fc1_nvfp4_scaled_tma256n32_20423.json"
    ).write_text(text + "\n")
    print(text)
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
