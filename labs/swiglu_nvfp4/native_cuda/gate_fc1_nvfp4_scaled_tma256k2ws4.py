#!/usr/bin/env python3
"""Byte-exact gate: 12-warp 4-producer k2 == kitchen == k2.

Kitchen launch structure (4 TMA warps + 8 MMA) on the best lab
tile (256x64 K=128 paired-N). 9-warp k2ws serializes every TMA
on one lane; this issues A / Bg / Bu / scales in parallel.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import comfy_kitchen as ck

from swiglu_nvfp4 import (
    fc1_paired_nvfp4_scaled_tma256k2,
    fc1_paired_nvfp4_scaled_tma256k2ws4,
    fc1_paired_nvfp4_scaled_tma256k2ws4_attrs,
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
    torch.manual_seed(26081274)
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
    ws4 = fc1_paired_nvfp4_scaled_tma256k2ws4(qx_live, qxs, qw, qws, alpha)
    attrs = fc1_paired_nvfp4_scaled_tma256k2ws4_attrs()
    payload = {
        "identity": (
            "12-warp 4-producer + 8-MMA on k2 256x64 K=128 == kitchen "
            "== unified k2. Kitchen launch split: A / Bg / Bu / scales "
            "issue in parallel. Same atom and smem as k2."
        ),
        "rows": ROWS,
        "k": K,
        "n": 2 * N,
        "device": torch.cuda.get_device_name(device),
        "ws4_vs_kitchen_mismatches": byte_diff(ws4, kitchen),
        "ws4_vs_kitchen_max_abs": round(max_abs_diff(ws4, kitchen), 6),
        "ws4_vs_k2_mismatches": byte_diff(ws4, k2),
        "full_finite": bool(torch.isfinite(ws4.float()).all().item()),
        "ws4_regs": int(attrs["regs"]),
        "ws4_smem": int(attrs["smem"]),
        "ws4_occupancy": int(attrs["occupancy"]),
        "ws4_threads": int(attrs["threads"]),
        "ws4_warps": int(attrs["warps"]),
        "ws4_prod_warps": int(attrs["prod_warps"]),
        "kitchen_gemm_min_ms": round(
            time_ms(lambda: kitchen_gemm(qx, qxs, qw, qws, sx, sw)), 4
        ),
        "ws4_full_min_ms": round(
            time_ms(
                lambda: fc1_paired_nvfp4_scaled_tma256k2ws4(
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
        payload["ws4_vs_kitchen_mismatches"] == 0
        and payload["ws4_vs_k2_mismatches"] == 0
        and payload["full_finite"]
    )
    if payload["pass"]:
        payload["vs_kitchen_ms"] = round(
            payload["kitchen_gemm_min_ms"] - payload["ws4_full_min_ms"], 4
        )
        payload["vs_k2_ms"] = round(
            payload["k2_full_min_ms"] - payload["ws4_full_min_ms"], 4
        )
    text = json.dumps(payload, indent=2, sort_keys=True)
    Path(__file__).with_name(
        "gate_fc1_nvfp4_scaled_tma256k2ws4_20423.json"
    ).write_text(text + "\n")
    print(text)
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
