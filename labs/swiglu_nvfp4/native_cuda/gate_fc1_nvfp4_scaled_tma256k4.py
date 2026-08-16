#!/usr/bin/env python3
"""Byte-exact gate: K=256 TMA box on the 256x64 tile.

One TMA of 128 packed bytes feeds four m16n8k64 atoms. Keeps the
256x64 k2 register budget (2x8x4). 1-stage: 2-stage at this box
is 112 KiB and exceeds the 99 KiB static smem cap.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

import comfy_kitchen as ck

from swiglu_nvfp4 import (
    bf16_nvfp4_dynamic,
    fc1_paired_nvfp4_scaled_tma256k2,
    fc1_paired_nvfp4_scaled_tma256k4,
    load_extension,
    swiglu_nvfp4_dynamic,
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


def eager_act(raw: torch.Tensor) -> torch.Tensor:
    gate, up = raw.chunk(2, dim=-1)
    return F.silu(gate).mul_(up)


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
    torch.manual_seed(26081246)
    x = torch.randn(ROWS, K, device=device, dtype=torch.bfloat16)
    weight = torch.randn(2 * N, K, device=device, dtype=torch.bfloat16) * 0.02
    sx = nvfp4_scale(x)
    sw = nvfp4_scale(weight)
    qx, qxs = ck.quantize_nvfp4(x, sx, pad_16x=True, hi_first=True)
    qw, qws = ck.quantize_nvfp4(weight, sw, pad_16x=True, hi_first=True)
    qx_live = qx[:ROWS].contiguous()
    alpha = (sx * sw).reshape(1).contiguous()
    kitchen = kitchen_gemm(qx, qxs, qw, qws, sx, sw)
    full = fc1_paired_nvfp4_scaled_tma256k4(
        qx_live, qxs, qw, qws, alpha, product=False
    )
    prod = fc1_paired_nvfp4_scaled_tma256k4(
        qx_live, qxs, qw, qws, alpha, product=True
    )
    k2 = fc1_paired_nvfp4_scaled_tma256k2(
        qx_live, qxs, qw, qws, alpha, product=True
    )
    eager_kit = eager_act(kitchen.contiguous())
    eager_full = eager_act(full)
    two_q, two_s, two_g = swiglu_nvfp4_dynamic(kitchen.contiguous())
    prod_q, prod_s, prod_g = bf16_nvfp4_dynamic(prod)
    payload = {
        "identity": (
            "K=256 TMA box == four sequential K=64 MMA steps on the "
            "256x64 tile: one 128-byte packed row feeds four m16n8k64 "
            "atoms; four 128x4 scale slabs cover the 16 K-scale columns. "
            "Same 2x8x4 register acc as k2. 1-stage: 2-stage box "
            "exceeds the 99 KiB static smem cap."
        ),
        "rows": ROWS,
        "k": K,
        "n": N,
        "device": torch.cuda.get_device_name(device),
        "k4_full_vs_kitchen_mismatches": byte_diff(full, kitchen),
        "k4_full_vs_kitchen_max_abs": round(max_abs_diff(full, kitchen), 6),
        "k4_prod_vs_k2_prod": byte_diff(prod, k2),
        "product_vs_eager_full_mismatches": byte_diff(prod, eager_full),
        "product_vs_eager_kitchen_mismatches": byte_diff(prod, eager_kit),
        "from_product_vs_kitchen_q": byte_diff(prod_q, two_q),
        "from_product_vs_kitchen_s": byte_diff(prod_s, two_s),
        "from_product_vs_kitchen_scale_exact": bool(torch.equal(prod_g, two_g)),
        "full_finite": bool(torch.isfinite(full.float()).all().item()),
        "prod_finite": bool(torch.isfinite(prod.float()).all().item()),
        "kitchen_gemm_min_ms": round(
            time_ms(lambda: kitchen_gemm(qx, qxs, qw, qws, sx, sw)), 4
        ),
        "k4_full_min_ms": round(
            time_ms(
                lambda: fc1_paired_nvfp4_scaled_tma256k4(
                    qx_live, qxs, qw, qws, alpha, product=False
                )
            ),
            4,
        ),
        "k4_prod_min_ms": round(
            time_ms(
                lambda: fc1_paired_nvfp4_scaled_tma256k4(
                    qx_live, qxs, qw, qws, alpha, product=True
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
        "from_product_min_ms": round(
            time_ms(lambda: bf16_nvfp4_dynamic(prod)), 4
        ),
    }
    payload["pass"] = (
        payload["k4_full_vs_kitchen_mismatches"] == 0
        and payload["k4_prod_vs_k2_prod"] == 0
        and payload["product_vs_eager_full_mismatches"] == 0
        and payload["product_vs_eager_kitchen_mismatches"] == 0
        and payload["from_product_vs_kitchen_q"] == 0
        and payload["from_product_vs_kitchen_s"] == 0
        and payload["from_product_vs_kitchen_scale_exact"]
        and payload["full_finite"]
        and payload["prod_finite"]
    )
    if payload["pass"]:
        payload["vs_k2_ms"] = round(
            payload["k2_prod_min_ms"] - payload["k4_prod_min_ms"], 4
        )
    text = json.dumps(payload, indent=2, sort_keys=True)
    Path(__file__).with_name(
        "gate_fc1_nvfp4_scaled_tma256k4_20423.json"
    ).write_text(text + "\n")
    print(text)
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
