#!/usr/bin/env python3
"""Byte-exact gate: SWIZZLE_64B remapped on the K=128 TMA box.

phys_col = col XOR (((row >> 1) & 3) << 4). Same two m16n8k64 atoms,
256x64 3-stage tile, and cuBLAS 128x4 scale slabs.
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
    fc1_paired_nvfp4_scaled_tma256k2_sw,
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
    torch.manual_seed(26081243)
    x = torch.randn(ROWS, K, device=device, dtype=torch.bfloat16)
    weight = torch.randn(2 * N, K, device=device, dtype=torch.bfloat16) * 0.02
    sx = nvfp4_scale(x)
    sw = nvfp4_scale(weight)
    qx, qxs = ck.quantize_nvfp4(x, sx, pad_16x=True, hi_first=True)
    qw, qws = ck.quantize_nvfp4(weight, sw, pad_16x=True, hi_first=True)
    qx_live = qx[:ROWS].contiguous()
    alpha = (sx * sw).reshape(1).contiguous()
    kitchen = kitchen_gemm(qx, qxs, qw, qws, sx, sw)
    full = fc1_paired_nvfp4_scaled_tma256k2_sw(
        qx_live, qxs, qw, qws, alpha, product=False
    )
    prod = fc1_paired_nvfp4_scaled_tma256k2_sw(
        qx_live, qxs, qw, qws, alpha, product=True
    )
    base = fc1_paired_nvfp4_scaled_tma256k2(
        qx_live, qxs, qw, qws, alpha, product=True
    )
    eager_kit = eager_act(kitchen.contiguous())
    eager_full = eager_act(full)
    two_q, two_s, two_g = swiglu_nvfp4_dynamic(kitchen.contiguous())
    prod_q, prod_s, prod_g = bf16_nvfp4_dynamic(prod)
    payload = {
        "identity": (
            "SWIZZLE_64B TMA == linear fragment under "
            "phys_col = col XOR (((row>>1)&3)<<4) on the kitchen-legal "
            "K=128 / 256x64 3-stage m16n8k64 paired-N atom"
        ),
        "rows": ROWS,
        "k": K,
        "n": N,
        "device": torch.cuda.get_device_name(device),
        "sw_full_vs_kitchen_mismatches": byte_diff(full, kitchen),
        "sw_full_vs_kitchen_max_abs": round(max_abs_diff(full, kitchen), 6),
        "sw_prod_vs_k2_prod": byte_diff(prod, base),
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
        "sw_full_min_ms": round(
            time_ms(
                lambda: fc1_paired_nvfp4_scaled_tma256k2_sw(
                    qx_live, qxs, qw, qws, alpha, product=False
                )
            ),
            4,
        ),
        "sw_prod_min_ms": round(
            time_ms(
                lambda: fc1_paired_nvfp4_scaled_tma256k2_sw(
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
        payload["sw_full_vs_kitchen_mismatches"] == 0
        and payload["sw_prod_vs_k2_prod"] == 0
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
            payload["k2_prod_min_ms"] - payload["sw_prod_min_ms"], 4
        )
    text = json.dumps(payload, indent=2, sort_keys=True)
    Path(__file__).with_name(
        "gate_fc1_nvfp4_scaled_tma256k2_sw_20423.json"
    ).write_text(text + "\n")
    print(text)
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
