#!/usr/bin/env python3
"""Byte-exact gate: k2 [gate|up] epilogue amax == two-pass G.

Amax is the reduction of the same BF16 products the store already
holds. No second HBM read of [gate|up]. live_amax-sourced rebind
then matches two-pass pack. Kitchen GEMM stays; this is lab identity.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import comfy_kitchen as ck

from swiglu_nvfp4 import (
    fc1_paired_nvfp4_scaled_tma256k2,
    fc1_paired_nvfp4_scaled_tma256k2_amax,
    load_extension,
    swiglu_nvfp4_dynamic,
    swiglu_nvfp4_static_rebind,
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
    torch.manual_seed(26081258)
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
    epi_raw, epi_g = fc1_paired_nvfp4_scaled_tma256k2_amax(
        qx_live, qxs, qw, qws, alpha
    )
    two_q, two_s, two_g = swiglu_nvfp4_dynamic(kitchen.contiguous())
    pack_q, pack_s = swiglu_nvfp4_static_rebind(
        epi_raw, epi_g, certified_scale=epi_g, source="live_amax"
    )
    payload = {
        "identity": (
            "k2 [gate|up] epilogue amax is the same reduction as "
            "two-pass on those BF16 products. G_epi == G_two_pass. "
            "live_amax rebind == two-pass q/s. No second HBM read."
        ),
        "rows": ROWS,
        "k": K,
        "n": 2 * N,
        "device": torch.cuda.get_device_name(device),
        "epi_vs_kitchen_mismatches": byte_diff(epi_raw, kitchen),
        "epi_vs_k2_mismatches": byte_diff(epi_raw, k2),
        "g_epi_vs_twopass_exact": bool(torch.equal(epi_g, two_g)),
        "g_epi": float(epi_g.item()),
        "g_two": float(two_g.item()),
        "rebind_vs_twopass_q": byte_diff(pack_q, two_q),
        "rebind_vs_twopass_s": byte_diff(pack_s, two_s),
        "kitchen_gemm_min_ms": round(
            time_ms(lambda: kitchen_gemm(qx, qxs, qw, qws, sx, sw)), 4
        ),
        "k2_full_min_ms": round(
            time_ms(
                lambda: fc1_paired_nvfp4_scaled_tma256k2(
                    qx_live, qxs, qw, qws, alpha, product=False
                )
            ),
            4,
        ),
        "k2_amax_min_ms": round(
            time_ms(
                lambda: fc1_paired_nvfp4_scaled_tma256k2_amax(
                    qx_live, qxs, qw, qws, alpha
                )
            ),
            4,
        ),
        "twopass_min_ms": round(
            time_ms(lambda: swiglu_nvfp4_dynamic(kitchen.contiguous())), 4
        ),
        "rebind_min_ms": round(
            time_ms(
                lambda: swiglu_nvfp4_static_rebind(
                    epi_raw, epi_g, certified_scale=epi_g, source="live_amax"
                )
            ),
            4,
        ),
    }
    payload["pass"] = (
        payload["epi_vs_kitchen_mismatches"] == 0
        and payload["epi_vs_k2_mismatches"] == 0
        and payload["g_epi_vs_twopass_exact"]
        and payload["rebind_vs_twopass_q"] == 0
        and payload["rebind_vs_twopass_s"] == 0
    )
    if payload["pass"]:
        payload["amax_in_epilogue_ms"] = round(
            payload["k2_amax_min_ms"] - payload["k2_full_min_ms"], 4
        )
        payload["vs_kitchen_plus_twopass_ms"] = round(
            (payload["kitchen_gemm_min_ms"] + payload["twopass_min_ms"])
            - (payload["k2_amax_min_ms"] + payload["rebind_min_ms"]),
            4,
        )
    text = json.dumps(payload, indent=2, sort_keys=True)
    Path(__file__).with_name(
        "gate_fc1_epilogue_amax_20423.json"
    ).write_text(text + "\n")
    print(text)
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
