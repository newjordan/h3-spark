#!/usr/bin/env python3
"""Byte-exact gate: 12-warp 128x128x256 == kitchen / 8-warp clone.

Kitchen launches 384 threads, 88064 B dynamic smem. Four producer
warps issue A/B/SFA/SFB; eight MMA warps keep the m16n8k64 atom.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import comfy_kitchen as ck

from swiglu_nvfp4 import (
    fc1_nvfp4_scaled_tma128n128k4,
    fc1_nvfp4_scaled_tma128n128k4ws,
    fc1_nvfp4_scaled_tma128n128k4ws_attrs,
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
    torch.manual_seed(26081255)
    x = torch.randn(ROWS, K, device=device, dtype=torch.bfloat16)
    weight = torch.randn(2 * N, K, device=device, dtype=torch.bfloat16) * 0.02
    sx = nvfp4_scale(x)
    sw = nvfp4_scale(weight)
    qx, qxs = ck.quantize_nvfp4(x, sx, pad_16x=True, hi_first=True)
    qw, qws = ck.quantize_nvfp4(weight, sw, pad_16x=True, hi_first=True)
    qx_live = qx[:ROWS].contiguous()
    alpha = (sx * sw).reshape(1).contiguous()
    kitchen = kitchen_gemm(qx, qxs, qw, qws, sx, sw)
    clone8 = fc1_nvfp4_scaled_tma128n128k4(qx_live, qxs, qw, qws, alpha)
    ws = fc1_nvfp4_scaled_tma128n128k4ws(qx_live, qxs, qw, qws, alpha)
    attrs = fc1_nvfp4_scaled_tma128n128k4ws_attrs()
    payload = {
        "identity": (
            "Kitchen 128x128x256 12-warp launch: 4 producer warps "
            "issue A/B/SFA/SFB TMA; 8 MMA warps keep the m16n8k64 "
            "atom and 16x4 acc. 88064 B dynamic smem. Same tile as "
            "the 8-warp clone (already byte-exact)."
        ),
        "rows": ROWS,
        "k": K,
        "n": 2 * N,
        "device": torch.cuda.get_device_name(device),
        "launch": attrs,
        "ws_vs_kitchen_mismatches": byte_diff(ws, kitchen),
        "ws_vs_kitchen_max_abs": round(max_abs_diff(ws, kitchen), 6),
        "ws_vs_clone8_mismatches": byte_diff(ws, clone8),
        "ws_vs_clone8_max_abs": round(max_abs_diff(ws, clone8), 6),
        "full_finite": bool(torch.isfinite(ws.float()).all().item()),
        "kitchen_gemm_min_ms": round(
            time_ms(lambda: kitchen_gemm(qx, qxs, qw, qws, sx, sw)), 4
        ),
        "ws_full_min_ms": round(
            time_ms(
                lambda: fc1_nvfp4_scaled_tma128n128k4ws(
                    qx_live, qxs, qw, qws, alpha
                )
            ),
            4,
        ),
        "clone8_full_min_ms": round(
            time_ms(
                lambda: fc1_nvfp4_scaled_tma128n128k4(
                    qx_live, qxs, qw, qws, alpha
                )
            ),
            4,
        ),
    }
    payload["pass"] = (
        payload["ws_vs_kitchen_mismatches"] == 0
        and payload["ws_vs_clone8_mismatches"] == 0
        and payload["full_finite"]
        and int(attrs["threads"]) == 384
        and int(attrs["dynamic_smem"]) == 88064
    )
    if payload["pass"]:
        payload["vs_kitchen_ms"] = round(
            payload["kitchen_gemm_min_ms"] - payload["ws_full_min_ms"], 4
        )
        payload["vs_clone8_ms"] = round(
            payload["clone8_full_min_ms"] - payload["ws_full_min_ms"], 4
        )
    text = json.dumps(payload, indent=2, sort_keys=True)
    Path(__file__).with_name(
        "gate_fc1_nvfp4_scaled_tma128n128k4ws_20423.json"
    ).write_text(text + "\n")
    print(text)
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
