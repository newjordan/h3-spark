#!/usr/bin/env python3
"""Byte-exact gate: split-M kitchen == full kitchen; amax associates.

NVFP4 GEMM rows are independent. Amax is a max of per-row-block bit
maxima. If a 128-row panel (7 MiB) stays in the 24 MiB L2 after the
write, panel amax is not a second HBM read of the 1171 MiB pair.
Does not patch serving. Kitchen stays the GEMM.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import comfy_kitchen as ck

from swiglu_nvfp4 import load_extension, swiglu_nvfp4_dynamic, swiglu_winner_lu


ROWS = 20423
K = 5376
N = 14336
TILE = 128


def nvfp4_scale(x: torch.Tensor) -> torch.Tensor:
    return (x.abs().amax() / (6.0 * 448.0)).to(torch.float32).reshape(1)


def kitchen_full(qx, qxs, qw, qws, sx, sw) -> torch.Tensor:
    alpha = (sx * sw).reshape(1)
    y = ck.scaled_mm_nvfp4(
        qx, qw, sx, sw, qxs, qws, out_dtype=torch.bfloat16, alpha=alpha
    )
    return y[:ROWS]


def kitchen_panel(qx, qxs, qw, qws, sx, sw, m0: int, rows: int) -> torch.Tensor:
    alpha = (sx * sw).reshape(1)
    sl = slice(m0, m0 + rows)
    y = ck.scaled_mm_nvfp4(
        qx[sl], qw, sx, sw, qxs[sl], qws,
        out_dtype=torch.bfloat16, alpha=alpha,
    )
    return y


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


def g_from_bits(bits: int, device) -> torch.Tensor:
    absmax = torch.tensor([bits & 0x7FFF], dtype=torch.uint16, device="cpu")
    absmax = absmax.view(torch.bfloat16).float()
    g = (absmax / 2688.0).to(torch.bfloat16).to(torch.float32)
    return g.to(device).reshape(1)


def main() -> int:
    load_extension(verbose=False)
    device = torch.device("cuda")
    torch.manual_seed(26081260)
    x = torch.randn(ROWS, K, device=device, dtype=torch.bfloat16)
    weight = torch.randn(2 * N, K, device=device, dtype=torch.bfloat16) * 0.02
    sx = nvfp4_scale(x)
    sw = nvfp4_scale(weight)
    qx, qxs = ck.quantize_nvfp4(x, sx, pad_16x=True, hi_first=True)
    qw, qws = ck.quantize_nvfp4(weight, sw, pad_16x=True, hi_first=True)
    kitchen = kitchen_full(qx, qxs, qw, qws, sx, sw)
    pad_m = qx.size(0)
    panels = []
    bit_max = 0
    for m0 in range(0, pad_m, TILE):
        live = min(TILE, ROWS - m0) if m0 < ROWS else 0
        y = kitchen_panel(qx, qxs, qw, qws, sx, sw, m0, TILE)
        if live > 0:
            panels.append(y[:live].contiguous())
            lo, _hi = swiglu_winner_lu(y[:live].contiguous())
            if lo > bit_max:
                bit_max = lo
    split = torch.cat(panels, dim=0)
    two_q, two_s, two_g = swiglu_nvfp4_dynamic(kitchen.contiguous())
    del two_q, two_s
    g_assoc = g_from_bits(bit_max, device)
    g_equal = bool(torch.equal(g_assoc, two_g))
    payload = {
        "identity": (
            "Kitchen NVFP4 GEMM is M-independent: concat of 128-row "
            "panels == full GEMM. Amax bits associate over panels "
            "(max of per-panel L). Panel is 7 MiB; L2 is 24 MiB."
        ),
        "rows": ROWS,
        "k": K,
        "n": 2 * N,
        "tile_m": TILE,
        "panels": (pad_m + TILE - 1) // TILE,
        "panel_bytes": TILE * 2 * N * 2,
        "l2_bytes": 25165824,
        "device": torch.cuda.get_device_name(device),
        "split_vs_full_mismatches": byte_diff(split, kitchen),
        "g_assoc_vs_twopass_exact": g_equal,
        "g_assoc": float(g_assoc.item()),
        "g_two": float(two_g.item()),
        "bit_max": bit_max,
        "kitchen_full_min_ms": round(
            time_ms(lambda: kitchen_full(qx, qxs, qw, qws, sx, sw)), 4
        ),
        "twopass_min_ms": round(
            time_ms(lambda: swiglu_nvfp4_dynamic(kitchen.contiguous())), 4
        ),
    }

    def split_gemm_only() -> None:
        for m0 in range(0, pad_m, TILE):
            kitchen_panel(qx, qxs, qw, qws, sx, sw, m0, TILE)

    def split_gemm_amax() -> None:
        bits = 0
        for m0 in range(0, pad_m, TILE):
            y = kitchen_panel(qx, qxs, qw, qws, sx, sw, m0, TILE)
            live = min(TILE, max(0, ROWS - m0))
            if live > 0:
                lo, _ = swiglu_winner_lu(y[:live].contiguous())
                if lo > bits:
                    bits = lo

    payload["split_gemm_min_ms"] = round(time_ms(split_gemm_only), 4)
    payload["split_gemm_amax_min_ms"] = round(time_ms(split_gemm_amax), 4)
    payload["pass"] = (
        payload["split_vs_full_mismatches"] == 0
        and payload["g_assoc_vs_twopass_exact"]
    )
    if payload["pass"]:
        payload["vs_full_gemm_ms"] = round(
            payload["kitchen_full_min_ms"] - payload["split_gemm_min_ms"], 4
        )
        payload["amax_in_l2_ms"] = round(
            payload["split_gemm_amax_min_ms"] - payload["split_gemm_min_ms"], 4
        )
    text = json.dumps(payload, indent=2, sort_keys=True)
    Path(__file__).with_name("gate_kitchen_splitm_20423.json").write_text(
        text + "\n"
    )
    print(text)
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
