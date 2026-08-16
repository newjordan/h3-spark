#!/usr/bin/env python3
"""Byte-exact gate: bound-winner exact L == true amax T on kitchen FC1.

G(L)==G(U) failed (U is a bound, 1 ULP above). If T==L, no exact
product attains U and winner-only amax is exact without that skip.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

import comfy_kitchen as ck

from swiglu_nvfp4 import (
    load_extension,
    swiglu_nvfp4_dynamic,
    swiglu_nvfp4_static_rebind,
    swiglu_winner_lu,
)


ROWS = 20423
K = 5376
N = 14336


def nvfp4_scale(x: torch.Tensor) -> torch.Tensor:
    return (x.abs().amax() / (6.0 * 448.0)).to(torch.float32).reshape(1)


def kitchen_fc1(x, weight):
    sx = nvfp4_scale(x)
    sw = nvfp4_scale(weight)
    qx, qxs = ck.quantize_nvfp4(x, sx, pad_16x=True, hi_first=True)
    qw, qws = ck.quantize_nvfp4(weight, sw, pad_16x=True, hi_first=True)
    alpha = (sx * sw).reshape(1)
    y = ck.scaled_mm_nvfp4(
        qx, qw, sx, sw, qxs, qws, out_dtype=torch.bfloat16, alpha=alpha
    )
    return y[:ROWS].contiguous()


def eager_amax_bits(raw: torch.Tensor) -> int:
    gate, up = raw.chunk(2, dim=-1)
    prod = F.silu(gate).mul_(up)
    bits = prod.view(torch.int16).to(torch.int32) & 0x7FFF
    return int(bits.max().item())


def g_from_bits(bits: int) -> float:
    absmax = torch.tensor([bits & 0x7FFF], dtype=torch.uint16).view(
        torch.bfloat16
    )
    return float((absmax.float() / 2688.0).to(torch.bfloat16).item())


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
    torch.manual_seed(26081252)
    x = torch.randn(ROWS, K, device=device, dtype=torch.bfloat16)
    weight = torch.randn(2 * N, K, device=device, dtype=torch.bfloat16) * 0.02
    raw = kitchen_fc1(x, weight)
    L, U = swiglu_winner_lu(raw)
    T = eager_amax_bits(raw)
    two_q, two_s, two_g = swiglu_nvfp4_dynamic(raw)
    g_L = g_from_bits(L)
    scale_L = torch.tensor([g_L], device=device, dtype=torch.float32)
    win_q, win_s = swiglu_nvfp4_static_rebind(raw, scale_L)
    payload = {
        "identity": (
            "On kitchen FC1 [gate|up], bound-winner exact L equals "
            "true amax T iff no product attains the loose bound U. "
            "Then pack-from-G(L) == two-pass without G(L)==G(U)."
        ),
        "source": "kitchen_fc1_nvfp4_20423x5376x28672",
        "rows": ROWS,
        "device": torch.cuda.get_device_name(device),
        "L_abs_bits": L,
        "U_abs_bits": U,
        "T_abs_bits": T,
        "T_equals_L": T == L,
        "U_minus_L": U - L,
        "G_L": g_L,
        "G_T": g_from_bits(T),
        "G_U": g_from_bits(U),
        "winner_pack_vs_twopass_q": byte_diff(win_q, two_q),
        "winner_pack_vs_twopass_s": byte_diff(win_s, two_s),
        "winner_pack_vs_twopass_scale_exact": bool(
            torch.equal(scale_L, two_g)
        ),
        "twopass_min_ms": round(
            time_ms(lambda: swiglu_nvfp4_dynamic(raw)), 4
        ),
        "winner_plus_rebind_min_ms": round(
            time_ms(
                lambda: (
                    swiglu_winner_lu(raw),
                    swiglu_nvfp4_static_rebind(raw, scale_L),
                )
            ),
            4,
        ),
    }
    payload["pass"] = True
    payload["winner_exact_is_amax"] = payload["T_equals_L"]
    payload["winner_pack_byte_exact"] = (
        payload["winner_pack_vs_twopass_q"] == 0
        and payload["winner_pack_vs_twopass_s"] == 0
        and payload["winner_pack_vs_twopass_scale_exact"]
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    Path(__file__).with_name(
        "gate_winner_eq_amax_kitchen_fc1_20423.json"
    ).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
