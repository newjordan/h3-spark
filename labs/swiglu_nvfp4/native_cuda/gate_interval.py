#!/usr/bin/env python3
"""Byte-exact gate: G(L)==G(U) interval skip vs two-pass + eager.

Does not patch serving. Rebuilds the local lab extension on first import.
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
    swiglu_nvfp4_dynamic_interval,
    swiglu_nvfp4_dynamic_interval_stats,
)


ROWS = 20423
WIDTH = 28672
REPEATS = 3


def eager_dynamic(raw: torch.Tensor):
    gate, up = raw.chunk(2, dim=-1)
    activated = F.silu(gate).mul_(up)
    scale = (activated.abs().amax() / (6.0 * 448.0)).to(torch.float32).reshape(1)
    packed, block_scales = ck.quantize_nvfp4(
        activated, scale, pad_16x=True, hi_first=True
    )
    return packed, block_scales, scale


def byte_diff(left: torch.Tensor, right: torch.Tensor) -> int:
    return int((left.view(torch.uint8) != right.view(torch.uint8)).sum().item())


def scale_from_abs_bits(bits: int) -> float:
    absmax = torch.tensor([bits & 0x7FFF], dtype=torch.uint16).view(torch.bfloat16)
    return float((absmax.float() / 2688.0).to(torch.bfloat16).item())


def time_ms(fn, raw: torch.Tensor) -> float:
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    for _ in range(2):
        fn(raw)
    torch.cuda.synchronize()
    samples = []
    for _ in range(REPEATS):
        starter.record()
        fn(raw)
        ender.record()
        torch.cuda.synchronize()
        samples.append(starter.elapsed_time(ender))
    return min(samples)


def main() -> int:
    load_extension(verbose=False)
    device = torch.device("cuda")
    torch.manual_seed(26081226)
    raw = torch.randn(ROWS, WIDTH, device=device, dtype=torch.bfloat16)
    two_q, two_s, two_g = swiglu_nvfp4_dynamic(raw)
    lvl_q, lvl_s, lvl_g, sparse_exact, scanned, lo, hi = (
        swiglu_nvfp4_dynamic_interval_stats(raw)
    )
    ref_q, ref_s, ref_g = eager_dynamic(raw)
    g_lo = scale_from_abs_bits(lo)
    g_hi = scale_from_abs_bits(hi)
    payload = {
        "identity": (
            "G(x)=(bf16(x)/2688).to(bf16) constant on [L,U] skips sparse "
            "HBM amax scan; pack from amax bits; no 558 MiB store"
        ),
        "rows": ROWS,
        "width": WIDTH,
        "L_abs_bits": lo,
        "U_abs_bits": hi,
        "G_L": g_lo,
        "G_U": g_hi,
        "scale_interval_equal": g_lo == g_hi,
        "sparse_scanned": scanned,
        "sparse_exact_evals": sparse_exact,
        "device": torch.cuda.get_device_name(device),
        "interval_vs_twopass_q": byte_diff(lvl_q, two_q),
        "interval_vs_twopass_s": byte_diff(lvl_s, two_s),
        "interval_vs_twopass_scale_exact": bool(torch.equal(lvl_g, two_g)),
        "interval_vs_eager_q": byte_diff(lvl_q, ref_q),
        "interval_vs_eager_s": byte_diff(lvl_s, ref_s),
        "interval_vs_eager_scale_exact": bool(torch.equal(lvl_g, ref_g)),
        "twopass_min_ms": round(time_ms(swiglu_nvfp4_dynamic, raw), 4),
        "interval_min_ms": round(time_ms(swiglu_nvfp4_dynamic_interval, raw), 4),
    }
    payload["pass"] = (
        payload["interval_vs_twopass_q"] == 0
        and payload["interval_vs_twopass_s"] == 0
        and payload["interval_vs_twopass_scale_exact"]
        and payload["interval_vs_eager_q"] == 0
        and payload["interval_vs_eager_s"] == 0
        and payload["interval_vs_eager_scale_exact"]
    )
    if payload["pass"]:
        payload["saved_ms"] = round(
            payload["twopass_min_ms"] - payload["interval_min_ms"], 4
        )
        payload["projected_s_per_20_step"] = round(
            payload["saved_ms"] * 50 * 20 / 1000.0, 4
        )
    text = json.dumps(payload, indent=2, sort_keys=True)
    out = Path(__file__).with_name("gate_interval_20423.json")
    out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
