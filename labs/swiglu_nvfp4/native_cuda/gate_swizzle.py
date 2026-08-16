#!/usr/bin/env python3
"""Byte-exact gate: 128x64 swizzle-tile amax+pack vs two-pass and kitchen.

Does not patch serving.
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
    swiglu_nvfp4_dynamic_swizzle,
    swiglu_nvfp4_dynamic_vec,
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


def time_ms(fn) -> float:
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(REPEATS):
        starter.record()
        fn()
        ender.record()
        torch.cuda.synchronize()
        samples.append(starter.elapsed_time(ender))
    return min(samples)


def main() -> int:
    load_extension(verbose=False)
    device = torch.device("cuda")
    torch.manual_seed(26081230)
    raw = torch.randn(ROWS, WIDTH, device=device, dtype=torch.bfloat16)
    two_q, two_s, two_g = swiglu_nvfp4_dynamic(raw)
    sw_q, sw_s, sw_g = swiglu_nvfp4_dynamic_swizzle(raw)
    ref_q, ref_s, ref_g = eager_dynamic(raw)
    payload = {
        "identity": (
            "amax+pack tiled to the cuBLAS 128x4 E4M3 slab; "
            "same eager SwiGLU and NVFP4 algebra as the 8-wide two-pass"
        ),
        "rows": ROWS,
        "width": WIDTH,
        "device": torch.cuda.get_device_name(device),
        "swizzle_vs_twopass_q": byte_diff(sw_q, two_q),
        "swizzle_vs_twopass_s": byte_diff(sw_s, two_s),
        "swizzle_vs_twopass_scale_exact": bool(torch.equal(sw_g, two_g)),
        "swizzle_vs_eager_q": byte_diff(sw_q, ref_q),
        "swizzle_vs_eager_s": byte_diff(sw_s, ref_s),
        "swizzle_vs_eager_scale_exact": bool(torch.equal(sw_g, ref_g)),
        "twopass_min_ms": round(time_ms(lambda: swiglu_nvfp4_dynamic(raw)), 4),
        "vec_min_ms": round(time_ms(lambda: swiglu_nvfp4_dynamic_vec(raw)), 4),
        "swizzle_min_ms": round(
            time_ms(lambda: swiglu_nvfp4_dynamic_swizzle(raw)), 4
        ),
    }
    payload["pass"] = (
        payload["swizzle_vs_twopass_q"] == 0
        and payload["swizzle_vs_twopass_s"] == 0
        and payload["swizzle_vs_twopass_scale_exact"]
        and payload["swizzle_vs_eager_q"] == 0
        and payload["swizzle_vs_eager_s"] == 0
        and payload["swizzle_vs_eager_scale_exact"]
    )
    if payload["pass"]:
        payload["saved_ms"] = round(
            payload["twopass_min_ms"] - payload["swizzle_min_ms"], 4
        )
        payload["saved_vs_vec_ms"] = round(
            payload["vec_min_ms"] - payload["swizzle_min_ms"], 4
        )
    text = json.dumps(payload, indent=2, sort_keys=True)
    Path(__file__).with_name("gate_swizzle_20423.json").write_text(text + "\n")
    print(text)
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
