#!/usr/bin/env python3
"""Byte-exact gate: 16-wide two-pass == 8-wide two-pass.

One thread owns one NVFP4 scale block. Block amax is thread-local;
no 2-thread shuffle. Same eager SiLU and (amax/2688).to(bf16).to(fp32).
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
    swiglu_nvfp4_dynamic_vec,
    swiglu_nvfp4_dynamic_vec16,
)


ROWS = 20423
WIDTH = 28672


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
    torch.manual_seed(26081262)
    raw = torch.randn(ROWS, WIDTH, device=device, dtype=torch.bfloat16)
    two_q, two_s, two_g = swiglu_nvfp4_dynamic(raw)
    v8_q, v8_s, v8_g = swiglu_nvfp4_dynamic_vec(raw)
    v16_q, v16_s, v16_g = swiglu_nvfp4_dynamic_vec16(raw)
    ref_q, ref_s, ref_g = eager_dynamic(raw)
    payload = {
        "identity": (
            "16-wide two-pass: one thread owns one 16-value NVFP4 "
            "scale block. Block amax is thread-local. == 8-wide and "
            "eager two-pass (q/s/G)."
        ),
        "rows": ROWS,
        "width": WIDTH,
        "device": torch.cuda.get_device_name(device),
        "v16_vs_twopass_q": byte_diff(v16_q, two_q),
        "v16_vs_twopass_s": byte_diff(v16_s, two_s),
        "v16_vs_twopass_scale_exact": bool(torch.equal(v16_g, two_g)),
        "v16_vs_vec8_q": byte_diff(v16_q, v8_q),
        "v16_vs_vec8_s": byte_diff(v16_s, v8_s),
        "v16_vs_eager_q": byte_diff(v16_q, ref_q),
        "v16_vs_eager_s": byte_diff(v16_s, ref_s),
        "v16_vs_eager_scale_exact": bool(torch.equal(v16_g, ref_g)),
        "twopass_min_ms": round(time_ms(lambda: swiglu_nvfp4_dynamic(raw)), 4),
        "vec8_min_ms": round(
            time_ms(lambda: swiglu_nvfp4_dynamic_vec(raw)), 4
        ),
        "vec16_min_ms": round(
            time_ms(lambda: swiglu_nvfp4_dynamic_vec16(raw)), 4
        ),
    }
    payload["pass"] = (
        payload["v16_vs_twopass_q"] == 0
        and payload["v16_vs_twopass_s"] == 0
        and payload["v16_vs_twopass_scale_exact"]
        and payload["v16_vs_vec8_q"] == 0
        and payload["v16_vs_vec8_s"] == 0
        and payload["v16_vs_eager_q"] == 0
        and payload["v16_vs_eager_s"] == 0
        and payload["v16_vs_eager_scale_exact"]
    )
    if payload["pass"]:
        payload["vs_vec8_ms"] = round(
            payload["vec8_min_ms"] - payload["vec16_min_ms"], 4
        )
        payload["vs_twopass_ms"] = round(
            payload["twopass_min_ms"] - payload["vec16_min_ms"], 4
        )
    text = json.dumps(payload, indent=2, sort_keys=True)
    Path(__file__).with_name("gate_vec16_20423.json").write_text(text + "\n")
    print(text)
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
