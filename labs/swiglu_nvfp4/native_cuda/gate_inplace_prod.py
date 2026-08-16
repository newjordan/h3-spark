#!/usr/bin/env python3
"""Byte-exact gate: in-place product-over-up amax + half-width pack.

Mutates a copy of raw. Does not patch serving.
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
    swiglu_nvfp4_dynamic_inplace_prod,
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


def time_ms(fn, raw: torch.Tensor) -> float:
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    for _ in range(2):
        fn(raw.clone())
    torch.cuda.synchronize()
    samples = []
    for _ in range(REPEATS):
        work = raw.clone()
        starter.record()
        fn(work)
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
    prod_q, prod_s, prod_g = swiglu_nvfp4_dynamic_inplace_prod(raw.clone())
    ref_q, ref_s, ref_g = eager_dynamic(raw)
    payload = {
        "identity": (
            "amax writes eager product over the dead up half; "
            "pack is half-width NVFP4 of that product; no SiLU, no new "
            "558 MiB workspace"
        ),
        "rows": ROWS,
        "width": WIDTH,
        "device": torch.cuda.get_device_name(device),
        "prod_vs_twopass_q": byte_diff(prod_q, two_q),
        "prod_vs_twopass_s": byte_diff(prod_s, two_s),
        "prod_vs_twopass_scale_exact": bool(torch.equal(prod_g, two_g)),
        "prod_vs_eager_q": byte_diff(prod_q, ref_q),
        "prod_vs_eager_s": byte_diff(prod_s, ref_s),
        "prod_vs_eager_scale_exact": bool(torch.equal(prod_g, ref_g)),
        "twopass_min_ms": round(time_ms(swiglu_nvfp4_dynamic, raw), 4),
        "vec_min_ms": round(time_ms(swiglu_nvfp4_dynamic_vec, raw), 4),
        "prod_min_ms": round(
            time_ms(swiglu_nvfp4_dynamic_inplace_prod, raw), 4
        ),
    }
    payload["pass"] = (
        payload["prod_vs_twopass_q"] == 0
        and payload["prod_vs_twopass_s"] == 0
        and payload["prod_vs_twopass_scale_exact"]
        and payload["prod_vs_eager_q"] == 0
        and payload["prod_vs_eager_s"] == 0
        and payload["prod_vs_eager_scale_exact"]
    )
    if payload["pass"]:
        payload["saved_ms"] = round(
            payload["twopass_min_ms"] - payload["prod_min_ms"], 4
        )
        payload["projected_s_per_20_step"] = round(
            payload["saved_ms"] * 50 * 20 / 1000.0, 4
        )
    text = json.dumps(payload, indent=2, sort_keys=True)
    Path(__file__).with_name("gate_inplace_prod_20423.json").write_text(
        text + "\n"
    )
    print(text)
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
