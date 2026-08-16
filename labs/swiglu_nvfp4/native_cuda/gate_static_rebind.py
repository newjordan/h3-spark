#!/usr/bin/env python3
"""Byte-exact gate: static pack-from-scale vs dynamic two-pass + eager.

Fail-closed fingerprint. Does not patch serving.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

import comfy_kitchen as ck

from swiglu_nvfp4 import (
    load_extension,
    swiglu_nvfp4,
    swiglu_nvfp4_dynamic,
    swiglu_nvfp4_static_rebind,
    _static_rebind_fingerprint,
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


def time_ms(fn, raw: torch.Tensor, scale=None) -> float:
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    call = (lambda: fn(raw, scale)) if scale is not None else (lambda: fn(raw))
    for _ in range(2):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(REPEATS):
        starter.record()
        call()
        ender.record()
        torch.cuda.synchronize()
        samples.append(starter.elapsed_time(ender))
    return min(samples)


def fingerprint_rejects() -> dict:
    device = torch.device("cuda")
    raw = torch.zeros(2, WIDTH, device=device, dtype=torch.bfloat16)
    scale = torch.ones(1, device=device, dtype=torch.float32)
    cases = {}
    try:
        _static_rebind_fingerprint(
            torch.zeros(2, 16, device=device, dtype=torch.bfloat16), scale
        )
        cases["wrong_width"] = "accepted"
    except ValueError:
        cases["wrong_width"] = "rejected"
    try:
        _static_rebind_fingerprint(raw.cpu(), scale.cpu())
        cases["cpu"] = "accepted"
    except ValueError:
        cases["cpu"] = "rejected"
    try:
        _static_rebind_fingerprint(
            raw, torch.ones(2, device=device, dtype=torch.float32)
        )
        cases["scale_numel"] = "accepted"
    except ValueError:
        cases["scale_numel"] = "rejected"
    other = scale.clone()
    try:
        _static_rebind_fingerprint(raw, other, certified_scale=scale)
        cases["scale_identity"] = "accepted"
    except ValueError:
        cases["scale_identity"] = "rejected"
    return cases


def main() -> int:
    load_extension(verbose=False)
    device = torch.device("cuda")
    torch.manual_seed(26081226)
    raw = torch.randn(ROWS, WIDTH, device=device, dtype=torch.bfloat16)
    two_q, two_s, two_g = swiglu_nvfp4_dynamic(raw)
    st_q, st_s = swiglu_nvfp4_static_rebind(
        raw, two_g, certified_scale=two_g
    )
    old_q, old_s = swiglu_nvfp4(raw, two_g)
    ref_q, ref_s, ref_g = eager_dynamic(raw)
    rejects = fingerprint_rejects()
    payload = {
        "identity": (
            "static 8-wide pack-from-supplied-scale skips amax; "
            "byte-identical iff G is the live (amax/2688).to(bf16).to(fp32); "
            "fingerprint is shape/dtype/device/identity"
        ),
        "rows": ROWS,
        "width": WIDTH,
        "device": torch.cuda.get_device_name(device),
        "static_vs_twopass_q": byte_diff(st_q, two_q),
        "static_vs_twopass_s": byte_diff(st_s, two_s),
        "static_vs_twopass_scale_exact": bool(torch.equal(two_g, ref_g)),
        "static_vs_eager_q": byte_diff(st_q, ref_q),
        "static_vs_eager_s": byte_diff(st_s, ref_s),
        "static_vs_4wide_q": byte_diff(st_q, old_q),
        "static_vs_4wide_s": byte_diff(st_s, old_s),
        "fingerprint_rejects": rejects,
        "twopass_min_ms": round(time_ms(swiglu_nvfp4_dynamic, raw), 4),
        "static_min_ms": round(
            time_ms(swiglu_nvfp4_static_rebind, raw, two_g), 4
        ),
        "legacy4_min_ms": round(time_ms(swiglu_nvfp4, raw, two_g), 4),
    }
    payload["pass"] = (
        payload["static_vs_twopass_q"] == 0
        and payload["static_vs_twopass_s"] == 0
        and payload["static_vs_eager_q"] == 0
        and payload["static_vs_eager_s"] == 0
        and payload["static_vs_4wide_q"] == 0
        and payload["static_vs_4wide_s"] == 0
        and payload["static_vs_twopass_scale_exact"]
        and all(value == "rejected" for value in rejects.values())
    )
    if payload["pass"]:
        payload["saved_ms"] = round(
            payload["twopass_min_ms"] - payload["static_min_ms"], 4
        )
        payload["projected_s_per_20_step"] = round(
            payload["saved_ms"] * 50 * 20 / 1000.0, 4
        )
    text = json.dumps(payload, indent=2, sort_keys=True)
    out = Path(__file__).with_name("gate_static_rebind_20423.json")
    out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
