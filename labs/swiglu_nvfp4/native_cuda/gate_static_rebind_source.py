#!/usr/bin/env python3
"""Byte-exact gate: live_amax source bind vs shape-only rebind.

Shape-only fingerprint accepts a foreign G (another FC1 draw).
source='live_amax' binds the two-pass scale object to this raw's
storage. Not a margin. Does not patch serving.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import comfy_kitchen as ck

from swiglu_nvfp4 import (
    clear_live_amax_certs,
    load_extension,
    swiglu_nvfp4_dynamic,
    swiglu_nvfp4_static_rebind,
)


ROWS = 20423
K = 5376
N = 14336


def nvfp4_scale(x: torch.Tensor) -> torch.Tensor:
    return (x.abs().amax() / (6.0 * 448.0)).to(torch.float32).reshape(1)


def kitchen_fc1(x, weight) -> torch.Tensor:
    sx = nvfp4_scale(x)
    sw = nvfp4_scale(weight)
    qx, qxs = ck.quantize_nvfp4(x, sx, pad_16x=True, hi_first=True)
    qw, qws = ck.quantize_nvfp4(weight, sw, pad_16x=True, hi_first=True)
    alpha = (sx * sw).reshape(1)
    y = ck.scaled_mm_nvfp4(
        qx, qw, sx, sw, qxs, qws, out_dtype=torch.bfloat16, alpha=alpha
    )
    return y[:ROWS].contiguous()


def byte_diff(left: torch.Tensor, right: torch.Tensor) -> int:
    return int((left.view(torch.uint8) != right.view(torch.uint8)).sum().item())


def reject_reason(fn) -> str:
    try:
        fn()
        return "accepted"
    except ValueError as exc:
        return f"rejected:{exc}"


def main() -> int:
    load_extension(verbose=False)
    clear_live_amax_certs()
    device = torch.device("cuda")
    torch.manual_seed(26081257)
    x_a = torch.randn(ROWS, K, device=device, dtype=torch.bfloat16)
    x_b = torch.randn(ROWS, K, device=device, dtype=torch.bfloat16)
    weight = torch.randn(2 * N, K, device=device, dtype=torch.bfloat16) * 0.02
    raw_a = kitchen_fc1(x_a, weight)
    raw_b = kitchen_fc1(x_b, weight)
    two_qa, two_sa, g_a = swiglu_nvfp4_dynamic(raw_a)
    two_qb, two_sb, g_b = swiglu_nvfp4_dynamic(raw_b)
    g_a_clone = g_a.clone()
    sourced_a = swiglu_nvfp4_static_rebind(
        raw_a, g_a, certified_scale=g_a, source="live_amax"
    )
    shape_only_foreign = swiglu_nvfp4_static_rebind(raw_b, g_a)
    foreign_q, foreign_s = shape_only_foreign
    rejects = {
        "unknown_source": reject_reason(
            lambda: swiglu_nvfp4_static_rebind(
                raw_a, g_a, certified_scale=g_a, source="plan_v3"
            )
        ),
        "missing_certified": reject_reason(
            lambda: swiglu_nvfp4_static_rebind(
                raw_a, g_a, source="live_amax"
            )
        ),
        "clone_not_object": reject_reason(
            lambda: swiglu_nvfp4_static_rebind(
                raw_a, g_a_clone, certified_scale=g_a_clone, source="live_amax"
            )
        ),
        "foreign_draw": reject_reason(
            lambda: swiglu_nvfp4_static_rebind(
                raw_b, g_a, certified_scale=g_a, source="live_amax"
            )
        ),
        "wrong_certified_object": reject_reason(
            lambda: swiglu_nvfp4_static_rebind(
                raw_a, g_a, certified_scale=g_b, source="live_amax"
            )
        ),
    }
    payload = {
        "identity": (
            "live_amax source binds the two-pass G object to this raw "
            "storage. Shape-only rebind accepts a foreign kitchen-FC1 G; "
            "that pack is not two-pass. Sourced rebind == two-pass and "
            "rejects unknown source / clone / other draw."
        ),
        "rows": ROWS,
        "k": K,
        "n": 2 * N,
        "device": torch.cuda.get_device_name(device),
        "g_a_equals_g_b": bool(torch.equal(g_a, g_b)),
        "g_a": float(g_a.item()),
        "g_b": float(g_b.item()),
        "sourced_vs_twopass_q": byte_diff(sourced_a[0], two_qa),
        "sourced_vs_twopass_s": byte_diff(sourced_a[1], two_sa),
        "foreign_shapeonly_vs_twopass_q": byte_diff(foreign_q, two_qb),
        "foreign_shapeonly_vs_twopass_s": byte_diff(foreign_s, two_sb),
        "fingerprint_rejects": {
            key: ("rejected" if value.startswith("rejected") else value)
            for key, value in rejects.items()
        },
        "fingerprint_reject_details": rejects,
    }
    payload["pass"] = (
        payload["sourced_vs_twopass_q"] == 0
        and payload["sourced_vs_twopass_s"] == 0
        and payload["foreign_shapeonly_vs_twopass_q"] != 0
        and not payload["g_a_equals_g_b"]
        and all(
            value == "rejected"
            for value in payload["fingerprint_rejects"].values()
        )
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    Path(__file__).with_name(
        "gate_static_rebind_source_20423.json"
    ).write_text(text + "\n")
    print(text)
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
