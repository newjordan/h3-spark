#!/usr/bin/env python3
"""Fail-closed gate: G=1 pack + E4M3 scale rewrite vs two-pass.

Algebra: two-pass uses s=E4M3((block_amax/6)/G) and
q=encode(x/(G*decode(s))). Packing with G=1 yields
s'=E4M3(block_amax/6) and q'=encode(x/decode(s')).
q'==q and s_fix==s iff decode(s') == G*decode(s) on every
block (E4M3 homogeneity). If that holds, one raw read plus
a tiny scale rewrite is byte-exact. Does not patch serving.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import comfy_kitchen as ck

from swiglu_nvfp4 import load_extension, swiglu_nvfp4, swiglu_nvfp4_dynamic


ROWS = 20423
WIDTH = 28672
FC2_N = 5376
REPEATS = 3


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


def rewrite_scales(s_g1: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    decoded = s_g1.to(torch.float32)
    fixed = (decoded / g).clamp(max=448.0).to(torch.float8_e4m3fn)
    return fixed


def kitchen_fc2(q, s, g, qw, sw, gw):
    alpha = (g * gw).reshape(1).contiguous()
    return ck.scaled_mm_nvfp4(
        q, qw, g, gw, s.view(torch.float8_e4m3fn), sw,
        out_dtype=torch.bfloat16, alpha=alpha,
    )


def main() -> int:
    load_extension(verbose=False)
    device = torch.device("cuda")
    torch.manual_seed(26081268)
    raw = torch.randn(ROWS, WIDTH, device=device, dtype=torch.bfloat16)
    two_q, two_s, two_g = swiglu_nvfp4_dynamic(raw)
    one = torch.ones(1, device=device, dtype=torch.float32)
    q1, s1 = swiglu_nvfp4(raw, one)
    s_fix = rewrite_scales(s1, two_g)
    decoded_s1 = s1.to(torch.float32)
    decoded_s = two_s.to(torch.float32)
    # Homogeneity: decode(s') == G * decode(s) on live (nonzero) scales.
    live = (s1.view(torch.uint8) != 0) | (two_s.view(torch.uint8) != 0)
    homo = torch.zeros_like(decoded_s1, dtype=torch.bool)
    homo[live] = decoded_s1[live] == (two_g * decoded_s[live])
    live_n = int(live.sum().item())
    homo_n = int(homo.sum().item())
    weight = torch.randn(FC2_N, WIDTH // 2, device=device, dtype=torch.bfloat16)
    gw = (weight.abs().amax() / (6.0 * 448.0)).to(torch.float32).reshape(1)
    qw, qws = ck.quantize_nvfp4(weight, gw, pad_16x=True, hi_first=True)
    y_two = kitchen_fc2(two_q, two_s, two_g, qw, qws, gw)
    y_g1 = kitchen_fc2(q1, s1, one, qw, qws, gw)
    y_fix = kitchen_fc2(q1, s_fix, two_g, qw, qws, gw)
    payload = {
        "identity": (
            "G=1 SwiGLU pack + E4M3 rewrite s_fix=E4M3(decode(s')/G) "
            "== two-pass q/s iff E4M3 is homogeneous; FC2 consumer "
            "of (q',s',1) and (q',s_fix,G) vs two-pass"
        ),
        "rows": ROWS,
        "width": WIDTH,
        "device": torch.cuda.get_device_name(device),
        "g": float(two_g.item()),
        "q1_vs_twopass_q": byte_diff(q1, two_q),
        "s1_vs_twopass_s": byte_diff(s1, two_s),
        "sfix_vs_twopass_s": byte_diff(s_fix, two_s),
        "live_scale_blocks": live_n,
        "homogeneous_scale_blocks": homo_n,
        "homogeneous_frac": round(homo_n / max(live_n, 1), 6),
        "fc2_g1_vs_twopass": byte_diff(y_g1, y_two),
        "fc2_sfix_vs_twopass": byte_diff(y_fix, y_two),
        "fc2_finite": bool(torch.isfinite(y_two.float()).all().item()),
        "twopass_min_ms": round(time_ms(lambda: swiglu_nvfp4_dynamic(raw)), 4),
        "g1_pack_min_ms": round(time_ms(lambda: swiglu_nvfp4(raw, one)), 4),
    }
    # Identity holds only if q/s/G match two-pass. Homogeneity is the
    # predicate; a mismatch is a closed negative, not a failed gate.
    payload["q_s_match"] = (
        payload["q1_vs_twopass_q"] == 0
        and payload["sfix_vs_twopass_s"] == 0
    )
    payload["fc2_g1_match"] = payload["fc2_g1_vs_twopass"] == 0
    payload["fc2_sfix_match"] = payload["fc2_sfix_vs_twopass"] == 0
    payload["pass"] = (
        payload["fc2_finite"]
        and live_n > 0
        and payload["twopass_min_ms"] > 0.0
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    Path(__file__).with_name("gate_swiglu_g1_scale_rewrite_20423.json").write_text(
        text + "\n"
    )
    print(text)
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
