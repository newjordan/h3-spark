#!/usr/bin/env python3
"""Algebra gate: same T>G*2688 pre-clamp on QKV/out/FC1 bf16_nvfp4.

SwiGLU sat cert closed the 28672 [gate|up] pack. This is the direct BF16
pack used by QKV (5376), attn-out (7168), FC1 input (5376), and FC2
(14336). 28672 is FC1 out_features / SwiGLU raw — not an H3 bf16_nvfp4
width.

The pack already computes per-16 absmax and clamps decode_scale =
absmax/6/G at 448. Pre-clamp > 448 is T > G*2688. Two-pass G is
bf16(T/2688) and can self-ratio > 1; that is the association, not a
fallback. Validator overflow is T versus a *calibrated* G. Layer
quarantine of 10/11/22/46 is the wrong object (1/20 spike calls).

Algebra only. Fail-closed. Does not patch serving. Not promoted.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from swiglu_nvfp4 import (
    H3_NVFP4_WIDTHS,
    bf16_nvfp4,
    bf16_nvfp4_dynamic,
    load_extension,
)


HERE = Path(__file__).resolve().parent
ROWS = 20423
DENOM = 6.0 * 448.0
# Ticket listed 28672; that is FC1 out_features / SwiGLU raw, already
# receipted. Legal bf16_nvfp4 widths are the three consumers below.
WIDTHS = (5376, 7168, 14336)
ROLE = {
    5376: "qkv_and_fc1_input",
    7168: "attn_out_input",
    14336: "fc2_input_swiglu_product",
}
CHEF_QKV10_T = 24.875
CHEF_QKV10_THRESHOLD = 24.740625500679016
CHEF_RATIO = CHEF_QKV10_T / CHEF_QKV10_THRESHOLD


def byte_diff(left: torch.Tensor, right: torch.Tensor) -> int:
    return int((left.view(torch.uint8) != right.view(torch.uint8)).sum().item())


def block_absmax(x: torch.Tensor) -> torch.Tensor:
    rows, width = x.shape
    return x.abs().reshape(rows, width // 16, 16).amax(dim=-1)


def case_for_g(
    x: torch.Tensor,
    t: float,
    g: torch.Tensor,
    *,
    expect_sat: int | None = None,
) -> dict:
    # Pack clamp is FP32 divide: decode = absmax/6/G, saturate if > 448.
    # Validator overflow is the multiply form T/(G*2688) > 1 in Python
    # float. Over reals they are the same identity. At 1 ULP of
    # G=fp32(T/2688) they can disagree; the pack is the divide.
    g_f = float(g.item())
    g_dev = g.to(dtype=torch.float32)
    blocks = block_absmax(x).float()
    t_from_blocks = float(blocks.amax().item())
    decode = torch.div(torch.div(blocks, 6.0), g_dev)
    sat_pack = bool((decode > 448.0).any().item())
    decode_max = float(decode.amax().item())
    ratio = t / (g_f * DENOM)
    sat_validator = ratio > 1.0
    if expect_sat is None:
        expect_sat = int(sat_pack)
    return {
        "g": g_f,
        "ratio": ratio,
        "decode_unclamped_max": decode_max,
        "t_equals_max_block": t_from_blocks == t,
        "sat_pack_divide": int(sat_pack),
        "sat_validator_multiply": int(sat_validator),
        "expect_sat": int(expect_sat),
        "pack_matches_expect": int(sat_pack) == int(expect_sat),
        "multiply_divide_agree": int(sat_pack) == int(sat_validator),
    }


def run_width(width: int, device: torch.device) -> dict:
    torch.manual_seed(26081301 + width)
    x = torch.randn(ROWS, width, device=device, dtype=torch.bfloat16)
    two_q, two_s, two_g = bf16_nvfp4_dynamic(x)
    live_t = float(x.abs().amax().float().item())
    two_g_from_t = (
        torch.tensor([live_t / DENOM], device=device, dtype=torch.float32)
        .to(torch.bfloat16)
        .float()
    )
    g_is_bf16_t = float(two_g.item()) == float(two_g_from_t.item())

    stock_q, stock_s = bf16_nvfp4(x, two_g)
    live_bytes = (
        byte_diff(stock_q[:ROWS], two_q[:ROWS]) == 0
        and byte_diff(stock_s, two_s) == 0
    )

    live = case_for_g(x, live_t, two_g)
    # FixedRef 1.005 and the 0.416 median are far from 1 ULP: both forms
    # must fire / not fire. Exact G=fp32(T/2688) is the 1-ULP seam.
    spike_g = (two_g / CHEF_RATIO).contiguous()
    spike = case_for_g(x, live_t, spike_g, expect_sat=1)

    safe_g = (two_g / 0.416).contiguous()
    safe = case_for_g(x, live_t, safe_g, expect_sat=0)

    exact_g = (
        x.abs().amax().float() / DENOM
    ).reshape(1).contiguous()
    exact = case_for_g(x, live_t, exact_g)
    exact_is_one = abs(exact["ratio"] - 1.0) < 1e-6

    margin_g = (two_g * 1.30).contiguous()
    margin = case_for_g(x, live_t, margin_g, expect_sat=0)
    margin_q, _margin_s = bf16_nvfp4(x, margin_g)
    margin_foreign = byte_diff(margin_q[:ROWS], two_q[:ROWS]) > 0

    far_from_seam = all(
        case["pack_matches_expect"]
        and case["multiply_divide_agree"]
        and case["t_equals_max_block"]
        for case in (live, spike, safe, margin)
    )
    exact_ok = (
        exact["t_equals_max_block"]
        and exact["pack_matches_expect"]
        and exact_is_one
    )
    return {
        "width": width,
        "role": ROLE[width],
        "live_t": live_t,
        "live_g": float(two_g.item()),
        "live_g_is_bf16_t_over_2688": g_is_bf16_t,
        "live_bytes_exact": live_bytes,
        "exact_ratio_is_one": exact_is_one,
        "margin_is_foreign_g": margin_foreign,
        "margin_vs_twopass_q": byte_diff(margin_q[:ROWS], two_q[:ROWS]),
        "cases": {
            "live_g": live,
            "chef_qkv10_1p005": {
                **spike,
                "chef_heldout_ratio": CHEF_RATIO,
            },
            "median_safe_0p416": safe,
            "exact_ratio_1": exact,
            "margin_1p30_foreign": margin,
        },
        "multiply_divide_1ulp_at_exact_g": (
            not exact["multiply_divide_agree"]
        ),
        "pass": bool(
            far_from_seam
            and exact_ok
            and g_is_bf16_t
            and live_bytes
            and margin_foreign
        ),
    }


def main() -> int:
    load_extension(verbose=False)
    device = torch.device("cuda")

    rejected_28672 = False
    reject_msg = ""
    try:
        bad = torch.randn(16, 28672, device=device, dtype=torch.bfloat16)
        bf16_nvfp4_dynamic(bad)
    except (ValueError, RuntimeError) as exc:
        rejected_28672 = True
        reject_msg = str(exc)
        if len(reject_msg) > 240:
            reject_msg = reject_msg[:240]

    widths = [run_width(w, device) for w in WIDTHS]
    algebra_holds = all(w["pass"] for w in widths) and rejected_28672
    # Same predicate as the validator (T > G_cal*2688). That is per-call,
    # so a named layer quarantine is the wrong object. Live two-pass G
    # self-saturating is the BF16 association, not a fallback signal.
    # Not hooked. Not promoted.
    payload = {
        "identity": (
            "bf16_nvfp4 (QKV 5376 / out 7168 / FC1-in 5376 / FC2 14336) "
            "uses the same pre-clamp as SwiGLU sat cert. Pack saturate is "
            "FP32 decode=absmax/6/G > 448. Validator overflow is the "
            "multiply T/(G*2688)>1. Over reals they are T>G*2688. At "
            "G=fp32(T/2688) they can 1-ULP disagree (14336 this draw). "
            "FixedRef 1.005 / median 0.416 agree. Two-pass G is bf16(T/2688) "
            "and can self-ratio > 1; that is the association, not a "
            "fallback. The four exclusions are 1/20 spike calls vs a "
            "calibrated G, not layers. 28672 is not a bf16_nvfp4 width "
            "(FC1 out / SwiGLU raw; sat cert). Algebra only. Not a "
            "margin raise. Per-call fallback not hooked. Not promoted."
        ),
        "device": torch.cuda.get_device_name(0),
        "rows": ROWS,
        "legal_widths": list(sorted(H3_NVFP4_WIDTHS)),
        "ticket_width_28672": {
            "is_bf16_nvfp4_width": 28672 in H3_NVFP4_WIDTHS,
            "rejected_by_pack": rejected_28672,
            "reject": reject_msg,
            "actual_object": "FC1 out_features / SwiGLU raw; see gate_sat_cert_20423.json",
        },
        "widths": widths,
        "exclusion_algebra": {
            "object_wrong": "per-layer max scale + named quarantine",
            "object_right": (
                "per-call pack decode=absmax/6/G > 448 vs calibrated G"
            ),
            "live_g_is_not_fallback": (
                "two-pass bf16(T/2688) can self-ratio > 1; saturate on "
                "live G is the stock association, not 'fall back'"
            ),
            "multiply_vs_divide": (
                "validator T/(G*2688)>1 and pack absmax/6/G>448 agree "
                "on 1.005/0.416/live/margin; they can 1-ULP disagree "
                "at G=fp32(T/2688). Fallback must name the divide form."
            ),
            "m130_after_fc2_quarantine": {
                "blocks.10.attn.qkv_proj": {
                    "width": 5376,
                    "max": 1.0054313299119013,
                    "median": 0.7250220896600895,
                    "overflow_calls": 1,
                    "calls": 20,
                },
                "blocks.11.attn.qkv_proj": {
                    "width": 5376,
                    "max": 1.0033699633699633,
                    "median": 0.41611721611721614,
                    "overflow_calls": 1,
                    "calls": 20,
                },
            },
            "two_seed_unexcluded": {
                "blocks.49.attn.qkv_proj": 1.0721414640767217,
                "blocks.48.mlp.fc2": 1.0374150130985875,
            },
            "do_not": "raise every scale; spikes migrate; do not hook fallback",
        },
        "promoted": False,
        "serving_default": "stock",
        "one_ulp_multiply_divide_at_exact_g": {
            str(w["width"]): w["multiply_divide_1ulp_at_exact_g"]
            for w in widths
        },
        "pass": algebra_holds,
    }
    out = HERE / "gate_bf16_sat_cert_20423.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "pass": payload["pass"],
                "promoted": False,
                "receipt": str(out),
                "widths_pass": {w["width"]: w["pass"] for w in widths},
                "rejected_28672": rejected_28672,
            },
            indent=2,
        )
    )
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
