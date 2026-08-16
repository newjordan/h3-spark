#!/usr/bin/env python3
"""Algebra gate: NVFP4 overflow is per-call scale saturation, not a layer.

Held-out static failures migrate by seed (fc2 22/46, then qkv 10/11, then
qkv 49 / fc2 48). Each is a single-step spike just over G*2688. The static
pack already computes per-block absmax and clamps decode_scale at 448.
That pre-clamp test is the exact validator predicate T > G*2688.

Not a margin raise. Not a tile knob. Does not patch serving.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from swiglu_nvfp4 import (
    load_extension,
    swiglu_nvfp4_dynamic,
    swiglu_nvfp4_static_rebind,
)


HERE = Path(__file__).resolve().parent
ROWS = 20423
WIDTH = 28672
DENOM = 6.0 * 448.0
REPEATS = 3
# FixedRef held-out qkv.10 spike: observed 24.875 vs threshold 24.7406255.
CHEF_QKV10_T = 24.875
CHEF_QKV10_THRESHOLD = 24.740625500679016


def load_sat():
    import importlib.util

    so = HERE / "cmake-build-sat" / "h3_swiglu_sat_cert.so"
    if not so.is_file():
        raise FileNotFoundError(
            f"missing {so}; build with cmake --target h3_swiglu_sat_cert"
        )
    spec = importlib.util.spec_from_file_location("h3_swiglu_sat_cert", so)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {so}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def eager_product(raw: torch.Tensor) -> torch.Tensor:
    gate, up = raw.chunk(2, dim=-1)
    return F.silu(gate).mul_(up)


def main() -> int:
    load_extension(verbose=False)
    sat = load_sat()
    device = torch.device("cuda")
    torch.manual_seed(26081301)
    raw = torch.randn(ROWS, WIDTH, device=device, dtype=torch.bfloat16)

    two_q, two_s, two_g = swiglu_nvfp4_dynamic(raw)
    product = eager_product(raw)
    live_t = float(product.abs().amax().float().item())
    live_threshold = float(two_g.item()) * DENOM
    live_ratio = live_t / live_threshold
    # Stock G is bf16(T/2688). That rounding can sit under T/2688, so
    # T > G*2688 on the two-pass path itself. The pack still matches
    # two-pass because both use that G; the clamp is in the association.
    two_g_from_t = torch.tensor(
        [live_t / DENOM], device=device, dtype=torch.float32
    ).to(torch.bfloat16).float()
    two_pass_g_is_bf16_t = (
        float(two_g.item()) == float(two_g_from_t.item())
    )

    live_pack = sat.swiglu_static_rebind_sat(raw, two_g)
    live_q, live_s, live_flag = live_pack
    stock_q, stock_s = swiglu_nvfp4_static_rebind(raw, two_g)

    # 1.005x FixedRef-style spike: scale the live G down so T / (G*2688) matches
    # the held-out qkv.10 ratio. Same algebra, production numbers.
    chef_ratio = CHEF_QKV10_T / CHEF_QKV10_THRESHOLD
    spike_g = (two_g / chef_ratio).contiguous()
    spike_threshold = float(spike_g.item()) * DENOM
    spike_ratio = live_t / spike_threshold
    spike_pack = sat.swiglu_static_rebind_sat(raw, spike_g)
    spike_flag = int(spike_pack[2].item())

    # Typical safe call at the qkv.11 median ratio 0.416.
    safe_g = (two_g / 0.416).contiguous()
    safe_pack = sat.swiglu_static_rebind_sat(raw, safe_g)
    safe_flag = int(safe_pack[2].item())
    safe_ratio = live_t / (float(safe_g.item()) * DENOM)

    # Exact threshold: G = T/2688 so ratio == 1. Not overflow.
    exact_g = (product.abs().amax().float() / DENOM).reshape(1).contiguous()
    exact_pack = sat.swiglu_static_rebind_sat(raw, exact_g)
    exact_flag = int(exact_pack[2].item())
    exact_ratio = live_t / (float(exact_g.item()) * DENOM)

    # Foreign larger G (static margin 1.30): no saturate, not two-pass bytes.
    margin_g = (two_g * 1.30).contiguous()
    margin_pack = sat.swiglu_static_rebind_sat(raw, margin_g)
    margin_flag = int(margin_pack[2].item())

    predicate = {
        "live_g": {
            "flag": int(live_flag.item()),
            "ratio": live_ratio,
            "expect_flag": int(live_ratio > 1.0),
            "pack_vs_twopass_q": byte_diff(live_q[:ROWS], two_q[:ROWS]),
            "pack_vs_stock_q": byte_diff(live_q, stock_q),
            "pack_vs_stock_s": byte_diff(live_s, stock_s),
            "note": (
                "two-pass G is not a no-overflow certificate; "
                "bf16(T/2688) may round down"
            ),
        },
        "chef_qkv10_1p005": {
            "flag": spike_flag,
            "ratio": spike_ratio,
            "expect_flag": 1,
            "chef_heldout_ratio": chef_ratio,
        },
        "median_safe_0p416": {
            "flag": safe_flag,
            "ratio": safe_ratio,
            "expect_flag": 0,
        },
        "exact_ratio_1": {
            "flag": exact_flag,
            "ratio": exact_ratio,
            "expect_flag": 0,
        },
        "margin_1p30_foreign": {
            "flag": margin_flag,
            "ratio": live_t / (float(margin_g.item()) * DENOM),
            "expect_flag": 0,
            "pack_vs_twopass_q": byte_diff(margin_pack[0][:ROWS], two_q[:ROWS]),
        },
    }

    flags_match = all(
        case["flag"] == case["expect_flag"] for case in predicate.values()
    )
    live_bytes = (
        predicate["live_g"]["pack_vs_twopass_q"] == 0
        and predicate["live_g"]["pack_vs_stock_q"] == 0
        and predicate["live_g"]["pack_vs_stock_s"] == 0
    )
    exact_is_one = abs(exact_ratio - 1.0) < 1e-6
    # Foreign G must not be two-pass; saturation is a different predicate.
    margin_is_foreign = predicate["margin_1p30_foreign"]["pack_vs_twopass_q"] > 0

    sat_ms = time_ms(lambda: sat.swiglu_static_rebind_sat(raw, two_g))
    stock_ms = time_ms(lambda: swiglu_nvfp4_static_rebind(raw, two_g))
    two_ms = time_ms(lambda: swiglu_nvfp4_dynamic(raw))

    payload = {
        "identity": (
            "NVFP4 supplied-scale overflow is decode_scale = absmax/6/G > 448 "
            "on any 16-wide block (T > G*2688). The static pack already "
            "computes that absmax and clamps it. A per-call saturate flag "
            "is that exact predicate. Two-pass G is bf16(T/2688) and can "
            "round down, so T/(G*2688) > 1 on the stock path itself; that "
            "is the association, not a fallback. Validator overflow is T "
            "versus a *calibrated* G. Quarantining layers 10/11/22/46 is "
            "the wrong object: held-out spikes migrate by seed and are "
            "one-step. Not a margin raise. Not promoted."
        ),
        "device": torch.cuda.get_device_name(0),
        "rows": ROWS,
        "width": WIDTH,
        "live_t": live_t,
        "live_g": float(two_g.item()),
        "live_g_is_bf16_t_over_2688": two_pass_g_is_bf16_t,
        "live_ratio": live_ratio,
        "predicate": predicate,
        "flags_match": flags_match,
        "live_bytes_exact": live_bytes,
        "exact_ratio_is_one": exact_is_one,
        "margin_is_foreign_g": margin_is_foreign,
        "sat_pack_min_ms": round(sat_ms, 4),
        "stock_rebind_min_ms": round(stock_ms, 4),
        "twopass_min_ms": round(two_ms, 4),
        "exclusion_algebra": {
            "object_wrong": "per-layer max scale + named quarantine",
            "object_right": "per-call pre-clamp decode_scale > 448",
            "m130_after_fc2_quarantine": {
                "blocks.10.attn.qkv_proj": {
                    "max": 1.0054313299119013,
                    "median": 0.7250220896600895,
                    "overflow_calls": 1,
                    "calls": 20,
                },
                "blocks.11.attn.qkv_proj": {
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
            "do_not": "raise every scale; spikes migrate",
        },
        "pass": bool(
            flags_match
            and live_bytes
            and exact_is_one
            and margin_is_foreign
            and two_pass_g_is_bf16_t
        ),
    }
    out = HERE / "gate_sat_cert_20423.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"pass": payload["pass"], "receipt": str(out)}, indent=2))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
