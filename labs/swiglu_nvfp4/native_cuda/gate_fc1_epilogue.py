#!/usr/bin/env python3
"""FC1 SwiGLU epilogue contract: eager BF16 association + product consumer.

Does not patch serving. Does not implement a kitchen/CUTLASS MMA.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

import comfy_kitchen as ck

from swiglu_nvfp4 import (
    bf16_nvfp4_dynamic,
    fc1_epilogue_assoc_mismatches,
    load_extension,
    swiglu_nvfp4,
    swiglu_nvfp4_dynamic,
    swiglu_nvfp4_dynamic_from_product,
    swiglu_nvfp4_dynamic_vec,
)


ROWS = 20423
WIDTH = 28672
REPEATS = 3


def eager_act(raw: torch.Tensor) -> torch.Tensor:
    gate, up = raw.chunk(2, dim=-1)
    return F.silu(gate).mul_(up)


def eager_dynamic(raw: torch.Tensor):
    activated = eager_act(raw)
    scale = (activated.abs().amax() / (6.0 * 448.0)).to(torch.float32).reshape(1)
    packed, block_scales = ck.quantize_nvfp4(
        activated, scale, pad_16x=True, hi_first=True
    )
    return packed, block_scales, scale, activated


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
    torch.manual_seed(26081227)
    raw = torch.randn(ROWS, WIDTH, device=device, dtype=torch.bfloat16)
    two_q, two_s, two_g = swiglu_nvfp4_dynamic(raw)
    ref_q, ref_s, ref_g, act = eager_dynamic(raw)
    prod_q, prod_s, prod_g = swiglu_nvfp4_dynamic_from_product(act)
    mismatches = fc1_epilogue_assoc_mismatches(raw)
    acc = raw.float()
    acc = torch.nextafter(acc, torch.copysign(torch.full_like(acc, float("inf")), acc))
    store = acc.to(torch.bfloat16)
    store_roundtrip = int((store.view(torch.uint8) != raw.view(torch.uint8)).sum().item())
    fused_act = (F.silu(acc.chunk(2, dim=-1)[0]) * acc.chunk(2, dim=-1)[1]).to(
        torch.bfloat16
    )
    fused_product_mismatches = byte_diff(act, fused_act)
    fused_q, fused_s = swiglu_nvfp4(raw, two_g, eager_bf16_rounding=False)
    payload = {
        "identity": (
            "FC1 epilogue must store eager round(silu(round(acc_g))*round(acc_u)); "
            "fused FP32 SiLU on the accumulator is a different function; "
            "bf16_nvfp4_dynamic on that 14336-wide product is the pack consumer"
        ),
        "rows": ROWS,
        "width": WIDTH,
        "device": torch.cuda.get_device_name(device),
        "fp32_ulp_store_roundtrip_mismatches": store_roundtrip,
        "native_assoc_product_mismatches": mismatches,
        "python_fused_product_mismatches": fused_product_mismatches,
        "fused_fp32_vs_eager_q": byte_diff(fused_q, two_q),
        "fused_fp32_vs_eager_s": byte_diff(fused_s, two_s),
        "from_product_vs_twopass_q": byte_diff(prod_q, two_q),
        "from_product_vs_twopass_s": byte_diff(prod_s, two_s),
        "from_product_vs_twopass_scale_exact": bool(torch.equal(prod_g, two_g)),
        "from_product_vs_eager_q": byte_diff(prod_q, ref_q),
        "from_product_vs_eager_s": byte_diff(prod_s, ref_s),
        "from_product_vs_eager_scale_exact": bool(torch.equal(prod_g, ref_g)),
        "twopass_min_ms": round(time_ms(lambda: swiglu_nvfp4_dynamic(raw)), 4),
        "vec_min_ms": round(time_ms(lambda: swiglu_nvfp4_dynamic_vec(raw)), 4),
        "from_product_min_ms": round(
            time_ms(lambda: bf16_nvfp4_dynamic(act)), 4
        ),
    }
    payload["pass"] = (
        payload["fp32_ulp_store_roundtrip_mismatches"] == 0
        and payload["native_assoc_product_mismatches"] > 0
        and payload["python_fused_product_mismatches"] > 0
        and payload["fused_fp32_vs_eager_q"] > 0
        and payload["from_product_vs_twopass_q"] == 0
        and payload["from_product_vs_twopass_s"] == 0
        and payload["from_product_vs_twopass_scale_exact"]
        and payload["from_product_vs_eager_q"] == 0
        and payload["from_product_vs_eager_s"] == 0
        and payload["from_product_vs_eager_scale_exact"]
    )
    if payload["pass"]:
        payload["pack_side_saved_ms"] = round(
            payload["twopass_min_ms"] - payload["from_product_min_ms"], 4
        )
        payload["projected_s_per_20_step"] = round(
            payload["pack_side_saved_ms"] * 50 * 20 / 1000.0, 4
        )
    text = json.dumps(payload, indent=2, sort_keys=True)
    Path(__file__).with_name("gate_fc1_epilogue_20423.json").write_text(
        text + "\n"
    )
    print(text)
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
