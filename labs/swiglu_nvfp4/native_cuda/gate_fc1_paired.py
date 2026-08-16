#!/usr/bin/env python3
"""Byte-exact gate: paired-N FC1 MMA store of eager product vs [gate|up].

Same mainloop, two epilogues. Kitchen GEMM is not modified.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

import comfy_kitchen as ck

from swiglu_nvfp4 import (
    bf16_nvfp4_dynamic,
    fc1_paired_store,
    load_extension,
    swiglu_nvfp4_dynamic,
)


ROWS = 20423
K = 5376
N = 14336
REPEATS = 1


def eager_act(raw: torch.Tensor) -> torch.Tensor:
    gate, up = raw.chunk(2, dim=-1)
    return F.silu(gate).mul_(up)


def eager_pack(activated: torch.Tensor):
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
    torch.manual_seed(26081228)
    x = torch.randn(ROWS, K, device=device, dtype=torch.bfloat16)
    weight = torch.randn(2 * N, K, device=device, dtype=torch.bfloat16) * 0.02
    full = fc1_paired_store(x, weight, product=False)
    prod = fc1_paired_store(x, weight, product=True)
    eager = eager_act(full)
    two_q, two_s, two_g = swiglu_nvfp4_dynamic(full)
    prod_q, prod_s, prod_g = bf16_nvfp4_dynamic(prod)
    ref_q, ref_s, ref_g = eager_pack(eager)
    payload = {
        "identity": (
            "paired-N MMA accumulates gate and up together; "
            "product epilogue stores eager round(silu(round(g))*round(u)); "
            "same mainloop as the [gate|up] store"
        ),
        "rows": ROWS,
        "k": K,
        "n": N,
        "device": torch.cuda.get_device_name(device),
        "product_vs_eager_full_mismatches": byte_diff(prod, eager),
        "from_product_vs_twopass_q": byte_diff(prod_q, two_q),
        "from_product_vs_twopass_s": byte_diff(prod_s, two_s),
        "from_product_vs_twopass_scale_exact": bool(torch.equal(prod_g, two_g)),
        "from_product_vs_eager_q": byte_diff(prod_q, ref_q),
        "from_product_vs_eager_s": byte_diff(prod_s, ref_s),
        "from_product_vs_eager_scale_exact": bool(torch.equal(prod_g, ref_g)),
        "full_store_min_ms": round(
            time_ms(lambda: fc1_paired_store(x, weight, product=False)), 4
        ),
        "product_store_min_ms": round(
            time_ms(lambda: fc1_paired_store(x, weight, product=True)), 4
        ),
        "twopass_min_ms": round(time_ms(lambda: swiglu_nvfp4_dynamic(full)), 4),
        "from_product_min_ms": round(time_ms(lambda: bf16_nvfp4_dynamic(prod)), 4),
    }
    payload["pass"] = (
        payload["product_vs_eager_full_mismatches"] == 0
        and payload["from_product_vs_twopass_q"] == 0
        and payload["from_product_vs_twopass_s"] == 0
        and payload["from_product_vs_twopass_scale_exact"]
        and payload["from_product_vs_eager_q"] == 0
        and payload["from_product_vs_eager_s"] == 0
        and payload["from_product_vs_eager_scale_exact"]
    )
    if payload["pass"]:
        payload["store_delta_ms"] = round(
            payload["full_store_min_ms"] - payload["product_store_min_ms"], 4
        )
        payload["pack_side_saved_ms"] = round(
            payload["twopass_min_ms"] - payload["from_product_min_ms"], 4
        )
    text = json.dumps(payload, indent=2, sort_keys=True)
    Path(__file__).with_name("gate_fc1_paired_20423.json").write_text(text + "\n")
    print(text)
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
