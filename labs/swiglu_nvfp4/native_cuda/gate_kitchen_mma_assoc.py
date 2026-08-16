#!/usr/bin/env python3
"""Kitchen NVFP4 MMA vs dequant-then-GEMM association.

If they match, a software paired-N epilogue on dequantized BF16 is legal.
If they do not, the epilogue must sit on the tensor-core accumulators.
Does not patch serving.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

import comfy_kitchen as ck

from swiglu_nvfp4 import bf16_nvfp4_dynamic, swiglu_nvfp4_dynamic


ROWS = 20423
K = 5376
N = 14336
REPEATS = 1


def nvfp4_scale(x: torch.Tensor) -> torch.Tensor:
    return (x.abs().amax() / (6.0 * 448.0)).to(torch.float32).reshape(1)


def quantize(x: torch.Tensor):
    scale = nvfp4_scale(x)
    packed, block = ck.quantize_nvfp4(x, scale, pad_16x=True, hi_first=True)
    return packed, block, scale


def kitchen_gemm(qx, sx, qw, sw, scale_x, scale_w) -> torch.Tensor:
    alpha = (scale_x * scale_w).reshape(1)
    y = ck.scaled_mm_nvfp4(
        qx, qw, scale_x, scale_w, sx, sw, out_dtype=torch.bfloat16, alpha=alpha
    )
    return y[:ROWS]


def eager_act(raw: torch.Tensor) -> torch.Tensor:
    gate, up = raw.chunk(2, dim=-1)
    return F.silu(gate).mul_(up)


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
    device = torch.device("cuda")
    torch.manual_seed(26081233)
    x = torch.randn(ROWS, K, device=device, dtype=torch.bfloat16)
    weight = torch.randn(2 * N, K, device=device, dtype=torch.bfloat16) * 0.02
    qx, sx, scale_x = quantize(x)
    qw, sw, scale_w = quantize(weight)
    y_kit = kitchen_gemm(qx, sx, qw, sw, scale_x, scale_w)

    a_bf16 = ck.dequantize_nvfp4(
        qx, scale_x, sx, output_type=torch.bfloat16, hi_first=True
    )[:ROWS]
    b_bf16 = ck.dequantize_nvfp4(
        qw, scale_w, sw, output_type=torch.bfloat16, hi_first=True
    )
    y_bf16 = torch.nn.functional.linear(a_bf16, b_bf16)
    y_fp32 = torch.nn.functional.linear(
        a_bf16.float(), b_bf16.float()
    ).to(torch.bfloat16)

    eager_kit = eager_act(y_kit.contiguous())
    eager_bf16 = eager_act(y_bf16.contiguous())
    kit_q, kit_s, kit_g = swiglu_nvfp4_dynamic(y_kit.contiguous())
    bf16_q, bf16_s, bf16_g = swiglu_nvfp4_dynamic(y_bf16.contiguous())
    prod_q, prod_s, prod_g = bf16_nvfp4_dynamic(eager_kit)

    payload = {
        "identity": (
            "kitchen NVFP4 MMA vs dequant(A)@dequant(B) in BF16 linear "
            "and in FP32-then-BF16; SwiGLU pack of each path"
        ),
        "rows": ROWS,
        "k": K,
        "n": N,
        "device": torch.cuda.get_device_name(device),
        "dequant_a_rows": int(a_bf16.shape[0]),
        "y_bf16_vs_kitchen": byte_diff(y_bf16, y_kit),
        "y_fp32_vs_kitchen": byte_diff(y_fp32, y_kit),
        "y_bf16_vs_fp32": byte_diff(y_bf16, y_fp32),
        "eager_bf16_vs_kitchen": byte_diff(eager_bf16, eager_kit),
        "pack_bf16_vs_kitchen_q": byte_diff(bf16_q, kit_q),
        "pack_bf16_vs_kitchen_s": byte_diff(bf16_s, kit_s),
        "pack_bf16_vs_kitchen_scale_exact": bool(torch.equal(bf16_g, kit_g)),
        "from_product_vs_kitchen_q": byte_diff(prod_q, kit_q),
        "from_product_vs_kitchen_s": byte_diff(prod_s, kit_s),
        "from_product_vs_kitchen_scale_exact": bool(torch.equal(prod_g, kit_g)),
        "kitchen_gemm_min_ms": round(
            time_ms(lambda: kitchen_gemm(qx, sx, qw, sw, scale_x, scale_w)), 4
        ),
        "dequant_bf16_linear_min_ms": round(
            time_ms(lambda: torch.nn.functional.linear(a_bf16, b_bf16)), 4
        ),
    }
    # Pass = the measurement completed and the consumer identity on the
    # kitchen path still holds. MMA mismatch is the result, not a failure.
    payload["kitchen_mma_equals_dequant_bf16"] = payload["y_bf16_vs_kitchen"] == 0
    payload["kitchen_mma_equals_dequant_fp32"] = payload["y_fp32_vs_kitchen"] == 0
    payload["pass"] = (
        payload["from_product_vs_kitchen_q"] == 0
        and payload["from_product_vs_kitchen_s"] == 0
        and payload["from_product_vs_kitchen_scale_exact"]
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    Path(__file__).with_name("gate_kitchen_mma_assoc_20423.json").write_text(
        text + "\n"
    )
    print(text)
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
