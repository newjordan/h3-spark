#!/usr/bin/env python3
"""Kitchen NVFP4 split-N vs concat-N association for the FC1 epilogue.

Same quantized weights. Does not patch serving or kitchen.
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
REPEATS = 3


def nvfp4_scale(x: torch.Tensor) -> torch.Tensor:
    return (x.abs().amax() / (6.0 * 448.0)).to(torch.float32).reshape(1)


def quantize(x: torch.Tensor, scale: torch.Tensor | None = None):
    if scale is None:
        scale = nvfp4_scale(x)
    packed, block = ck.quantize_nvfp4(x, scale, pad_16x=True, hi_first=True)
    return packed, block, scale


def gemm(qx, sx, qw, sw, scale_x, scale_w) -> torch.Tensor:
    alpha = (scale_x * scale_w).reshape(1)
    return ck.scaled_mm_nvfp4(
        qx, qw, scale_x, scale_w, sx, sw, out_dtype=torch.bfloat16, alpha=alpha
    )


def eager_act(raw: torch.Tensor) -> torch.Tensor:
    gate, up = raw.chunk(2, dim=-1)
    return F.silu(gate).mul_(up)


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
    device = torch.device("cuda")
    torch.manual_seed(26081229)
    x = torch.randn(ROWS, K, device=device, dtype=torch.bfloat16)
    weight = torch.randn(2 * N, K, device=device, dtype=torch.bfloat16) * 0.02
    qx, sx, scale_x = quantize(x)
    qw, sw, scale_w = quantize(weight)
    qw_g, sw_g = qw[:N].contiguous(), sw[:N].contiguous()
    qw_u, sw_u = qw[N:].contiguous(), sw[N:].contiguous()

    y_full = gemm(qx, sx, qw, sw, scale_x, scale_w)
    y_g = gemm(qx, sx, qw_g, sw_g, scale_x, scale_w)
    y_u = gemm(qx, sx, qw_u, sw_u, scale_x, scale_w)
    # Kitchen may pad M; compare the live rows only.
    y_full = y_full[:ROWS]
    y_g = y_g[:ROWS]
    y_u = y_u[:ROWS]
    split = torch.cat((y_g, y_u), dim=-1)
    eager_full = eager_act(y_full)
    eager_split = eager_act(split)
    two_q, two_s, two_g = swiglu_nvfp4_dynamic(y_full.contiguous())
    split_q, split_s, split_g = bf16_nvfp4_dynamic(eager_split.contiguous())
    payload = {
        "identity": (
            "kitchen NVFP4 MMA is N-independent: mm(X,W)[:, :N] and "
            "mm(X,W)[:, N:] equal mm(X,Wg) and mm(X,Wu) on a prefix slice "
            "of the same packed codes and swizzled scales (14336 % 128 == 0)"
        ),
        "rows": ROWS,
        "k": K,
        "n": N,
        "device": torch.cuda.get_device_name(device),
        "qw_g_vs_prefix": byte_diff(qw_g, qw[:N]),
        "sw_g_vs_prefix": byte_diff(sw_g, sw[:N]),
        "yg_vs_full_left": byte_diff(y_g, y_full[:, :N]),
        "yu_vs_full_right": byte_diff(y_u, y_full[:, N:]),
        "split_vs_full": byte_diff(split, y_full),
        "eager_split_vs_eager_full": byte_diff(eager_split, eager_full),
        "pack_split_vs_twopass_q": byte_diff(split_q, two_q),
        "pack_split_vs_twopass_s": byte_diff(split_s, two_s),
        "pack_split_vs_twopass_scale_exact": bool(torch.equal(split_g, two_g)),
        "full_gemm_min_ms": round(
            time_ms(lambda: gemm(qx, sx, qw, sw, scale_x, scale_w)), 4
        ),
        "split_gemm_min_ms": round(
            time_ms(
                lambda: (
                    gemm(qx, sx, qw_g, sw_g, scale_x, scale_w),
                    gemm(qx, sx, qw_u, sw_u, scale_x, scale_w),
                )
            ),
            4,
        ),
    }
    payload["pass"] = (
        payload["qw_g_vs_prefix"] == 0
        and payload["sw_g_vs_prefix"] == 0
        and payload["yg_vs_full_left"] == 0
        and payload["yu_vs_full_right"] == 0
        and payload["split_vs_full"] == 0
        and payload["eager_split_vs_eager_full"] == 0
        and payload["pack_split_vs_twopass_q"] == 0
        and payload["pack_split_vs_twopass_s"] == 0
        and payload["pack_split_vs_twopass_scale_exact"]
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    Path(__file__).with_name("gate_kitchen_splitn_20423.json").write_text(
        text + "\n"
    )
    print(text)
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
