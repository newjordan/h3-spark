#!/usr/bin/env python3
"""Launch-structure probe: kitchen 128x128x256 vs lab clone.

Prints a marker then runs one kitchen GEMM and one lab clone so ncu
can attach. Not a byte gate (identity already proven).
"""

from __future__ import annotations

import os
import sys

import torch
import comfy_kitchen as ck

from swiglu_nvfp4 import fc1_nvfp4_scaled_tma128n128k4, load_extension


ROWS = 20423
K = 5376
N = 14336


def nvfp4_scale(x: torch.Tensor) -> torch.Tensor:
    return (x.abs().amax() / (6.0 * 448.0)).to(torch.float32).reshape(1)


def main() -> int:
    load_extension(verbose=False)
    device = torch.device("cuda")
    torch.manual_seed(26081254)
    x = torch.randn(ROWS, K, device=device, dtype=torch.bfloat16)
    weight = torch.randn(2 * N, K, device=device, dtype=torch.bfloat16) * 0.02
    sx = nvfp4_scale(x)
    sw = nvfp4_scale(weight)
    qx, qxs = ck.quantize_nvfp4(x, sx, pad_16x=True, hi_first=True)
    qw, qws = ck.quantize_nvfp4(weight, sw, pad_16x=True, hi_first=True)
    qx_live = qx[:ROWS].contiguous()
    alpha = (sx * sw).reshape(1).contiguous()
    which = os.environ.get("H3_NCU_WHICH", "kitchen")
    if which == "kitchen":
        print("H3_NCU_MARK kitchen", flush=True)
        y = ck.scaled_mm_nvfp4(
            qx, qw, sx, sw, qxs, qws, out_dtype=torch.bfloat16, alpha=alpha
        )
        torch.cuda.synchronize()
        print("H3_NCU_DONE", tuple(y.shape), flush=True)
    else:
        print("H3_NCU_MARK lab128n128k4", flush=True)
        y = fc1_nvfp4_scaled_tma128n128k4(qx_live, qxs, qw, qws, alpha)
        torch.cuda.synchronize()
        print("H3_NCU_DONE", tuple(y.shape), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
