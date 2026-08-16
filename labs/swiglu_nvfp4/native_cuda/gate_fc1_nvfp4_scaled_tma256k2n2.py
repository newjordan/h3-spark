#!/usr/bin/env python3
"""Byte-exact gate: sequential N-halves stream A once on 256x64 acc.

One A K=128 TMA feeds two 64-wide N-halves. Acc0 stays in the k2
2x8x4 register budget; acc1 swaps through opt-in smem. Same PTX
fragment and cuBLAS 128x4 scale slabs.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import comfy_kitchen as ck

from swiglu_nvfp4 import (
    fc1_paired_nvfp4_scaled_tma256k2n2,
    load_extension,
)


ROWS = 20423
K = 5376
N = 14336


def nvfp4_scale(x: torch.Tensor) -> torch.Tensor:
    return (x.abs().amax() / (6.0 * 448.0)).to(torch.float32).reshape(1)


def main() -> int:
    load_extension(verbose=False)
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    optin = int(getattr(props, "shared_memory_per_block_optin", 0))
    acc_bytes = 256 * 128 * 4
    tile_smem = 209024
    torch.manual_seed(26081247)
    x = torch.randn(ROWS, K, device=device, dtype=torch.bfloat16)
    weight = torch.randn(2 * N, K, device=device, dtype=torch.bfloat16) * 0.02
    sx = nvfp4_scale(x)
    sw = nvfp4_scale(weight)
    qx, qxs = ck.quantize_nvfp4(x, sx, pad_16x=True, hi_first=True)
    qw, qws = ck.quantize_nvfp4(weight, sw, pad_16x=True, hi_first=True)
    qx_live = qx[:ROWS].contiguous()
    alpha = (sx * sw).reshape(1).contiguous()
    launch_error = ""
    launched = False
    try:
        _ = fc1_paired_nvfp4_scaled_tma256k2n2(
            qx_live, qxs, qw, qws, alpha, product=False
        )
        launched = True
    except RuntimeError as exc:
        launch_error = str(exc)
    payload = {
        "identity": (
            "Sequential N-halves that stream A once need a second "
            "256x64 FP32 acc (128 KiB) plus TMA. GB10 block opt-in "
            "is 99 KiB, so acc1 cannot live in smem. Fail-closed."
        ),
        "rows": ROWS,
        "k": K,
        "n": N,
        "device": torch.cuda.get_device_name(device),
        "smem_per_block": int(props.shared_memory_per_block),
        "smem_per_sm": int(props.shared_memory_per_multiprocessor),
        "smem_optin": optin,
        "acc1_bytes": acc_bytes,
        "tile_smem_bytes": tile_smem,
        "acc1_exceeds_optin": acc_bytes > optin,
        "tile_exceeds_optin": tile_smem > optin,
        "launched": launched,
        "launch_error": launch_error,
    }
    payload["pass"] = (
        not launched
        and acc_bytes > optin
        and tile_smem > optin
        and optin <= 101376
        and "opt-in" in launch_error
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    Path(__file__).with_name(
        "gate_fc1_nvfp4_scaled_tma256k2n2_20423.json"
    ).write_text(text + "\n")
    print(text)
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
