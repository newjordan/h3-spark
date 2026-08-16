#!/usr/bin/env python3
"""HBM-wall identity: 8-wide two-pass is a 2-read of [gate|up].

GB10 copy is ~212 GB/s. Raw is 1171 MiB, so two reads are ~11 ms.
Lab two-pass sits on that wall. Static rebind is the 1-read bound.
No-store exact pack cannot beat the 2-read wall: pack needs the
products, and they do not fit in 24 MiB L2 or 99 KiB smem.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import comfy_kitchen as ck

from swiglu_nvfp4 import (
    load_extension,
    swiglu_nvfp4_dynamic,
    swiglu_nvfp4_static_rebind,
)


ROWS = 20423
K = 5376
N = 14336
RAW_BYTES = ROWS * (2 * N) * 2
PROD_BYTES = ROWS * N * 2
COPY_BYTES = 256 * 1024 * 1024


def nvfp4_scale(x: torch.Tensor) -> torch.Tensor:
    return (x.abs().amax() / (6.0 * 448.0)).to(torch.float32).reshape(1)


def kitchen_fc1(x, weight):
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
    load_extension(verbose=False)
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    src = torch.empty(COPY_BYTES, device=device, dtype=torch.uint8)
    dst = torch.empty_like(src)
    copy_ms = time_ms(lambda: dst.copy_(src))
    copy_gbs = (2 * COPY_BYTES / 1e6) / copy_ms
    two_read_ms = (2 * RAW_BYTES / 1e6) / copy_gbs
    one_read_ms = (RAW_BYTES / 1e6) / copy_gbs
    torch.manual_seed(26081253)
    x = torch.randn(ROWS, K, device=device, dtype=torch.bfloat16)
    weight = torch.randn(2 * N, K, device=device, dtype=torch.bfloat16) * 0.02
    raw = kitchen_fc1(x, weight)
    two_q, two_s, two_g = swiglu_nvfp4_dynamic(raw)
    st_q, st_s = swiglu_nvfp4_static_rebind(raw, two_g)
    payload = {
        "identity": (
            "8-wide two-pass is a 2-read of the 1171 MiB [gate|up] pair. "
            "GB10 copy bandwidth sets a ~11 ms wall. Static rebind is the "
            "1-read wall. L2 is 24 MiB; products are 586 MiB, so a no-store "
            "exact pack must re-read. Coop/fused still read twice."
        ),
        "device": torch.cuda.get_device_name(device),
        "l2_bytes": int(props.L2_cache_size),
        "smem_optin": int(props.shared_memory_per_block_optin),
        "raw_bytes": RAW_BYTES,
        "prod_bytes": PROD_BYTES,
        "l2_holds_raw": int(props.L2_cache_size) >= RAW_BYTES,
        "l2_holds_prod": int(props.L2_cache_size) >= PROD_BYTES,
        "copy_256mib_ms": round(copy_ms, 4),
        "copy_gbs": round(copy_gbs, 2),
        "two_read_wall_ms": round(two_read_ms, 4),
        "one_read_wall_ms": round(one_read_ms, 4),
        "twopass_min_ms": round(
            time_ms(lambda: swiglu_nvfp4_dynamic(raw)), 4
        ),
        "static_rebind_min_ms": round(
            time_ms(lambda: swiglu_nvfp4_static_rebind(raw, two_g)), 4
        ),
        "static_vs_twopass_q": byte_diff(st_q, two_q),
        "static_vs_twopass_s": byte_diff(st_s, two_s),
        "source": "kitchen_fc1_nvfp4_20423x5376x28672",
    }
    payload["twopass_over_two_read"] = round(
        payload["twopass_min_ms"] / payload["two_read_wall_ms"], 3
    )
    payload["static_over_one_read"] = round(
        payload["static_rebind_min_ms"] / payload["one_read_wall_ms"], 3
    )
    payload["pass"] = (
        payload["static_vs_twopass_q"] == 0
        and payload["static_vs_twopass_s"] == 0
        and not payload["l2_holds_prod"]
        and payload["twopass_over_two_read"] < 1.3
        and payload["static_over_one_read"] < 1.5
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    Path(__file__).with_name("gate_twopass_hbm_wall_20423.json").write_text(
        text + "\n"
    )
    print(text)
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
