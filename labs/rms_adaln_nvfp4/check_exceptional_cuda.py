#!/usr/bin/env python3
"""Byte-parity gate for exceptional values at every fused input boundary.

This is a hardening test, not an expansion of the finite-inference serving
contract.  It compares only the tensors consumed by the downstream NVFP4
linear: packed data, block scales, and the float32 global scale bits.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
STAGED = HERE.parent / "adaln_nvfp4" / "native_cuda"
sys.path.insert(0, str(STAGED))

from adaln_nvfp4 import adaln_nvfp4_dynamic  # noqa: E402
from rms_adaln_nvfp4 import (  # noqa: E402
    WIDTH,
    load_extension,
    rms_adaln_nvfp4_dynamic,
)


ROWS = 17
MOD_ROWS = 21
EPS = 1e-5
NATIVE_PATH = HERE / "build" / "h3_rms_adaln_nvfp4_native_v2.so"


def byte_mismatches(left: torch.Tensor, right: torch.Tensor) -> int:
    return int(
        (left.view(torch.uint8) != right.view(torch.uint8)).sum().item()
    )


def uint32_bits(value: torch.Tensor) -> int:
    if value.dtype != torch.float32 or value.numel() != 1:
        raise ValueError("expected one float32 global scale")
    return int(value.view(torch.int32).item()) & 0xFFFFFFFF


def make_base_inputs():
    device = torch.device("cuda")
    x = torch.randn(ROWS, WIDTH, dtype=torch.bfloat16, device=device)
    weight = torch.randn(WIDTH, dtype=torch.bfloat16, device=device)
    table = torch.randn(
        MOD_ROWS,
        WIDTH * 6,
        dtype=torch.float32,
        device=device,
    )
    row_ids = (
        torch.arange(ROWS, dtype=torch.int64, device=device)
        .mul_(MOD_ROWS)
        .div_(ROWS, rounding_mode="floor")
        .to(torch.int32)
    )
    return x, weight, table, row_ids


def check_case(base, source: str, label: str, value: float) -> dict[str, object]:
    base_x, base_weight, base_table, row_ids = base
    x = base_x.clone()
    weight = base_weight.clone()
    table = base_table.clone()
    scale = table[:, WIDTH : 2 * WIDTH]
    shift = table[:, 4 * WIDTH : 5 * WIDTH]

    if source == "x":
        x[0, 0] = value
    elif source == "weight":
        weight[0] = value
    elif source == "scale":
        scale[0, 0] = value
    elif source == "shift":
        shift[0, 0] = value
    else:
        raise ValueError(source)

    normed, _ = torch.ops.aten._fused_rms_norm(
        x,
        [WIDTH],
        weight,
        EPS,
    )
    reference_q, reference_s, reference_global = adaln_nvfp4_dynamic(
        normed,
        scale,
        shift,
        row_ids,
    )
    fused_q, fused_s, fused_global = rms_adaln_nvfp4_dynamic(
        x,
        weight,
        scale,
        shift,
        row_ids,
        EPS,
    )

    reference_bits = uint32_bits(reference_global)
    fused_bits = uint32_bits(fused_global)
    result = {
        "source": source,
        "exception": label,
        "packed_byte_mismatches": byte_mismatches(fused_q, reference_q),
        "block_scale_byte_mismatches": byte_mismatches(fused_s, reference_s),
        "reference_global_scale_bits": f"0x{reference_bits:08x}",
        "fused_global_scale_bits": f"0x{fused_bits:08x}",
        "global_scale_uint32_exact": reference_bits == fused_bits,
    }
    result["downstream_bytes_exact"] = (
        result["packed_byte_mismatches"] == 0
        and result["block_scale_byte_mismatches"] == 0
        and result["global_scale_uint32_exact"]
    )
    return result


def main() -> None:
    torch.manual_seed(260812)
    native = load_extension()
    base = make_base_inputs()
    cases = [
        check_case(base, source, label, value)
        for source in ("x", "weight", "scale", "shift")
        for label, value in (("nan", float("nan")), ("+inf", float("inf")))
    ]
    torch.cuda.synchronize()
    payload = {
        "rows": ROWS,
        "width": WIDTH,
        "activation_dtype": "torch.bfloat16",
        "modulation_dtype": "torch.float32",
        "eps": EPS,
        "native_sha256": hashlib.sha256(NATIVE_PATH.read_bytes()).hexdigest(),
        "native_abi_version": native.abi_version(),
        "native_build_id": native.build_id(),
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "all_downstream_bytes_exact": all(
            case["downstream_bytes_exact"] for case in cases
        ),
        "cases": cases,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["all_downstream_bytes_exact"]:
        raise SystemExit("exceptional-value downstream byte-parity gate failed")


if __name__ == "__main__":
    main()
