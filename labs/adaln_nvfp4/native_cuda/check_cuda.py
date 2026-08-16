"""Small opt-in CUDA equality gate for staged AdaLN -> NVFP4."""

from __future__ import annotations

import argparse
import json

import comfy_kitchen as ck
import torch

try:
    from .adaln_nvfp4 import (
        WIDTH,
        adaln_nvfp4,
        adaln_nvfp4_dynamic,
        eager_adaln_reference,
        load_extension,
    )
except ImportError:
    from adaln_nvfp4 import (
        WIDTH,
        adaln_nvfp4,
        adaln_nvfp4_dynamic,
        eager_adaln_reference,
        load_extension,
    )


def byte_diff(left: torch.Tensor, right: torch.Tensor) -> int:
    return int((left.view(torch.uint8) != right.view(torch.uint8)).sum().item())


def check_case(rows: int, modulation_dtype: torch.dtype) -> dict[str, object]:
    device = torch.device("cuda")
    normed = torch.randn(rows, WIDTH, dtype=torch.bfloat16, device=device)
    # Model chunks have a six-wide row stride, not a compact [M, D] stride.
    table = torch.randn(5, WIDTH * 6, dtype=modulation_dtype, device=device)
    scale = table[:, WIDTH:2 * WIDTH]
    shift = table[:, 4 * WIDTH:5 * WIDTH]
    row_ids = torch.arange(rows, device=device, dtype=torch.int32).remainder_(5)

    materialized = eager_adaln_reference(normed, scale, shift, row_ids)
    reference_scale = (
        materialized.abs().amax() / (6.0 * 448.0)
    ).to(torch.float32).reshape(1)
    reference_q, reference_s = ck.quantize_nvfp4(
        materialized,
        reference_scale,
        pad_16x=True,
        hi_first=True,
    )
    static_q, static_s = adaln_nvfp4(
        normed, scale, shift, row_ids, reference_scale
    )
    dynamic_q, dynamic_s, dynamic_scale = adaln_nvfp4_dynamic(
        normed, scale, shift, row_ids
    )

    result = {
        "rows": rows,
        "modulation_dtype": str(modulation_dtype),
        "modulation_stride": list(scale.stride()),
        "dynamic_global_scale_exact": bool(
            torch.equal(dynamic_scale, reference_scale)
        ),
        "static_packed_byte_mismatches": byte_diff(static_q, reference_q),
        "static_scale_byte_mismatches": byte_diff(static_s, reference_s),
        "dynamic_packed_byte_mismatches": byte_diff(dynamic_q, reference_q),
        "dynamic_scale_byte_mismatches": byte_diff(dynamic_s, reference_s),
    }
    result["byte_exact"] = all(
        (
            result["dynamic_global_scale_exact"],
            result["static_packed_byte_mismatches"] == 0,
            result["static_scale_byte_mismatches"] == 0,
            result["dynamic_packed_byte_mismatches"] == 0,
            result["dynamic_scale_byte_mismatches"] == 0,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=129)
    parser.add_argument("--seed", type=int, default=260812)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.rows <= 0:
        raise SystemExit("rows must be positive")

    torch.manual_seed(args.seed)
    load_extension()
    results = [
        check_case(args.rows, torch.bfloat16),
        check_case(args.rows, torch.float32),
    ]
    print(json.dumps(results, indent=2, sort_keys=True))
    if not all(result["byte_exact"] for result in results):
        raise SystemExit("AdaLN-to-NVFP4 byte equality failed")


if __name__ == "__main__":
    main()
