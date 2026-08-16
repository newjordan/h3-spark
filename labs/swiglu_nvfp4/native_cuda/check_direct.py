"""Small correctness gate for H3's direct BF16-to-NVFP4 CUDA path.

The default workload is intentionally tiny: 129 rows exercises both kinds of
padding at each H3 activation width. Timing is opt-in and capped at three
samples so this is suitable while a serving process owns most GPU memory.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

import comfy_kitchen as ck

from swiglu_nvfp4 import (
    H3_NVFP4_WIDTHS,
    bf16_nvfp4,
    bf16_nvfp4_dynamic,
    load_extension,
)


def reference(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Preserve TensorCoreNVFP4Layout.quantize's BF16 division before widening.
    scale = (value.abs().amax() / (6.0 * 448.0)).to(torch.float32).reshape(1)
    packed, block_scales = ck.quantize_nvfp4(
        value,
        scale,
        pad_16x=True,
        hi_first=True,
    )
    return packed, block_scales, scale


def byte_diff(left: torch.Tensor, right: torch.Tensor) -> int:
    return int((left.view(torch.uint8) != right.view(torch.uint8)).sum().item())


@torch.inference_mode()
def check_width(value: torch.Tensor) -> dict[str, object]:
    ref_q, ref_s, ref_scale = reference(value)
    static_q, static_s = bf16_nvfp4(value, ref_scale)
    dynamic_q, dynamic_s, dynamic_scale = bf16_nvfp4_dynamic(value)
    result = {
        "shape": list(value.shape),
        "packed_shape": list(dynamic_q.shape),
        "scale_shape": list(dynamic_s.shape),
        "static_packed_byte_mismatches": byte_diff(static_q, ref_q),
        "static_scale_byte_mismatches": byte_diff(static_s, ref_s),
        "dynamic_global_scale_exact": bool(torch.equal(dynamic_scale, ref_scale)),
        "dynamic_packed_byte_mismatches": byte_diff(dynamic_q, ref_q),
        "dynamic_scale_byte_mismatches": byte_diff(dynamic_s, ref_s),
    }
    result["byte_exact"] = all(
        (
            result["static_packed_byte_mismatches"] == 0,
            result["static_scale_byte_mismatches"] == 0,
            result["dynamic_global_scale_exact"],
            result["dynamic_packed_byte_mismatches"] == 0,
            result["dynamic_scale_byte_mismatches"] == 0,
        )
    )
    return result


@torch.inference_mode()
def time_three(
    value: torch.Tensor,
    repeats: int,
) -> dict[str, object] | None:
    if repeats == 0:
        return None
    if repeats < 0 or repeats > 3:
        raise ValueError("timing repeats must be between 0 and 3")

    # One untimed call initializes dispatcher state; the only measured calls
    # are the requested samples.
    reference(value)
    bf16_nvfp4_dynamic(value)
    torch.cuda.synchronize(value.device)

    def measure(fn) -> list[float]:
        samples = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = fn()
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
            del output
        return samples

    stock = measure(lambda: reference(value))
    native = measure(lambda: bf16_nvfp4_dynamic(value))
    return {
        "rows": value.shape[0],
        "width": value.shape[1],
        "repeats": repeats,
        "stock_ms": stock,
        "native_ms": native,
        "stock_median_ms": statistics.median(stock),
        "native_median_ms": statistics.median(native),
        "median_speedup": statistics.median(stock) / statistics.median(native),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=129)
    parser.add_argument("--seed", type=int, default=260812)
    parser.add_argument("--timing-repeats", type=int, choices=range(4), default=0)
    parser.add_argument(
        "--timing-width",
        type=int,
        choices=sorted(H3_NVFP4_WIDTHS),
        default=5376,
    )
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.rows <= 0:
        raise SystemExit("rows must be positive")

    load_extension()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    values = {
        width: torch.randn(
            args.rows,
            width,
            dtype=torch.bfloat16,
            device=device,
        )
        for width in sorted(H3_NVFP4_WIDTHS)
    }
    checks = []
    for width, value in values.items():
        check = check_width(value)
        checks.append(check)
        if not check["byte_exact"]:
            raise SystemExit(
                "direct BF16-to-NVFP4 correctness mismatch: "
                + json.dumps(check, sort_keys=True)
            )

    timing = time_three(values[args.timing_width], args.timing_repeats)
    result = {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "rows": args.rows,
        "checks": checks,
        "timing": timing,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        args.json_out.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
