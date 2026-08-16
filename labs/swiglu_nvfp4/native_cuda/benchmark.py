"""Correctness and latency comparison against comfy-kitchen v0.2.27.

This script intentionally does nothing at import time. Running it compiles a
local CUDA extension and uses the active CUDA device.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

import comfy_kitchen as ck

from swiglu_nvfp4 import (
    load_extension,
    swiglu_nvfp4,
    swiglu_nvfp4_dynamic,
)


WIDTH = 14336


def eager_reference(
    raw: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    gate, up = raw.chunk(2, dim=-1)
    activated = F.silu(gate).mul_(up)
    return ck.quantize_nvfp4(
        activated,
        scale,
        pad_16x=True,
        hi_first=True,
    )


def eager_dynamic_reference(
    raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gate, up = raw.chunk(2, dim=-1)
    activated = F.silu(gate).mul_(up)
    # Preserve ComfyUI's operation order: BF16 amax and BF16 division, followed
    # by widening the scalar to FP32 inside TensorCoreNVFP4Layout.quantize.
    scale = (activated.abs().amax() / (6.0 * 448.0)).to(torch.float32)
    packed, block_scales = ck.quantize_nvfp4(
        activated,
        scale,
        pad_16x=True,
        hi_first=True,
    )
    return packed, block_scales, scale.reshape(1)


@torch.inference_mode()
def compare_outputs(
    raw: torch.Tensor,
    scale: torch.Tensor,
    *,
    eager_bf16_rounding: bool,
) -> dict[str, object]:
    ref_q, ref_s = eager_reference(raw, scale)
    got_q, got_s = swiglu_nvfp4(
        raw, scale, eager_bf16_rounding=eager_bf16_rounding
    )

    q_diff = got_q != ref_q
    s_diff = got_s.view(torch.uint8) != ref_s.view(torch.uint8)
    q_mismatches = int(q_diff.sum().item())
    s_mismatches = int(s_diff.sum().item())
    nibble_report = _packed_difference_report(ref_q, got_q)
    ref_dq = ck.dequantize_nvfp4(
        ref_q, scale, ref_s, output_type=torch.bfloat16, hi_first=True
    )[: raw.shape[0]]
    got_dq = ck.dequantize_nvfp4(
        got_q, scale, got_s, output_type=torch.bfloat16, hi_first=True
    )[: raw.shape[0]]
    torch.cuda.synchronize(raw.device)

    ref_bits = ref_dq.view(torch.uint16)
    got_bits = got_dq.view(torch.uint16)
    return {
        "packed_shape": list(got_q.shape),
        "scale_shape": list(got_s.shape),
        "packed_byte_exact": q_mismatches == 0,
        "scale_byte_exact": s_mismatches == 0,
        "packed_mismatches": q_mismatches,
        "packed_mismatch_fraction": q_mismatches / max(1, got_q.numel()),
        **nibble_report,
        "scale_mismatches": s_mismatches,
        "scale_mismatch_fraction": s_mismatches / max(1, got_s.numel()),
        "dequant_logical_rows_bit_exact": bool(torch.equal(ref_bits, got_bits)),
        "dequant_logical_rows_numeric_exact": bool(torch.equal(ref_dq, got_dq)),
        "dequant_logical_rows_bit_mismatches": int(
            (ref_bits != got_bits).sum().item()
        ),
    }


def _packed_difference_report(
    ref_q: torch.Tensor,
    got_q: torch.Tensor,
) -> dict[str, object]:
    ref_nibbles = torch.stack((ref_q >> 4, ref_q & 0x0F), dim=-1)
    got_nibbles = torch.stack((got_q >> 4, got_q & 0x0F), dim=-1)
    nibble_diff = ref_nibbles != got_nibbles
    signed_zero_diff = (
        nibble_diff
        & ((ref_nibbles & 0x07) == 0)
        & ((got_nibbles & 0x07) == 0)
        & ((ref_nibbles ^ got_nibbles) == 0x08)
    )
    nibble_mismatches = int(nibble_diff.sum().item())
    signed_zero_mismatches = int(signed_zero_diff.sum().item())
    return {
        "packed_mismatched_nibbles": nibble_mismatches,
        "signed_zero_only": signed_zero_mismatches == nibble_mismatches,
        "signed_zero_nibble_mismatches": signed_zero_mismatches,
        "non_signed_zero_nibble_mismatches": (
            nibble_mismatches - signed_zero_mismatches
        ),
    }


@torch.inference_mode()
def compare_dynamic_outputs(raw: torch.Tensor) -> dict[str, object]:
    ref_q, ref_s, ref_scale = eager_dynamic_reference(raw)
    got_q, got_s, got_scale = swiglu_nvfp4_dynamic(raw)

    q_mismatches = int((got_q != ref_q).sum().item())
    s_mismatches = int(
        (got_s.view(torch.uint8) != ref_s.view(torch.uint8)).sum().item()
    )
    nibble_report = _packed_difference_report(ref_q, got_q)
    ref_dq = ck.dequantize_nvfp4(
        ref_q, ref_scale, ref_s, output_type=torch.bfloat16, hi_first=True
    )[: raw.shape[0]]
    got_dq = ck.dequantize_nvfp4(
        got_q, got_scale, got_s, output_type=torch.bfloat16, hi_first=True
    )[: raw.shape[0]]
    torch.cuda.synchronize(raw.device)

    scale_exact = bool(torch.equal(got_scale, ref_scale))
    dequant_bit_exact = bool(
        torch.equal(ref_dq.view(torch.uint16), got_dq.view(torch.uint16))
    )
    dequant_numeric_exact = bool(torch.equal(ref_dq, got_dq))
    dequant_max_abs = float((ref_dq.float() - got_dq.float()).abs().max().item())
    return {
        "global_scale_exact": scale_exact,
        "reference_global_scale": float(ref_scale.item()),
        "native_global_scale": float(got_scale.item()),
        "packed_byte_exact": q_mismatches == 0,
        "scale_byte_exact": s_mismatches == 0,
        "packed_mismatches": q_mismatches,
        "packed_mismatch_fraction": q_mismatches / max(1, got_q.numel()),
        **nibble_report,
        "scale_mismatches": s_mismatches,
        "scale_mismatch_fraction": s_mismatches / max(1, got_s.numel()),
        "dequant_logical_rows_bit_exact": dequant_bit_exact,
        "dequant_logical_rows_numeric_exact": dequant_numeric_exact,
        "dequant_logical_rows_max_abs": dequant_max_abs,
    }


@torch.inference_mode()
def time_cuda(
    fn: Callable[[], object], *, warmup: int, repeats: int
) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
        del result
    return samples


def summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(samples),
        "min_ms": ordered[0],
        "p90_ms": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=20423)
    parser.add_argument("--correctness-rows", type=int, default=129)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Global decode scale. Default derives max(abs(SwiGLU))/(6*448).",
    )
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument(
        "--direct-fp32",
        action="store_true",
        help="Skip eager-equivalent BF16 intermediate rounding (experimental).",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Also time exact dynamic scale reduction plus quantization.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path(__file__).resolve().parent / "results.json",
    )
    return parser.parse_args()


def derive_scale(raw: torch.Tensor) -> torch.Tensor:
    gate, up = raw.chunk(2, dim=-1)
    activated = F.silu(gate).mul_(up)
    # Division happens in BF16; only the resulting scalar is widened to FP32.
    return (activated.abs().amax() / (6.0 * 448.0)).to(torch.float32)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.rows <= 0 or args.correctness_rows <= 0:
        raise SystemExit("row counts must be positive")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    load_extension(verbose=args.verbose_build)

    # A non-multiple of both 16 and 128 exercises output and scale padding.
    check_raw = torch.randn(
        args.correctness_rows,
        2 * WIDTH,
        device=device,
        dtype=torch.bfloat16,
    )
    check_scale = (
        torch.tensor([args.scale], device=device, dtype=torch.float32)
        if args.scale is not None
        else derive_scale(check_raw).reshape(1)
    )
    correctness = compare_outputs(
        check_raw,
        check_scale,
        eager_bf16_rounding=not args.direct_fp32,
    )
    dynamic_correctness = compare_dynamic_outputs(check_raw)

    raw = torch.randn(
        args.rows,
        2 * WIDTH,
        device=device,
        dtype=torch.bfloat16,
    )
    scale = (
        torch.tensor([args.scale], device=device, dtype=torch.float32)
        if args.scale is not None
        else derive_scale(raw).reshape(1)
    )

    baseline_samples = time_cuda(
        lambda: eager_reference(raw, scale),
        warmup=args.warmup,
        repeats=args.repeats,
    )
    fused_samples = time_cuda(
        lambda: swiglu_nvfp4(
            raw,
            scale,
            eager_bf16_rounding=not args.direct_fp32,
        ),
        warmup=args.warmup,
        repeats=args.repeats,
    )
    baseline = summarize(baseline_samples)
    fused = summarize(fused_samples)
    dynamic_timing = None
    if args.dynamic:
        dynamic_baseline_samples = time_cuda(
            lambda: eager_dynamic_reference(raw),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        dynamic_native_samples = time_cuda(
            lambda: swiglu_nvfp4_dynamic(raw),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        dynamic_baseline = summarize(dynamic_baseline_samples)
        dynamic_native = summarize(dynamic_native_samples)
        dynamic_saved = (
            dynamic_baseline["median_ms"] - dynamic_native["median_ms"]
        )
        dynamic_timing = {
            "baseline": dynamic_baseline,
            "native": dynamic_native,
            "median_speedup": (
                dynamic_baseline["median_ms"] / dynamic_native["median_ms"]
            ),
            "median_saved_per_layer_ms": dynamic_saved,
            "estimated_saved_50_layers_ms": dynamic_saved * 50.0,
            "estimated_saved_50_layers_20_steps_s": dynamic_saved,
        }
    static_saved = baseline["median_ms"] - fused["median_ms"]
    result = {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "rows": args.rows,
        "raw_shape": [args.rows, 2 * WIDTH],
        "eager_bf16_rounding": not args.direct_fp32,
        "correctness": correctness,
        "dynamic_correctness": dynamic_correctness,
        "baseline": baseline,
        "fused": fused,
        "median_speedup": baseline["median_ms"] / fused["median_ms"],
        "median_saved_per_layer_ms": static_saved,
        "estimated_saved_50_layers_ms": static_saved * 50.0,
        "estimated_saved_50_layers_20_steps_s": static_saved,
        "dynamic_timing": dynamic_timing,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
