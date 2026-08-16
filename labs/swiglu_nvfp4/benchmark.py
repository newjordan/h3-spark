"""Correctness and exact-shape benchmark for fused SwiGLU -> NVFP4."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch
import comfy_kitchen as ck

from swiglu_nvfp4 import fused_swiglu_nvfp4


K = 14336


def eager_swiglu(x: torch.Tensor) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate).mul_(up)


def reference(x: torch.Tensor, scale: torch.Tensor):
    act = eager_swiglu(x)
    with ck.use_backend("cuda"):
        return ck.quantize_nvfp4(act, scale, pad_16x=x.shape[0] % 16 != 0)


def event_samples(fn, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        result = fn()
    del result
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
        del result
    return values


def tensor_error(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    af = a.float()
    bf = b.float()
    diff = af - bf
    return {
        "mean_abs": float(diff.abs().mean()),
        "max_abs": float(diff.abs().max()),
        "rel_l2": float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(af).clamp_min(1e-12)),
    }


def run_shape(rows: int, warmup: int, repeats: int, seed: int) -> dict:
    torch.manual_seed(seed + rows)
    x = torch.randn((rows, K * 2), device="cuda", dtype=torch.bfloat16)
    act = eager_swiglu(x)
    dynamic_scale = (torch.amax(act.abs()) / 2688.0).to(torch.float32)
    scale = dynamic_scale * 1.20

    ref_q, ref_s = reference(x, scale)
    fused_q, fused_s = fused_swiglu_nvfp4(x, scale)
    torch.cuda.synchronize()

    q_equal = bool(torch.equal(ref_q, fused_q))
    s_equal = bool(torch.equal(ref_s, fused_s))
    q_match = float((ref_q == fused_q).float().mean())
    s_match = float((ref_s.view(torch.uint8) == fused_s.view(torch.uint8)).float().mean())

    with ck.use_backend("cuda"):
        ref_dq = ck.dequantize_nvfp4(ref_q, scale, ref_s, output_type=torch.bfloat16)
        fused_dq = ck.dequantize_nvfp4(fused_q, scale, fused_s, output_type=torch.bfloat16)
    padded_act = torch.nn.functional.pad(act, (0, 0, 0, ref_dq.shape[0] - rows))
    correctness = {
        "qdata_equal": q_equal,
        "block_scale_equal": s_equal,
        "qdata_byte_match_fraction": q_match,
        "block_scale_byte_match_fraction": s_match,
        "dequant_fused_vs_reference": tensor_error(ref_dq, fused_dq),
        "reference_dequant_vs_bf16_activation": tensor_error(padded_act, ref_dq),
        "fused_dequant_vs_bf16_activation": tensor_error(padded_act, fused_dq),
    }
    del ref_q, ref_s, fused_q, fused_s, ref_dq, fused_dq, padded_act, act
    torch.cuda.empty_cache()

    eager_ms = event_samples(lambda: reference(x, scale), warmup, repeats)
    fused_ms = event_samples(lambda: fused_swiglu_nvfp4(x, scale), warmup, repeats)
    eager_median = statistics.median(eager_ms)
    fused_median = statistics.median(fused_ms)
    result = {
        "rows": rows,
        "raw_shape": [rows, K * 2],
        "activated_shape": [rows, K],
        "scale": float(scale),
        "correctness": correctness,
        "timing_ms": {
            "eager_swiglu_plus_cuda_nvfp4": eager_ms,
            "fused": fused_ms,
            "eager_median": eager_median,
            "fused_median": fused_median,
            "saved_per_layer": eager_median - fused_median,
            "speedup": eager_median / fused_median,
            "estimated_saved_50_layers_per_step": (eager_median - fused_median) * 50.0,
            "estimated_saved_20_steps_seconds": (eager_median - fused_median),
        },
    }
    del x
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="+", type=int, default=[20393, 16775])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default="benchmark_results.json")
    args = parser.parse_args()
    output = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "rows": [],
    }
    for rows in args.rows:
        result = run_shape(rows, args.warmup, args.repeats, args.seed)
        output["rows"].append(result)
        print(json.dumps(result, indent=2), flush=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()

