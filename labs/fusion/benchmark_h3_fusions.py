import argparse
import json
import math
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

import comfy_kitchen
from h3_fused_ops import segmented_adaln_bf16_, segmented_gate_bf16_, segmented_rms_adaln_bf16


S = 20_393
D = 5_376
EPS = 1e-5
SEGMENTS = [
    (0, 6, 1),
    (6, 399, 0),
    (399, 405, 1),
    (405, 807, 0),
    (807, 813, 1),
    (813, 1215, 0),
    (1215, 1221, 1),
    (1221, 1643, 0),
    (1643, 1649, 1),
    (1649, 2042, 0),
    (2042, 2048, 1),
    (2048, 2441, 0),
    (2441, 2601, 1),
    (2601, 2992, 6),
    (2992, 3392, 6),
    (3392, 3792, 6),
    (3792, 4212, 6),
    (4212, 4603, 6),
    (4603, 4994, 6),
    (4994, 5408, 5),
    (5408, 20393, 0),
]


def coalesce(segments):
    out = []
    for a, b, row in segments:
        if out and out[-1][1] == a and out[-1][2] == row:
            out[-1] = (out[-1][0], b, row)
        else:
            out.append((a, b, row))
    return out


def make_row_ids(segments, device):
    ids = torch.empty(S, dtype=torch.int32)
    for a, b, row in segments:
        ids[a:b] = row
    return ids.to(device)


def current_rms_adaln(x, weight, scale, shift, segments):
    h = F.rms_norm(x, (D,), weight, EPS)
    for a, b, row in segments:
        h[a:b].mul_(1.0 + scale[row].to(h.dtype)).add_(shift[row].to(h.dtype))
    return h


def torch_rms_fused_adaln(x, weight, scale, shift, row_ids):
    h = F.rms_norm(x, (D,), weight, EPS)
    return segmented_adaln_bf16_(h, scale, shift, row_ids)


def kitchen_rms_adaln(x, effective_scale, shift, segments):
    chunks = []
    for a, b, row in segments:
        chunks.append(comfy_kitchen.rms_adaln(x[a:b], effective_scale[row], shift[row], EPS))
    return torch.cat(chunks)


def current_gate_(x, other, gate, segments):
    for a, b, row in segments:
        x[a:b].addcmul_(other[a:b], gate[row].to(x.dtype))
    return x


def event_samples(fn, warmup, repetitions):
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def errors(actual, reference):
    diff = (actual.float() - reference.float()).abs()
    ref_abs = reference.float().abs()
    finite = torch.isfinite(actual)
    return {
        "finite_fraction": finite.float().mean().item(),
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "rmse": diff.square().mean().sqrt().item(),
        "max_rel_where_ref_gt_1e-2": (diff[ref_abs > 1e-2] / ref_abs[ref_abs > 1e-2]).max().item(),
        "exact_fraction": (actual == reference).float().mean().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    args = parser.parse_args()

    torch.manual_seed(1234)
    device = torch.device("cuda")
    segments = SEGMENTS
    compact_segments = coalesce(segments)
    row_ids = make_row_ids(segments, device)
    x = torch.randn((S, D), device=device, dtype=torch.bfloat16)
    other = torch.randn_like(x).mul_(0.2)
    weight = (torch.randn(D, device=device, dtype=torch.float32) * 0.15 + 0.92).to(torch.bfloat16)
    scale = torch.randn((7, D), device=device, dtype=torch.float32).mul_(0.20)
    shift = torch.randn((7, D), device=device, dtype=torch.float32).mul_(0.20)
    gate = torch.randn((7, D), device=device, dtype=torch.float32).mul_(0.20)
    effective_scale = weight.float().unsqueeze(0) * (1.0 + scale) - 1.0

    with torch.inference_mode():
        ref = current_rms_adaln(x, weight, scale, shift, segments)
        fused_staged = segmented_rms_adaln_bf16(x, weight, scale, shift, row_ids, EPS, staged_rounding=True)
        fused_fp32 = segmented_rms_adaln_bf16(x, weight, scale, shift, row_ids, EPS, staged_rounding=False)
        exact_two_launch = torch_rms_fused_adaln(x, weight, scale, shift, row_ids)
        with comfy_kitchen.use_backend("cuda"):
            kitchen = kitchen_rms_adaln(x, effective_scale, shift, segments)
        with comfy_kitchen.use_backend("triton"):
            kitchen_triton = kitchen_rms_adaln(x, effective_scale, shift, segments)

        gate_ref = x.clone()
        current_gate_(gate_ref, other, gate, segments)
        gate_fused = x.clone()
        segmented_gate_bf16_(gate_fused, other, gate, row_ids, staged_rounding=True)

        calibration_rows = 128
        x_gold = x[:calibration_rows].float()
        ids_gold = row_ids[:calibration_rows].long()
        gold = x_gold * torch.rsqrt(x_gold.square().mean(dim=-1, keepdim=True) + EPS)
        gold = gold * weight.float() * (1.0 + scale[ids_gold]) + shift[ids_gold]

        timings = {}
        timings["current_rms_adaln_21_segments"] = event_samples(
            lambda: current_rms_adaln(x, weight, scale, shift, segments), args.warmup, args.repetitions)
        timings["current_rms_adaln_16_coalesced"] = event_samples(
            lambda: current_rms_adaln(x, weight, scale, shift, compact_segments), args.warmup, args.repetitions)
        timings["fused_rms_adaln_staged"] = event_samples(
            lambda: segmented_rms_adaln_bf16(x, weight, scale, shift, row_ids, EPS, staged_rounding=True), args.warmup, args.repetitions)
        timings["fused_rms_adaln_fp32_math"] = event_samples(
            lambda: segmented_rms_adaln_bf16(x, weight, scale, shift, row_ids, EPS, staged_rounding=False), args.warmup, args.repetitions)
        timings["torch_rms_plus_fused_adaln_exact"] = event_samples(
            lambda: torch_rms_fused_adaln(x, weight, scale, shift, row_ids), args.warmup, args.repetitions)
        with comfy_kitchen.use_backend("cuda"):
            timings["comfy_kitchen_cuda_21_segments_plus_cat"] = event_samples(
                lambda: kitchen_rms_adaln(x, effective_scale, shift, segments), args.warmup, args.repetitions)
        with comfy_kitchen.use_backend("triton"):
            timings["comfy_kitchen_triton_21_segments_plus_cat"] = event_samples(
                lambda: kitchen_rms_adaln(x, effective_scale, shift, segments), args.warmup, args.repetitions)

        gate_current_input = x.clone()
        timings["current_gate_21_segments_inplace"] = event_samples(
            lambda: current_gate_(gate_current_input, other, gate, segments), args.warmup, args.repetitions)
        gate_current_compact_input = x.clone()
        timings["current_gate_16_coalesced_inplace"] = event_samples(
            lambda: current_gate_(gate_current_compact_input, other, gate, compact_segments), args.warmup, args.repetitions)
        gate_fused_input = x.clone()
        timings["fused_gate_inplace"] = event_samples(
            lambda: segmented_gate_bf16_(gate_fused_input, other, gate, row_ids, staged_rounding=True), args.warmup, args.repetitions)

    for name, result in timings.items():
        result["effective_gib_per_s_minimum_io"] = (
            (2 if "rms_adaln" in name else 3) * S * D * 2 / (1024 ** 3) / (result["median_ms"] / 1000)
        )

    report = {
        "environment": {
            "device": torch.cuda.get_device_name(),
            "capability": torch.cuda.get_device_capability(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "dtype": "bfloat16",
        },
        "shape": [S, D],
        "segments": len(segments),
        "segments_after_safe_coalescing": len(compact_segments),
        "accuracy_vs_current_bf16": {
            "fused_staged": errors(fused_staged, ref),
            "fused_fp32_math": errors(fused_fp32, ref),
            "torch_rms_plus_fused_adaln_exact": errors(exact_two_launch, ref),
            "comfy_kitchen_cuda_weight_folded": errors(kitchen, ref),
            "comfy_kitchen_triton_weight_folded": errors(kitchen_triton, ref),
            "fused_gate": errors(gate_fused, gate_ref),
        },
        "accuracy_vs_fp32_formula_first_128_rows": {
            "current_bf16": errors(ref[:calibration_rows], gold),
            "fused_staged": errors(fused_staged[:calibration_rows], gold),
            "fused_fp32_math": errors(fused_fp32[:calibration_rows], gold),
            "torch_rms_plus_fused_adaln_exact": errors(exact_two_launch[:calibration_rows], gold),
            "comfy_kitchen_triton_weight_folded": errors(kitchen_triton[:calibration_rows], gold),
        },
        "timings": timings,
        "source_launch_count_estimate": {
            "current_rms_adaln_21_segments": 1 + 5 * len(segments),
            "current_rms_adaln_16_coalesced": 1 + 5 * len(compact_segments),
            "fused_rms_adaln": 1,
            "torch_rms_plus_fused_adaln_exact": 2,
            "comfy_kitchen_21_segments_plus_cat": len(segments) + 1,
            "current_gate_21_segments": 2 * len(segments),
            "current_gate_16_coalesced": 2 * len(compact_segments),
            "fused_gate": 1,
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
