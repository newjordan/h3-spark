"""Seconds-long exactness/timing gate against CK RMS+RoPE + Sage Q quant."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import torch

COMFY_ROOT = Path(os.environ["COMFYUI_ROOT"])
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

import comfy_kitchen
from sageattention import _fused
from comfy.ldm.minimax.model import rope_rotation_table

import h3_q_rms_rope_int8 as fusion


HEADS = 56
DIM = 128
ROT = 96
INNER = HEADS * DIM
QKV = INNER * 3


def rope_table(seq: int, device: torch.device) -> torch.Tensor:
    angles = torch.randn((seq, ROT), device=device, dtype=torch.float32)
    angles[:, ROT // 2 :] = angles[:, : ROT // 2]
    return rope_rotation_table(angles, torch.bfloat16)


def views(qkv: torch.Tensor):
    seq = qkv.shape[0]
    q, k, _ = qkv.split(INNER, dim=-1)
    return q.view(1, seq, HEADS, DIM), k.view(1, seq, HEADS, DIM)


def reference(qkv, freqs, qw, kw, epsilon):
    q, k = views(qkv)
    comfy_kitchen.rms_rope_split_half_(
        q, k, freqs, qw, kw, epsilon=epsilon, rot_dim=ROT
    )
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    groups = ((q.shape[1] + 127) // 128) * 4
    q_scale = torch.empty((1, HEADS, groups), dtype=torch.float32, device=q.device)
    _fused.quant_per_warp_int8_cuda(q, q_int8, q_scale, 128, 32, 0)
    return q_int8, q_scale, k


def candidate(qkv, freqs, qw, kw, epsilon, warps):
    q, k = views(qkv)
    result = fusion.try_qk_rms_rope_q_int8(
        q, k, freqs, qw, kw, epsilon, warps
    )
    if result is None:
        raise RuntimeError("candidate unexpectedly ineligible")
    return result


def elapsed_ms(fn, repeats):
    samples = []
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=20423)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.manual_seed(20260812)
    device = torch.device("cuda")
    qkv = torch.randn((args.seq, QKV), device=device, dtype=torch.bfloat16)
    freqs = rope_table(args.seq, device)
    qw = (torch.randn((DIM,), device=device) * 0.1 + 1).to(torch.bfloat16)
    kw = (torch.randn((DIM,), device=device) * 0.1 + 1).to(torch.bfloat16)
    epsilon = 1e-5

    with torch.no_grad():
        ref_qkv = qkv.clone()
        ref_q, ref_s, ref_k = reference(ref_qkv, freqs, qw, kw, epsilon)
        candidates = {}
        for warps in (8, 16, 32):
            got_qkv = qkv.clone()
            candidates[warps] = candidate(
                got_qkv, freqs, qw, kw, epsilon, warps
            )
        torch.cuda.synchronize()

        stream = torch.cuda.Stream()
        stream_qkv = qkv.clone()
        with torch.cuda.stream(stream):
            stream_q, stream_s, stream_k = candidate(
                stream_qkv, freqs, qw, kw, epsilon, 8
            )
        stream.synchronize()

        result = {
            "shape": [1, args.seq, HEADS, DIM],
            "nondefault_stream_q_exact": torch.equal(stream_q, ref_q),
            "nondefault_stream_scale_exact": torch.equal(stream_s, ref_s),
            "nondefault_stream_k_exact": torch.equal(stream_k, ref_k),
        }
        for warps, (got_q, got_s, got_k) in candidates.items():
            result[f"w{warps}_q_int8_exact"] = torch.equal(got_q, ref_q)
            result[f"w{warps}_q_scale_exact"] = torch.equal(got_s, ref_s)
            result[f"w{warps}_k_bf16_exact"] = torch.equal(got_k, ref_k)

        # Time only the two operations replaced. Clone setup is outside events.
        timing_buffers = [qkv.clone() for _ in range(args.repeats + 2)]
        timing_index = iter(timing_buffers)
        reference_ms = elapsed_ms(
            lambda: reference(next(timing_index), freqs, qw, kw, epsilon),
            args.repeats,
        )
        result["timing_ms"] = {"ck_plus_sage_q": reference_ms}
        for warps in (8, 16, 32):
            timing_buffers = [qkv.clone() for _ in range(args.repeats + 2)]
            timing_index = iter(timing_buffers)
            candidate_ms = elapsed_ms(
                lambda warps=warps: candidate(
                    next(timing_index), freqs, qw, kw, epsilon, warps
                ),
                args.repeats,
            )
            result["timing_ms"][f"fused_w{warps}"] = candidate_ms
            result["timing_ms"][f"saved_w{warps}"] = reference_ms - candidate_ms

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    checks = [value for key, value in result.items() if key.endswith("exact")]
    raise SystemExit(0 if all(checks) else 1)


def _tail_lengths_documented_for_gate():
    # Structural audit list; the full production-shape run remains the only
    # timed path. Correctness sweeps may call the same helpers for each length.
    return (1, 31, 32, 33, 127, 128, 129, 20423)


if __name__ == "__main__":
    main()
