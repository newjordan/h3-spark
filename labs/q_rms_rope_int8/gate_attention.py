"""Short full-attention equivalence gate for the staged prequantized-Q API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import comfy_kitchen


LAB = Path(__file__).resolve().parent
SAGE_LAB = LAB.parent / "sageattention_amax_lab"
COMFY_ROOT = Path(os.environ["COMFYUI_ROOT"])
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(SAGE_LAB))

import h3_q_rms_rope_int8 as fusion
from comfy.ldm.minimax.model import rope_rotation_table
from sageattention import (
    sageattn_h3_prequantized_q_pv_fp8_cuda,
    sageattn_qk_int8_pv_fp8_cuda,
)


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
    q, k, v = qkv.split(INNER, dim=-1)
    return (
        q.view(1, seq, HEADS, DIM),
        k.view(1, seq, HEADS, DIM),
        v.view(1, seq, HEADS, DIM),
    )


def baseline(qkv, freqs, qw, kw):
    q, k, v = views(qkv)
    comfy_kitchen.rms_rope_split_half_(
        q, k, freqs, qw, kw, epsilon=1e-5, rot_dim=ROT
    )
    return sageattn_qk_int8_pv_fp8_cuda(
        q,
        k,
        v,
        tensor_layout="NHD",
        is_causal=False,
        qk_quant_gran="per_warp",
        sm_scale=1.0 / DIM**0.5,
        pv_accum_dtype="fp32+fp32",
        smooth_k=True,
    )


def candidate(qkv, freqs, qw, kw):
    q, k, v = views(qkv)
    preprocessed = fusion.try_qk_rms_rope_q_int8(
        q, k, freqs, qw, kw, 1e-5, 16
    )
    if preprocessed is None:
        raise RuntimeError("fused Q preprocessor unexpectedly ineligible")
    q_int8, q_scale, k = preprocessed
    return sageattn_h3_prequantized_q_pv_fp8_cuda(q_int8, q_scale, k, v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=20423)
    args = parser.parse_args()
    torch.manual_seed(20260812)
    device = torch.device("cuda")
    qkv = torch.randn((args.seq, QKV), device=device, dtype=torch.bfloat16)
    freqs = rope_table(args.seq, device)
    qw = (torch.randn((DIM,), device=device) * 0.1 + 1).to(torch.bfloat16)
    kw = (torch.randn((DIM,), device=device) * 0.1 + 1).to(torch.bfloat16)

    with torch.no_grad():
        reference = baseline(qkv.clone(), freqs, qw, kw)
        got = candidate(qkv.clone(), freqs, qw, kw)
        torch.cuda.synchronize()

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            got_stream = candidate(qkv.clone(), freqs, qw, kw)
        stream.synchronize()

    result = {
        "shape": list(reference.shape),
        "attention_bf16_exact": torch.equal(got, reference),
        "nondefault_stream_attention_exact": torch.equal(got_stream, reference),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if all(v for k, v in result.items() if k.endswith("exact")) else 1)


if __name__ == "__main__":
    main()
