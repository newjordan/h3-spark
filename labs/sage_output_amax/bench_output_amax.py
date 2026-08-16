"""Exactness and latency gate for SageAttention output-amax side production."""

from __future__ import annotations

import argparse
import json

import torch

from sageattention import sm89_compile
from sageattention.quant import per_channel_fp8, per_warp_int8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=20423)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--expect-amax", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(260812)
    shape = (1, args.seq, 56, 128)
    q = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    km = k.mean(dim=1, keepdim=True)
    qi, qs, ki, ks = per_warp_int8(
        q, k, km, tensor_layout="NHD", BLKQ=128, WARPQ=32, BLKK=64
    )
    vi, vs, _ = per_channel_fp8(
        v, tensor_layout="NHD", scale_max=448.0, smooth_v=False
    )
    output = torch.empty_like(q)

    def run():
        return sm89_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(
            qi, ki, vi, output, qs, ks, vs, 0, 0, 2, 128**-0.5, 0
        )

    for _ in range(3):
        marker = run()
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        marker = run()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop))

    result = {
        "sequence": args.seq,
        "median_ms": sorted(samples)[len(samples) // 2],
        "samples_ms": samples,
        "marker_shape": list(marker.shape),
    }
    if args.expect_amax:
        output_absmax_bits = int(
            output.view(torch.int16).bitwise_and(0x7FFF).max().item()
        )
        marker_bits = int(marker.item())
        result.update(
            output_amax_bits=output_absmax_bits,
            producer_amax_bits=marker_bits,
            output_amax_exact=marker_bits == output_absmax_bits,
        )
    print(json.dumps(result, indent=2))
    if args.expect_amax and not result["output_amax_exact"]:
        raise SystemExit("SageAttention output amax mismatch")


if __name__ == "__main__":
    main()
