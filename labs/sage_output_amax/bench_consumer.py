"""Full-shape exactness and latency gate for producer-supplied BF16 amax."""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from swiglu_nvfp4 import bf16_nvfp4_dynamic, bf16_nvfp4_from_amax_bits


def timed_ms(function, *, warmups: int, repeats: int) -> dict[str, object]:
    for _ in range(warmups):
        function()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop))
    return {
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
        "samples_ms": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=int, default=20423)
    parser.add_argument("--width", type=int, default=7168)
    parser.add_argument("--repeats", type=int, default=11)
    args = parser.parse_args()

    torch.manual_seed(20260812)
    source = (
        torch.randn(
            (args.seq, args.width), dtype=torch.bfloat16, device="cuda"
        )
        * 3
    ).contiguous()
    max_bits = (
        source.view(torch.int16)
        .bitwise_and(0x7FFF)
        .max()
        .to(torch.int32)
        .reshape(1)
    )

    dynamic = bf16_nvfp4_dynamic(source)
    supplied = bf16_nvfp4_from_amax_bits(source, max_bits)
    torch.cuda.synchronize()
    correctness = {
        "packed_equal": torch.equal(dynamic[0], supplied[0]),
        "scales_equal": torch.equal(
            dynamic[1].view(torch.uint8), supplied[1].view(torch.uint8)
        ),
        "global_equal": torch.equal(dynamic[2], supplied[2]),
        "amax_bits": int(max_bits.item()),
        "global_scale": float(supplied[2].item()),
    }
    dynamic_timing = timed_ms(
        lambda: bf16_nvfp4_dynamic(source), warmups=4, repeats=args.repeats
    )
    supplied_timing = timed_ms(
        lambda: bf16_nvfp4_from_amax_bits(source, max_bits),
        warmups=4,
        repeats=args.repeats,
    )
    saved_ms = (
        float(dynamic_timing["median_ms"])
        - float(supplied_timing["median_ms"])
    )
    result = {
        "shape": [args.seq, args.width],
        "correctness": correctness,
        "dynamic": dynamic_timing,
        "from_amax_bits": supplied_timing,
        "saved_ms_per_call": saved_ms,
        "projected_saved_s_20_steps_50_blocks": saved_ms,
    }
    print(json.dumps(result, indent=2))
    if not all(
        correctness[key]
        for key in ("packed_equal", "scales_equal", "global_equal")
    ):
        raise SystemExit("producer-supplied amax changed NVFP4 output")


if __name__ == "__main__":
    main()
