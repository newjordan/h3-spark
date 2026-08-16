"""Measure the exact abs+amax work that a static input scale removes."""

from __future__ import annotations

import argparse
import json
import platform
import statistics

import torch


WIDTHS = (5376, 5376, 7168, 14336)
LAYERS_PER_WIDTH = (100, 0, 50, 50)  # two 5376-input linears + one each per block


def timed_reduction(value: torch.Tensor, repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.amax(value.abs()) / 2688.0
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=20393)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    torch.cuda.set_device(0)
    results = []
    counts = {5376: 100, 7168: 50, 14336: 50}
    for width, layer_count in counts.items():
        value = torch.randn(args.rows, width, device="cuda", dtype=torch.bfloat16)
        timed_reduction(value, 2)
        samples = timed_reduction(value, args.repeats)
        median = statistics.median(samples)
        results.append(
            {
                "shape": [args.rows, width],
                "layers_per_h3_forward": layer_count,
                "median_ms": median,
                "samples_ms": samples,
                "estimated_h3_forward_ms": median * layer_count,
            }
        )
        del value

    estimated = sum(item["estimated_h3_forward_ms"] for item in results)
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "torch": torch.__version__,
                "python": platform.python_version(),
                "rows": args.rows,
                "results": results,
                "estimated_abs_amax_ms_per_h3_forward": estimated,
                "estimated_abs_amax_seconds_for_20_steps": estimated * 20 / 1000,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
