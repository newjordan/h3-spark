#!/usr/bin/env python3
"""Merge independent v3 H3 NVFP4 calibration artifact pairs offline."""

from __future__ import annotations

import argparse

from safetensors import SafetensorError

from h3_nvfp4_static_scales import ArtifactError, merge_calibrations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge two or more compatible v3 NVFP4 calibrations using per-layer "
            "maximum exact dynamic scales."
        )
    )
    parser.add_argument(
        "sources",
        nargs="+",
        help="Source artifact prefixes (with or without the v3 suffix/extension)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Distinct output artifact prefix",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=None,
        help=(
            "Optional output safety margin. It may increase, but not decrease, "
            "the common source margin."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        json_path, tensor_path = merge_calibrations(
            args.sources,
            args.output,
            output_margin=args.margin,
        )
    except (ArtifactError, OSError, ValueError, SafetensorError) as error:
        raise SystemExit(f"merge rejected: {error}") from error
    print(f"merged_json={json_path}")
    print(f"merged_safetensors={tensor_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
