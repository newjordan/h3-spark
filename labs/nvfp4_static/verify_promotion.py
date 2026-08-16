#!/usr/bin/env python3
"""Fail-closed promotion check for an H3 NVFP4 artifact/report pair."""

from __future__ import annotations

import argparse
import json

from h3_nvfp4_static_scales import ArtifactError, verify_promotion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", help="v3 artifact prefix")
    parser.add_argument(
        "--exclude",
        default="",
        help="comma/newline-separated layers that must remain dynamic",
    )
    parser.add_argument("--report", default=None, help="override validation report path")
    args = parser.parse_args()
    try:
        result = verify_promotion(
            args.prefix,
            excluded_layers=args.exclude,
            report_path=args.report,
        )
    except ArtifactError as error:
        print(json.dumps({"pass": False, "reason": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
