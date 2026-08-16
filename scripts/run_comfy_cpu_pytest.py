#!/usr/bin/env python3
"""Run pytest with ComfyUI forced into CPU mode before test collection."""

from __future__ import annotations

import sys

import comfy.options


def main() -> int:
    pytest_args = sys.argv[1:]
    # comfy.cli_args parses the process argv lazily when the tested modules
    # import model_management. Keep pytest's arguments out of that parser and
    # make device discovery CPU-only.
    sys.argv = [sys.argv[0], "--cpu"]
    comfy.options.enable_args_parsing()

    import pytest

    return pytest.main(pytest_args)


if __name__ == "__main__":
    raise SystemExit(main())
