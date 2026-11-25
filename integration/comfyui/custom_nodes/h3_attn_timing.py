"""Opt-in attention timing for the H3 SageAttention node.

This module does not import torch. Timing stays off unless
``H3_ATTN_TIMING`` is set, and the log file defaults to the working
directory so a run does not write into a machine-local home path.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

_OFF = frozenset({"", "0", "off", "false", "no"})
_DEFAULT_FILE = "h3_attn_timing.jsonl"
_DEFAULT_FLOOR = 900


def _env(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def timing_enabled(environ: Mapping[str, str] | None = None) -> bool:
    raw = _env(environ).get("H3_ATTN_TIMING", "off")
    return str(raw).strip().lower() not in _OFF


def timing_path(environ: Mapping[str, str] | None = None) -> Path:
    raw = str(_env(environ).get("H3_ATTN_TIMING_FILE", _DEFAULT_FILE)).strip()
    if not raw:
        raw = _DEFAULT_FILE
    return Path(raw).expanduser()


def sync_floor(
    environ: Mapping[str, str] | None = None,
    default: int = _DEFAULT_FLOOR,
) -> int:
    env = _env(environ)
    raw = env.get("H3_ATTN_SYNC_FLOOR", env.get("_ATTN_SYNC_FLOOR", str(default)))
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    if value < 0:
        return default
    return value
