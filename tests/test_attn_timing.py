from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load():
    path = ROOT / "integration" / "comfyui" / "custom_nodes" / "h3_attn_timing.py"
    spec = importlib.util.spec_from_file_location("h3_attn_timing", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_timing_is_off_unless_requested():
    timing = load()
    assert timing.timing_enabled({}) is False
    assert timing.timing_enabled({"H3_ATTN_TIMING": "off"}) is False
    assert timing.timing_enabled({"H3_ATTN_TIMING": "false"}) is False
    assert timing.timing_enabled({"H3_ATTN_TIMING": "on"}) is True
    assert timing.timing_enabled({"H3_ATTN_TIMING": " ON "}) is True


def test_timing_path_stays_in_the_working_directory_by_default():
    timing = load()
    assert timing.timing_path({}) == Path("h3_attn_timing.jsonl")
    assert timing.timing_path({"H3_ATTN_TIMING_FILE": "runs/attn.jsonl"}) == Path(
        "runs/attn.jsonl"
    )
    assert timing.timing_path({"H3_ATTN_TIMING_FILE": "~/attn.jsonl"}) == (
        Path.home() / "attn.jsonl"
    )


def test_sync_floor_rejects_garbage_and_keeps_the_legacy_name():
    timing = load()
    assert timing.sync_floor({}) == 900
    assert timing.sync_floor({"H3_ATTN_SYNC_FLOOR": "12"}) == 12
    assert timing.sync_floor({"_ATTN_SYNC_FLOOR": "4"}) == 4
    assert timing.sync_floor({"H3_ATTN_SYNC_FLOOR": "nope"}) == 900
    assert timing.sync_floor({"H3_ATTN_SYNC_FLOOR": "-1"}) == 900
