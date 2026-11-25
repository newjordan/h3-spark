from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load():
    path = ROOT / "scripts" / "h3_paths.py"
    spec = importlib.util.spec_from_file_location("h3_paths", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_missing_env_is_a_usage_error(monkeypatch):
    monkeypatch.delenv("COMFYUI_ROOT", raising=False)
    paths = load()
    with pytest.raises(SystemExit, match="COMFYUI_ROOT is required"):
        paths.require_env_path("COMFYUI_ROOT", kind="dir")


def test_require_file_and_dir(tmp_path, monkeypatch):
    paths = load()
    source = tmp_path / "model.py"
    source.write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setenv("H3_LIVE_MODEL", str(source))
    assert paths.require_env_path("H3_LIVE_MODEL", kind="file") == source

    monkeypatch.setenv("COMFYUI_ROOT", str(tmp_path))
    assert paths.require_env_path("COMFYUI_ROOT", kind="dir") == tmp_path

    monkeypatch.setenv("COMFYUI_ROOT", str(source))
    with pytest.raises(SystemExit, match="not a directory"):
        paths.require_env_path("COMFYUI_ROOT", kind="dir")


def test_optional_expands_home(monkeypatch):
    paths = load()
    monkeypatch.setenv("H3_PROFILE_TXT", "~/h3-not-a-real-profile.txt")
    resolved = paths.optional_env_path("H3_PROFILE_TXT")
    assert resolved == Path.home() / "h3-not-a-real-profile.txt"
    monkeypatch.delenv("H3_PROFILE_TXT")
    assert paths.optional_env_path("H3_PROFILE_TXT") is None
