from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_native.sh"


def test_list_matches_cmake_projects():
    syntax = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr

    listed = subprocess.run(
        ["bash", str(SCRIPT), "--list"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert listed.returncode == 0, listed.stderr
    lines = [line for line in listed.stdout.splitlines() if line]
    assert lines
    assert lines == sorted(lines)
    assert "labs/q_rms_rope_int8" in lines
    assert "labs/swiglu_nvfp4/native_cuda" in lines
    for line in lines:
        assert (ROOT / line / "CMakeLists.txt").is_file()
        assert "cmake-build" not in line
        assert not line.endswith("/build")

    usage = subprocess.run(
        ["bash", str(SCRIPT), "--bogus"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert usage.returncode == 2
    assert "usage:" in usage.stderr
