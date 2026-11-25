from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALL = ROOT / "scripts" / "install_comfyui_overlay.sh"
GIT_ENV = {
    "GIT_AUTHOR_NAME": "h3-test",
    "GIT_AUTHOR_EMAIL": "h3-test@example.com",
    "GIT_COMMITTER_NAME": "h3-test",
    "GIT_COMMITTER_EMAIL": "h3-test@example.com",
}


def run(cmd, cwd, extra_env=None):
    env = os.environ.copy()
    env.update(GIT_ENV)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def init_repo(path: Path) -> str:
    path.mkdir()
    assert run(["git", "init"], path).returncode == 0
    (path / "README.md").write_text("upstream\n", encoding="utf-8")
    assert run(["git", "add", "README.md"], path).returncode == 0
    assert run(["git", "commit", "-m", "init"], path).returncode == 0
    revision = run(["git", "rev-parse", "HEAD"], path)
    assert revision.returncode == 0
    return revision.stdout.strip()


def test_usage_and_non_checkout(tmp_path):
    missing = run([str(INSTALL)], ROOT)
    assert missing.returncode == 2
    assert "usage:" in missing.stderr

    plain = tmp_path / "plain"
    plain.mkdir()
    not_git = run([str(INSTALL), str(plain)], ROOT)
    assert not_git.returncode == 2
    assert "not a Git checkout" in not_git.stderr


def test_wrong_revision_is_refused(tmp_path):
    checkout = tmp_path / "ComfyUI"
    init_repo(checkout)
    result = run([str(INSTALL), str(checkout)], ROOT)
    assert result.returncode == 2
    assert "expected ComfyUI" in result.stderr


def test_installs_overlay_and_skips_tests(tmp_path):
    checkout = tmp_path / "ComfyUI"
    revision = init_repo(checkout)
    result = run(
        [str(INSTALL), str(checkout)],
        ROOT,
        extra_env={"H3_COMFYUI_REVISION": revision},
    )
    assert result.returncode == 0, result.stderr
    assert "warning: H3_COMFYUI_REVISION overrides" in result.stderr
    assert (checkout / "custom_nodes" / "h3_sage_attention.py").is_file()
    assert (checkout / "custom_nodes" / "h3_attn_timing.py").is_file()
    assert not (checkout / "tests" / "test_h3_attn_timing_off.py").exists()
    assert (checkout / "README.md").read_text(encoding="utf-8") == "upstream\n"
    installed = sum(
        1
        for path in (ROOT / "integration" / "comfyui").rglob("*")
        if path.is_file() and "tests" not in path.relative_to(ROOT / "integration" / "comfyui").parts
        and path.name != "README.md"
    )
    assert f"Installed {installed} H3 overlay files" in result.stdout

    readme = checkout / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + "local edit\n", encoding="utf-8")
    dirty = run(
        [str(INSTALL), str(checkout)],
        ROOT,
        extra_env={"H3_COMFYUI_REVISION": revision},
    )
    assert dirty.returncode == 2
    assert "dirty" in dirty.stderr
