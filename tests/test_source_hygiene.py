from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".sh", ".md", ".env", ".yml", ".yaml", ".txt", ".cmake"}
SKIP_PARTS = {".git", "third_party_licenses"}
GENERATED_SUFFIXES = {".so", ".o", ".cubin", ".ptx", ".fatbin"}
SECRET_MARKERS = (
    "BEGIN RSA PRIVATE KEY",
    "BEGIN OPENSSH PRIVATE KEY",
    "BEGIN PRIVATE KEY",
    "AKIA",
    "ghp_",
    "github_pat_",
    "xoxb-",
    "xoxp-",
)


def _text_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.name == "test_source_hygiene.py":
            continue
        if path.suffix not in TEXT_SUFFIXES and path.name not in {"CMakeLists.txt", "Makefile"}:
            continue
        if path.stat().st_size > 2_000_000:
            continue
        yield path


def test_source_does_not_use_an_unexpanded_home_token():
    offenders = []
    for path in _text_files():
        if path.suffix == ".json":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "$HOME" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_no_generated_binaries_or_cmake_build_trees():
    generated = []
    for path in ROOT.rglob("*"):
        if ".git" in path.parts:
            continue
        if "cmake-build" in path.parts or path.suffix in GENERATED_SUFFIXES:
            generated.append(str(path.relative_to(ROOT)))
    assert generated == []


def test_no_obvious_secrets_in_text_files():
    hits = []
    for path in _text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for marker in SECRET_MARKERS:
            if marker in text:
                hits.append(f"{path.relative_to(ROOT)}:{marker}")
    assert hits == []


def test_attention_node_does_not_advertise_the_missing_backend():
    node = (
        ROOT / "integration" / "comfyui" / "custom_nodes" / "h3_sage_attention.py"
    ).read_text(encoding="utf-8")
    vae = (
        ROOT / "integration" / "comfyui" / "comfy" / "ldm" / "minimax" / "vae.py"
    ).read_text(encoding="utf-8")
    assert '["quality", "fast", "stock"]' in node
    assert "sage3_kfold is not shipped" in node
    assert "falling back to the production kernel" not in node
    assert "silu_mul_native_v1.so" in vae
    assert "vae_swiglu_lab" not in vae
