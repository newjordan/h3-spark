from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / "config" / "h3_spark_serving.env"


def parse_env(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition("=")
        assert separator == "=", stripped
        values[key] = value
    return values


def test_serving_defaults_match_the_fail_closed_config():
    values = parse_env(ENV_PATH.read_text(encoding="utf-8"))
    assert values["H3_ATTENTION"] == "sage2-quality"
    assert values["H3_MODULATION_FUSION"] == "exact"
    assert values["H3_SWIGLU_NVFP4_FUSION"] == "auto"
    assert values["H3_RMS_ADALN_NVFP4_FUSION"] == "auto"
    assert values["H3_Q_RMS_ROPE_INT8_FUSION"] == "auto"
    assert values["H3_CROSS_STEP_CACHE"] == "off"
    assert values["H3_NVFP4_STATIC_ARTIFACT"] == ""
    assert values["H3_NVFP4_EXCLUDED_LAYERS"] == ""
    assert values["COMFY_MINIMAX_H3_VAE_RMS_ROPE"] == "off"
    assert "H3_CROSSBLOCK_GATE_QKV_FUSION" not in values


def test_serving_env_has_no_secret_assignments():
    values = parse_env(ENV_PATH.read_text(encoding="utf-8"))
    for key, value in values.items():
        lowered = f"{key}={value}".lower()
        assert "password" not in lowered
        assert "token" not in lowered
        assert "api_key" not in lowered
        assert not value.startswith("sk-")
        assert "BEGIN " not in value
