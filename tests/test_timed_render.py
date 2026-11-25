from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

BUILDER = '''
def workflow(seed, name, prompt, refs, **kwargs):
    return {
        "prompt": {
            "124": {"inputs": {"steps": 20}},
            "126": {"inputs": {}},
            "127": {"inputs": {}},
            "136": {"inputs": {}},
            "125": {"inputs": {}},
        },
        "seen": {"refs": list(refs), "kwargs": kwargs, "name": name},
    }
'''


def load():
    path = ROOT / "scripts" / "h3_timed_render.py"
    spec = importlib.util.spec_from_file_location("h3_timed_render", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def args_for(module, builder: Path, *extra: str):
    return module.parse_args(
        [
            "--workflow-builder",
            str(builder),
            "--tag",
            "warm",
            "--prompt",
            "a knight",
            "--refs",
            " a.png, b.png ",
            *extra,
        ]
    )


def test_lab_preset_and_serving_preset_build_different_wrappers(tmp_path):
    builder = tmp_path / "builder.py"
    builder.write_text(BUILDER, encoding="utf-8")
    module = load()

    lab = module.prepare_job(args_for(module, builder))
    assert lab["seen"]["refs"] == ["a.png", "b.png"]
    assert lab["prompt"]["202"]["inputs"]["mode"] == "stock"
    assert lab["prompt"]["202"]["inputs"]["q_rms_rope_int8_fusion"] == "auto"
    assert lab["prompt"]["202"]["inputs"]["swiglu_nvfp4_fusion"] == "stock"
    assert "203" not in lab["prompt"]

    serving = module.prepare_job(args_for(module, builder, "--preset", "serving"))
    assert serving["prompt"]["202"]["inputs"]["mode"] == "quality"
    assert serving["prompt"]["202"]["inputs"]["swiglu_nvfp4_fusion"] == "auto"
    assert serving["prompt"]["202"]["inputs"]["rms_adaln_nvfp4_fusion"] == "auto"
    assert serving["prompt"]["202"]["inputs"]["crossblock_gate_qkv_fusion"] == "stock"
    assert serving["prompt"]["124"]["inputs"]["model"] == ["202", 0]
    assert serving["prompt"]["126"]["inputs"]["model"] == ["202", 0]


def test_explicit_stock_flags_omit_the_wrapper(tmp_path):
    builder = tmp_path / "builder.py"
    builder.write_text(BUILDER, encoding="utf-8")
    module = load()
    job = module.prepare_job(
        args_for(
            module,
            builder,
            "--preset",
            "serving",
            "--attention",
            "stock",
            "--fusion",
            "exact",
            "--swiglu-nvfp4-fusion",
            "stock",
            "--rms-adaln-nvfp4-fusion",
            "stock",
            "--q-rms-rope-int8-fusion",
            "stock",
            "--crossblock-gate-qkv-fusion",
            "stock",
        )
    )
    assert "202" not in job["prompt"]
    assert job["prompt"]["124"]["inputs"]["model"] == ["127", 0]


def test_sampler_only_and_missing_nodes(tmp_path):
    builder = tmp_path / "builder.py"
    builder.write_text(BUILDER, encoding="utf-8")
    module = load()
    job = module.prepare_job(args_for(module, builder, "--sampler-only"))
    assert job["prompt"]["92"]["class_type"] == "PreviewAny"
    assert job["prompt"]["92"]["inputs"]["source"] == ["125", 0]

    broken = tmp_path / "broken.py"
    broken.write_text(
        "def workflow(*args, **kwargs):\n    return {'prompt': {}}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing node id"):
        module.prepare_job(args_for(module, broken))


def test_result_record_keeps_every_fusion_flag(tmp_path):
    builder = tmp_path / "builder.py"
    builder.write_text(BUILDER, encoding="utf-8")
    module = load()
    parsed = args_for(module, builder, "--preset", "serving")
    record = module.result_record(
        parsed,
        status="success",
        wall_seconds=1.04,
        executor_seconds=1.0,
        outputs=["clip.mp4"],
        host="test-host",
    )
    assert record["preset"] == "serving"
    assert record["attention"] == "sage2-quality"
    assert record["q_rms_rope_int8_fusion"] == "auto"
    assert record["crossblock_gate_qkv_fusion"] == "stock"
    assert record["wall_seconds"] == 1.0
    assert record["outputs"] == ["clip.mp4"]


def test_unreachable_api_is_json_not_a_traceback(tmp_path, capsys):
    builder = tmp_path / "builder.py"
    builder.write_text(BUILDER, encoding="utf-8")
    module = load()
    status = module.main(
        [
            "--workflow-builder",
            str(builder),
            "--tag",
            "warm",
            "--prompt",
            "a knight",
            "--refs",
            "a.png",
            "--api",
            "http://127.0.0.1:9",
            "--preset",
            "lab",
        ]
    )
    assert status == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["tag"] == "warm"
    assert payload["error"].startswith("API_UNREACHABLE")


def test_static_scales_require_a_prefix():
    module = load()
    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "--workflow-builder",
                "builder.py",
                "--tag",
                "t",
                "--prompt",
                "p",
                "--refs",
                "a.png",
                "--nvfp4-scales",
                "static",
            ]
        )
