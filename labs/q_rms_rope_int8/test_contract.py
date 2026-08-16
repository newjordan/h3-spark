from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest
import torch

import h3_q_rms_rope_int8 as fusion


class _FakeNative:
    def qk_rms_rope_q_int8(self, *args):
        return args


def _cpu_inputs(seq=33):
    q = torch.empty_strided((1, seq, 56, 128), (seq * 21504, 21504, 128, 1), dtype=torch.bfloat16)
    k = torch.empty_strided((1, seq, 56, 128), (seq * 21504, 21504, 128, 1), dtype=torch.bfloat16)
    freqs = torch.empty((1, seq, 1, 48, 2, 2), dtype=torch.bfloat16)
    weight = torch.empty((128,), dtype=torch.bfloat16)
    return q, k, freqs, weight, weight.clone()


def test_source_parses_and_uses_current_stream():
    ast.parse((Path(__file__).parent / "h3_q_rms_rope_int8.py").read_text())
    source = (Path(__file__).parent / "q_rms_rope_int8_cuda.cu").read_text()
    assert "getCurrentCUDAStream" in source
    assert "CUDAGuard" in source
    assert "C10_CUDA_KERNEL_LAUNCH_CHECK" in source
    assert "<<<grid, WARPS * 32, 0, stream>>>" in source
    assert "#include <torch/csrc/autograd/VariableTypeUtils.h>" in source
    launch_tail = source.split("#undef H3_LAUNCH", 1)[1]
    assert launch_tail.index("C10_CUDA_KERNEL_LAUNCH_CHECK();") < launch_tail.index(
        "torch::autograd::increment_version(k);"
    )
    assert source.count("torch::autograd::increment_version(k);") == 1
    facade = (Path(__file__).parent / "h3_q_rms_rope_int8.py").read_text()
    assert "glob(" not in facade
    assert '"build" / f"{_MODULE_NAME}.so"' in facade
    assert "build_id = module.build_id()" in facade
    assert "build_id != _BUILD_ID" in facade
    assert 'torch.__version__ != _TORCH_VERSION' in facade
    assert 'torch.version.git_version != _TORCH_GIT' in facade
    assert 'torch.version.cuda != _CUDA_VERSION' in facade
    assert 'torch.cuda.get_device_capability(index) == _DEVICE_CAPABILITY' in facade
    assert 'except (ImportError, OSError, RuntimeError)' in facade
    assert 'except (AttributeError, RuntimeError)' in facade
    assert 'and q.shape[1] > 1024' in facade
    assert 'h3-q-rms-rope96-qint8-v3-torch2.13-sm120' in facade


def test_cpu_and_grad_fail_closed(monkeypatch):
    monkeypatch.setattr(fusion, "_NATIVE", _FakeNative())
    inputs = _cpu_inputs()
    assert fusion.try_qk_rms_rope_q_int8(*inputs, 1e-5) is None
    with torch.no_grad():
        assert fusion.try_qk_rms_rope_q_int8(*inputs, 1e-5) is None


def test_missing_native_fails_closed(monkeypatch):
    monkeypatch.setattr(fusion, "_NATIVE", None)
    with torch.no_grad():
        assert fusion.try_qk_rms_rope_q_int8(*_cpu_inputs(), 1e-5) is None


def test_invalid_warp_count_is_explicitly_rejected():
    assert "warps_per_block not in (8, 16, 32)" in Path(
        fusion.__file__
    ).read_text()


def test_contract_is_narrow():
    source = (Path(__file__).parent / "q_rms_rope_int8_cuda.cu").read_text()
    for fragment in (
        "q.size(0) == 1",
        "q.size(2) == kHeads",
        "q.size(3) == kHeadDim",
        "q.size(1) > 1024",
        "constexpr int kRotDim = 96",
        "const int tail = kRotDim + lane",
        "epsilon == 1.0e-5",
        "k.stride(1) == kQkvRowWidth",
        "k.storage_offset() == q.storage_offset() + kHeads * kHeadDim",
        "freqs.is_contiguous()",
        "q_weight.is_contiguous()",
        "!q.requires_grad()",
    ):
        assert fragment in source


def test_gates_import_real_h3_rope_builder_and_production_epsilon():
    for name in ("gate.py", "gate_attention.py"):
        source = (Path(__file__).parent / name).read_text()
        assert "from comfy.ldm.minimax.model import rope_rotation_table" in source
        assert 'os.environ["COMFYUI_ROOT"]' in source
        assert "rope_rotation_table(angles, torch.bfloat16)" in source
        assert "1e-5" in source
    gate_source = (Path(__file__).parent / "gate.py").read_text()
    assert "(1, 31, 32, 33, 127, 128, 129, 20423)" in gate_source
