from __future__ import annotations

from types import SimpleNamespace

import torch

from comfy.ldm.minimax import rms_adaln_nvfp4 as fused


def _cpu_inputs():
    norm = torch.nn.RMSNorm(
        fused.WIDTH, eps=fused.CERTIFIED_H3_RMS_EPS, dtype=torch.bfloat16
    )
    norm.weight.requires_grad_(False)
    linear = SimpleNamespace(in_features=fused.WIDTH, out_features=21504)
    x = torch.empty(2, fused.WIDTH, dtype=torch.bfloat16)
    table = torch.empty(1, fused.WIDTH, dtype=torch.float32)
    row_ids = torch.zeros(2, dtype=torch.int32)
    return linear, norm, x, table, row_ids


def test_cpu_input_fails_before_native_load(monkeypatch):
    monkeypatch.setattr(
        fused,
        "_load_native",
        lambda: (_ for _ in ()).throw(AssertionError("native loader called")),
    )
    linear, norm, x, table, row_ids = _cpu_inputs()
    assert fused.try_linear(
        linear,
        norm,
        x,
        table,
        table,
        row_ids,
        fused.ROW_IDS_TOKEN,
        "auto",
        "exact",
    ) is None


def test_stock_and_untrusted_map_fail_before_guards(monkeypatch):
    monkeypatch.setattr(
        fused,
        "_tensor_inputs_eligible",
        lambda *_args: (_ for _ in ()).throw(AssertionError("guard evaluated")),
    )
    for mode, token in (("stock", fused.ROW_IDS_TOKEN), ("auto", object())):
        assert fused.try_linear(
            object(), object(), object(), object(), object(), object(),
            token, mode, "exact"
        ) is None


def test_native_abi_requires_version_build_id_and_entrypoint():
    good = SimpleNamespace(
        abi_version=lambda: fused.NATIVE_ABI_VERSION,
        build_id=lambda: fused.NATIVE_BUILD_ID,
        rms_adaln_nvfp4_dynamic=lambda *_args: None,
    )
    assert fused._native_abi_eligible(good)
    assert not fused._native_abi_eligible(
        SimpleNamespace(**{**good.__dict__, "abi_version": lambda: 1})
    )
    assert not fused._native_abi_eligible(
        SimpleNamespace(**{**good.__dict__, "build_id": lambda: "stale"})
    )


def test_certified_epsilon_is_exact():
    _linear, norm, x, _table, _row_ids = _cpu_inputs()
    with torch.inference_mode():
        assert fused._norm_eligible(norm, x)
        norm.eps = 1e-6
        assert not fused._norm_eligible(norm, x)
