from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import comfy.ldm.minimax.adaln_nvfp4 as staged_adaln
import comfy.ldm.minimax.model as AUDIT_MODEL

import comfy_adapter as adapter


class TorchOperations:
    Linear = torch.nn.Linear
    RMSNorm = torch.nn.RMSNorm


def make_block():
    block = AUDIT_MODEL.DiTBlock(
        hidden=4,
        heads=2,
        head_dim=2,
        ffn=8,
        t_dim=2,
        eps=1e-6,
        qk_eps=1e-6,
        operations=TorchOperations,
    )
    for parameter in block.parameters():
        parameter.requires_grad_(False)
    return block


def run_block(block, options=None, token=None):
    return block(
        torch.randn(3, 4),
        torch.randn(1, 2),
        [(0, 3, 0)],
        None,
        transformer_options=options or {},
        mod_row_ids=torch.zeros(3, dtype=torch.int32),
        adaln_nvfp4_token=token,
    )


def identity_attention(_q, _k, v, _heads, **_kwargs):
    return v


def test_cpu_input_fails_before_native_load(monkeypatch):
    monkeypatch.setattr(
        adapter,
        "_load_native",
        lambda: (_ for _ in ()).throw(AssertionError("native loader called")),
    )
    norm = torch.nn.RMSNorm(
        adapter.WIDTH, eps=adapter.CERTIFIED_H3_RMS_EPS,
        dtype=torch.bfloat16
    )
    norm.weight.requires_grad_(False)
    linear = SimpleNamespace(in_features=adapter.WIDTH, out_features=21504)
    x = torch.empty(2, adapter.WIDTH, dtype=torch.bfloat16)
    table = torch.empty(1, adapter.WIDTH, dtype=torch.float32)
    row_ids = torch.zeros(2, dtype=torch.int32)
    assert adapter.try_linear(
        linear,
        norm,
        x,
        table,
        table,
        row_ids,
        staged_adaln.ROW_IDS_TOKEN,
        "auto",
        "exact",
    ) is None


@pytest.mark.parametrize(
    ("mode", "modulation", "trusted"),
    [
        ("stock", "exact", True),
        ("static", "exact", True),
        ("auto", "stock", True),
        ("auto", "aggressive", True),
        ("auto", "exact", False),
    ],
)
def test_non_exact_dynamic_routes_never_evaluate_guards(
    monkeypatch, mode, modulation, trusted
):
    monkeypatch.setattr(
        adapter,
        "_tensor_inputs_eligible",
        lambda *_args: (_ for _ in ()).throw(AssertionError("guard evaluated")),
    )
    token = staged_adaln.ROW_IDS_TOKEN if trusted else object()
    assert adapter.try_linear(
        object(), object(), object(), object(), object(), object(), token,
        mode, modulation
    ) is None


def test_norm_guard_rejects_hooks_patches_lowvram_and_override(monkeypatch):
    monkeypatch.setattr(adapter.comfy.model_management, "in_training", False)
    x = torch.empty(2, adapter.WIDTH, dtype=torch.bfloat16)

    def fresh_norm():
        value = torch.nn.RMSNorm(
            adapter.WIDTH,
            eps=adapter.CERTIFIED_H3_RMS_EPS,
            dtype=torch.bfloat16,
        )
        value.weight.requires_grad_(False)
        return value

    norm = fresh_norm()
    with torch.no_grad():
        assert adapter._norm_eligible(norm, x)
    assert not adapter._norm_eligible(norm, x)

    handle = norm.register_forward_hook(lambda *_args: None)
    try:
        assert not adapter._norm_eligible(norm, x)
    finally:
        handle.remove()

    norm = fresh_norm()
    norm.weight_function = [lambda weight: weight]
    assert not adapter._norm_eligible(norm, x)

    norm = fresh_norm()
    norm.weight_lowvram_function = object()
    assert not adapter._norm_eligible(norm, x)

    class OverriddenRMSNorm(torch.nn.RMSNorm):
        def forward(self, value):
            return super().forward(value)

    norm = OverriddenRMSNorm(
        adapter.WIDTH, eps=adapter.CERTIFIED_H3_RMS_EPS,
        dtype=torch.bfloat16
    )
    norm.weight.requires_grad_(False)
    assert not adapter._norm_eligible(norm, x)

    norm = fresh_norm()
    norm.eps = 1e-6
    with torch.no_grad():
        assert not adapter._norm_eligible(norm, x)

    norm = fresh_norm()
    global_handle = torch.nn.modules.module.register_module_forward_pre_hook(
        lambda *_args: None
    )
    try:
        assert not adapter._norm_eligible(norm, x)
    finally:
        global_handle.remove()


def test_missing_native_and_missing_abi_fail_before_cast(monkeypatch):
    monkeypatch.setattr(adapter, "_tensor_inputs_eligible", lambda *_args: True)
    monkeypatch.setattr(adapter, "_norm_eligible", lambda *_args: True)
    monkeypatch.setattr(adapter, "_linear_eligible", lambda *_args: True)
    monkeypatch.setattr(adapter, "_dynamic_eligible", lambda *_args: True)
    monkeypatch.setattr(
        adapter,
        "cast_bias_weight",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cast attempted")
        ),
    )
    for native in (None, SimpleNamespace(other_entrypoint=lambda: None)):
        monkeypatch.setattr(adapter, "_load_native", lambda value=native: value)
        assert adapter.try_linear(
            object(), object(), object(), object(), object(), object(),
            staged_adaln.ROW_IDS_TOKEN, "auto", "exact"
        ) is None


def test_native_abi_contract():
    entrypoint = lambda *_args: None
    assert adapter._native_abi_eligible(
        SimpleNamespace(
            abi_version=lambda: adapter.NATIVE_ABI_VERSION,
            build_id=lambda: adapter.NATIVE_BUILD_ID,
            rms_adaln_nvfp4_dynamic=entrypoint,
        )
    )
    assert not adapter._native_abi_eligible(
        SimpleNamespace(
            abi_version=lambda: adapter.NATIVE_ABI_VERSION + 1,
            build_id=lambda: adapter.NATIVE_BUILD_ID,
            rms_adaln_nvfp4_dynamic=entrypoint,
        )
    )
    assert not adapter._native_abi_eligible(
        SimpleNamespace(
            abi_version=lambda: adapter.NATIVE_ABI_VERSION,
            build_id=lambda: "wrong-build",
            rms_adaln_nvfp4_dynamic=entrypoint,
        )
    )
    assert not adapter._native_abi_eligible(
        SimpleNamespace(
            abi_version=lambda: adapter.NATIVE_ABI_VERSION,
            build_id=lambda: adapter.NATIVE_BUILD_ID,
        )
    )


def test_nested_norm_and_linear_cast_lifecycle(monkeypatch):
    class FakeQuantizedTensor:
        def __init__(self, qdata=None, layout=None, params=None):
            self.qdata = qdata
            self._layout_cls = layout or adapter.NVFP4_LAYOUT
            self._params = params or SimpleNamespace(transposed=False)

    class FakeLayout:
        Params = staticmethod(lambda **kwargs: SimpleNamespace(**kwargs))

    norm = SimpleNamespace(eps=1e-5)
    linear = SimpleNamespace()
    x = torch.empty(2, 4, dtype=torch.bfloat16)
    norm_weight = torch.ones(4, dtype=torch.bfloat16)
    linear_weight = FakeQuantizedTensor()
    qdata = torch.zeros(16, 2, dtype=torch.uint8)
    block_scale = torch.zeros(128, 4, dtype=torch.uint8)
    input_scale = torch.tensor([0.25], dtype=torch.float32)
    expected = object()
    lifecycle = []
    casts = []

    monkeypatch.setattr(adapter, "QuantizedTensor", FakeQuantizedTensor)
    monkeypatch.setattr(adapter, "TensorCoreNVFP4Layout", FakeLayout)
    monkeypatch.setattr(adapter, "_tensor_inputs_eligible", lambda *_args: True)
    monkeypatch.setattr(adapter, "_norm_eligible", lambda *_args: True)
    monkeypatch.setattr(adapter, "_linear_eligible", lambda *_args: True)
    monkeypatch.setattr(adapter, "_dynamic_eligible", lambda *_args: True)
    monkeypatch.setattr(adapter, "_cast_norm_weight_eligible", lambda *_args: True)
    monkeypatch.setattr(adapter, "run_every_op", lambda: lifecycle.append("run"))
    monkeypatch.setattr(
        adapter,
        "_load_native",
        lambda: SimpleNamespace(
            rms_adaln_nvfp4_dynamic=lambda *_args: (
                qdata,
                block_scale,
                input_scale,
            )
        ),
    )

    def cast(module, *_args, **_kwargs):
        casts.append(module)
        if module is norm:
            return norm_weight, None, "norm_stream"
        assert module is linear
        assert lifecycle[-1] == (norm, norm_weight, None, "norm_stream")
        return linear_weight, None, "linear_stream"

    monkeypatch.setattr(adapter, "cast_bias_weight", cast)
    monkeypatch.setattr(
        adapter,
        "uncast_bias_weight",
        lambda module, weight, bias, stream: lifecycle.append(
            (module, weight, bias, stream)
        ),
    )

    def fake_linear(qinput, weight, bias):
        assert isinstance(qinput, FakeQuantizedTensor)
        assert qinput.qdata is qdata
        assert qinput._params.scale is input_scale
        assert weight is linear_weight and bias is None
        return expected

    monkeypatch.setattr(torch.nn.functional, "linear", fake_linear)
    assert adapter.try_linear(
        linear,
        norm,
        x,
        object(),
        object(),
        object(),
        staged_adaln.ROW_IDS_TOKEN,
        "auto",
        "exact",
    ) is expected
    assert casts == [norm, linear]
    assert lifecycle == [
        "run",
        (norm, norm_weight, None, "norm_stream"),
        (linear, linear_weight, None, "linear_stream"),
    ]


def test_quantizer_exception_uncasts_norm_and_never_casts_linear(monkeypatch):
    norm = SimpleNamespace(eps=1e-5)
    linear = SimpleNamespace()
    x = torch.empty(2, 4, dtype=torch.bfloat16)
    norm_weight = torch.ones(4, dtype=torch.bfloat16)
    casts = []
    lifecycle = []
    monkeypatch.setattr(adapter, "_tensor_inputs_eligible", lambda *_args: True)
    monkeypatch.setattr(adapter, "_norm_eligible", lambda *_args: True)
    monkeypatch.setattr(adapter, "_linear_eligible", lambda *_args: True)
    monkeypatch.setattr(adapter, "_dynamic_eligible", lambda *_args: True)
    monkeypatch.setattr(adapter, "_cast_norm_weight_eligible", lambda *_args: True)
    monkeypatch.setattr(adapter, "run_every_op", lambda: None)
    monkeypatch.setattr(
        adapter,
        "_load_native",
        lambda: SimpleNamespace(
            rms_adaln_nvfp4_dynamic=lambda *_args: (_ for _ in ()).throw(
                RuntimeError("launch failed")
            )
        ),
    )

    def cast(module, *_args, **_kwargs):
        casts.append(module)
        assert module is norm
        return norm_weight, None, "norm_stream"

    monkeypatch.setattr(adapter, "cast_bias_weight", cast)
    monkeypatch.setattr(
        adapter,
        "uncast_bias_weight",
        lambda *args: lifecycle.append(args),
    )
    with pytest.raises(RuntimeError, match="launch failed"):
        adapter.try_linear(
            linear, norm, x, object(), object(), object(),
            staged_adaln.ROW_IDS_TOKEN, "auto", "exact"
        )
    assert casts == [norm]
    assert lifecycle == [(norm, norm_weight, None, "norm_stream")]


def test_rms_success_skips_both_norm_and_staged_paths(monkeypatch):
    block = make_block()
    attempted = []
    norm_calls = []
    originals = (block.norm1.forward, block.norm2.forward)
    monkeypatch.setattr(
        block.norm1,
        "forward",
        lambda value: norm_calls.append("norm1") or originals[0](value),
    )
    monkeypatch.setattr(
        block.norm2,
        "forward",
        lambda value: norm_calls.append("norm2") or originals[1](value),
    )

    def fused(linear, *_args):
        attempted.append(linear)
        if linear is block.attn.qkv_proj:
            return torch.randn(3, 12)
        assert linear is block.mlp.fc1
        return torch.randn(3, 16)

    monkeypatch.setattr(AUDIT_MODEL, "_h3_rms_adaln_linear", fused)
    monkeypatch.setattr(
        AUDIT_MODEL,
        "_h3_adaln_linear",
        lambda *_args: (_ for _ in ()).throw(AssertionError("staged attempted")),
    )
    monkeypatch.setattr(AUDIT_MODEL, "optimized_attention", identity_attention)
    output = run_block(
        block,
        {"minimax_h3_modulation_fusion": "exact"},
        AUDIT_MODEL._ADALN_NVFP4_TOKEN,
    )
    assert output.shape == (3, 4)
    assert attempted == [block.attn.qkv_proj, block.mlp.fc1]
    assert norm_calls == []


def test_rms_failure_reenters_current_norm_and_staged_paths_once(monkeypatch):
    block = make_block()
    rms_attempts = []
    staged_attempts = []
    norm_calls = []
    originals = (block.norm1.forward, block.norm2.forward)
    monkeypatch.setattr(
        block.norm1,
        "forward",
        lambda value: norm_calls.append("norm1") or originals[0](value),
    )
    monkeypatch.setattr(
        block.norm2,
        "forward",
        lambda value: norm_calls.append("norm2") or originals[1](value),
    )
    monkeypatch.setattr(
        AUDIT_MODEL,
        "_h3_rms_adaln_linear",
        lambda linear, *_args: rms_attempts.append(linear) or None,
    )
    monkeypatch.setattr(
        AUDIT_MODEL,
        "_h3_adaln_linear",
        lambda linear, *_args: staged_attempts.append(linear) or None,
    )
    monkeypatch.setattr(AUDIT_MODEL, "optimized_attention", identity_attention)
    output = run_block(
        block,
        {"minimax_h3_modulation_fusion": "exact"},
        AUDIT_MODEL._ADALN_NVFP4_TOKEN,
    )
    assert output.shape == (3, 4)
    assert rms_attempts == [block.attn.qkv_proj, block.mlp.fc1]
    assert staged_attempts == [block.attn.qkv_proj, block.mlp.fc1]
    assert norm_calls == ["norm1", "norm2"]


def test_untrusted_map_and_nonexact_modes_never_attempt_rms(monkeypatch):
    block = make_block()
    monkeypatch.setattr(
        AUDIT_MODEL,
        "_h3_rms_adaln_linear",
        lambda *_args: (_ for _ in ()).throw(AssertionError("RMS fusion attempted")),
    )
    monkeypatch.setattr(AUDIT_MODEL, "optimized_attention", identity_attention)
    assert run_block(
        block,
        {"minimax_h3_modulation_fusion": "exact"},
        token=None,
    ).shape == (3, 4)
    for mode in ("stock", "aggressive"):
        assert run_block(
            block,
            {"minimax_h3_modulation_fusion": mode},
            AUDIT_MODEL._ADALN_NVFP4_TOKEN,
        ).shape == (3, 4)


def test_rms_mode_inheritance_and_dedicated_rollback(monkeypatch):
    seen = []
    monkeypatch.setattr(
        AUDIT_MODEL,
        "try_rms_adaln_nvfp4_linear",
        lambda *args: seen.append(args[-2:]) or None,
    )
    common = (object(), object(), object(), object(), object(), object())
    AUDIT_MODEL._h3_rms_adaln_linear(
        *common, {"minimax_h3_swiglu_nvfp4_fusion": "auto"}
    )
    AUDIT_MODEL._h3_rms_adaln_linear(
        *common,
        {
            "minimax_h3_swiglu_nvfp4_fusion": "auto",
            "minimax_h3_rms_adaln_nvfp4_fusion": "stock",
        },
    )
    assert seen == [("auto", "exact"), ("stock", "exact")]
