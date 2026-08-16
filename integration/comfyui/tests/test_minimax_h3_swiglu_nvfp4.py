from types import SimpleNamespace

import torch

import comfy.ldm.minimax.model as minimax_model
from comfy.ldm.minimax import swiglu_nvfp4


class TorchOperations:
    Linear = torch.nn.Linear


def test_mlp_uses_h3_fusion_mode_and_returns_fused_result(monkeypatch):
    mlp = minimax_model.MLP(4, 8, operations=TorchOperations)
    x = torch.randn(3, 4)
    expected = torch.randn(3, 4)
    seen = {}

    def fake_try(linear, raw, mode):
        seen.update(linear=linear, raw=raw, mode=mode)
        return expected

    monkeypatch.setattr(minimax_model, "try_swiglu_nvfp4_linear", fake_try)
    actual = mlp(
        x,
        transformer_options={"minimax_h3_swiglu_nvfp4_fusion": "auto"},
    )

    assert actual is expected
    assert seen["linear"] is mlp.fc2
    assert seen["mode"] == "auto"
    assert seen["raw"].shape == (3, 16)


def test_mlp_falls_back_to_stock_activation(monkeypatch):
    mlp = minimax_model.MLP(4, 8, operations=TorchOperations)
    x = torch.randn(3, 4)
    expected = torch.randn(3, 4)
    seen = {}

    monkeypatch.setattr(
        minimax_model,
        "try_swiglu_nvfp4_linear",
        lambda linear, raw, mode: None,
    )

    def fake_stock(linear, raw, activation):
        seen.update(linear=linear, raw=raw, activation=activation)
        return expected

    monkeypatch.setattr(minimax_model.comfy.ops, "linear_input_act", fake_stock)
    actual = mlp(x)

    assert actual is expected
    assert seen["linear"] is mlp.fc2
    assert seen["activation"] == "swiglu"
    assert seen["raw"].shape == (3, 16)


def test_dynamic_route_requires_no_scale_bridge_or_hooks():
    linear = SimpleNamespace(input_scale=None)
    assert swiglu_nvfp4._dynamic_eligible(linear)

    linear.input_scale = torch.tensor(1.0)
    assert not swiglu_nvfp4._dynamic_eligible(linear)
    linear.input_scale = None

    for name in swiglu_nvfp4.BRIDGE_ATTRS:
        setattr(linear, name, object())
        assert not swiglu_nvfp4._dynamic_eligible(linear)
        delattr(linear, name)

    for name in swiglu_nvfp4.HOOK_ATTRS:
        setattr(linear, name, {1: object()})
        assert not swiglu_nvfp4._dynamic_eligible(linear)
        delattr(linear, name)


def test_static_scale_requires_exact_shape_dtype_and_marker():
    x = torch.empty(17, 28672, dtype=torch.bfloat16)
    linear = SimpleNamespace(
        input_scale=torch.tensor(0.25, dtype=torch.float32),
        _h3_nvfp4_fusion_enabled=True,
        _h3_nvfp4_allowed_shapes=frozenset({(17, 14336)}),
        _h3_nvfp4_allowed_dtypes=frozenset({str(torch.bfloat16)}),
        _h3_nvfp4_plan_hook_ids=(1, 2),
        _forward_pre_hooks={1: object()},
        _forward_hooks={2: object()},
        _forward_hooks_always_called={2},
    )
    linear._h3_nvfp4_validated_input_scale = linear.input_scale
    assert swiglu_nvfp4._static_scale(linear, x) is linear.input_scale

    linear._h3_nvfp4_allowed_shapes = frozenset({(18, 14336)})
    assert swiglu_nvfp4._static_scale(linear, x) is None
    linear._h3_nvfp4_allowed_shapes = frozenset({(17, 14336)})
    linear._h3_nvfp4_fusion_enabled = 1
    assert swiglu_nvfp4._static_scale(linear, x) is None


def test_static_scale_requires_producer_certified_tensor_identity():
    x = torch.empty(17, 28672, dtype=torch.bfloat16)
    scale = torch.tensor(0.25, dtype=torch.float32)
    linear = SimpleNamespace(
        input_scale=scale,
        _h3_nvfp4_fusion_enabled=True,
        _h3_nvfp4_allowed_shapes=frozenset({(17, 14336)}),
        _h3_nvfp4_allowed_dtypes=frozenset({str(torch.bfloat16)}),
        _h3_nvfp4_validated_input_scale=torch.tensor(0.25, dtype=torch.float32),
        _h3_nvfp4_plan_hook_ids=(1, 2),
        _forward_pre_hooks={1: object()},
        _forward_hooks={2: object()},
        _forward_hooks_always_called={2},
    )
    assert swiglu_nvfp4._static_scale(linear, x) is None

    linear._h3_nvfp4_validated_input_scale = scale
    assert swiglu_nvfp4._static_scale(linear, x) is scale


def test_stock_and_unknown_modes_fail_closed_without_loading_native(monkeypatch):
    monkeypatch.setattr(
        swiglu_nvfp4,
        "_load_native",
        lambda: (_ for _ in ()).throw(AssertionError("native loader called")),
    )
    linear = SimpleNamespace()
    x = torch.empty(2, 28672, dtype=torch.bfloat16)
    assert swiglu_nvfp4.try_linear(linear, x, "stock") is None
    assert swiglu_nvfp4.try_linear(linear, x, "unknown") is None


def test_ineligible_static_and_auto_routes_do_not_load_native(monkeypatch):
    monkeypatch.setattr(swiglu_nvfp4, "_raw_eligible", lambda _x: True)
    monkeypatch.setattr(swiglu_nvfp4, "_linear_eligible", lambda _linear: True)
    monkeypatch.setattr(swiglu_nvfp4, "_static_scale", lambda _linear, _x: None)
    monkeypatch.setattr(swiglu_nvfp4, "_dynamic_eligible", lambda _linear: False)
    monkeypatch.setattr(
        swiglu_nvfp4,
        "_load_native",
        lambda: (_ for _ in ()).throw(AssertionError("native loader called")),
    )
    linear = SimpleNamespace()
    x = object()
    assert swiglu_nvfp4.try_linear(linear, x, "static") is None
    assert swiglu_nvfp4.try_linear(linear, x, "auto") is None


def test_native_loader_rejects_unsupported_cuda_architecture(monkeypatch):
    swiglu_nvfp4._load_native.cache_clear()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    try:
        assert swiglu_nvfp4._load_native() is None
    finally:
        swiglu_nvfp4._load_native.cache_clear()


def test_input_device_architecture_gate_is_cached_and_fail_closed(monkeypatch):
    swiglu_nvfp4._cuda_device_supported.cache_clear()
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda device: calls.append(device) or ((12, 1) if device.index == 0 else (9, 0)),
    )
    try:
        supported = torch.device("cuda", 0)
        unsupported = torch.device("cuda", 1)
        assert swiglu_nvfp4._cuda_device_supported(supported)
        assert swiglu_nvfp4._cuda_device_supported(supported)
        assert not swiglu_nvfp4._cuda_device_supported(unsupported)
        assert calls == [supported, unsupported]
    finally:
        swiglu_nvfp4._cuda_device_supported.cache_clear()


def test_global_forward_hooks_disable_direct_fusion_routes():
    x = torch.empty(17, 28672, dtype=torch.bfloat16)
    scale = torch.tensor(0.25, dtype=torch.float32)
    linear = SimpleNamespace(
        input_scale=scale,
        _h3_nvfp4_fusion_enabled=True,
        _h3_nvfp4_allowed_shapes=frozenset({(17, 14336)}),
        _h3_nvfp4_allowed_dtypes=frozenset({str(torch.bfloat16)}),
        _h3_nvfp4_validated_input_scale=scale,
        _h3_nvfp4_plan_hook_ids=(1, 2),
        _forward_pre_hooks={1: object()},
        _forward_hooks={2: object()},
        _forward_hooks_always_called={2},
    )
    handle = torch.nn.modules.module.register_module_forward_hook(
        lambda _module, _args, _output: None
    )
    try:
        assert swiglu_nvfp4._static_scale(linear, x) is None
        assert not swiglu_nvfp4._dynamic_eligible(SimpleNamespace(input_scale=None))
    finally:
        handle.remove()
