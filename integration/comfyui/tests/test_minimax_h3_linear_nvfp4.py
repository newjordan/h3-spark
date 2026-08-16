from types import SimpleNamespace

import torch

import comfy.ldm.minimax.model as minimax_model
from comfy.ldm.minimax import linear_nvfp4


class TorchOperations:
    Linear = torch.nn.Linear
    RMSNorm = torch.nn.RMSNorm


def test_h3_linear_explicit_mode_overrides_swiglu_mode(monkeypatch):
    linear = torch.nn.Linear(4, 6, bias=False)
    x = torch.randn(3, 4)
    expected = torch.randn(3, 6)
    seen = {}

    def fake_try(module, value, mode):
        seen.update(module=module, value=value, mode=mode)
        return expected

    monkeypatch.setattr(minimax_model, "try_nvfp4_linear", fake_try)
    actual = minimax_model._h3_linear(
        linear,
        x,
        {
            "minimax_h3_swiglu_nvfp4_fusion": "static",
            "minimax_h3_linear_nvfp4_fusion": "auto",
        },
    )

    assert actual is expected
    assert seen == {"module": linear, "value": x, "mode": "auto"}


def test_h3_linear_inherits_swiglu_mode_and_falls_back_to_module(monkeypatch):
    linear = torch.nn.Linear(4, 6, bias=False)
    x = torch.randn(3, 4)
    seen = {}

    def fake_try(module, value, mode):
        seen.update(module=module, value=value, mode=mode)
        return None

    monkeypatch.setattr(minimax_model, "try_nvfp4_linear", fake_try)
    actual = minimax_model._h3_linear(
        linear,
        x,
        {"minimax_h3_swiglu_nvfp4_fusion": "auto"},
    )

    assert torch.equal(actual, linear(x))
    assert seen == {"module": linear, "value": x, "mode": "auto"}


def test_attention_routes_qkv_and_output_projection(monkeypatch):
    attention = minimax_model.Attention(
        hidden=4,
        heads=2,
        head_dim=2,
        eps=1e-6,
        operations=TorchOperations,
    )
    x = torch.randn(3, 4)
    calls = []

    def fake_h3_linear(linear, value, options):
        calls.append((linear, value.shape, options))
        return linear(value)

    monkeypatch.setattr(minimax_model, "_h3_linear", fake_h3_linear)
    monkeypatch.setattr(
        minimax_model,
        "optimized_attention",
        lambda _q, _k, v, _heads, **_kwargs: v,
    )
    options = {"minimax_h3_linear_nvfp4_fusion": "auto"}
    output = attention(x, transformer_options=options)

    assert output.shape == (3, 4)
    assert calls == [
        (attention.qkv_proj, torch.Size((3, 4)), options),
        (attention.out_proj, torch.Size((3, 4)), options),
    ]


def test_mlp_routes_fc1_before_swiglu_fusion(monkeypatch):
    mlp = minimax_model.MLP(4, 8, operations=TorchOperations)
    x = torch.randn(3, 4)
    raw = torch.randn(3, 16)
    expected = torch.randn(3, 4)
    seen = {}

    def fake_h3_linear(linear, value, options):
        seen.update(linear=linear, value=value, options=options)
        return raw

    monkeypatch.setattr(minimax_model, "_h3_linear", fake_h3_linear)
    monkeypatch.setattr(
        minimax_model,
        "try_swiglu_nvfp4_linear",
        lambda linear, value, mode: expected,
    )
    options = {"minimax_h3_swiglu_nvfp4_fusion": "auto"}
    output = mlp(x, transformer_options=options)

    assert output is expected
    assert seen == {"linear": mlp.fc1, "value": x, "options": options}


def test_dynamic_route_requires_no_scale_bridge_or_hooks():
    linear = SimpleNamespace(input_scale=None)
    assert linear_nvfp4._dynamic_eligible(linear)

    linear.input_scale = torch.tensor(1.0)
    assert not linear_nvfp4._dynamic_eligible(linear)
    linear.input_scale = None

    for name in linear_nvfp4.BRIDGE_ATTRS:
        setattr(linear, name, object())
        assert not linear_nvfp4._dynamic_eligible(linear)
        delattr(linear, name)

    for name in linear_nvfp4.HOOK_ATTRS:
        setattr(linear, name, {1: object()})
        assert not linear_nvfp4._dynamic_eligible(linear)
        delattr(linear, name)


def test_static_scale_requires_exact_shape_dtype_and_certified_identity():
    x = torch.empty(17, 5376, dtype=torch.bfloat16)
    scale = torch.tensor(0.25, dtype=torch.float32)
    linear = SimpleNamespace(
        input_scale=scale,
        _h3_nvfp4_fusion_enabled=True,
        _h3_nvfp4_allowed_shapes=frozenset({(17, 5376)}),
        _h3_nvfp4_allowed_dtypes=frozenset({str(torch.bfloat16)}),
        _h3_nvfp4_validated_input_scale=scale,
        _h3_nvfp4_plan_hook_ids=(1, 2),
        _forward_pre_hooks={1: object()},
        _forward_hooks={2: object()},
        _forward_hooks_always_called={2},
    )
    assert linear_nvfp4._static_scale(linear, x) is scale

    linear._h3_nvfp4_allowed_shapes = frozenset({(18, 5376)})
    assert linear_nvfp4._static_scale(linear, x) is None
    linear._h3_nvfp4_allowed_shapes = frozenset({(17, 5376)})
    linear._h3_nvfp4_validated_input_scale = scale.clone()
    assert linear_nvfp4._static_scale(linear, x) is None


def test_linear_eligibility_rejects_non_h3_shapes_and_patches(monkeypatch):
    class FakeQuantizedTensor:
        _layout_cls = linear_nvfp4.NVFP4_LAYOUT
        _params = SimpleNamespace(transposed=False)

    monkeypatch.setattr(linear_nvfp4, "QuantizedTensor", FakeQuantizedTensor)
    linear = SimpleNamespace(
        in_features=5376,
        out_features=21504,
        bias=None,
        quant_format="nvfp4",
        layout_type=linear_nvfp4.NVFP4_LAYOUT,
        weight=FakeQuantizedTensor(),
        pre_quant_scale=None,
        _full_precision_mm=False,
        comfy_force_cast_weights=False,
        weight_function=[],
        bias_function=[],
        weight_lowvram_function=None,
        bias_lowvram_function=None,
    )
    for in_features, out_features in sorted(linear_nvfp4.H3_LINEAR_SHAPES):
        linear.in_features = in_features
        linear.out_features = out_features
        assert linear_nvfp4._linear_eligible(
            linear, torch.empty(2, in_features)
        )

    x = torch.empty(2, 5376)
    linear.in_features = 5376
    linear.out_features = 5376
    assert not linear_nvfp4._linear_eligible(linear, x)
    linear.out_features = 21504
    linear.weight_function.append(object())
    assert not linear_nvfp4._linear_eligible(linear, x)


def test_stock_and_unknown_modes_fail_closed_without_native(monkeypatch):
    monkeypatch.setattr(
        linear_nvfp4,
        "_load_native",
        lambda: (_ for _ in ()).throw(AssertionError("native loader called")),
    )
    linear = SimpleNamespace()
    x = torch.empty(2, 5376, dtype=torch.bfloat16)
    assert linear_nvfp4.try_linear(linear, x, "stock") is None
    assert linear_nvfp4.try_linear(linear, x, "unknown") is None


def test_missing_native_or_direct_entrypoint_fails_closed_before_cast(monkeypatch):
    monkeypatch.setattr(linear_nvfp4, "_raw_eligible", lambda _x: True)
    monkeypatch.setattr(linear_nvfp4, "_linear_eligible", lambda _linear, _x: True)
    monkeypatch.setattr(linear_nvfp4, "_static_scale", lambda _linear, _x: None)
    monkeypatch.setattr(linear_nvfp4, "_dynamic_eligible", lambda _linear: True)
    monkeypatch.setattr(
        linear_nvfp4,
        "cast_bias_weight",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("weight cast called")
        ),
    )
    linear = SimpleNamespace()
    x = object()
    for native in (None, SimpleNamespace(swiglu_nvfp4_dynamic=lambda _x: None)):
        monkeypatch.setattr(linear_nvfp4, "_load_native", lambda value=native: value)
        assert linear_nvfp4.try_linear(linear, x, "auto") is None


def test_staged_plain_weight_uses_eager_linear_and_uncasts(monkeypatch):
    weight = torch.randn(6, 4, dtype=torch.bfloat16)
    bias = torch.randn(6, dtype=torch.bfloat16)
    x = torch.randn(2, 4, dtype=torch.bfloat16)
    linear = SimpleNamespace()
    lifecycle = []

    monkeypatch.setattr(linear_nvfp4, "_raw_eligible", lambda _x: True)
    monkeypatch.setattr(linear_nvfp4, "_linear_eligible", lambda _linear, _x: True)
    monkeypatch.setattr(linear_nvfp4, "_static_scale", lambda _linear, _x: None)
    monkeypatch.setattr(linear_nvfp4, "_dynamic_eligible", lambda _linear: True)
    monkeypatch.setattr(linear_nvfp4, "run_every_op", lambda: None)
    monkeypatch.setattr(
        linear_nvfp4,
        "_load_native",
        lambda: SimpleNamespace(
            bf16_nvfp4_dynamic=lambda _x: (_ for _ in ()).throw(
                AssertionError("quantizer called")
            )
        ),
    )
    monkeypatch.setattr(
        linear_nvfp4,
        "cast_bias_weight",
        lambda *args, **kwargs: (weight, bias, "offload"),
    )
    monkeypatch.setattr(
        linear_nvfp4,
        "uncast_bias_weight",
        lambda *args: lifecycle.append(args),
    )

    expected = torch.nn.functional.linear(x, weight, bias)
    actual = linear_nvfp4.try_linear(linear, x, "auto")
    assert torch.equal(actual, expected)
    assert lifecycle == [(linear, weight, bias, "offload")]


def test_global_forward_hook_disables_dynamic_and_static_routes():
    x = torch.empty(17, 5376, dtype=torch.bfloat16)
    scale = torch.tensor(0.25, dtype=torch.float32)
    linear = SimpleNamespace(
        input_scale=scale,
        _h3_nvfp4_fusion_enabled=True,
        _h3_nvfp4_allowed_shapes=frozenset({tuple(x.shape)}),
        _h3_nvfp4_allowed_dtypes=frozenset({str(torch.bfloat16)}),
        _h3_nvfp4_validated_input_scale=scale,
        _h3_nvfp4_plan_hook_ids=(1, 2),
        _forward_pre_hooks={1: object()},
        _forward_hooks={2: object()},
        _forward_hooks_always_called={2},
    )
    handle = torch.nn.modules.module.register_module_forward_pre_hook(
        lambda _module, _args: None
    )
    try:
        assert linear_nvfp4._static_scale(linear, x) is None
        assert not linear_nvfp4._dynamic_eligible(SimpleNamespace(input_scale=None))
    finally:
        handle.remove()


def test_dynamic_native_route_preserves_cast_lifecycle(monkeypatch):
    class FakeQuantizedTensor:
        def __init__(self, qdata=None, layout=None, params=None):
            self.qdata = qdata
            self._layout_cls = layout or linear_nvfp4.NVFP4_LAYOUT
            self._params = params or SimpleNamespace(transposed=False)

    class FakeLayout:
        Params = staticmethod(lambda **kwargs: SimpleNamespace(**kwargs))

    weight = FakeQuantizedTensor()
    qdata = torch.zeros(16, 2688, dtype=torch.uint8)
    block_scale = torch.zeros(128, 336, dtype=torch.uint8)
    scale = torch.tensor([0.25], dtype=torch.float32)
    expected = object()
    lifecycle = []

    monkeypatch.setattr(linear_nvfp4, "QuantizedTensor", FakeQuantizedTensor)
    monkeypatch.setattr(linear_nvfp4, "TensorCoreNVFP4Layout", FakeLayout)
    monkeypatch.setattr(linear_nvfp4, "_raw_eligible", lambda _x: True)
    monkeypatch.setattr(linear_nvfp4, "_linear_eligible", lambda _linear, _x: True)
    monkeypatch.setattr(linear_nvfp4, "_static_scale", lambda _linear, _x: None)
    monkeypatch.setattr(linear_nvfp4, "_dynamic_eligible", lambda _linear: True)
    monkeypatch.setattr(linear_nvfp4, "run_every_op", lambda: None)
    monkeypatch.setattr(
        linear_nvfp4,
        "_load_native",
        lambda: SimpleNamespace(
            bf16_nvfp4_dynamic=lambda _x: (qdata, block_scale, scale)
        ),
    )
    monkeypatch.setattr(
        linear_nvfp4,
        "cast_bias_weight",
        lambda *args, **kwargs: (weight, None, "offload"),
    )
    monkeypatch.setattr(
        linear_nvfp4,
        "uncast_bias_weight",
        lambda *args: lifecycle.append(args),
    )

    def fake_linear(qinput, got_weight, bias):
        assert isinstance(qinput, FakeQuantizedTensor)
        assert qinput.qdata is qdata
        assert qinput._params.scale is scale
        assert qinput._params.orig_shape == (2, 5376)
        assert got_weight is weight
        assert bias is None
        return expected

    monkeypatch.setattr(torch.nn.functional, "linear", fake_linear)
    x = torch.empty(2, 5376, dtype=torch.bfloat16)
    assert linear_nvfp4.try_linear(SimpleNamespace(), x, "auto") is expected
    assert lifecycle == [(SimpleNamespace(), weight, None, "offload")]
