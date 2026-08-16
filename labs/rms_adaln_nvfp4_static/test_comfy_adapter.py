from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import comfy.ldm.minimax.adaln_nvfp4 as adaln
import comfy.ldm.minimax.rms_adaln_nvfp4 as dynamic_v2
from comfy.ldm.minimax.swiglu_nvfp4 import BF16_DTYPE, BRIDGE_ATTRS

import comfy_adapter as adapter


def make_owned(pair=(5376, 21504)):
    linear = SimpleNamespace(in_features=pair[0], out_features=pair[1])
    field = "qkv_proj" if pair == (5376, 21504) else "fc1"
    owner = SimpleNamespace(**{field: linear})
    return owner, linear


def call_adapter(
    owner,
    linear,
    *,
    norm=None,
    x=None,
    scale=None,
    shift=None,
    rows=None,
    token=adaln.ROW_IDS_TOKEN,
    mode="auto",
    modulation="exact",
):
    args = (
        linear,
        norm if norm is not None else object(),
        x if x is not None else object(),
        scale if scale is not None else object(),
        shift if shift is not None else object(),
        rows if rows is not None else object(),
        token,
        mode,
        modulation,
    )
    result = adapter.try_static_route(*args)
    return dynamic_v2.try_linear(*args) if result is adapter.NO_STATIC_ROUTE else result


def allow_common_route(monkeypatch):
    monkeypatch.setattr(adapter, "_tensor_inputs_eligible", lambda *_args: True)
    monkeypatch.setattr(adapter, "_spark_device_eligible", lambda *_args: True)
    monkeypatch.setattr(adapter, "_norm_eligible", lambda *_args: True)
    monkeypatch.setattr(adapter, "_linear_eligible", lambda *_args: True)


def install_bridge(linear, x, input_scale):
    linear.input_scale = input_scale
    setattr(linear, BRIDGE_ATTRS[0], True)
    setattr(linear, BRIDGE_ATTRS[1], frozenset({tuple(x.shape)}))
    setattr(linear, BRIDGE_ATTRS[2], frozenset({BF16_DTYPE}))
    setattr(linear, BRIDGE_ATTRS[3], input_scale)
    setattr(linear, BRIDGE_ATTRS[4], (1, 2))
    linear._forward_pre_hooks = {1: object()}
    linear._forward_hooks = {2: object()}
    linear._forward_hooks_always_called = {2}


def test_reuses_current_dynamic_v2_guards_and_exact_five_bridge_attrs():
    assert adapter._tensor_inputs_eligible is dynamic_v2._tensor_inputs_eligible
    assert adapter._norm_eligible is dynamic_v2._norm_eligible
    assert adapter._cast_norm_weight_eligible is dynamic_v2._cast_norm_weight_eligible
    assert adapter.CERTIFIED_TORCH_VERSION == dynamic_v2.CERTIFIED_TORCH_VERSION
    assert adapter.CERTIFIED_TORCH_GIT_VERSION == dynamic_v2.CERTIFIED_TORCH_GIT_VERSION
    assert BRIDGE_ATTRS == (
        "_h3_nvfp4_fusion_enabled",
        "_h3_nvfp4_allowed_shapes",
        "_h3_nvfp4_allowed_dtypes",
        "_h3_nvfp4_validated_input_scale",
        "_h3_nvfp4_plan_hook_ids",
    )


def test_pinned_model_source_owns_exact_qkv_fc1_and_row_token_call_sites():
    path = (
        Path(__file__).resolve().parents[2]
        / "integration"
        / "comfyui"
        / "comfy"
        / "ldm"
        / "minimax"
        / "model.py"
    )
    source = path.read_bytes()
    assert hashlib.sha256(source).hexdigest() == (
        "d7b196cc8b2ca753fe2a590e0741e6e7979aa52bdd6f5429592ddf1fcb7d77ff"
    )
    text = source.decode("utf-8")
    assert (
        'trusted_row_map = adaln_nvfp4_token is _ADALN_NVFP4_TOKEN'
        in text
    )
    assert (
        'trusted_row_map and adaln_parent_eligible(self.attn)' in text
    )
    assert (
        'trusted_row_map and adaln_parent_eligible(self.mlp)' in text
    )
    assert text.count("self.attn.qkv_proj, self.norm1, x, shift_msa, scale_msa") == 1
    assert text.count("self.mlp.fc1, self.norm2, x, shift_mlp, scale_mlp") == 1
    assert (
        "linear, norm, x, scale, shift, row_ids, _ADALN_NVFP4_TOKEN,\n"
        "        _h3_rms_adaln_mode(transformer_options), \"exact\""
        in text
    )
    assert adapter.CERTIFIED_LINEAR_SHAPES == {
        (5376, 21504),
        (5376, 28672),
    }


@pytest.mark.parametrize(
    ("mode", "modulation", "token"),
    [
        ("stock", "exact", adaln.ROW_IDS_TOKEN),
        ("static", "exact", adaln.ROW_IDS_TOKEN),
        ("auto", "stock", adaln.ROW_IDS_TOKEN),
        ("auto", "aggressive", adaln.ROW_IDS_TOKEN),
        ("auto", "exact", object()),
    ],
)
def test_mode_row_token_and_owner_fail_before_tensor_guards(
    monkeypatch, mode, modulation, token
):
    owner, linear = make_owned()
    setattr(linear, BRIDGE_ATTRS[0], object())
    monkeypatch.setattr(
        adapter,
        "_tensor_inputs_eligible",
        lambda *_args: (_ for _ in ()).throw(AssertionError("tensor guard entered")),
    )
    monkeypatch.setattr(
        dynamic_v2,
        "try_linear",
        lambda *_args: (_ for _ in ()).throw(AssertionError("dynamic route entered")),
    )
    assert call_adapter(
        owner,
        linear,
        mode=mode,
        modulation=modulation,
        token=token,
    ) is None


def test_unbridged_exact_owner_delegates_dynamic_v2_verbatim(monkeypatch):
    owner, linear = make_owned()
    norm, x, scale, shift, rows = (object() for _ in range(5))
    expected = object()
    calls = []
    allow_common_route(monkeypatch)
    monkeypatch.setattr(adapter, "_static_scale", lambda *_args: None)

    def dynamic(*args):
        calls.append(args)
        return expected

    monkeypatch.setattr(dynamic_v2, "try_linear", dynamic)
    assert call_adapter(
        owner,
        linear,
        norm=norm,
        x=x,
        scale=scale,
        shift=shift,
        rows=rows,
    ) is expected
    assert calls == [
        (
            linear,
            norm,
            x,
            scale,
            shift,
            rows,
            adaln.ROW_IDS_TOKEN,
            "auto",
            "exact",
        )
    ]


def test_dynamic_fallback_exception_is_not_swallowed(monkeypatch):
    owner, linear = make_owned()
    allow_common_route(monkeypatch)
    monkeypatch.setattr(adapter, "_static_scale", lambda *_args: None)
    monkeypatch.setattr(
        dynamic_v2,
        "try_linear",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("dynamic failure")),
    )
    with pytest.raises(RuntimeError, match="dynamic failure"):
        call_adapter(owner, linear)


def test_valid_bridge_routes_static_and_never_dynamic(monkeypatch):
    owner, linear = make_owned()
    for attribute in BRIDGE_ATTRS:
        setattr(linear, attribute, object())
    input_scale = object()
    expected = object()
    allow_common_route(monkeypatch)
    monkeypatch.setattr(adapter, "_static_scale", lambda *_args: input_scale)
    monkeypatch.setattr(adapter, "_try_static_linear", lambda *_args: expected)
    monkeypatch.setattr(
        dynamic_v2,
        "try_linear",
        lambda *_args: (_ for _ in ()).throw(AssertionError("dynamic route entered")),
    )
    assert call_adapter(owner, linear) is expected


@pytest.mark.parametrize("state", BRIDGE_ATTRS)
def test_partial_or_stale_static_state_never_delegates_dynamic(
    monkeypatch, state
):
    owner, linear = make_owned()
    allow_common_route(monkeypatch)
    setattr(linear, state, object())
    monkeypatch.setattr(adapter, "_static_scale", lambda *_args: None)
    monkeypatch.setattr(
        dynamic_v2,
        "try_linear",
        lambda *_args: (_ for _ in ()).throw(AssertionError("dynamic route entered")),
    )
    assert call_adapter(owner, linear) is None


def test_input_scale_without_bridge_remains_current_dynamic_v2_contract(monkeypatch):
    owner, linear = make_owned()
    linear.input_scale = object()
    allow_common_route(monkeypatch)
    expected = object()
    calls = []
    monkeypatch.setattr(
        dynamic_v2,
        "try_linear",
        lambda *args: calls.append(args) or expected,
    )
    assert call_adapter(owner, linear) is expected
    assert len(calls) == 1


def test_real_bridge_requires_shape_dtype_and_scale_object_identity():
    x = torch.empty(2, 5376, dtype=torch.bfloat16)
    _, linear = make_owned()
    input_scale = torch.tensor(0.25, dtype=torch.float32)
    install_bridge(linear, x, input_scale)
    assert adapter._static_scale(linear, x) is input_scale

    cases = [
        (BRIDGE_ATTRS[0], 1),
        (BRIDGE_ATTRS[1], frozenset({(3, 5376)})),
        (BRIDGE_ATTRS[1], {(2, 5376)}),
        (BRIDGE_ATTRS[2], frozenset({"torch.float32"})),
        (BRIDGE_ATTRS[2], {BF16_DTYPE}),
        (BRIDGE_ATTRS[3], input_scale.clone()),
    ]
    for attribute, bad_value in cases:
        install_bridge(linear, x, input_scale)
        setattr(linear, attribute, bad_value)
        assert adapter._static_scale(linear, x) is None

    install_bridge(linear, x, input_scale)
    linear.input_scale = input_scale.clone()
    assert adapter._static_scale(linear, x) is None

    install_bridge(linear, x, torch.tensor([0.25], dtype=torch.float32))
    assert adapter._static_scale(linear, x) is None

    install_bridge(linear, x, torch.tensor(0.25, dtype=torch.float64))
    assert adapter._static_scale(linear, x) is None


@pytest.mark.parametrize("pair", [(5376, 21504), (5376, 28672)])
def test_real_qkv_fc1_bridge_reaches_static_only_at_exact_5376_shape(
    monkeypatch, pair
):
    owner, linear = make_owned(pair)
    x = torch.empty(20423, 5376, dtype=torch.bfloat16)
    input_scale = torch.tensor(0.25, dtype=torch.float32)
    install_bridge(linear, x, input_scale)
    allow_common_route(monkeypatch)
    calls = []
    monkeypatch.setattr(
        adapter,
        "_try_static_linear",
        lambda *args: calls.append(args) or object(),
    )
    monkeypatch.setattr(
        dynamic_v2,
        "try_linear",
        lambda *_args: (_ for _ in ()).throw(AssertionError("dynamic route entered")),
    )
    result = call_adapter(owner, linear, x=x)
    assert result is not None
    assert len(calls) == 1
    assert calls[0][0] is linear
    assert calls[0][2] is x
    assert calls[0][-1] is input_scale

    calls.clear()
    wrong_rows = torch.empty(129, 5376, dtype=torch.bfloat16)
    assert call_adapter(owner, linear, x=wrong_rows) is None
    assert calls == []


def test_fc2_bridge_is_never_claimed_by_static_adapter(monkeypatch):
    # FC2 exclusions are applied by StaticScalePlan, but even adversarial bridge
    # metadata on the 14336-wide down projection is outside this kernel contract.
    linear = SimpleNamespace(in_features=14336, out_features=5376)
    owner = SimpleNamespace(fc2=linear)
    x = torch.empty(20423, 14336, dtype=torch.bfloat16)
    input_scale = torch.tensor(0.25, dtype=torch.float32)
    install_bridge(linear, x, input_scale)
    monkeypatch.setattr(
        adapter,
        "_tensor_inputs_eligible",
        lambda *_args: (_ for _ in ()).throw(AssertionError("tensor guard entered")),
    )
    monkeypatch.setattr(
        adapter,
        "_try_static_linear",
        lambda *_args: (_ for _ in ()).throw(AssertionError("static launch entered")),
    )
    monkeypatch.setattr(
        dynamic_v2,
        "try_linear",
        lambda *_args: (_ for _ in ()).throw(AssertionError("dynamic route entered")),
    )
    assert call_adapter(owner, linear, x=x) is None


@pytest.mark.parametrize(
    "break_bridge",
    [
        lambda linear, _x, _scale: setattr(linear, BRIDGE_ATTRS[0], False),
        lambda linear, _x, _scale: setattr(linear, BRIDGE_ATTRS[1], frozenset()),
        lambda linear, _x, _scale: setattr(linear, BRIDGE_ATTRS[2], frozenset()),
        lambda linear, _x, scale: setattr(linear, BRIDGE_ATTRS[3], scale.clone()),
        lambda linear, _x, scale: setattr(linear, "input_scale", scale.clone()),
    ],
)
def test_malformed_real_bridge_fails_before_static_native_load(
    monkeypatch, break_bridge
):
    owner, linear = make_owned()
    x = torch.empty(20423, 5376, dtype=torch.bfloat16)
    input_scale = torch.tensor(0.25, dtype=torch.float32)
    install_bridge(linear, x, input_scale)
    break_bridge(linear, x, input_scale)
    allow_common_route(monkeypatch)
    monkeypatch.setattr(
        adapter,
        "_load_native",
        lambda: (_ for _ in ()).throw(AssertionError("static native loaded")),
    )
    monkeypatch.setattr(
        dynamic_v2,
        "try_linear",
        lambda *_args: (_ for _ in ()).throw(AssertionError("dynamic route entered")),
    )
    assert call_adapter(owner, linear, x=x) is None


def test_shape_mismatched_live_bridge_uses_staged_not_dynamic(monkeypatch):
    owner, linear = make_owned()
    x = torch.empty(3, 5376, dtype=torch.bfloat16)
    input_scale = torch.tensor(0.25, dtype=torch.float32)
    install_bridge(linear, torch.empty(2, 5376, dtype=torch.bfloat16), input_scale)
    allow_common_route(monkeypatch)
    monkeypatch.setattr(
        dynamic_v2,
        "try_linear",
        lambda *_args: (_ for _ in ()).throw(AssertionError("dynamic route entered")),
    )
    assert call_adapter(owner, linear, x=x) is None


def test_linear_hook_rejects_before_bridge_read_or_dynamic(monkeypatch):
    owner, linear = make_owned()
    setattr(linear, BRIDGE_ATTRS[0], object())
    linear._forward_pre_hooks = {1: object()}
    monkeypatch.setattr(adapter, "_tensor_inputs_eligible", lambda *_args: True)
    monkeypatch.setattr(adapter, "_spark_device_eligible", lambda *_args: True)
    monkeypatch.setattr(adapter, "_norm_eligible", lambda *_args: True)
    monkeypatch.setattr(adapter, "_linear_eligible", lambda *_args: True)
    monkeypatch.setattr(
        adapter,
        "_static_scale",
        lambda *_args: (_ for _ in ()).throw(AssertionError("bridge read")),
    )
    monkeypatch.setattr(
        dynamic_v2,
        "try_linear",
        lambda *_args: (_ for _ in ()).throw(AssertionError("dynamic route")),
    )
    assert call_adapter(owner, linear) is None


def test_native_abi_and_build_contract():
    entry = lambda *_args: None
    good = SimpleNamespace(
        abi_version=lambda: adapter.NATIVE_ABI_VERSION,
        build_id=lambda: adapter.NATIVE_BUILD_ID,
        rms_adaln_nvfp4_static=entry,
    )
    assert adapter._native_abi_eligible(good)
    assert not adapter._native_abi_eligible(
        SimpleNamespace(**{**good.__dict__, "abi_version": lambda: 2})
    )
    assert not adapter._native_abi_eligible(
        SimpleNamespace(**{**good.__dict__, "build_id": lambda: "stale"})
    )
    assert not adapter._native_abi_eligible(
        SimpleNamespace(
            abi_version=good.abi_version,
            build_id=good.build_id,
            rms_adaln_nvfp4_static=None,
        )
    )


def test_hash_mismatched_aot_binary_never_loads(monkeypatch, tmp_path):
    binary = tmp_path / "candidate.so"
    binary.write_bytes(b"not the certified binary")
    monkeypatch.setenv(adapter.LIBRARY_ENV, str(binary))
    monkeypatch.setattr(adapter, "_spark_device_eligible", lambda *_args: True)
    monkeypatch.setattr(torch, "__version__", adapter.CERTIFIED_TORCH_VERSION)
    monkeypatch.setattr(torch.version, "git_version", adapter.CERTIFIED_TORCH_GIT_VERSION)
    monkeypatch.setattr(torch.version, "cuda", adapter.CERTIFIED_CUDA_VERSION)
    monkeypatch.setattr(
        adapter.importlib.util,
        "spec_from_file_location",
        lambda *_args: (_ for _ in ()).throw(AssertionError("uncertified binary loaded")),
    )
    adapter._load_native.cache_clear()
    try:
        assert adapter._load_native() is None
    finally:
        adapter._load_native.cache_clear()


class FakeQuantizedTensor:
    def __init__(self, qdata=None, layout=None, params=None):
        self.qdata = qdata
        self._layout_cls = layout or adapter.NVFP4_LAYOUT
        self._params = params or SimpleNamespace(transposed=False)


class FakeLayout:
    Params = staticmethod(lambda **kwargs: SimpleNamespace(**kwargs))


def static_fixture():
    norm = SimpleNamespace(eps=adapter.CERTIFIED_H3_RMS_EPS)
    linear = SimpleNamespace(
        in_features=adapter.WIDTH,
        out_features=21504,
        bias=None,
        quant_format="nvfp4",
        layout_type=adapter.NVFP4_LAYOUT,
        weight=FakeQuantizedTensor(),
        pre_quant_scale=None,
        _full_precision_mm=False,
        comfy_force_cast_weights=False,
        weight_function=[],
        bias_function=[],
        weight_lowvram_function=None,
        bias_lowvram_function=None,
    )
    x = torch.empty(2, adapter.WIDTH, dtype=torch.bfloat16)
    mod_scale = torch.empty(1, adapter.WIDTH, dtype=torch.float32)
    shift = torch.empty_like(mod_scale)
    rows = torch.zeros(2, dtype=torch.int32)
    input_scale = torch.tensor(0.25, dtype=torch.float32)
    install_bridge(linear, x, input_scale)
    norm_weight = torch.ones(adapter.WIDTH, dtype=torch.bfloat16)
    linear_weight = FakeQuantizedTensor()
    qdata = torch.zeros(16, adapter.WIDTH // 2, dtype=torch.uint8)
    block_scale = torch.zeros(128, adapter.WIDTH // 16, dtype=torch.uint8)
    return SimpleNamespace(
        norm=norm,
        linear=linear,
        x=x,
        mod_scale=mod_scale,
        shift=shift,
        rows=rows,
        input_scale=input_scale,
        norm_weight=norm_weight,
        linear_weight=linear_weight,
        qdata=qdata,
        block_scale=block_scale,
    )


def test_static_quant_then_linear_cast_lifetime_is_sequential(monkeypatch):
    case = static_fixture()
    expected = object()
    lifecycle = []
    casts = []
    monkeypatch.setattr(adapter, "QuantizedTensor", FakeQuantizedTensor)
    monkeypatch.setattr(adapter, "_norm_eligible", lambda *_args: True)
    monkeypatch.setattr(adapter, "_linear_eligible", lambda *_args: True)
    monkeypatch.setattr(adapter, "TensorCoreNVFP4Layout", FakeLayout)
    monkeypatch.setattr(adapter, "_cast_norm_weight_eligible", lambda *_args: True)
    monkeypatch.setattr(adapter, "run_every_op", lambda: lifecycle.append("run"))

    def quantize(x, weight, scale, shift, rows, input_scale, eps):
        assert (x, weight, scale, shift, rows, input_scale) == (
            case.x,
            case.norm_weight,
            case.mod_scale,
            case.shift,
            case.rows,
            case.input_scale,
        )
        assert eps == adapter.CERTIFIED_H3_RMS_EPS
        lifecycle.append("quantize")
        return case.qdata, case.block_scale

    monkeypatch.setattr(
        adapter,
        "_load_native",
        lambda: SimpleNamespace(rms_adaln_nvfp4_static=quantize),
    )

    def cast(module, *_args, **_kwargs):
        casts.append(module)
        if module is case.norm:
            return case.norm_weight, None, "norm-stream"
        assert module is case.linear
        assert lifecycle[-1] == ("uncast", case.norm, "norm-stream")
        return case.linear_weight, None, "linear-stream"

    monkeypatch.setattr(adapter, "cast_bias_weight", cast)
    monkeypatch.setattr(
        adapter,
        "uncast_bias_weight",
        lambda module, _weight, _bias, stream: lifecycle.append(("uncast", module, stream)),
    )

    def linear_call(qinput, weight, bias):
        assert isinstance(qinput, FakeQuantizedTensor)
        assert qinput.qdata is case.qdata
        assert qinput._params.scale is case.input_scale
        assert qinput._params.orig_shape == tuple(case.x.shape)
        assert weight is case.linear_weight and bias is None
        lifecycle.append("linear")
        return expected

    monkeypatch.setattr(torch.nn.functional, "linear", linear_call)
    actual = adapter._try_static_linear(
        case.linear,
        case.norm,
        case.x,
        case.mod_scale,
        case.shift,
        case.rows,
        case.input_scale,
    )
    assert actual is expected
    assert casts == [case.norm, case.linear]
    assert lifecycle == [
        "run",
        "quantize",
        ("uncast", case.norm, "norm-stream"),
        "linear",
        ("uncast", case.linear, "linear-stream"),
    ]


def test_run_every_op_norm_invalidation_stops_before_cast_or_native(monkeypatch):
    case = static_fixture()
    state = {"norm_ok": True}
    monkeypatch.setattr(adapter, "_norm_eligible", lambda *_args: state["norm_ok"])
    monkeypatch.setattr(adapter, "_linear_eligible", lambda *_args: True)
    monkeypatch.setattr(
        adapter,
        "_load_native",
        lambda: SimpleNamespace(
            rms_adaln_nvfp4_static=lambda *_args: (_ for _ in ()).throw(
                AssertionError("native launch")
            )
        ),
    )
    monkeypatch.setattr(
        adapter,
        "run_every_op",
        lambda: state.__setitem__("norm_ok", False),
    )
    monkeypatch.setattr(
        adapter,
        "cast_bias_weight",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("cast")),
    )

    assert adapter._try_static_linear(
        case.linear,
        case.norm,
        case.x,
        case.mod_scale,
        case.shift,
        case.rows,
        case.input_scale,
    ) is None


def test_norm_cast_callback_invalidation_stops_before_native(monkeypatch):
    case = static_fixture()
    state = {"norm_ok": True}
    uncast = []
    monkeypatch.setattr(adapter, "_norm_eligible", lambda *_args: state["norm_ok"])
    monkeypatch.setattr(adapter, "_linear_eligible", lambda *_args: True)
    monkeypatch.setattr(adapter, "_cast_norm_weight_eligible", lambda *_args: True)
    monkeypatch.setattr(adapter, "run_every_op", lambda: None)
    monkeypatch.setattr(
        adapter,
        "_load_native",
        lambda: SimpleNamespace(
            rms_adaln_nvfp4_static=lambda *_args: (_ for _ in ()).throw(
                AssertionError("native launch")
            )
        ),
    )

    def cast(module, *_args, **_kwargs):
        assert module is case.norm
        state["norm_ok"] = False
        return case.norm_weight, None, "norm-stream"

    monkeypatch.setattr(adapter, "cast_bias_weight", cast)
    monkeypatch.setattr(
        adapter,
        "uncast_bias_weight",
        lambda module, *_args: uncast.append(module),
    )

    assert adapter._try_static_linear(
        case.linear,
        case.norm,
        case.x,
        case.mod_scale,
        case.shift,
        case.rows,
        case.input_scale,
    ) is None
    assert uncast == [case.norm]


def test_linear_cast_callback_norm_invalidation_stops_before_gemm(monkeypatch):
    case = static_fixture()
    state = {"norm_ok": True}
    uncast = []
    monkeypatch.setattr(adapter, "QuantizedTensor", FakeQuantizedTensor)
    monkeypatch.setattr(adapter, "_norm_eligible", lambda *_args: state["norm_ok"])
    monkeypatch.setattr(adapter, "_linear_eligible", lambda *_args: True)
    monkeypatch.setattr(adapter, "_cast_norm_weight_eligible", lambda *_args: True)
    monkeypatch.setattr(adapter, "run_every_op", lambda: None)
    monkeypatch.setattr(
        adapter,
        "_load_native",
        lambda: SimpleNamespace(
            rms_adaln_nvfp4_static=lambda *_args: (case.qdata, case.block_scale)
        ),
    )

    def cast(module, *_args, **_kwargs):
        if module is case.norm:
            return case.norm_weight, None, "norm-stream"
        assert module is case.linear
        state["norm_ok"] = False
        return case.linear_weight, None, "linear-stream"

    monkeypatch.setattr(adapter, "cast_bias_weight", cast)
    monkeypatch.setattr(
        adapter,
        "uncast_bias_weight",
        lambda module, *_args: uncast.append(module),
    )
    monkeypatch.setattr(
        torch.nn.functional,
        "linear",
        lambda *_args: (_ for _ in ()).throw(AssertionError("GEMM")),
    )

    assert adapter._try_static_linear(
        case.linear,
        case.norm,
        case.x,
        case.mod_scale,
        case.shift,
        case.rows,
        case.input_scale,
    ) is None
    assert uncast == [case.norm, case.linear]


def test_norm_cast_mismatch_uncasts_and_never_launches_or_casts_linear(monkeypatch):
    case = static_fixture()
    monkeypatch.setattr(adapter, "_norm_eligible", lambda *_args: True)
    monkeypatch.setattr(adapter, "_linear_eligible", lambda *_args: True)
    events = []
    monkeypatch.setattr(
        adapter,
        "_load_native",
        lambda: SimpleNamespace(
            rms_adaln_nvfp4_static=lambda *_args: (_ for _ in ()).throw(
                AssertionError("native launch")
            )
        ),
    )
    monkeypatch.setattr(adapter, "run_every_op", lambda: None)
    monkeypatch.setattr(adapter, "_cast_norm_weight_eligible", lambda *_args: False)
    monkeypatch.setattr(
        adapter,
        "cast_bias_weight",
        lambda module, *_args, **_kwargs: (
            (case.norm_weight, None, "norm-stream")
            if module is case.norm
            else (_ for _ in ()).throw(AssertionError("linear cast"))
        ),
    )
    monkeypatch.setattr(
        adapter,
        "uncast_bias_weight",
        lambda module, *_args: events.append(module),
    )
    assert adapter._try_static_linear(
        case.linear,
        case.norm,
        case.x,
        case.mod_scale,
        case.shift,
        case.rows,
        case.input_scale,
    ) is None
    assert events == [case.norm]


def test_native_exception_uncasts_norm_and_propagates_without_linear_cast(monkeypatch):
    case = static_fixture()
    monkeypatch.setattr(adapter, "_norm_eligible", lambda *_args: True)
    monkeypatch.setattr(adapter, "_linear_eligible", lambda *_args: True)
    events = []
    monkeypatch.setattr(
        adapter,
        "_load_native",
        lambda: SimpleNamespace(
            rms_adaln_nvfp4_static=lambda *_args: (_ for _ in ()).throw(
                RuntimeError("native failure")
            )
        ),
    )
    monkeypatch.setattr(adapter, "run_every_op", lambda: None)
    monkeypatch.setattr(adapter, "_cast_norm_weight_eligible", lambda *_args: True)

    def cast(module, *_args, **_kwargs):
        if module is case.norm:
            return case.norm_weight, None, "norm-stream"
        raise AssertionError("linear cast")

    monkeypatch.setattr(adapter, "cast_bias_weight", cast)
    monkeypatch.setattr(
        adapter,
        "uncast_bias_weight",
        lambda module, *_args: events.append(module),
    )
    with pytest.raises(RuntimeError, match="native failure"):
        adapter._try_static_linear(
            case.linear,
            case.norm,
            case.x,
            case.mod_scale,
            case.shift,
            case.rows,
            case.input_scale,
        )
    assert events == [case.norm]


def test_invalid_native_output_uncasts_norm_and_never_casts_linear(monkeypatch):
    case = static_fixture()
    monkeypatch.setattr(adapter, "_norm_eligible", lambda *_args: True)
    monkeypatch.setattr(adapter, "_linear_eligible", lambda *_args: True)
    events = []
    monkeypatch.setattr(
        adapter,
        "_load_native",
        lambda: SimpleNamespace(
            rms_adaln_nvfp4_static=lambda *_args: (
                torch.empty(1, dtype=torch.uint8),
                case.block_scale,
            )
        ),
    )
    monkeypatch.setattr(adapter, "run_every_op", lambda: None)
    monkeypatch.setattr(adapter, "_cast_norm_weight_eligible", lambda *_args: True)

    def cast(module, *_args, **_kwargs):
        if module is case.norm:
            return case.norm_weight, None, "norm-stream"
        raise AssertionError("linear cast")

    monkeypatch.setattr(adapter, "cast_bias_weight", cast)
    monkeypatch.setattr(
        adapter,
        "uncast_bias_weight",
        lambda module, *_args: events.append(module),
    )
    assert adapter._try_static_linear(
        case.linear,
        case.norm,
        case.x,
        case.mod_scale,
        case.shift,
        case.rows,
        case.input_scale,
    ) is None
    assert events == [case.norm]


def test_post_quant_linear_weight_change_uncasts_both_and_returns_staged(monkeypatch):
    case = static_fixture()
    events = []
    monkeypatch.setattr(adapter, "QuantizedTensor", FakeQuantizedTensor)
    monkeypatch.setattr(adapter, "_norm_eligible", lambda *_args: True)
    monkeypatch.setattr(adapter, "_linear_eligible", lambda *_args: True)
    monkeypatch.setattr(
        adapter,
        "_load_native",
        lambda: SimpleNamespace(
            rms_adaln_nvfp4_static=lambda *_args: (case.qdata, case.block_scale)
        ),
    )
    monkeypatch.setattr(adapter, "run_every_op", lambda: None)
    monkeypatch.setattr(adapter, "_cast_norm_weight_eligible", lambda *_args: True)

    def cast(module, *_args, **_kwargs):
        if module is case.norm:
            return case.norm_weight, None, "norm-stream"
        return object(), None, "linear-stream"

    monkeypatch.setattr(adapter, "cast_bias_weight", cast)
    monkeypatch.setattr(
        adapter,
        "uncast_bias_weight",
        lambda module, *_args: events.append(module),
    )
    assert adapter._try_static_linear(
        case.linear,
        case.norm,
        case.x,
        case.mod_scale,
        case.shift,
        case.rows,
        case.input_scale,
    ) is None
    assert events == [case.norm, case.linear]


def test_linear_exception_uncasts_linear_and_propagates(monkeypatch):
    case = static_fixture()
    events = []
    monkeypatch.setattr(adapter, "QuantizedTensor", FakeQuantizedTensor)
    monkeypatch.setattr(adapter, "_norm_eligible", lambda *_args: True)
    monkeypatch.setattr(adapter, "_linear_eligible", lambda *_args: True)
    monkeypatch.setattr(adapter, "TensorCoreNVFP4Layout", FakeLayout)
    monkeypatch.setattr(
        adapter,
        "_load_native",
        lambda: SimpleNamespace(
            rms_adaln_nvfp4_static=lambda *_args: (case.qdata, case.block_scale)
        ),
    )
    monkeypatch.setattr(adapter, "run_every_op", lambda: None)
    monkeypatch.setattr(adapter, "_cast_norm_weight_eligible", lambda *_args: True)

    def cast(module, *_args, **_kwargs):
        if module is case.norm:
            return case.norm_weight, None, "norm-stream"
        return case.linear_weight, None, "linear-stream"

    monkeypatch.setattr(adapter, "cast_bias_weight", cast)
    monkeypatch.setattr(
        adapter,
        "uncast_bias_weight",
        lambda module, *_args: events.append(module),
    )
    monkeypatch.setattr(
        torch.nn.functional,
        "linear",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("linear failure")),
    )
    with pytest.raises(RuntimeError, match="linear failure"):
        adapter._try_static_linear(
            case.linear,
            case.norm,
            case.x,
            case.mod_scale,
            case.shift,
            case.rows,
            case.input_scale,
        )
    assert events == [case.norm, case.linear]


def test_missing_static_native_does_not_cross_into_dynamic(monkeypatch):
    owner, linear = make_owned()
    setattr(linear, BRIDGE_ATTRS[0], object())
    input_scale = object()
    allow_common_route(monkeypatch)
    monkeypatch.setattr(adapter, "_static_scale", lambda *_args: input_scale)
    monkeypatch.setattr(adapter, "_load_native", lambda: None)
    monkeypatch.setattr(
        dynamic_v2,
        "try_linear",
        lambda *_args: (_ for _ in ()).throw(AssertionError("dynamic route entered")),
    )
    assert call_adapter(owner, linear) is None


def test_adapter_never_mutates_bridge_metadata(monkeypatch):
    owner, linear = make_owned()
    x = torch.empty(2, 5376, dtype=torch.bfloat16)
    input_scale = torch.tensor(0.25, dtype=torch.float32)
    install_bridge(linear, x, input_scale)
    before = {name: getattr(linear, name) for name in ("input_scale", *BRIDGE_ATTRS)}
    allow_common_route(monkeypatch)
    monkeypatch.setattr(adapter, "_try_static_linear", lambda *_args: None)
    assert call_adapter(owner, linear, x=x) is None
    after = {name: getattr(linear, name) for name in ("input_scale", *BRIDGE_ATTRS)}
    assert before.keys() == after.keys()
    assert all(after[name] is before[name] for name in before)
