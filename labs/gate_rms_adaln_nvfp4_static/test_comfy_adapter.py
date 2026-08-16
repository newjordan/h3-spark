from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import comfy_adapter as fused


HERE = Path(__file__).resolve().parent
GATE_V2_SOURCE = (
    HERE.parent / "gate_rms_adaln_nvfp4" / "gate_rms_adaln_nvfp4_cuda.cu"
)
STATIC_SOURCE = (
    HERE.parent / "rms_adaln_nvfp4_static" / "rms_adaln_nvfp4_static_cuda.cu"
)
BINARY = HERE / "build" / f"{fused.NATIVE_MODULE_NAME}.so"


class FakeQuantizedTensor:
    def __init__(self, data=None, layout=None, params=None, *, layout_cls=None):
        self.data = data
        self._layout_cls = layout if layout is not None else layout_cls
        self._params = params or SimpleNamespace(transposed=False)


class FakeLayout:
    Params = staticmethod(lambda **kwargs: SimpleNamespace(**kwargs))


def make_call(pair=fused.FC1_PAIR):
    linear = SimpleNamespace(in_features=pair[0], out_features=pair[1])
    owner = SimpleNamespace(fc1=linear)
    return SimpleNamespace(
        owner=owner,
        linear=linear,
        norm=SimpleNamespace(eps=1e-5),
        x=torch.empty(2, 5376, dtype=torch.bfloat16),
        other=object(),
        gate=object(),
        scale=object(),
        shift=object(),
        rows=object(),
    )


def args(case, *, token=fused.ROW_IDS_TOKEN, mode="auto", modulation="exact"):
    return (
        case.owner,
        case.linear,
        case.norm,
        case.x,
        case.other,
        case.gate,
        case.scale,
        case.shift,
        case.rows,
        token,
        mode,
        modulation,
    )


def install_bridge(linear, x, input_scale):
    linear.input_scale = input_scale
    setattr(linear, fused.BRIDGE_ATTRS[0], True)
    setattr(linear, fused.BRIDGE_ATTRS[1], frozenset({tuple(x.shape)}))
    setattr(linear, fused.BRIDGE_ATTRS[2], frozenset({str(torch.bfloat16)}))
    setattr(linear, fused.BRIDGE_ATTRS[3], input_scale)
    setattr(linear, fused.BRIDGE_ATTRS[4], (1, 2))
    linear._forward_pre_hooks = {1: object()}
    linear._forward_hooks = {2: object()}
    linear._forward_hooks_always_called = {2}


def complete_bridge(linear):
    for name in fused.BRIDGE_ATTRS:
        setattr(linear, name, object())


def test_no_bridge_delegates_dynamic_v2_verbatim(monkeypatch):
    case = make_call()
    expected = object()
    calls = []
    monkeypatch.setattr(
        fused.dynamic_gate,
        "try_linear",
        lambda *values: calls.append(values) or expected,
    )
    assert fused.try_linear(*args(case)) is expected
    assert calls == [args(case)]


@pytest.mark.parametrize("present", fused.BRIDGE_ATTRS)
def test_partial_bridge_falls_stock_without_guards_native_or_dynamic(
    monkeypatch, present
):
    case = make_call()
    setattr(case.linear, present, object())
    monkeypatch.setattr(
        fused,
        "_preflight_eligible",
        lambda *_args: (_ for _ in ()).throw(AssertionError("guard entered")),
    )
    monkeypatch.setattr(
        fused,
        "_load_native",
        lambda: (_ for _ in ()).throw(AssertionError("native entered")),
    )
    monkeypatch.setattr(
        fused.dynamic_gate,
        "try_linear",
        lambda *_args: (_ for _ in ()).throw(AssertionError("dynamic entered")),
    )
    assert fused.try_linear(*args(case)) is None


def test_complete_malformed_bridge_falls_stock_not_dynamic(monkeypatch):
    case = make_call()
    complete_bridge(case.linear)
    monkeypatch.setattr(fused, "_preflight_eligible", lambda *_args: True)
    monkeypatch.setattr(fused, "_static_scale", lambda *_args: None)
    monkeypatch.setattr(
        fused.dynamic_gate,
        "try_linear",
        lambda *_args: (_ for _ in ()).throw(AssertionError("dynamic entered")),
    )
    monkeypatch.setattr(
        fused,
        "_load_native",
        lambda: (_ for _ in ()).throw(AssertionError("native entered")),
    )
    assert fused.try_linear(*args(case)) is None


def test_v3_scale_bridge_requires_exact_shape_dtype_and_object_identity():
    case = make_call()
    scale = torch.tensor(0.25, dtype=torch.float32)
    install_bridge(case.linear, case.x, scale)
    assert fused._static_scale(case.linear, case.x) is scale

    setattr(case.linear, fused.BRIDGE_ATTRS[1], frozenset({(3, 5376)}))
    assert fused._static_scale(case.linear, case.x) is None
    setattr(case.linear, fused.BRIDGE_ATTRS[1], frozenset({tuple(case.x.shape)}))

    setattr(case.linear, fused.BRIDGE_ATTRS[2], frozenset({str(torch.float32)}))
    assert fused._static_scale(case.linear, case.x) is None
    setattr(case.linear, fused.BRIDGE_ATTRS[2], frozenset({str(torch.bfloat16)}))

    setattr(case.linear, fused.BRIDGE_ATTRS[3], scale.clone())
    assert fused._static_scale(case.linear, case.x) is None
    setattr(case.linear, fused.BRIDGE_ATTRS[3], scale)
    case.linear.input_scale = scale.to(torch.float64)
    assert fused._static_scale(case.linear, case.x) is None


def test_input_scale_without_bridge_delegates_only_to_dynamic_adapter(monkeypatch):
    case = make_call()
    case.linear.input_scale = torch.tensor(0.25)
    expected = object()
    monkeypatch.setattr(fused.dynamic_gate, "try_linear", lambda *_args: expected)
    assert fused.try_linear(*args(case)) is expected


@pytest.mark.parametrize(
    ("mode", "modulation", "token"),
    [
        ("stock", "exact", fused.ROW_IDS_TOKEN),
        ("auto", "stock", fused.ROW_IDS_TOKEN),
        ("auto", "aggressive", fused.ROW_IDS_TOKEN),
        ("auto", "exact", object()),
    ],
)
def test_mode_and_row_token_fail_before_native_or_dynamic(
    monkeypatch, mode, modulation, token
):
    case = make_call()
    complete_bridge(case.linear)
    monkeypatch.setattr(
        fused,
        "_preflight_eligible",
        lambda *_args: (_ for _ in ()).throw(AssertionError("guards entered")),
    )
    monkeypatch.setattr(
        fused.dynamic_gate,
        "try_linear",
        lambda *_args: (_ for _ in ()).throw(AssertionError("dynamic entered")),
    )
    assert fused.try_linear(
        *args(case, mode=mode, modulation=modulation, token=token)
    ) is None


def test_preflight_binds_exact_fc1_owner_parent_and_bridge_certified_hooks(monkeypatch):
    case = make_call()
    common = {
        "parent_eligible": lambda *_args: True,
        "_tensor_inputs_eligible": lambda *_args: True,
        "_spark_device_eligible": lambda *_args: True,
        "_gate_inputs_eligible": lambda *_args: True,
        "_norm_eligible": lambda *_args: True,
        "_linear_eligible": lambda *_args: True,
    }
    for name, value in common.items():
        monkeypatch.setattr(fused, name, value)
    assert fused._preflight_eligible(*args(case)[:9])

    case.owner.fc1 = object()
    assert not fused._preflight_eligible(*args(case)[:9])
    case.owner.fc1 = case.linear
    case.linear.out_features = 5376
    assert not fused._preflight_eligible(*args(case)[:9])
    case.linear.out_features = 28672
    # The exact plan-owned hook set is certified by `_static_scale`; preflight
    # must not reject those local hooks before reaching that shared predicate.
    case.linear._forward_hooks = {1: object()}
    assert fused._preflight_eligible(*args(case)[:9])


def test_gate_contract_is_fp32_and_rejects_every_x_alias():
    x = torch.empty(2, 5376, dtype=torch.bfloat16)
    other = torch.empty_like(x)
    table = torch.empty(3, 5376 * 3, dtype=torch.float32)
    gate = table[:, :5376]
    scale = table[:, 5376 : 2 * 5376]
    shift = table[:, 2 * 5376 :]
    rows = torch.tensor([0, 2], dtype=torch.int32)
    assert fused._gate_inputs_eligible(x, other, gate, scale, shift, rows)

    aliases = (
        x,
        x.view(torch.float32).reshape(1, 5376),
        x.view(torch.float32).reshape(1, 5376),
        x.view(torch.float32).reshape(1, 5376),
        x.view(torch.int32).reshape(-1)[:2],
    )
    original = [other, gate, scale, shift, rows]
    for index, alias in enumerate(aliases):
        candidate = list(original)
        candidate[index] = alias
        assert not fused._gate_inputs_eligible(x, *candidate)

    assert not fused._gate_inputs_eligible(
        x,
        other,
        gate.to(torch.bfloat16),
        scale.to(torch.bfloat16),
        shift.to(torch.bfloat16),
        rows,
    )


def test_complete_bridge_valid_route_never_calls_dynamic(monkeypatch):
    case = make_call()
    complete_bridge(case.linear)
    input_scale = torch.tensor(0.25)
    expected = object()
    monkeypatch.setattr(fused, "_preflight_eligible", lambda *_args: True)
    monkeypatch.setattr(fused, "_static_scale", lambda *_args: input_scale)
    monkeypatch.setattr(fused, "_storage_disjoint", lambda *_args: True)
    monkeypatch.setattr(fused, "_load_native", lambda: None)
    monkeypatch.setattr(
        fused.dynamic_gate,
        "try_linear",
        lambda *_args: (_ for _ in ()).throw(AssertionError("dynamic entered")),
    )
    # Missing candidate native is a pre-mutation stock fallback, not dynamic.
    assert fused.try_linear(*args(case)) is None

    monkeypatch.setattr(fused, "try_static_linear", lambda *_args: expected)
    assert fused.try_linear(*args(case)) is expected


def test_deferred_callback_bridge_change_returns_before_cast_or_mutation(monkeypatch):
    case = make_call()
    complete_bridge(case.linear)
    first_scale = torch.tensor(0.25)
    second_scale = torch.tensor(0.5)
    scales = iter((first_scale, second_scale))
    native_call = pytest.fail
    native = SimpleNamespace(gate_rms_adaln_nvfp4_static=native_call)
    monkeypatch.setattr(fused, "_preflight_eligible", lambda *_args: True)
    monkeypatch.setattr(fused, "_static_scale", lambda *_args: next(scales))
    monkeypatch.setattr(fused, "_storage_disjoint", lambda *_args: True)
    monkeypatch.setattr(fused, "_load_native", lambda: native)
    monkeypatch.setattr(fused, "run_every_op", lambda: None)
    monkeypatch.setattr(
        fused,
        "cast_bias_weight",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cast entered")
        ),
    )
    assert fused.try_linear(*args(case)) is None


def test_norm_cast_mismatch_uncasts_and_returns_pre_mutation(monkeypatch):
    case = make_call()
    complete_bridge(case.linear)
    input_scale = torch.tensor(0.25)
    order = []
    native = SimpleNamespace(
        gate_rms_adaln_nvfp4_static=lambda *_args: (_ for _ in ()).throw(
            AssertionError("mutation entered")
        )
    )
    monkeypatch.setattr(fused, "_preflight_eligible", lambda *_args: True)
    monkeypatch.setattr(fused, "_static_scale", lambda *_args: input_scale)
    monkeypatch.setattr(fused, "_storage_disjoint", lambda *_args: True)
    monkeypatch.setattr(fused, "_load_native", lambda: native)
    monkeypatch.setattr(fused, "run_every_op", lambda: None)
    monkeypatch.setattr(
        fused,
        "cast_bias_weight",
        lambda *_args, **_kwargs: ("bad", None, "norm-stream"),
    )
    monkeypatch.setattr(fused, "_cast_norm_weight_eligible", lambda *_args: False)
    monkeypatch.setattr(
        fused,
        "uncast_bias_weight",
        lambda *_args: order.append("uncast-norm"),
    )
    assert fused.try_linear(*args(case)) is None
    assert order == ["uncast-norm"]


def test_norm_cast_callback_reruns_full_preflight_before_mutation(monkeypatch):
    case = make_call()
    complete_bridge(case.linear)
    input_scale = torch.tensor(0.25)
    norm_weight = object()
    eligible = {"value": True}
    preflight_values = []
    order = []
    native = SimpleNamespace(
        gate_rms_adaln_nvfp4_static=lambda *_args: (_ for _ in ()).throw(
            AssertionError("mutation entered")
        )
    )

    def preflight(*_args):
        preflight_values.append(eligible["value"])
        return eligible["value"]

    def cast(module, *_args, **_kwargs):
        assert module is case.norm
        order.append("cast-norm")
        eligible["value"] = False
        return norm_weight, None, "norm-stream"

    monkeypatch.setattr(fused, "_preflight_eligible", preflight)
    monkeypatch.setattr(fused, "_static_scale", lambda *_args: input_scale)
    monkeypatch.setattr(fused, "_storage_disjoint", lambda *_args: True)
    monkeypatch.setattr(fused, "_load_native", lambda: native)
    monkeypatch.setattr(fused, "run_every_op", lambda: None)
    monkeypatch.setattr(fused, "cast_bias_weight", cast)
    monkeypatch.setattr(fused, "_cast_norm_weight_eligible", lambda *_args: True)
    monkeypatch.setattr(
        fused,
        "uncast_bias_weight",
        lambda *_args: order.append("uncast-norm"),
    )

    assert fused.try_linear(*args(case)) is None
    assert preflight_values == [True, True, False]
    assert order == ["cast-norm", "uncast-norm"]


def test_native_failure_propagates_without_linear_cast(monkeypatch):
    case = make_call()
    complete_bridge(case.linear)
    input_scale = torch.tensor(0.25)
    norm_weight = object()
    order = []
    native = SimpleNamespace(
        gate_rms_adaln_nvfp4_static=lambda *_args: (_ for _ in ()).throw(
            RuntimeError("native failure")
        )
    )
    monkeypatch.setattr(fused, "_preflight_eligible", lambda *_args: True)
    monkeypatch.setattr(fused, "_static_scale", lambda *_args: input_scale)
    monkeypatch.setattr(fused, "_storage_disjoint", lambda *_args: True)
    monkeypatch.setattr(fused, "_load_native", lambda: native)
    monkeypatch.setattr(fused, "run_every_op", lambda: None)
    monkeypatch.setattr(
        fused,
        "cast_bias_weight",
        lambda module, *_args, **_kwargs: (
            (norm_weight, None, "norm-stream")
            if module is case.norm
            else (_ for _ in ()).throw(AssertionError("linear cast entered"))
        ),
    )
    monkeypatch.setattr(fused, "_cast_norm_weight_eligible", lambda *_args: True)
    monkeypatch.setattr(
        fused,
        "uncast_bias_weight",
        lambda *_args: order.append("uncast-norm"),
    )
    with pytest.raises(RuntimeError, match="native failure"):
        fused.try_linear(*args(case))
    assert order == ["uncast-norm"]


def patch_success_guards(monkeypatch, case, input_scale, native):
    monkeypatch.setattr(fused, "_preflight_eligible", lambda *_args: True)
    monkeypatch.setattr(fused, "parent_eligible", lambda *_args: True)
    monkeypatch.setattr(fused, "_norm_eligible", lambda *_args: True)
    monkeypatch.setattr(fused, "_linear_eligible", lambda *_args: True)
    monkeypatch.setattr(fused, "_static_scale", lambda *_args: input_scale)
    monkeypatch.setattr(fused, "_storage_disjoint", lambda *_args: True)
    monkeypatch.setattr(fused, "_load_native", lambda: native)
    monkeypatch.setattr(fused, "run_every_op", lambda: None)
    monkeypatch.setattr(fused, "_cast_norm_weight_eligible", lambda *_args: True)


def test_invalid_native_outputs_raise_after_mutation(monkeypatch):
    case = make_call()
    complete_bridge(case.linear)
    input_scale = torch.tensor(0.25)
    native = SimpleNamespace(
        gate_rms_adaln_nvfp4_static=lambda *_args: (object(), object())
    )
    patch_success_guards(monkeypatch, case, input_scale, native)
    order = []
    monkeypatch.setattr(
        fused,
        "cast_bias_weight",
        lambda module, *_args, **_kwargs: (
            (object(), None, "norm-stream")
            if module is case.norm
            else (_ for _ in ()).throw(AssertionError("linear cast entered"))
        ),
    )
    monkeypatch.setattr(
        fused,
        "uncast_bias_weight",
        lambda *_args: order.append("uncast-norm"),
    )
    with pytest.raises(RuntimeError, match="invalid NVFP4 outputs"):
        fused.try_linear(*args(case))
    assert order == ["uncast-norm"]


def test_post_mutation_weight_change_raises_and_restores_both_casts(monkeypatch):
    case = make_call()
    complete_bridge(case.linear)
    input_scale = torch.tensor(0.25)
    qdata = torch.empty(16, 2688, dtype=torch.uint8)
    block_scale = torch.empty(128, 336, dtype=torch.uint8)
    native = SimpleNamespace(
        gate_rms_adaln_nvfp4_static=lambda *_args: (qdata, block_scale)
    )
    patch_success_guards(monkeypatch, case, input_scale, native)
    order = []

    def cast(module, *_args, **_kwargs):
        order.append("cast-norm" if module is case.norm else "cast-linear")
        return object(), None, "stream"

    def uncast(module, *_args):
        order.append("uncast-norm" if module is case.norm else "uncast-linear")

    monkeypatch.setattr(fused, "cast_bias_weight", cast)
    monkeypatch.setattr(fused, "uncast_bias_weight", uncast)
    with pytest.raises(RuntimeError, match="weight changed"):
        fused.try_linear(*args(case))
    assert order == ["cast-norm", "uncast-norm", "cast-linear", "uncast-linear"]


@pytest.mark.parametrize("changed_guard", ["norm", "linear"])
def test_linear_cast_callback_eligibility_change_raises_after_mutation(
    monkeypatch, changed_guard
):
    case = make_call()
    complete_bridge(case.linear)
    input_scale = torch.tensor(0.25)
    qdata = torch.empty(16, 2688, dtype=torch.uint8)
    block_scale = torch.empty(128, 336, dtype=torch.uint8)
    native = SimpleNamespace(
        gate_rms_adaln_nvfp4_static=lambda *_args: (qdata, block_scale)
    )
    patch_success_guards(monkeypatch, case, input_scale, native)
    eligible = {"norm": True, "linear": True}
    order = []
    weight = FakeQuantizedTensor(
        layout_cls=fused.NVFP4_LAYOUT,
        params=SimpleNamespace(transposed=False),
    )

    def cast(module, *_args, **_kwargs):
        if module is case.norm:
            order.append("cast-norm")
            return object(), None, "norm-stream"
        order.append("cast-linear")
        eligible[changed_guard] = False
        return weight, None, "linear-stream"

    def uncast(module, *_args):
        order.append("uncast-norm" if module is case.norm else "uncast-linear")

    monkeypatch.setattr(fused, "_norm_eligible", lambda *_args: eligible["norm"])
    monkeypatch.setattr(
        fused, "_linear_eligible", lambda *_args: eligible["linear"]
    )
    monkeypatch.setattr(fused, "cast_bias_weight", cast)
    monkeypatch.setattr(fused, "uncast_bias_weight", uncast)

    with pytest.raises(RuntimeError, match="eligibility.*changed"):
        fused.try_linear(*args(case))
    assert order == ["cast-norm", "uncast-norm", "cast-linear", "uncast-linear"]


def test_success_preserves_scale_identity_and_sequential_cast_lifetimes(monkeypatch):
    case = make_call()
    complete_bridge(case.linear)
    input_scale = torch.tensor(0.25)
    qdata = torch.empty(16, 2688, dtype=torch.uint8)
    block_scale = torch.empty(128, 336, dtype=torch.uint8)
    native_scales = []
    order = []

    def quantize(*native_args):
        order.append("quantize")
        native_scales.append(native_args[-2])
        return qdata, block_scale

    native = SimpleNamespace(gate_rms_adaln_nvfp4_static=quantize)
    patch_success_guards(monkeypatch, case, input_scale, native)
    weight = FakeQuantizedTensor(
        layout_cls=fused.NVFP4_LAYOUT,
        params=SimpleNamespace(transposed=False),
    )

    def cast(module, *_args, **_kwargs):
        if module is case.norm:
            order.append("cast-norm")
            return object(), None, "norm-stream"
        order.append("cast-linear")
        return weight, None, "linear-stream"

    def uncast(module, *_args):
        order.append("uncast-norm" if module is case.norm else "uncast-linear")

    expected = object()
    seen_params = []
    monkeypatch.setattr(fused, "cast_bias_weight", cast)
    monkeypatch.setattr(fused, "uncast_bias_weight", uncast)
    monkeypatch.setattr(fused, "QuantizedTensor", FakeQuantizedTensor)
    monkeypatch.setattr(fused, "TensorCoreNVFP4Layout", FakeLayout)
    monkeypatch.setattr(
        fused.torch.nn.functional,
        "linear",
        lambda qinput, got_weight, bias: (
            order.append("linear")
            or seen_params.append(qinput._params)
            or expected
        ),
    )
    monkeypatch.setattr(
        fused.dynamic_gate,
        "try_linear",
        lambda *_args: (_ for _ in ()).throw(AssertionError("dynamic entered")),
    )

    assert fused.try_linear(*args(case)) is expected
    assert order == [
        "cast-norm",
        "quantize",
        "uncast-norm",
        "cast-linear",
        "linear",
        "uncast-linear",
    ]
    assert native_scales == [input_scale]
    assert seen_params[0].scale is input_scale
    assert seen_params[0].orig_shape == tuple(case.x.shape)


def test_transition_enters_private_mlp_only_after_producer_success(monkeypatch):
    case = make_call()
    options = {"sentinel": True}
    calls = []
    case.owner._forward_raw = lambda raw, transformer_options: (
        calls.append((raw, transformer_options)) or "mlp"
    )
    monkeypatch.setattr(fused, "try_linear", lambda *_args: None)
    assert fused.try_mlp_transition(
        case.owner,
        case.norm,
        case.x,
        case.other,
        case.gate,
        case.scale,
        case.shift,
        case.rows,
        fused.ROW_IDS_TOKEN,
        "auto",
        "exact",
        options,
    ) is None
    assert calls == []

    raw = object()
    monkeypatch.setattr(fused, "try_linear", lambda *_args: raw)
    assert fused.try_mlp_transition(
        case.owner,
        case.norm,
        case.x,
        case.other,
        case.gate,
        case.scale,
        case.shift,
        case.rows,
        fused.ROW_IDS_TOKEN,
        "auto",
        "exact",
        options,
    ) == "mlp"
    assert calls == [(raw, options)]


def test_native_abi_build_and_hash_contract(monkeypatch, tmp_path):
    entry = lambda *_args: None
    good = SimpleNamespace(
        abi_version=lambda: fused.NATIVE_ABI_VERSION,
        build_id=lambda: fused.NATIVE_BUILD_ID,
        gate_rms_adaln_nvfp4_static=entry,
    )
    assert fused._native_abi_eligible(good)
    assert not fused._native_abi_eligible(
        SimpleNamespace(**{**good.__dict__, "abi_version": lambda: 2})
    )
    assert not fused._native_abi_eligible(
        SimpleNamespace(**{**good.__dict__, "build_id": lambda: "stale"})
    )

    candidate = tmp_path / "candidate.so"
    candidate.write_bytes(b"wrong binary")
    monkeypatch.setenv(fused.LIBRARY_ENV, str(candidate))
    monkeypatch.setattr(fused, "_spark_device_eligible", lambda *_args: True)
    monkeypatch.setattr(torch, "__version__", fused.CERTIFIED_TORCH_VERSION)
    monkeypatch.setattr(torch.version, "git_version", fused.CERTIFIED_TORCH_GIT_VERSION)
    monkeypatch.setattr(torch.version, "cuda", fused.CERTIFIED_CUDA_VERSION)
    monkeypatch.setattr(
        fused.importlib.util,
        "spec_from_file_location",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("uncertified binary loaded")
        ),
    )
    fused._load_native.cache_clear()
    try:
        assert fused._load_native() is None
    finally:
        fused._load_native.cache_clear()


def test_source_composes_exact_qualified_contracts_and_mutation_boundary():
    assert hashlib.sha256(GATE_V2_SOURCE.read_bytes()).hexdigest() == (
        "5dcba067e7c81896f407ed5ed2fdf5b1f0a1140dffa5e706439667652b74da45"
    )
    assert hashlib.sha256(STATIC_SOURCE.read_bytes()).hexdigest() == (
        "f9090140eb81e0d8ba07a699ab9f7739dc9e1b44e5a995b85aac4550a86d38f5"
    )
    source = (HERE / "gate_rms_adaln_nvfp4_static_cuda.cu").read_text()
    for marker in (
        "fmaf(",
        "__float2bfloat16_rn(gated)",
        "const float left = warp_scratch[0] + warp_scratch[2]",
        "const float right = warp_scratch[1] + warp_scratch[3]",
        "staged_adaln_bf16",
        "scale_factor_swizzled_offset",
        "__nv_cvt_float2_to_fp4x2",
        "global_scale[0]",
        "torch::autograd::increment_version(x)",
        "at::cuda::getCurrentCUDAStream(x.get_device())",
        "at::assert_no_overlap(x, global_scale)",
        "gate.scalar_type() == torch::kFloat32",
        "eps == kCertifiedEps",
        fused.NATIVE_BUILD_ID,
    ):
        assert marker in source
    assert "atomicMax" not in source
    assert "global_max_bits" not in source


@pytest.mark.skipif(
    not BINARY.exists(), reason="source-only snapshot has no generated binary"
)
def test_built_sm120_cubin_has_no_stack_local_or_spills():
    assert hashlib.sha256(BINARY.read_bytes()).hexdigest() == fused.NATIVE_SHA256
    result = subprocess.run(
        ["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", str(BINARY)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout
    functions = [line.strip() for line in result.splitlines() if "REG:" in line]
    assert len(functions) == 2
    for record in functions:
        assert "REG:64" in record
        assert "STACK:0" in record
        assert "LOCAL:0" in record
    listing = subprocess.run(
        ["/usr/local/cuda/bin/cuobjdump", "--list-elf", str(BINARY)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout
    assert ".sm_120.cubin" in listing
