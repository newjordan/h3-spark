from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from comfy.ldm.minimax import linear_nvfp4, swiglu_nvfp4


ROOT = Path(__file__).resolve().parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


scales = _load(
    "h3_static_plan_certificate_test",
    ROOT / "nvfp4_static" / "h3_nvfp4_static_scales.py",
)
gate_static = _load(
    "h3_gate_static_certificate_test",
    ROOT / "gate_rms_adaln_nvfp4_static" / "comfy_adapter.py",
)
rms_static = _load(
    "h3_rms_static_certificate_test",
    ROOT / "rms_adaln_nvfp4_static" / "comfy_adapter.py",
)


class PlanLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._orig_shape = (out_features, in_features)
        self.quant_format = "nvfp4"
        self.layout_type = linear_nvfp4.NVFP4_LAYOUT
        self.weight_function = []
        self.bias_function = []
        self.weight = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, value):
        return value


class PlanRoot(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_out = PlanLinear(7168, 5376)
        self.qkv = PlanLinear(5376, 21504)
        self.fc1 = PlanLinear(5376, 28672)
        self.swiglu = PlanLinear(14336, 5376)


def _plan(tmp_path):
    root = PlanRoot()
    calibrator = scales.ActivationCalibrator(
        root,
        margin=1.0,
        model_id="hook-certificate-test",
        require_h3=False,
    )
    with calibrator.capture():
        root.q_out(torch.ones(2, 7168, dtype=torch.bfloat16))
        root.qkv(torch.ones(2, 5376, dtype=torch.bfloat16))
        root.fc1(torch.ones(2, 5376, dtype=torch.bfloat16))
        root.swiglu(torch.ones(2, 14336, dtype=torch.bfloat16))
    prefix = tmp_path / "tiny"
    scales.save_calibration(calibrator, prefix)
    wrapper = SimpleNamespace(model=root, load_device=torch.device("cuda:0"))
    return root, scales.StaticScalePlan(
        wrapper, prefix, strict=True, require_h3=False
    )


def _allow_gate_preflight(monkeypatch):
    for name in (
        "parent_eligible",
        "_tensor_inputs_eligible",
        "_spark_device_eligible",
        "_gate_inputs_eligible",
        "_norm_eligible",
        "_linear_eligible",
    ):
        monkeypatch.setattr(gate_static, name, lambda *_args: True)
    monkeypatch.setattr(gate_static, "_storage_disjoint", lambda *_args: True)


def _try_gate(monkeypatch, root, x, native_calls):
    _allow_gate_preflight(monkeypatch)
    monkeypatch.setattr(
        gate_static,
        "_load_native",
        lambda: native_calls.append(True) or None,
    )
    return gate_static.try_static_linear(
        root,
        root.fc1,
        SimpleNamespace(eps=1e-5),
        x,
        object(),
        object(),
        object(),
        object(),
        object(),
        gate_static.ROW_IDS_TOKEN,
        "auto",
        "exact",
    )


def _try_rms(monkeypatch, root, x, native_calls):
    for name in (
        "_tensor_inputs_eligible",
        "_spark_device_eligible",
        "_norm_eligible",
        "_linear_eligible",
    ):
        monkeypatch.setattr(rms_static, name, lambda *_args: True)
    monkeypatch.setattr(
        rms_static,
        "_load_native",
        lambda: native_calls.append(True) or None,
    )
    return rms_static.try_static_route(
        root.qkv,
        SimpleNamespace(eps=1e-5),
        x,
        object(),
        object(),
        object(),
        rms_static.ROW_IDS_TOKEN,
        "auto",
        "exact",
    )


def test_real_plan_certificate_allows_q_and_gate_then_rejects_foreign_hook(
    tmp_path, monkeypatch
):
    root, plan = _plan(tmp_path)
    monkeypatch.setattr(linear_nvfp4, "_cuda_device_supported", lambda device: True)
    monkeypatch.setattr(
        linear_nvfp4, "_linear_config_eligible", lambda linear, width: True
    )

    with FakeTensorMode(), torch.no_grad():
        with plan.activate(enable_fusion_bridge=True):
            q_ids = getattr(root.q_out, scales.FUSION_PLAN_HOOK_IDS_ATTR)
            fc_ids = getattr(root.fc1, scales.FUSION_PLAN_HOOK_IDS_ATTR)
            assert set(root.q_out._forward_pre_hooks) == {q_ids[0]}
            assert set(root.q_out._forward_hooks) == {q_ids[1]}
            assert set(root.q_out._forward_hooks_always_called) == {q_ids[1]}
            assert set(root.fc1._forward_pre_hooks) == {fc_ids[0]}
            assert set(root.fc1._forward_hooks) == {fc_ids[1]}
            assert set(root.fc1._forward_hooks_always_called) == {fc_ids[1]}

            assert linear_nvfp4.can_static_linear_for_output(
                root.q_out, 2, "cuda:0", torch.bfloat16, "auto"
            )
            gate_calls = []
            gate_x = torch.empty((2, 5376), device="cuda:0", dtype=torch.bfloat16)
            assert _try_gate(monkeypatch, root, gate_x, gate_calls) is None
            assert gate_calls == [True]
            rms_calls = []
            rms_x = torch.empty((2, 5376), device="cuda:0", dtype=torch.bfloat16)
            assert _try_rms(monkeypatch, root, rms_x, rms_calls) is None
            assert rms_calls == [True]
            swiglu_x = torch.empty(
                (2, 28672), device="cuda:0", dtype=torch.bfloat16
            )
            assert swiglu_nvfp4._static_scale(root.swiglu, swiglu_x) is (
                root.swiglu.input_scale
            )

            q_foreign = root.q_out.register_forward_hook(
                lambda _module, _args, output: output
            )
            fc_foreign = root.fc1.register_forward_pre_hook(
                lambda _module, _args: None
            )
            rms_foreign = root.qkv.register_forward_hook(
                lambda _module, _args, output: output
            )
            swiglu_foreign = root.swiglu.register_forward_pre_hook(
                lambda _module, _args: None
            )
            try:
                assert not linear_nvfp4.can_static_linear_for_output(
                    root.q_out, 2, "cuda:0", torch.bfloat16, "auto"
                )
                gate_calls.clear()
                assert _try_gate(monkeypatch, root, gate_x, gate_calls) is None
                assert gate_calls == []
                rms_calls.clear()
                assert _try_rms(monkeypatch, root, rms_x, rms_calls) is None
                assert rms_calls == []
                assert swiglu_nvfp4._static_scale(root.swiglu, swiglu_x) is None
            finally:
                q_foreign.remove()
                fc_foreign.remove()
                rms_foreign.remove()
                swiglu_foreign.remove()

    for module in (root.q_out, root.qkv, root.fc1, root.swiglu):
        assert not hasattr(module, "input_scale")
        for name in (
            scales.FUSION_ENABLED_ATTR,
            scales.FUSION_ALLOWED_SHAPES_ATTR,
            scales.FUSION_ALLOWED_DTYPES_ATTR,
            scales.FUSION_VALIDATED_SCALE_ATTR,
            scales.FUSION_PLAN_HOOK_IDS_ATTR,
        ):
            assert not hasattr(module, name)
        assert not module._forward_pre_hooks
        assert not module._forward_hooks


def test_real_plan_preexisting_hook_suppresses_bridge_and_restores(
    tmp_path, monkeypatch
):
    root, plan = _plan(tmp_path)
    monkeypatch.setattr(linear_nvfp4, "_cuda_device_supported", lambda device: True)
    monkeypatch.setattr(
        linear_nvfp4, "_linear_config_eligible", lambda linear, width: True
    )
    foreign = root.q_out.register_forward_hook(
        lambda _module, _args, output: output
    )
    try:
        with FakeTensorMode(), torch.no_grad():
            with plan.activate(enable_fusion_bridge=True):
                assert not hasattr(root.q_out, scales.FUSION_ENABLED_ATTR)
                assert not hasattr(root.q_out, scales.FUSION_PLAN_HOOK_IDS_ATTR)
                assert not linear_nvfp4.can_static_linear_for_output(
                    root.q_out, 2, "cuda:0", torch.bfloat16, "auto"
                )
        assert set(root.q_out._forward_hooks) == {foreign.id}
        assert not hasattr(root.q_out, "input_scale")
    finally:
        foreign.remove()
