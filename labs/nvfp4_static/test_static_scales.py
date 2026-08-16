from __future__ import annotations

import json
import os
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open

from h3_nvfp4_static_scales import (
    NVFP4_SCALE_DENOMINATOR,
    ActivationCalibrator,
    ArtifactError,
    FUSION_ALLOWED_DTYPES_ATTR,
    FUSION_ALLOWED_SHAPES_ATTR,
    FUSION_ENABLED_ATTR,
    FUSION_PLAN_HOOK_IDS_ATTR,
    FUSION_VALIDATED_SCALE_ATTR,
    StaticScalePlan,
    StaticScaleValidator,
    artifact_paths,
    _positive_finite_float,
    load_artifact,
    merge_calibrations,
    verify_promotion,
    save_calibration,
)


class FakeNVFP4Linear(torch.nn.Module):
    def __init__(self, width: int, pre_quant_scale: torch.Tensor | None = None):
        super().__init__()
        self.in_features = width
        self.out_features = width + 1
        self._orig_shape = (self.out_features, self.in_features)
        self.quant_format = "nvfp4"
        self.layout_type = "TensorCoreNVFP4Layout"
        self.weight_function = []
        self.bias_function = []
        self.weight = torch.nn.Parameter(
            torch.zeros(self.out_features, self.in_features), requires_grad=False
        )
        if pre_quant_scale is not None:
            self.register_parameter(
                "pre_quant_scale", torch.nn.Parameter(pre_quant_scale, requires_grad=False)
            )
        self.dynamic_reductions = 0
        self.static_hits = 0
        self.seen_scales: list[torch.Tensor] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        scale = getattr(self, "input_scale", None)
        if scale is None:
            self.dynamic_reductions += 1
            effective = value
            if hasattr(self, "pre_quant_scale"):
                effective = effective * self.pre_quant_scale.to(value)
            scale = (torch.amax(effective.abs()) / NVFP4_SCALE_DENOMINATOR).float()
        else:
            self.static_hits += 1
        self.seen_scales.append(scale.detach())
        return value


class FakeBlock(torch.nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.attn = torch.nn.Module()
        self.attn.qkv_proj = FakeNVFP4Linear(width)
        self.attn.out_proj = FakeNVFP4Linear(width + 1)
        self.mlp = torch.nn.Module()
        self.mlp.fc1 = FakeNVFP4Linear(width + 2)
        self.mlp.fc2 = FakeNVFP4Linear(
            width + 3, pre_quant_scale=torch.arange(1, width + 4, dtype=torch.float32)
        )


class FakeRoot(torch.nn.Module):
    def __init__(self, width: int = 4):
        super().__init__()
        self.blocks = torch.nn.ModuleList([FakeBlock(width), FakeBlock(width)])


def exercise(root: FakeRoot, multiplier: float, rows: int = 3) -> None:
    for index, module in enumerate(
        module for module in root.modules() if isinstance(module, FakeNVFP4Linear)
    ):
        width = module.in_features
        values = torch.arange(1, width * rows + 1, dtype=torch.float32).reshape(rows, width)
        module(values * multiplier * (index + 1))


class StaticScaleTests(unittest.TestCase):
    def test_fusion_bridge_certifies_only_plan_owned_hooks_and_restores(self):
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            root, calibrator = self.calibrate(directory)
            prefix = directory / "FixedRef"
            save_calibration(calibrator, prefix)
            plan = StaticScalePlan(root, prefix, strict=True, require_h3=False)
            module = root.blocks[0].attn.qkv_proj

            self.assertFalse(hasattr(module, FUSION_PLAN_HOOK_IDS_ATTR))
            with plan.activate(enable_fusion_bridge=True):
                hook_ids = getattr(module, FUSION_PLAN_HOOK_IDS_ATTR)
                self.assertEqual(set(module._forward_pre_hooks), {hook_ids[0]})
                self.assertEqual(set(module._forward_hooks), {hook_ids[1]})
                self.assertFalse(module._forward_pre_hooks_with_kwargs)
                self.assertFalse(module._forward_hooks_with_kwargs)
                self.assertEqual(
                    set(module._forward_hooks_always_called), {hook_ids[1]}
                )
                self.assertFalse(module._backward_pre_hooks)
                self.assertFalse(module._backward_hooks)

                foreign = module.register_forward_hook(
                    lambda _module, _args, output: output
                )
                try:
                    self.assertNotEqual(
                        set(module._forward_hooks), {hook_ids[1]}
                    )
                finally:
                    foreign.remove()
                self.assertEqual(set(module._forward_hooks), {hook_ids[1]})

            self.assertFalse(hasattr(module, "input_scale"))
            for attribute in (
                FUSION_ENABLED_ATTR,
                FUSION_ALLOWED_SHAPES_ATTR,
                FUSION_ALLOWED_DTYPES_ATTR,
                FUSION_VALIDATED_SCALE_ATTR,
                FUSION_PLAN_HOOK_IDS_ATTR,
            ):
                self.assertFalse(hasattr(module, attribute))
            self.assertFalse(module._forward_pre_hooks)
            self.assertFalse(module._forward_hooks)

    def test_preexisting_hook_suppresses_bridge_and_restores(self):
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            root, calibrator = self.calibrate(directory)
            prefix = directory / "FixedRef"
            save_calibration(calibrator, prefix)
            plan = StaticScalePlan(root, prefix, strict=True, require_h3=False)
            module = root.blocks[0].attn.qkv_proj
            foreign = module.register_forward_pre_hook(lambda _module, _args: None)
            try:
                with plan.activate(enable_fusion_bridge=True):
                    self.assertFalse(hasattr(module, FUSION_ENABLED_ATTR))
                    self.assertFalse(hasattr(module, FUSION_PLAN_HOOK_IDS_ATTR))
                    self.assertIn(foreign.id, module._forward_pre_hooks)
                self.assertEqual(set(module._forward_pre_hooks), {foreign.id})
                self.assertFalse(module._forward_hooks)
            finally:
                foreign.remove()

    def test_post_hook_registration_failure_restores_pre_hook_and_bridge(self):
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            root, calibrator = self.calibrate(directory)
            prefix = directory / "FixedRef"
            save_calibration(calibrator, prefix)
            plan = StaticScalePlan(root, prefix, strict=True, require_h3=False)
            module = root.blocks[0].attn.qkv_proj
            original = module.register_forward_hook

            def fail(*_args, **_kwargs):
                raise RuntimeError("registration failure")

            module.register_forward_hook = fail
            try:
                with self.assertRaisesRegex(RuntimeError, "registration failure"):
                    with plan.activate(enable_fusion_bridge=True):
                        self.fail("scope entered")
            finally:
                module.register_forward_hook = original

            self.assertFalse(module._forward_pre_hooks)
            self.assertFalse(module._forward_hooks)
            self.assertFalse(hasattr(module, "input_scale"))
            self.assertFalse(hasattr(module, FUSION_PLAN_HOOK_IDS_ATTR))

    def test_merge_numeric_validation_wraps_huge_integer_overflow(self):
        with self.assertRaisesRegex(ArtifactError, "positive finite number"):
            _positive_finite_float(10**10000, "malformed statistic")

    def save_run(
        self,
        directory: Path,
        prefix_name: str,
        *,
        calls: list[tuple[int, float]],
        margin: float = 1.25,
        model_id: str = "fake-h3-nvfp4",
        concept: Path | None = None,
        checkpoint: Path | None = None,
        width: int = 4,
    ):
        if concept is None:
            concept = directory / "fixed_ref.png"
            concept.write_bytes(b"representative-chefs-son-test")
        root = FakeRoot(width=width)
        calibrator = ActivationCalibrator(
            root,
            margin=margin,
            model_id=model_id,
            concept_path=concept,
            checkpoint_path=checkpoint,
            require_h3=False,
        )
        with calibrator.capture():
            for rows, multiplier in calls:
                exercise(root, multiplier, rows=rows)
        prefix = directory / prefix_name
        save_calibration(calibrator, prefix)
        return root, calibrator, prefix

    def calibrate(self, directory: Path):
        root = FakeRoot()
        concept = directory / "fixed_ref.png"
        concept.write_bytes(b"representative-chefs-son-test")
        calibrator = ActivationCalibrator(
            root,
            margin=1.25,
            model_id="fake-h3-nvfp4",
            concept_path=concept,
            require_h3=False,
        )
        with calibrator.capture():
            exercise(root, 1.0)
            exercise(root, 2.0)
        return root, calibrator

    def test_calibration_records_every_call_and_preserves_dynamic_math(self):
        with tempfile.TemporaryDirectory() as value:
            root, calibrator = self.calibrate(Path(value))
            records = calibrator.records()
            self.assertEqual(len(records), 8)
            for module in (m for m in root.modules() if isinstance(m, FakeNVFP4Linear)):
                self.assertEqual(module.dynamic_reductions, 0)
                self.assertEqual(module.static_hits, 2)
                self.assertFalse(hasattr(module, "input_scale"))
            for record in records.values():
                self.assertEqual(record["calls"], 2)
                self.assertEqual(len(record["input_shapes"]), 1)
                self.assertGreater(record["max_abs"], 0.0)
                self.assertEqual(record["activation_dtypes"], ["torch.float32"])
                self.assertEqual(
                    record["input_scale"],
                    torch.tensor(
                        record["max_dynamic_input_scale"] * 1.25,
                        dtype=torch.float32,
                    ).item(),
                )
                self.assertEqual(
                    record["call_dynamic_input_scale_stats"]["max"],
                    record["max_dynamic_input_scale"],
                )

            # The pre-quant-scale layer must record the post-AWQ activation max.
            record = records["blocks.0.mlp.fc2"]
            module = root.blocks[0].mlp.fc2
            width = module.in_features
            raw = torch.arange(1, width * 3 + 1, dtype=torch.float32).reshape(3, width)
            expected = float(torch.amax((raw * 2.0 * 4) * module.pre_quant_scale))
            self.assertEqual(record["max_abs"], expected)

    def test_json_safetensors_pair_and_scoped_restore(self):
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            root, calibrator = self.calibrate(directory)
            prefix = directory / "FixedRef"
            json_path, tensor_path = save_calibration(calibrator, prefix)
            self.assertEqual((json_path, tensor_path), artifact_paths(prefix))
            metadata, tensors = load_artifact(prefix)
            self.assertEqual(metadata["schema_version"], 3)
            self.assertEqual(metadata["calibration"]["captured_layer_count"], 8)
            self.assertEqual(metadata["calibration"]["activation_dtypes"], ["torch.float32"])
            self.assertEqual(len(tensors), 8)
            with safe_open(tensor_path, framework="pt", device="cpu") as handle:
                self.assertEqual(len(handle.keys()), 24)
                self.assertEqual(handle.metadata()["artifact_sha256"], metadata["artifact_sha256"])

            plan = StaticScalePlan(
                root,
                prefix,
                strict=True,
                expected_model_id="fake-h3-nvfp4",
                require_h3=False,
            )
            self.assertEqual(plan.report.applied_layers, 8)
            with plan.activate():
                for module in (m for m in root.modules() if isinstance(m, FakeNVFP4Linear)):
                    self.assertIsInstance(module.input_scale, torch.Tensor)
                exercise(root, 1.5)
            for module in (m for m in root.modules() if isinstance(m, FakeNVFP4Linear)):
                self.assertFalse(hasattr(module, "input_scale"))

            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                with plan.activate():
                    raise RuntimeError("synthetic failure")
            for module in (m for m in root.modules() if isinstance(m, FakeNVFP4Linear)):
                self.assertFalse(hasattr(module, "input_scale"))

    def test_architecture_mismatch_falls_back_or_raises(self):
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            _, calibrator = self.calibrate(directory)
            prefix = directory / "FixedRef"
            save_calibration(calibrator, prefix)
            different = FakeRoot(width=5)

            fallback = StaticScalePlan(
                different, prefix, strict=False, require_h3=False
            )
            self.assertFalse(fallback.report.architecture_match)
            self.assertEqual(fallback.report.applied_layers, 0)
            self.assertEqual(fallback.report.fallback_layers, 8)
            with fallback.activate():
                exercise(different, 1.0)
            self.assertTrue(
                all(
                    module.dynamic_reductions == 1
                    for module in different.modules()
                    if isinstance(module, FakeNVFP4Linear)
                )
            )

            with self.assertRaises(ArtifactError):
                StaticScalePlan(different, prefix, strict=True, require_h3=False)

    def test_incomplete_pair_is_rejected(self):
        with tempfile.TemporaryDirectory() as value:
            prefix = Path(value) / "missing"
            json_path, _ = artifact_paths(prefix)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps({"schema": "wrong"}), encoding="utf-8")
            with self.assertRaises(ArtifactError):
                load_artifact(prefix)

    def test_shape_guard_and_held_out_overflow_validator(self):
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            root, calibrator = self.calibrate(directory)
            prefix = directory / "FixedRef"
            save_calibration(calibrator, prefix)
            plan = StaticScalePlan(root, prefix, strict=True, require_h3=False)
            validator = StaticScaleValidator(
                plan,
                concept_path=directory / "fixed_ref.png",
            )

            # 1.10x of the calibration maximum stays under a 1.25 margin.
            with plan.activate():
                with validator.capture():
                    exercise(root, 2.20)
            report = validator.report()
            self.assertTrue(report["summary"]["pass"])
            self.assertEqual(report["summary"]["overflow_count"], 0)
            self.assertEqual(report["summary"]["static_calls"], 8)

            # A shorter, unobserved row count is conservatively dynamic.
            module = root.blocks[0].attn.qkv_proj
            dynamic_before = module.dynamic_reductions
            with plan.activate():
                module(torch.ones(2, module.in_features))
            self.assertEqual(module.dynamic_reductions, dynamic_before + 1)

            # The exact observed shape uses the static scale.
            static_before = module.static_hits
            with plan.activate():
                module(torch.ones(3, module.in_features))
            self.assertEqual(module.static_hits, static_before + 1)

            # A row count beyond the calibrated envelope is dynamic.
            dynamic_before = module.dynamic_reductions
            with plan.activate():
                module(torch.ones(4, module.in_features))
            self.assertEqual(module.dynamic_reductions, dynamic_before + 1)
            self.assertFalse(hasattr(module, "input_scale"))

            # An uncalibrated compute dtype must use the dynamic path.
            dynamic_before = module.dynamic_reductions
            with plan.activate():
                module(torch.ones(3, module.in_features, dtype=torch.bfloat16))
            self.assertEqual(module.dynamic_reductions, dynamic_before + 1)

            # A function patch installed after plan creation must also disable the
            # static scale for that call.
            dynamic_before = module.dynamic_reductions
            module.weight_function.append(object())
            try:
                with plan.activate():
                    module(torch.ones(3, module.in_features))
            finally:
                module.weight_function.clear()
            self.assertEqual(module.dynamic_reductions, dynamic_before + 1)

            # Deferred DynamicVRAM patches installed after plan construction are
            # equally ineligible. At scope entry they receive neither a static
            # scale nor fusion bridge metadata.
            for lowvram_attribute in (
                "weight_lowvram_function",
                "bias_lowvram_function",
            ):
                with self.subTest(lowvram_attribute=lowvram_attribute):
                    dynamic_before = module.dynamic_reductions
                    setattr(module, lowvram_attribute, object())
                    try:
                        with plan.activate(enable_fusion_bridge=True):
                            self.assertFalse(hasattr(module, "input_scale"))
                            self.assertFalse(hasattr(module, FUSION_ENABLED_ATTR))
                            self.assertFalse(
                                hasattr(module, FUSION_ALLOWED_SHAPES_ATTR)
                            )
                            self.assertFalse(
                                hasattr(module, FUSION_ALLOWED_DTYPES_ATTR)
                            )
                            module(torch.ones(3, module.in_features))
                    finally:
                        delattr(module, lowvram_attribute)
                    self.assertEqual(module.dynamic_reductions, dynamic_before + 1)

            # Same calibrated shape but a large outlier exercises static overflow reporting.
            with plan.activate():
                with validator.capture():
                    exercise(root, 4.0)
            overflow = validator.report()
            self.assertFalse(overflow["summary"]["pass"])
            self.assertEqual(overflow["summary"]["overflow_count"], 8)
            self.assertGreater(overflow["summary"]["max_ratio"], 1.0)

    def test_apply_fusion_bridge_attrs_unknown_shape_contract_and_restoration(self):
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            root, calibrator = self.calibrate(directory)
            prefix = directory / "FixedRef"
            save_calibration(calibrator, prefix)
            plan = StaticScalePlan(root, prefix, strict=True, require_h3=False)
            target = root.blocks[0].attn.qkv_proj
            preserved = root.blocks[0].attn.out_proj
            setattr(preserved, FUSION_ENABLED_ATTR, "prior-marker")
            setattr(preserved, FUSION_ALLOWED_SHAPES_ATTR, frozenset({(99, 5)}))
            setattr(preserved, FUSION_ALLOWED_DTYPES_ATTR, frozenset({"prior.dtype"}))
            prior_validated_scale = object()
            setattr(preserved, FUSION_VALIDATED_SCALE_ATTR, prior_validated_scale)

            with self.assertRaisesRegex(RuntimeError, "bridge failure"):
                with plan.activate(enable_fusion_bridge=True):
                    for name in plan.assignments:
                        module = plan.modules[name]
                        self.assertIs(getattr(module, FUSION_ENABLED_ATTR), True)
                        self.assertIsInstance(
                            getattr(module, FUSION_ALLOWED_SHAPES_ATTR), frozenset
                        )
                        self.assertEqual(
                            getattr(module, FUSION_ALLOWED_SHAPES_ATTR),
                            plan.allowed_shapes[name],
                        )
                        self.assertEqual(
                            getattr(module, FUSION_ALLOWED_DTYPES_ATTR),
                            plan.activation_dtypes[name],
                        )
                        self.assertIs(
                            getattr(module, FUSION_VALIDATED_SCALE_ATTR),
                            module.input_scale,
                        )

                    # The fused consumer's eligibility contract must reject an
                    # unseen activated Linear input shape. It then delegates to
                    # Linear.forward, whose plan pre-hook dynamically removes the
                    # scale for this call.
                    self.assertNotIn(
                        (2, target.in_features),
                        getattr(target, FUSION_ALLOWED_SHAPES_ATTR),
                    )
                    dynamic_before = target.dynamic_reductions
                    target(torch.ones(2, target.in_features))
                    self.assertEqual(target.dynamic_reductions, dynamic_before + 1)
                    self.assertIs(getattr(target, FUSION_ENABLED_ATTR), True)
                    self.assertTrue(hasattr(target, "input_scale"))
                    raise RuntimeError("bridge failure")

            for module in (m for m in root.modules() if isinstance(m, FakeNVFP4Linear)):
                self.assertFalse(hasattr(module, "input_scale"))
            for attribute_name in (
                FUSION_ENABLED_ATTR,
                FUSION_ALLOWED_SHAPES_ATTR,
                FUSION_ALLOWED_DTYPES_ATTR,
                FUSION_VALIDATED_SCALE_ATTR,
            ):
                self.assertFalse(hasattr(target, attribute_name))
            self.assertEqual(getattr(preserved, FUSION_ENABLED_ATTR), "prior-marker")
            self.assertEqual(
                getattr(preserved, FUSION_ALLOWED_SHAPES_ATTR), frozenset({(99, 5)})
            )
            self.assertEqual(
                getattr(preserved, FUSION_ALLOWED_DTYPES_ATTR), frozenset({"prior.dtype"})
            )
            self.assertIs(
                getattr(preserved, FUSION_VALIDATED_SCALE_ATTR),
                prior_validated_scale,
            )

    def test_validator_default_activate_has_no_fusion_marker(self):
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            root, calibrator = self.calibrate(directory)
            prefix = directory / "FixedRef"
            save_calibration(calibrator, prefix)
            plan = StaticScalePlan(root, prefix, strict=True, require_h3=False)
            validator = StaticScaleValidator(plan)

            with plan.activate():
                with validator.capture():
                    for module in (
                        m for m in root.modules() if isinstance(m, FakeNVFP4Linear)
                    ):
                        self.assertFalse(hasattr(module, FUSION_ENABLED_ATTR))
                        self.assertFalse(hasattr(module, FUSION_ALLOWED_SHAPES_ATTR))
                        self.assertFalse(hasattr(module, FUSION_ALLOWED_DTYPES_ATTR))
                    exercise(root, 1.0)
            report = validator.report()
            self.assertEqual(report["summary"]["calls"], 8)
            self.assertEqual(report["summary"]["static_calls"], 8)

    def test_explicit_layer_exclusion_stays_dynamic_and_is_reported(self):
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            root, calibrator = self.calibrate(directory)
            prefix = directory / "FixedRef"
            save_calibration(calibrator, prefix)
            excluded = "blocks.0.mlp.fc2"
            plan = StaticScalePlan(
                root,
                prefix,
                strict=True,
                require_h3=False,
                expected_model_id="fake-h3-nvfp4",
                excluded_layers=excluded,
            )
            self.assertNotIn(excluded, plan.assignments)
            self.assertEqual(plan.report.applied_layers, 7)
            self.assertEqual(plan.report.fallback_layers, 1)
            self.assertEqual(plan.report.excluded_layers, (excluded,))

            validator = StaticScaleValidator(
                plan,
                concept_path=directory / "fixed_ref.png",
            )
            target = plan.modules[excluded]
            dynamic_before = target.dynamic_reductions
            with plan.activate(enable_fusion_bridge=True):
                with validator.capture():
                    exercise(root, 1.0)
            self.assertEqual(target.dynamic_reductions, dynamic_before + 1)
            self.assertFalse(hasattr(target, "input_scale"))
            self.assertFalse(hasattr(target, FUSION_ENABLED_ATTR))
            report = validator.report()
            self.assertTrue(report["summary"]["pass"])
            self.assertEqual(report["summary"]["layers"], 7)
            self.assertEqual(report["summary"]["excluded_layers"], [excluded])
            self.assertEqual(report["artifact"]["excluded_layers"], [excluded])
            self.assertEqual(
                report["artifact"]["artifact_sha256"], plan.artifact_sha256
            )

            report_path = directory / "heldout.validation.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            promotion = verify_promotion(
                prefix,
                excluded_layers=excluded,
                report_path=report_path,
            )
            self.assertTrue(promotion["pass"])
            self.assertEqual(promotion["layers"], 7)
            self.assertEqual(promotion["excluded_layers"], [excluded])

            tampered = json.loads(report_path.read_text(encoding="utf-8"))
            tampered["artifact"]["artifact_sha256"] = "0" * 64
            report_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ArtifactError, "artifact SHA-256"):
                verify_promotion(
                    prefix,
                    excluded_layers=excluded,
                    report_path=report_path,
                )

            with self.assertRaisesRegex(ArtifactError, "Unknown excluded"):
                StaticScalePlan(
                    root,
                    prefix,
                    strict=True,
                    require_h3=False,
                    excluded_layers="blocks.99.mlp.fc2",
                )

    def test_preexisting_module_hooks_suppress_and_restore_fusion_bridge(self):
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            root, calibrator = self.calibrate(directory)
            prefix = directory / "FixedRef"
            save_calibration(calibrator, prefix)
            plan = StaticScalePlan(root, prefix, strict=True, require_h3=False)
            target = root.blocks[0].attn.qkv_proj

            hook_cases = {
                "forward_pre": lambda count: target.register_forward_pre_hook(
                    lambda _module, _args: count.append(1)
                ),
                "forward": lambda count: target.register_forward_hook(
                    lambda _module, _args, _output: count.append(1)
                ),
                "backward_pre": lambda count: target.register_full_backward_pre_hook(
                    lambda _module, _grad_output: count.append(1)
                ),
                "backward": lambda count: target.register_full_backward_hook(
                    lambda _module, _grad_input, _grad_output: count.append(1)
                ),
            }
            hook_attributes = (
                "_forward_pre_hooks",
                "_forward_hooks",
                "_backward_pre_hooks",
                "_backward_hooks",
            )

            for hook_name, register in hook_cases.items():
                with self.subTest(hook=hook_name):
                    prior_marker = object()
                    prior_shapes = object()
                    prior_dtypes = object()
                    prior_validated_scale = object()
                    setattr(target, FUSION_ENABLED_ATTR, prior_marker)
                    setattr(target, FUSION_ALLOWED_SHAPES_ATTR, prior_shapes)
                    setattr(target, FUSION_ALLOWED_DTYPES_ATTR, prior_dtypes)
                    setattr(
                        target,
                        FUSION_VALIDATED_SCALE_ATTR,
                        prior_validated_scale,
                    )
                    calls = []
                    handle = register(calls)
                    hook_sizes = {
                        name: len(getattr(target, name)) for name in hook_attributes
                    }
                    static_before = target.static_hits
                    try:
                        with plan.activate(enable_fusion_bridge=True):
                            self.assertTrue(hasattr(target, "input_scale"))
                            self.assertFalse(hasattr(target, FUSION_ENABLED_ATTR))
                            self.assertFalse(
                                hasattr(target, FUSION_ALLOWED_SHAPES_ATTR)
                            )
                            self.assertFalse(
                                hasattr(target, FUSION_ALLOWED_DTYPES_ATTR)
                            )
                            self.assertFalse(
                                hasattr(target, FUSION_VALIDATED_SCALE_ATTR)
                            )
                            needs_backward = hook_name.startswith("backward")
                            tensor = torch.ones(
                                3,
                                target.in_features,
                                requires_grad=needs_backward,
                            )
                            output = target(tensor)
                            if needs_backward:
                                output.sum().backward()

                        self.assertEqual(calls, [1])
                        self.assertEqual(target.static_hits, static_before + 1)
                        self.assertFalse(hasattr(target, "input_scale"))
                        self.assertIs(
                            getattr(target, FUSION_ENABLED_ATTR), prior_marker
                        )
                        self.assertIs(
                            getattr(target, FUSION_ALLOWED_SHAPES_ATTR), prior_shapes
                        )
                        self.assertIs(
                            getattr(target, FUSION_ALLOWED_DTYPES_ATTR), prior_dtypes
                        )
                        self.assertIs(
                            getattr(target, FUSION_VALIDATED_SCALE_ATTR),
                            prior_validated_scale,
                        )
                        self.assertEqual(
                            {
                                name: len(getattr(target, name))
                                for name in hook_attributes
                            },
                            hook_sizes,
                        )
                    finally:
                        handle.remove()
                        for attribute_name in (
                            FUSION_ENABLED_ATTR,
                            FUSION_ALLOWED_SHAPES_ATTR,
                            FUSION_ALLOWED_DTYPES_ATTR,
                            FUSION_VALIDATED_SCALE_ATTR,
                        ):
                            if hasattr(target, attribute_name):
                                delattr(target, attribute_name)

    def test_json_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            _, calibrator = self.calibrate(directory)
            prefix = directory / "FixedRef"
            json_path, _ = save_calibration(calibrator, prefix)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            payload["calibration"]["margin"] = 99.0
            json_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ArtifactError, "JSON checksum"):
                load_artifact(prefix)

    def test_nested_malformed_metadata_falls_back_without_attribute_error(self):
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            root, calibrator = self.calibrate(directory)
            prefix = directory / "FixedRef"
            json_path, _ = save_calibration(calibrator, prefix)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            payload["model"] = []
            checksum_value = {
                key: item
                for key, item in payload.items()
                if key not in ("artifact_sha256", "scale_file")
            }
            payload["artifact_sha256"] = hashlib.sha256(
                json.dumps(
                    checksum_value,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            json_path.write_text(json.dumps(payload), encoding="utf-8")

            plan = StaticScalePlan(root, prefix, strict=False, require_h3=False)
            self.assertEqual(plan.assignments, {})
            self.assertIn("model metadata", plan.report.reason)

    def test_function_patched_layer_is_rejected_at_plan_construction(self):
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            root, calibrator = self.calibrate(directory)
            prefix = directory / "FixedRef"
            save_calibration(calibrator, prefix)
            patched_module = root.blocks[0].attn.qkv_proj
            patched_module.weight_function.append(object())
            try:
                plan = StaticScalePlan(root, prefix, strict=False, require_h3=False)
            finally:
                patched_module.weight_function.clear()
            self.assertEqual(plan.report.applied_layers, 7)
            self.assertEqual(plan.report.fallback_layers, 1)
            self.assertNotIn("blocks.0.attn.qkv_proj", plan.assignments)

            for lowvram_attribute in (
                "weight_lowvram_function",
                "bias_lowvram_function",
            ):
                with self.subTest(lowvram_attribute=lowvram_attribute):
                    setattr(patched_module, lowvram_attribute, object())
                    try:
                        lowvram_plan = StaticScalePlan(
                            root, prefix, strict=False, require_h3=False
                        )
                        self.assertEqual(lowvram_plan.report.applied_layers, 7)
                        self.assertEqual(lowvram_plan.report.fallback_layers, 1)
                        self.assertNotIn(
                            "blocks.0.attn.qkv_proj", lowvram_plan.assignments
                        )
                        dynamic_before = patched_module.dynamic_reductions
                        with lowvram_plan.activate(enable_fusion_bridge=True):
                            self.assertFalse(hasattr(patched_module, "input_scale"))
                            self.assertFalse(
                                hasattr(patched_module, FUSION_ENABLED_ATTR)
                            )
                            self.assertFalse(
                                hasattr(patched_module, FUSION_ALLOWED_SHAPES_ATTR)
                            )
                            self.assertFalse(
                                hasattr(patched_module, FUSION_ALLOWED_DTYPES_ATTR)
                            )
                            patched_module(
                                torch.ones(3, patched_module.in_features)
                            )
                        self.assertEqual(
                            patched_module.dynamic_reductions, dynamic_before + 1
                        )
                    finally:
                        delattr(patched_module, lowvram_attribute)

    def test_lowvram_function_layer_is_dynamic_during_calibration(self):
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            concept = directory / "fixed_ref.png"
            concept.write_bytes(b"representative-chefs-son-test")
            for lowvram_attribute in (
                "weight_lowvram_function",
                "bias_lowvram_function",
            ):
                with self.subTest(lowvram_attribute=lowvram_attribute):
                    root = FakeRoot()
                    target = root.blocks[0].attn.qkv_proj
                    setattr(target, lowvram_attribute, object())
                    calibrator = ActivationCalibrator(
                        root,
                        margin=1.25,
                        model_id="fake-h3-nvfp4",
                        concept_path=concept,
                        require_h3=False,
                    )
                    self.assertNotIn("blocks.0.attn.qkv_proj", calibrator.modules)
                    with calibrator.capture():
                        exercise(root, 1.0)
                    records = calibrator.records()
                    self.assertNotIn("blocks.0.attn.qkv_proj", records)
                    self.assertEqual(target.dynamic_reductions, 1)
                    self.assertFalse(hasattr(target, "input_scale"))
                    self.assertFalse(hasattr(target, FUSION_ENABLED_ATTR))

    def test_merge_uses_max_exact_scale_unions_shapes_and_loads_applies(self):
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            concept = directory / "fixed_ref.png"
            concept.write_bytes(b"representative-chefs-son-test")
            checkpoint = directory / "model.safetensors"
            checkpoint.write_bytes(b"same-checkpoint-contract")
            _, _, first = self.save_run(
                directory,
                "chef_seed_1",
                calls=[(2, 1.0), (2, 2.0)],
                margin=1.0,
                concept=concept,
                checkpoint=checkpoint,
            )
            _, _, second = self.save_run(
                directory,
                "chef_seed_2",
                calls=[(3, 4.0)],
                margin=1.0,
                concept=concept,
                checkpoint=checkpoint,
            )
            first_metadata, _ = load_artifact(first)
            second_metadata, _ = load_artifact(second)

            output = directory / "chef_merged"
            json_path, tensor_path = merge_calibrations([first, second], output)
            self.assertEqual((json_path, tensor_path), artifact_paths(output))
            merged, scale_tensors = load_artifact(output)
            self.assertEqual(merged["merge"]["source_count"], 2)
            self.assertEqual(merged["calibration"]["captured_layer_count"], 8)
            self.assertEqual(merged["calibration"]["total_linear_calls"], 24)
            self.assertEqual(len(scale_tensors), 8)
            self.assertEqual(merged["model"], first_metadata["model"])
            self.assertEqual(
                merged["calibration"]["concept"],
                first_metadata["calibration"]["concept"],
            )

            for name, record in merged["layers"].items():
                first_record = first_metadata["layers"][name]
                second_record = second_metadata["layers"][name]
                expected_dynamic = max(
                    first_record["max_dynamic_input_scale"],
                    second_record["max_dynamic_input_scale"],
                )
                expected_max_abs = max(first_record["max_abs"], second_record["max_abs"])
                self.assertEqual(record["calls"], 3)
                self.assertEqual(record["max_dynamic_input_scale"], expected_dynamic)
                self.assertEqual(record["max_abs"], expected_max_abs)
                self.assertEqual(
                    record["input_scale"],
                    expected_dynamic,
                )
                self.assertEqual(
                    {shape[0] for shape in record["input_shapes"]},
                    {2, 3},
                )
                self.assertIsNone(record["call_max_abs_stats"]["median"])
                self.assertIsNone(
                    record["call_dynamic_input_scale_stats"]["median_over_max"]
                )

            with safe_open(tensor_path, framework="pt", device="cpu") as handle:
                self.assertEqual(len(handle.keys()), 24)
                self.assertEqual(
                    handle.metadata()["artifact_sha256"],
                    merged["artifact_sha256"],
                )

            runtime = FakeRoot()
            plan = StaticScalePlan(
                runtime,
                output,
                strict=True,
                expected_model_id="fake-h3-nvfp4",
                checkpoint_path=checkpoint,
                require_h3=False,
            )
            with plan.activate():
                exercise(runtime, 1.0, rows=2)
                exercise(runtime, 1.0, rows=3)
                exercise(runtime, 1.0, rows=4)
            for module in (m for m in runtime.modules() if isinstance(m, FakeNVFP4Linear)):
                self.assertEqual(module.static_hits, 2)
                self.assertEqual(module.dynamic_reductions, 1)
                self.assertFalse(hasattr(module, "input_scale"))

            wider = directory / "chef_merged_wider"
            merge_calibrations(
                [first, second],
                wider,
                output_margin=1.25,
            )
            wider_metadata, _ = load_artifact(wider)
            self.assertEqual(wider_metadata["calibration"]["margin"], 1.25)
            self.assertEqual(wider_metadata["merge"]["source_margin"], 1.0)
            self.assertEqual(wider_metadata["merge"]["output_margin"], 1.25)
            for name, record in wider_metadata["layers"].items():
                expected_dynamic = merged["layers"][name][
                    "max_dynamic_input_scale"
                ]
                expected_scale = float(
                    torch.tensor(expected_dynamic * 1.25, dtype=torch.float32).item()
                )
                self.assertEqual(record["input_scale"], expected_scale)

            with self.assertRaisesRegex(ArtifactError, "cannot be smaller"):
                merge_calibrations(
                    [first, second],
                    directory / "chef_merged_narrower",
                    output_margin=0.99,
                )

    def test_merge_rejects_contract_mismatches_and_source_aliases(self):
        with tempfile.TemporaryDirectory() as value:
            directory = Path(value)
            concept = directory / "fixed_ref.png"
            concept.write_bytes(b"representative-chefs-son-test")
            other_concept = directory / "other.png"
            other_concept.write_bytes(b"different-concept")
            checkpoint = directory / "model.safetensors"
            checkpoint.write_bytes(b"same-checkpoint-contract")
            other_checkpoint = directory / "other-model.safetensors"
            other_checkpoint.write_bytes(b"different-checkpoint-contract")
            _, _, baseline = self.save_run(
                directory,
                "baseline",
                calls=[(3, 1.0)],
                concept=concept,
                checkpoint=checkpoint,
            )
            variants = {
                "margin": self.save_run(
                    directory,
                    "margin",
                    calls=[(3, 1.0)],
                    margin=1.5,
                    concept=concept,
                    checkpoint=checkpoint,
                )[2],
                "model": self.save_run(
                    directory,
                    "model_id",
                    calls=[(3, 1.0)],
                    model_id="different-model-id",
                    concept=concept,
                    checkpoint=checkpoint,
                )[2],
                "concept": self.save_run(
                    directory,
                    "concept",
                    calls=[(3, 1.0)],
                    concept=other_concept,
                    checkpoint=checkpoint,
                )[2],
                "checkpoint": self.save_run(
                    directory,
                    "checkpoint",
                    calls=[(3, 1.0)],
                    concept=concept,
                    checkpoint=other_checkpoint,
                )[2],
                "architecture": self.save_run(
                    directory,
                    "architecture",
                    calls=[(3, 1.0)],
                    concept=concept,
                    checkpoint=checkpoint,
                    width=5,
                )[2],
            }
            for label, variant in variants.items():
                with self.subTest(label=label):
                    with self.assertRaisesRegex(ArtifactError, "contract differs"):
                        merge_calibrations(
                            [baseline, variant],
                            directory / f"rejected_{label}",
                        )

            with self.assertRaisesRegex(ArtifactError, "distinct artifact pair"):
                merge_calibrations([baseline, baseline], directory / "duplicate")
            with self.assertRaisesRegex(ArtifactError, "prefix distinct"):
                merge_calibrations([baseline, variants["margin"]], baseline)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for comfy-kitchen NVFP4")
    def test_real_comfy_nvfp4_exact_static_scale_matches_dynamic_quantization(self):
        comfy_root = os.environ.get("COMFYUI_ROOT", "")
        if comfy_root not in sys.path:
            sys.path.insert(0, comfy_root)
        try:
            from comfy.quant_ops import QuantizedTensor
        except ImportError as error:
            self.skipTest(f"ComfyUI/comfy-kitchen is unavailable: {error}")

        row = torch.tensor(
            [0.1, -5.0, 2.0, 1.0] * 4,
            device="cuda",
            dtype=torch.bfloat16,
        )
        value = row.repeat(16, 1)
        exact_dynamic_scale = (
            torch.amax(value.abs()) / NVFP4_SCALE_DENOMINATOR
        ).to(torch.float32)
        reconstructed_from_host_max = torch.tensor(
            float(torch.amax(value.abs()).item()) / NVFP4_SCALE_DENOMINATOR,
            device="cuda",
            dtype=torch.float32,
        )
        self.assertFalse(torch.equal(exact_dynamic_scale, reconstructed_from_host_max))
        dynamic = QuantizedTensor.from_float(value, "TensorCoreNVFP4Layout")
        static = QuantizedTensor.from_float(
            value,
            "TensorCoreNVFP4Layout",
            scale=exact_dynamic_scale,
        )
        self.assertTrue(torch.equal(dynamic._qdata, static._qdata))
        self.assertTrue(
            torch.equal(dynamic._params.block_scale, static._params.block_scale)
        )
        self.assertTrue(torch.equal(dynamic._params.scale, static._params.scale))


if __name__ == "__main__":
    unittest.main()
