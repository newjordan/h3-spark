"""CPU-only rollout tests for the non-installed Comfy adapter."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from comfy_integration_reference import (
    NVFP4_LAYOUT,
    _linear_is_eligible,
    _route_from_mode_state,
    _tensor_inputs_are_eligible,
)


class FakeQuantizedTensor:
    def __init__(self, *, transposed: bool = False):
        self._layout_cls = NVFP4_LAYOUT
        self._params = SimpleNamespace(transposed=transposed)


def eligible_linear():
    return SimpleNamespace(
        in_features=5376,
        quant_format="nvfp4",
        layout_type=NVFP4_LAYOUT,
        weight=FakeQuantizedTensor(),
        pre_quant_scale=None,
        _full_precision_mm=False,
        comfy_force_cast_weights=False,
        weight_function=[],
        bias_function=[],
        weight_lowvram_function=None,
        bias_lowvram_function=None,
    )


class IntegrationReferenceTests(unittest.TestCase):
    def test_rollout_routes_fail_closed(self) -> None:
        self.assertIsNone(
            _route_from_mode_state(
                "stock",
                static_bridge_valid=True,
                input_scale_is_none=True,
                bridge_attrs_absent=True,
                hooks_absent=True,
            )
        )
        self.assertEqual(
            _route_from_mode_state(
                "static",
                static_bridge_valid=True,
                input_scale_is_none=False,
                bridge_attrs_absent=False,
                hooks_absent=True,
            ),
            "static",
        )
        self.assertIsNone(
            _route_from_mode_state(
                "static",
                static_bridge_valid=False,
                input_scale_is_none=False,
                bridge_attrs_absent=False,
                hooks_absent=True,
            )
        )
        self.assertEqual(
            _route_from_mode_state(
                "auto",
                static_bridge_valid=False,
                input_scale_is_none=True,
                bridge_attrs_absent=True,
                hooks_absent=True,
            ),
            "dynamic",
        )

    def test_auto_dynamic_rejects_hooks_or_stale_bridge_state(self) -> None:
        for bridge_absent, hooks_absent in ((False, True), (True, False)):
            with self.subTest(
                bridge_absent=bridge_absent,
                hooks_absent=hooks_absent,
            ):
                self.assertIsNone(
                    _route_from_mode_state(
                        "auto",
                        static_bridge_valid=False,
                        input_scale_is_none=True,
                        bridge_attrs_absent=bridge_absent,
                        hooks_absent=hooks_absent,
                    )
                )

    def test_linear_guard_rejects_each_mutating_feature(self) -> None:
        self.assertTrue(_linear_is_eligible(eligible_linear(), FakeQuantizedTensor))
        mutations = {
            "in_features": 7168,
            "pre_quant_scale": torch.ones(1),
            "_full_precision_mm": True,
            "comfy_force_cast_weights": True,
            "weight_function": [object()],
            "bias_function": [object()],
            "weight_lowvram_function": object(),
            "bias_lowvram_function": object(),
        }
        for name, value in mutations.items():
            with self.subTest(name=name):
                linear = eligible_linear()
                setattr(linear, name, value)
                self.assertFalse(
                    _linear_is_eligible(linear, FakeQuantizedTensor)
                )
        linear = eligible_linear()
        linear.weight = FakeQuantizedTensor(transposed=True)
        self.assertFalse(_linear_is_eligible(linear, FakeQuantizedTensor))

    def test_cpu_tensor_bundle_is_not_eligible(self) -> None:
        normed = torch.empty((2, 5376), dtype=torch.bfloat16)
        table = torch.empty((3, 6 * 5376), dtype=torch.float32)
        scale = table[:, 5376:2 * 5376]
        shift = table[:, 4 * 5376:5 * 5376]
        row_ids = torch.zeros(2, dtype=torch.int32)
        self.assertFalse(
            _tensor_inputs_are_eligible(normed, scale, shift, row_ids)
        )


if __name__ == "__main__":
    unittest.main()
