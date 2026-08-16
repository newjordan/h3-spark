from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


COMFY_ROOT = Path(os.environ["COMFYUI_ROOT"])
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))
spec = importlib.util.spec_from_file_location(
    "crossblock_gate_qkv_adapter", Path(__file__).with_name("comfy_adapter.py")
)
fused = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(fused)


class FakeQuantizedTensor:
    def __init__(self, data=None, layout_cls=None, params=None):
        self.data = data
        self._layout_cls = layout_cls
        self._params = params or SimpleNamespace(transposed=False)


class FakeLayout:
    class Params:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)


class CrossBlockAdapterTests(unittest.TestCase):
    def setUp(self):
        self.linear = SimpleNamespace(in_features=5376, out_features=21504)
        self.norm = SimpleNamespace(eps=1e-5)
        self.previous = SimpleNamespace(mlp=object())
        self.current = SimpleNamespace(
            mlp=object(),
            attn=SimpleNamespace(qkv_proj=self.linear),
            norm1=self.norm,
        )
        self.x = SimpleNamespace(
            dtype=torch.bfloat16,
            shape=(2, 5376),
            device=torch.device("cuda:0"),
        )
        self.other = object()
        self.gate = object()
        self.scale = object()
        self.shift = object()
        self.rows = object()
        self.segments = object()
        self.blocks_replace = {}
        self.qdata = torch.empty(2, 2688, dtype=torch.uint8)
        self.block_scale = torch.empty(128, 336, dtype=torch.uint8)
        self.input_scale = torch.tensor(0.25, dtype=torch.float32)
        self.producer = mock.Mock(
            return_value=(self.qdata, self.block_scale, self.input_scale, object())
        )
        self.native = SimpleNamespace(
            gate_rms_adaln_nvfp4_dynamic=self.producer
        )

    @contextmanager
    def runtime(self):
        with mock.patch.object(fused, "_pending_inputs_eligible", return_value=True), mock.patch.object(
            fused, "_load_native", return_value=self.native
        ), mock.patch.object(fused, "_global_hooks_absent", return_value=True), mock.patch.object(
            fused.torch, "is_grad_enabled", return_value=False
        ), mock.patch.object(fused.comfy.model_management, "in_training", False):
            pending = fused.defer_qkv_transition(
                self.previous,
                self.current,
                self.x,
                self.other,
                self.gate,
                self.segments,
                self.rows,
                fused.CROSSBLOCK_TOKEN,
                "auto",
                "exact",
                self.blocks_replace,
            )
            self.assertTrue(fused.is_pending(pending))
            yield pending

    def call(
        self,
        pending,
        *,
        mode="auto",
        row_token=fused.ROW_IDS_TOKEN,
        cross_token=fused.CROSSBLOCK_TOKEN,
        blocks_replace=None,
    ):
        return fused.try_qkv_transition(
            self.previous,
            self.current,
            self.x,
            pending,
            self.scale,
            self.shift,
            self.segments,
            self.rows,
            row_token,
            cross_token,
            mode,
            "exact",
            self.blocks_replace if blocks_replace is None else blocks_replace,
        )

    @staticmethod
    def guard_values():
        return {
            name: (lambda *_args: True)
            for name in (
                "_tensor_inputs_eligible",
                "_gate_inputs_eligible",
                "_norm_eligible",
                "_linear_eligible",
                "_dynamic_eligible",
                "parent_eligible",
                "_spark_device_eligible",
                "_storage_disjoint",
                "_quantized_outputs_eligible",
            )
        }

    def materialize(self, pending):
        materialize = mock.Mock(return_value=self.x)
        result = fused.materialize_pending(
            pending,
            self.previous,
            self.current,
            self.x,
            self.segments,
            self.rows,
            fused.CROSSBLOCK_TOKEN,
            materialize,
        )
        return result, materialize

    def set_complete_bridge(self, scale):
        setattr(self.linear, fused.STATIC_BRIDGE_ATTRS[0], True)
        setattr(
            self.linear,
            fused.STATIC_BRIDGE_ATTRS[1],
            frozenset({tuple(self.x.shape)}),
        )
        setattr(
            self.linear,
            fused.STATIC_BRIDGE_ATTRS[2],
            frozenset({str(torch.bfloat16)}),
        )
        setattr(self.linear, fused.STATIC_BRIDGE_ATTRS[3], scale)
        setattr(self.linear, fused.STATIC_BRIDGE_ATTRS[4], (11, 12))
        self.linear.input_scale = scale

    def test_carrier_is_sealed_and_binds_every_identity(self):
        with self.assertRaisesRegex(TypeError, "internal"):
            fused.H3CrossBlockPending(
                object(),
                self.previous,
                self.current,
                self.x,
                self.other,
                self.gate,
                self.segments,
                self.rows,
                self.blocks_replace,
                fused.CROSSBLOCK_TOKEN,
                self.native,
                self.producer,
                "dynamic",
                None,
            )
        with self.runtime() as pending:
            for args in (
                (object(), self.current, self.x, self.segments, self.rows),
                (self.previous, object(), self.x, self.segments, self.rows),
                (self.previous, self.current, object(), self.segments, self.rows),
                (self.previous, self.current, self.x, object(), self.rows),
                (self.previous, self.current, self.x, self.segments, object()),
            ):
                with self.assertRaisesRegex(RuntimeError, "invalid"):
                    fused.materialize_pending(
                        pending,
                        *args,
                        fused.CROSSBLOCK_TOKEN,
                        mock.Mock(),
                    )

    def test_fresh_empty_blocks_replace_is_not_the_sealed_mapping(self):
        with self.runtime() as pending:
            self.assertIs(pending._blocks_replace, self.blocks_replace)
            self.assertFalse(
                fused._consumer_still_eligible(
                    pending,
                    pending._mlp,
                    pending._gate,
                    self.previous,
                    self.current,
                    self.x,
                    self.segments,
                    self.rows,
                    fused.CROSSBLOCK_TOKEN,
                    "auto",
                    "exact",
                    {},
                )
            )
            self.assertIsNot(pending._blocks_replace, {})

    def test_stock_hooks_replacements_and_grad_never_defer(self):
        with mock.patch.object(fused, "_pending_inputs_eligible", return_value=True), mock.patch.object(
            fused, "_load_native", side_effect=AssertionError("native")
        ), mock.patch.object(fused, "_global_hooks_absent", return_value=True), mock.patch.object(
            fused.torch, "is_grad_enabled", return_value=False
        ), mock.patch.object(fused.comfy.model_management, "in_training", False):
            args = (
                self.previous,
                self.current,
                self.x,
                self.other,
                self.gate,
                self.segments,
                self.rows,
                fused.CROSSBLOCK_TOKEN,
            )
            self.assertIsNone(fused.defer_qkv_transition(*args, "stock", "exact", {}))
            self.assertIsNone(fused.defer_qkv_transition(*args, "auto", "exact", {("double_block", 3): object()}))
            self.previous._forward_hooks = {1: object()}
            self.assertIsNone(fused.defer_qkv_transition(*args, "auto", "exact", {}))
            self.previous._forward_hooks = {}
        with mock.patch.object(fused, "_pending_inputs_eligible", return_value=True), mock.patch.object(
            fused, "_global_hooks_absent", return_value=True
        ), mock.patch.object(fused.torch, "is_grad_enabled", return_value=True):
            self.assertIsNone(
                fused.defer_qkv_transition(
                    self.previous,
                    self.current,
                    self.x,
                    self.other,
                    self.gate,
                    self.segments,
                    self.rows,
                    fused.CROSSBLOCK_TOKEN,
                    "auto",
                    "exact",
                    {},
                )
            )

    def test_static_bridge_certificate_is_exact_and_identity_bound(self):
        cpu_x = SimpleNamespace(
            shape=(2, 5376), dtype=torch.bfloat16, device=torch.device("cpu")
        )
        scale = torch.tensor(0.25, dtype=torch.float32)
        self.x = cpu_x
        self.set_complete_bridge(scale)
        hook_match = mock.Mock(return_value=True)
        with mock.patch.object(
            fused.swiglu_nvfp4, "_plan_hooks_match", hook_match, create=True
        ), mock.patch.object(fused, "_global_hooks_absent", return_value=True):
            self.assertIs(fused._certified_static_scale(self.linear, cpu_x), scale)
        hook_match.assert_called_once_with(self.linear, (11, 12))
        setattr(self.linear, fused.STATIC_BRIDGE_ATTRS[3], scale.clone())
        with mock.patch.object(
            fused.swiglu_nvfp4, "_plan_hooks_match", return_value=True, create=True
        ), mock.patch.object(fused, "_global_hooks_absent", return_value=True):
            self.assertIsNone(fused._certified_static_scale(self.linear, cpu_x))

    def test_partial_or_malformed_static_bridge_never_uses_dynamic(self):
        setattr(self.linear, fused.STATIC_BRIDGE_ATTRS[0], True)
        with mock.patch.object(fused, "_pending_inputs_eligible", return_value=True), mock.patch.object(
            fused, "_load_native", side_effect=AssertionError("dynamic")
        ), mock.patch.object(fused, "_global_hooks_absent", return_value=True), mock.patch.object(
            fused.torch, "is_grad_enabled", return_value=False
        ), mock.patch.object(fused.comfy.model_management, "in_training", False):
            self.assertIsNone(
                fused.defer_qkv_transition(
                    self.previous,
                    self.current,
                    self.x,
                    self.other,
                    self.gate,
                    self.segments,
                    self.rows,
                    fused.CROSSBLOCK_TOKEN,
                    "auto",
                    "exact",
                    self.blocks_replace,
                )
            )

        self.set_complete_bridge(object())
        static_loader = mock.Mock(side_effect=AssertionError("static"))
        with mock.patch.object(fused, "_pending_inputs_eligible", return_value=True), mock.patch.object(
            fused, "_certified_static_scale", return_value=None
        ), mock.patch.object(
            fused, "static_gate", SimpleNamespace(_load_native=static_loader)
        ), mock.patch.object(
            fused, "_load_native", side_effect=AssertionError("dynamic")
        ), mock.patch.object(fused, "_global_hooks_absent", return_value=True), mock.patch.object(
            fused.torch, "is_grad_enabled", return_value=False
        ), mock.patch.object(fused.comfy.model_management, "in_training", False):
            self.assertIsNone(
                fused.defer_qkv_transition(
                    self.previous,
                    self.current,
                    self.x,
                    self.other,
                    self.gate,
                    self.segments,
                    self.rows,
                    fused.CROSSBLOCK_TOKEN,
                    "auto",
                    "exact",
                    self.blocks_replace,
                )
            )
        static_loader.assert_not_called()

    def test_complete_static_bridge_uses_static_only(self):
        static_scale = object()
        self.set_complete_bridge(static_scale)
        static_producer = mock.Mock(return_value=(self.qdata, self.block_scale))
        static_native = SimpleNamespace(
            gate_rms_adaln_nvfp4_static=static_producer
        )
        weight = FakeQuantizedTensor(
            layout_cls=fused.NVFP4_LAYOUT,
            params=SimpleNamespace(transposed=False),
        )
        with mock.patch.object(fused, "_pending_inputs_eligible", return_value=True), mock.patch.object(
            fused, "_certified_static_scale", return_value=static_scale
        ), mock.patch.object(
            fused,
            "static_gate",
            SimpleNamespace(_load_native=lambda: static_native),
        ), mock.patch.object(
            fused, "_load_native", side_effect=AssertionError("dynamic")
        ), mock.patch.object(fused, "_global_hooks_absent", return_value=True), mock.patch.object(
            fused.torch, "is_grad_enabled", return_value=False
        ), mock.patch.object(fused.comfy.model_management, "in_training", False):
            pending = fused.defer_qkv_transition(
                self.previous,
                self.current,
                self.x,
                self.other,
                self.gate,
                self.segments,
                self.rows,
                fused.CROSSBLOCK_TOKEN,
                "auto",
                "exact",
                self.blocks_replace,
            )
            self.assertEqual(pending._route, "static")
            self.assertIs(pending._input_scale, static_scale)
            with mock.patch.multiple(fused, **self.guard_values()), mock.patch.object(
                fused, "cast_bias_weight", side_effect=((object(), None, "norm"), (weight, None, "linear"))
            ), mock.patch.object(
                fused, "uncast_bias_weight"
            ), mock.patch.object(
                fused, "_cast_norm_weight_eligible", return_value=True
            ), mock.patch.object(
                fused, "QuantizedTensor", FakeQuantizedTensor
            ), mock.patch.object(
                fused, "TensorCoreNVFP4Layout", FakeLayout
            ), mock.patch.object(
                fused.torch.nn.functional, "linear", return_value="qkv"
            ), mock.patch.object(fused, "run_every_op"):
                self.assertEqual(self.call(pending), "qkv")
        self.producer.assert_not_called()
        self.assertIs(static_producer.call_args.args[-2], static_scale)

    def test_static_certificate_loss_after_claim_propagates(self):
        static_scale = object()
        self.set_complete_bridge(static_scale)

        def certified(linear, _x):
            return (
                static_scale
                if getattr(linear, fused.STATIC_BRIDGE_ATTRS[3], None)
                is static_scale
                else None
            )

        def produce(*_args):
            setattr(self.linear, fused.STATIC_BRIDGE_ATTRS[3], object())
            return self.qdata, self.block_scale

        static_native = SimpleNamespace(
            gate_rms_adaln_nvfp4_static=mock.Mock(side_effect=produce)
        )
        with mock.patch.object(fused, "_pending_inputs_eligible", return_value=True), mock.patch.object(
            fused, "_certified_static_scale", side_effect=certified
        ), mock.patch.object(
            fused,
            "static_gate",
            SimpleNamespace(_load_native=lambda: static_native),
        ), mock.patch.object(
            fused, "_load_native", side_effect=AssertionError("dynamic")
        ), mock.patch.object(fused, "_global_hooks_absent", return_value=True), mock.patch.object(
            fused.torch, "is_grad_enabled", return_value=False
        ), mock.patch.object(fused.comfy.model_management, "in_training", False):
            pending = fused.defer_qkv_transition(
                self.previous,
                self.current,
                self.x,
                self.other,
                self.gate,
                self.segments,
                self.rows,
                fused.CROSSBLOCK_TOKEN,
                "auto",
                "exact",
                self.blocks_replace,
            )
            with mock.patch.multiple(fused, **self.guard_values()), mock.patch.object(
                fused, "cast_bias_weight", return_value=(object(), None, "norm")
            ), mock.patch.object(
                fused, "uncast_bias_weight"
            ), mock.patch.object(
                fused, "_cast_norm_weight_eligible", return_value=True
            ), mock.patch.object(fused, "run_every_op"):
                with self.assertRaisesRegex(RuntimeError, "contract changed"):
                    self.call(pending)
        self.producer.assert_not_called()

    def test_preflight_failure_materializes_old_gate_once(self):
        with self.runtime() as pending, mock.patch.object(
            fused, "_preflight_eligible", return_value=False
        ):
            self.assertIsNone(self.call(pending))
            result, materialize = self.materialize(pending)
        self.assertIs(result, self.x)
        materialize.assert_called_once_with(
            self.x, self.gate, self.other, self.segments, self.rows
        )
        with self.assertRaisesRegex(RuntimeError, "already-consumed"):
            self.materialize(pending)

    def test_deferred_callback_invalidation_is_pre_mutation(self):
        guards = self.guard_values()
        guards.pop("_tensor_inputs_eligible")
        with self.runtime() as pending, mock.patch.multiple(
            fused, **guards
        ), mock.patch.object(
            fused, "_tensor_inputs_eligible", side_effect=(True, False)
        ), mock.patch.object(
            fused, "run_every_op"
        ), mock.patch.object(
            fused, "cast_bias_weight", side_effect=AssertionError("cast")
        ):
            self.assertIsNone(self.call(pending))
            self.materialize(pending)
        self.producer.assert_not_called()

    def test_producer_identity_change_is_pre_mutation(self):
        with self.runtime() as pending:
            changed = SimpleNamespace(
                gate_rms_adaln_nvfp4_dynamic=mock.Mock()
            )
            with mock.patch.object(fused, "_load_native", return_value=changed):
                self.assertIsNone(self.call(pending))
            self.materialize(pending)
        self.producer.assert_not_called()

    def test_success_claims_once_and_projects_qkv(self):
        order = []
        self.producer.side_effect = lambda *_args: (
            order.append("quantize")
            or (self.qdata, self.block_scale, self.input_scale, object())
        )
        weight = FakeQuantizedTensor(
            layout_cls=fused.NVFP4_LAYOUT,
            params=SimpleNamespace(transposed=False),
        )
        norm_weight = object()

        def cast(module, *_args, **_kwargs):
            if module is self.norm:
                order.append("cast-norm")
                return norm_weight, None, "norm-stream"
            order.append("cast-linear")
            return weight, None, "linear-stream"

        def uncast(module, *_args):
            order.append("uncast-norm" if module is self.norm else "uncast-linear")

        expected = object()
        with self.runtime() as pending, mock.patch.multiple(
            fused, **self.guard_values()
        ), mock.patch.object(
            fused, "cast_bias_weight", side_effect=cast
        ), mock.patch.object(
            fused, "uncast_bias_weight", side_effect=uncast
        ), mock.patch.object(
            fused, "_cast_norm_weight_eligible", return_value=True
        ), mock.patch.object(
            fused, "QuantizedTensor", FakeQuantizedTensor
        ), mock.patch.object(
            fused, "TensorCoreNVFP4Layout", FakeLayout
        ), mock.patch.object(
            fused.torch.nn.functional,
            "linear",
            side_effect=lambda *_args: order.append("linear") or expected,
        ), mock.patch.object(fused, "run_every_op"):
            self.assertIs(self.call(pending), expected)
        self.assertEqual(
            order,
            [
                "cast-norm",
                "quantize",
                "uncast-norm",
                "cast-linear",
                "linear",
                "uncast-linear",
            ],
        )
        self.assertFalse(fused.is_pending(pending))
        with self.assertRaisesRegex(RuntimeError, "already-consumed"):
            self.materialize(pending)

    def test_native_failure_after_claim_propagates_without_materialization(self):
        self.producer.side_effect = RuntimeError("launch failed")
        with self.runtime() as pending, mock.patch.multiple(
            fused, **self.guard_values()
        ), mock.patch.object(
            fused, "cast_bias_weight", return_value=(object(), None, "stream")
        ), mock.patch.object(
            fused, "uncast_bias_weight"
        ), mock.patch.object(
            fused, "_cast_norm_weight_eligible", return_value=True
        ), mock.patch.object(fused, "run_every_op"):
            with self.assertRaisesRegex(RuntimeError, "launch failed"):
                self.call(pending)
        with self.assertRaisesRegex(RuntimeError, "already-consumed"):
            self.materialize(pending)

    def test_invalid_native_outputs_raise_after_claim(self):
        guards = self.guard_values()
        guards.pop("_quantized_outputs_eligible")
        with self.runtime() as pending, mock.patch.multiple(
            fused, **guards
        ), mock.patch.object(
            fused, "cast_bias_weight", return_value=(object(), None, "stream")
        ), mock.patch.object(
            fused, "uncast_bias_weight"
        ), mock.patch.object(
            fused, "_cast_norm_weight_eligible", return_value=True
        ), mock.patch.object(fused, "run_every_op"):
            with self.assertRaisesRegex(RuntimeError, "invalid NVFP4 outputs"):
                self.call(pending)
        with self.assertRaisesRegex(RuntimeError, "already-consumed"):
            self.materialize(pending)

    def test_postclaim_replacement_change_raises_before_projection(self):
        self.producer.side_effect = lambda *_args: (
            self.blocks_replace.__setitem__(("double_block", 1), object())
            or (self.qdata, self.block_scale, self.input_scale, object())
        )
        with self.runtime() as pending, mock.patch.multiple(
            fused, **self.guard_values()
        ), mock.patch.object(
            fused, "cast_bias_weight", return_value=(object(), None, "stream")
        ), mock.patch.object(
            fused, "uncast_bias_weight"
        ), mock.patch.object(
            fused, "_cast_norm_weight_eligible", return_value=True
        ), mock.patch.object(fused, "run_every_op"):
            with self.assertRaisesRegex(RuntimeError, "contract changed"):
                self.call(pending)

    def test_carrier_input_change_at_last_preflight_raises_before_mutation(self):
        with self.runtime() as pending, mock.patch.multiple(
            fused, **self.guard_values()
        ), mock.patch.object(
            fused,
            "run_every_op",
            side_effect=lambda: setattr(pending, "_mlp", object()),
        ), mock.patch.object(
            fused, "cast_bias_weight", side_effect=AssertionError("cast")
        ):
            with self.assertRaisesRegex(RuntimeError, "inputs changed"):
                self.call(pending)
        self.producer.assert_not_called()

    def test_post_mutation_weight_change_raises(self):
        def cast(module, *_args, **_kwargs):
            return object(), None, "stream"

        with self.runtime() as pending, mock.patch.multiple(
            fused, **self.guard_values()
        ), mock.patch.object(
            fused, "cast_bias_weight", side_effect=cast
        ), mock.patch.object(
            fused, "uncast_bias_weight"
        ), mock.patch.object(
            fused, "_cast_norm_weight_eligible", return_value=True
        ), mock.patch.object(fused, "run_every_op"):
            with self.assertRaisesRegex(RuntimeError, "QKV owner/hook/weight changed"):
                self.call(pending)


if __name__ == "__main__":
    unittest.main()
