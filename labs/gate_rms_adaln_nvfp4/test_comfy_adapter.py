from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

import comfy_adapter as fused


class FakeQuantizedTensor:
    def __init__(self, data=None, layout_cls=None, params=None):
        self.data = data
        self._layout_cls = layout_cls
        self._params = params or SimpleNamespace(transposed=False)


class FakeLayout:
    class Params:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.owner = SimpleNamespace()
        self.linear = SimpleNamespace(in_features=5376, out_features=28672)
        self.owner.fc1 = self.linear
        self.norm = SimpleNamespace(eps=1e-5)
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

    def call(self, *, mode="auto", token=fused.ROW_IDS_TOKEN):
        return fused.try_linear(
            self.owner,
            self.linear,
            self.norm,
            self.x,
            self.other,
            self.gate,
            self.scale,
            self.shift,
            self.rows,
            token,
            mode,
            "exact",
        )

    def eligible(self):
        return mock.patch.multiple(
            fused,
            _tensor_inputs_eligible=mock.DEFAULT,
            _gate_inputs_eligible=mock.DEFAULT,
            _norm_eligible=mock.DEFAULT,
            _linear_eligible=mock.DEFAULT,
            _dynamic_eligible=mock.DEFAULT,
            parent_eligible=mock.DEFAULT,
            _spark_device_eligible=mock.DEFAULT,
            _storage_disjoint=mock.DEFAULT,
        )

    def test_stock_untrusted_and_wrong_owner_fail_before_any_guard(self):
        with mock.patch.object(
            fused, "_tensor_inputs_eligible", side_effect=AssertionError("guard")
        ):
            self.assertIsNone(self.call(mode="stock"))
            self.assertIsNone(self.call(token=object()))
            other_owner = self.owner
            self.owner = SimpleNamespace(fc1=object())
            self.assertIsNone(self.call())
            self.owner = other_owner

    def test_parent_hook_guard_is_owned_by_adapter(self):
        with mock.patch.object(fused, "parent_eligible", return_value=False), mock.patch.object(
            fused, "_tensor_inputs_eligible", side_effect=AssertionError("later guard")
        ):
            self.assertIsNone(self.call())

    def test_native_abi_contract(self):
        good = SimpleNamespace(
            abi_version=lambda: fused.NATIVE_ABI_VERSION,
            build_id=lambda: fused.NATIVE_BUILD_ID,
            gate_rms_adaln_nvfp4_dynamic=lambda *_args: None,
        )
        self.assertTrue(fused._native_abi_eligible(good))
        self.assertFalse(
            fused._native_abi_eligible(
                SimpleNamespace(**{**good.__dict__, "build_id": lambda: "stale"})
            )
        )
        self.assertFalse(
            fused._native_abi_eligible(
                SimpleNamespace(
                    **{
                        **good.__dict__,
                        "abi_version": lambda: fused.NATIVE_ABI_VERSION + 1,
                    }
                )
            )
        )

    def test_gate_guard_is_fp32_and_rejects_all_writable_aliases(self):
        x = torch.empty(2, 5376, dtype=torch.bfloat16)
        other = torch.empty_like(x)
        scale = torch.empty(1, 5376, dtype=torch.float32)
        gate = torch.empty_like(scale)
        shift = torch.empty_like(scale)
        rows = torch.empty(2, dtype=torch.int32)
        args = (other, gate, scale, shift, rows)
        self.assertTrue(fused._gate_inputs_eligible(x, *args))
        for index, tensor in enumerate(args):
            aliased = list(args)
            if index == 0:
                aliased[index] = x
            elif index < 4:
                aliased[index] = x.view(torch.float32).reshape(1, 5376)
            else:
                aliased[index] = x.view(torch.int32).reshape(-1)[:2]
            self.assertFalse(
                fused._gate_inputs_eligible(x, *aliased),
                msg=f"writable x alias at argument {index} was accepted",
            )
        self.assertFalse(
            fused._gate_inputs_eligible(
                x,
                other,
                gate.to(torch.bfloat16),
                scale.to(torch.bfloat16),
                shift.to(torch.bfloat16),
                rows,
            )
        )

    def test_pre_mutation_norm_cast_mismatch_returns_none_and_uncasts(self):
        order = []
        native = SimpleNamespace(gate_rms_adaln_nvfp4_dynamic=lambda *_args: None)
        guards = {
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
            )
        }
        with mock.patch.multiple(fused, **guards), mock.patch.object(
            fused, "_load_native", return_value=native
        ), mock.patch.object(
            fused,
            "cast_bias_weight",
            side_effect=lambda *_a, **_k: ("bad", None, "norm-stream"),
        ), mock.patch.object(
            fused, "_cast_norm_weight_eligible", return_value=False
        ), mock.patch.object(
            fused,
            "uncast_bias_weight",
            side_effect=lambda *_a: order.append("uncast-norm"),
        ), mock.patch.object(fused, "run_every_op"):
            self.assertIsNone(self.call())
        self.assertEqual(order, ["uncast-norm"])

    def test_cast_norm_alias_returns_none_before_native_mutation(self):
        order = []
        native = SimpleNamespace(
            gate_rms_adaln_nvfp4_dynamic=lambda *_args: (_ for _ in ()).throw(
                AssertionError("native mutation entered")
            )
        )
        guards = {
            name: (lambda *_args: True)
            for name in (
                "_tensor_inputs_eligible",
                "_gate_inputs_eligible",
                "_norm_eligible",
                "_linear_eligible",
                "_dynamic_eligible",
                "parent_eligible",
                "_spark_device_eligible",
            )
        }
        with mock.patch.multiple(fused, **guards), mock.patch.object(
            fused, "_load_native", return_value=native
        ), mock.patch.object(
            fused,
            "cast_bias_weight",
            return_value=(object(), None, "norm-stream"),
        ), mock.patch.object(
            fused, "_cast_norm_weight_eligible", return_value=True
        ), mock.patch.object(
            fused, "_storage_disjoint", return_value=False
        ), mock.patch.object(
            fused,
            "uncast_bias_weight",
            side_effect=lambda *_a: order.append("uncast-norm"),
        ), mock.patch.object(fused, "run_every_op"):
            self.assertIsNone(self.call())
        self.assertEqual(order, ["uncast-norm"])

    def test_deferred_callback_invalidation_returns_before_cast_or_mutation(self):
        native_call = mock.Mock()
        native = SimpleNamespace(gate_rms_adaln_nvfp4_dynamic=native_call)
        tensor_guard = mock.Mock(side_effect=(True, False))
        guards = {
            name: (lambda *_args: True)
            for name in (
                "_gate_inputs_eligible",
                "_norm_eligible",
                "_linear_eligible",
                "_dynamic_eligible",
                "parent_eligible",
                "_spark_device_eligible",
            )
        }
        with mock.patch.multiple(fused, **guards), mock.patch.object(
            fused, "_tensor_inputs_eligible", tensor_guard
        ), mock.patch.object(
            fused, "_load_native", return_value=native
        ), mock.patch.object(fused, "run_every_op"), mock.patch.object(
            fused,
            "cast_bias_weight",
            side_effect=AssertionError("cast entered after invalidation"),
        ):
            self.assertIsNone(self.call())
        native_call.assert_not_called()

    def test_native_failure_propagates_and_never_casts_linear(self):
        order = []
        norm_weight = object()
        native = SimpleNamespace(
            gate_rms_adaln_nvfp4_dynamic=lambda *_args: (_ for _ in ()).throw(
                RuntimeError("native failure")
            )
        )
        guards = {
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
            )
        }
        with mock.patch.multiple(fused, **guards), mock.patch.object(
            fused, "_load_native", return_value=native
        ), mock.patch.object(
            fused,
            "cast_bias_weight",
            side_effect=lambda module, *_a, **_k: (
                order.append("cast-norm") or (norm_weight, None, "norm-stream")
                if module is self.norm
                else (_ for _ in ()).throw(AssertionError("linear cast after failure"))
            ),
        ), mock.patch.object(
            fused, "_cast_norm_weight_eligible", return_value=True
        ), mock.patch.object(
            fused,
            "uncast_bias_weight",
            side_effect=lambda *_a: order.append("uncast-norm"),
        ), mock.patch.object(fused, "run_every_op"):
            with self.assertRaisesRegex(RuntimeError, "native failure"):
                self.call()
        self.assertEqual(order, ["cast-norm", "uncast-norm"])

    def test_post_mutation_weight_change_raises_and_restores_casts(self):
        order = []
        native = SimpleNamespace(
            gate_rms_adaln_nvfp4_dynamic=lambda *_args: (
                object(), object(), object(), object()
            )
        )
        guards = {
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
            )
        }

        def cast(module, *_args, **_kwargs):
            if module is self.norm:
                order.append("cast-norm")
                return object(), None, "norm-stream"
            order.append("cast-linear")
            return object(), None, "linear-stream"

        def uncast(module, *_args):
            order.append("uncast-norm" if module is self.norm else "uncast-linear")

        with mock.patch.multiple(fused, **guards), mock.patch.object(
            fused, "_load_native", return_value=native
        ), mock.patch.object(fused, "cast_bias_weight", side_effect=cast), mock.patch.object(
            fused, "uncast_bias_weight", side_effect=uncast
        ), mock.patch.object(
            fused, "_cast_norm_weight_eligible", return_value=True
        ), mock.patch.object(fused, "QuantizedTensor", FakeQuantizedTensor), mock.patch.object(
            fused, "run_every_op"
        ):
            with self.assertRaisesRegex(RuntimeError, "weight changed"):
                self.call()
        self.assertEqual(
            order, ["cast-norm", "uncast-norm", "cast-linear", "uncast-linear"]
        )

    def test_success_has_single_cast_lifecycle_and_linear_call(self):
        order = []
        norm_weight = object()
        qdata = torch.empty(2, 2688, dtype=torch.uint8)
        block_scale = torch.empty(128, 336, dtype=torch.uint8)
        input_scale = torch.tensor(0.25, dtype=torch.float32)
        native = SimpleNamespace(
            gate_rms_adaln_nvfp4_dynamic=lambda *_args: (
                order.append("quantize")
                or (qdata, block_scale, input_scale, object())
            )
        )
        weight = FakeQuantizedTensor(
            layout_cls=fused.NVFP4_LAYOUT,
            params=SimpleNamespace(transposed=False),
        )
        guards = {
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
            )
        }

        def cast(module, *_args, **_kwargs):
            if module is self.norm:
                order.append("cast-norm")
                return norm_weight, None, "norm-stream"
            order.append("cast-linear")
            return weight, None, "linear-stream"

        def uncast(module, *_args):
            order.append("uncast-norm" if module is self.norm else "uncast-linear")

        expected = object()
        with mock.patch.multiple(fused, **guards), mock.patch.object(
            fused, "_load_native", return_value=native
        ), mock.patch.object(fused, "cast_bias_weight", side_effect=cast), mock.patch.object(
            fused, "uncast_bias_weight", side_effect=uncast
        ), mock.patch.object(
            fused, "_cast_norm_weight_eligible", return_value=True
        ), mock.patch.object(fused, "QuantizedTensor", FakeQuantizedTensor), mock.patch.object(
            fused, "TensorCoreNVFP4Layout", FakeLayout
        ), mock.patch.object(
            fused.torch.nn.functional,
            "linear",
            side_effect=lambda qinput, got_weight, bias: (
                order.append("linear") or expected
            ),
        ), mock.patch.object(fused, "run_every_op"):
            self.assertIs(self.call(), expected)

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

    def test_transition_none_does_not_enter_private_mlp(self):
        owner = SimpleNamespace(
            fc1=object(),
            _forward_raw=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("private MLP continuation entered")
            ),
        )
        with mock.patch.object(fused, "try_linear", return_value=None) as producer:
            self.assertIsNone(
                fused.try_mlp_transition(
                    owner,
                    object(),
                    object(),
                    object(),
                    object(),
                    object(),
                    object(),
                    object(),
                    fused.ROW_IDS_TOKEN,
                    "auto",
                    "exact",
                    {"sentinel": True},
                )
            )
        self.assertEqual(producer.call_count, 1)

    def test_transition_success_enters_private_mlp_once(self):
        calls = []
        expected = object()
        owner = SimpleNamespace(fc1=object())

        def forward_raw(raw, transformer_options):
            calls.append((raw, transformer_options))
            return expected

        owner._forward_raw = forward_raw
        raw = object()
        options = {"sentinel": True}
        with mock.patch.object(fused, "try_linear", return_value=raw):
            actual = fused.try_mlp_transition(
                owner,
                object(),
                object(),
                object(),
                object(),
                object(),
                object(),
                object(),
                fused.ROW_IDS_TOKEN,
                "auto",
                "exact",
                options,
            )
        self.assertIs(actual, expected)
        self.assertEqual(calls, [(raw, options)])

    def test_native_source_carries_mutation_and_stream_contract(self):
        source = Path(__file__).with_name(
            "gate_rms_adaln_nvfp4_cuda.cu"
        ).read_text(encoding="utf-8")
        for marker in (
            "at::assert_no_overlap(x, other)",
            "at::assert_no_overlap(x, gate)",
            "at::assert_no_overlap(x, scale)",
            "at::assert_no_overlap(x, weight)",
            "at::assert_no_overlap(x, shift)",
            "at::assert_no_overlap(x, row_ids)",
            "torch::autograd::increment_version(x)",
            "at::cuda::getCurrentCUDAStream(x.get_device())",
            "eps == kCertifiedEps",
            fused.NATIVE_BUILD_ID,
            "return 2",
        ):
            self.assertIn(marker, source)


if __name__ == "__main__":
    unittest.main()
