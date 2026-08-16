from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


LAB = Path(__file__).resolve().parent
COMFY_ROOT = Path(os.environ["COMFYUI_ROOT"])
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

adapter_spec = importlib.util.spec_from_file_location(
    "comfy.ldm.minimax.crossblock_gate_qkv",
    LAB
    / "candidate"
    / "comfy"
    / "ldm"
    / "minimax"
    / "crossblock_gate_qkv.py",
)
adapter = importlib.util.module_from_spec(adapter_spec)
assert adapter_spec.loader is not None
sys.modules[adapter_spec.name] = adapter
adapter_spec.loader.exec_module(adapter)

model_spec = importlib.util.spec_from_file_location(
    "crossblock_candidate_model",
    LAB / "candidate" / "comfy" / "ldm" / "minimax" / "model.py",
)
model = importlib.util.module_from_spec(model_spec)
assert model_spec.loader is not None
model_spec.loader.exec_module(model)
REAL_DIT_BLOCK = model.DiTBlock


class FakePending:
    def __init__(self, x):
        self.x = x


class FakeBlock:
    transitions = []

    def __init__(self, index):
        self.index = index
        self.mlp = object()
        self.attn = object()
        self.norm1 = object()
        self._forward_pre_hooks = {}
        self._forward_hooks = {}
        self._backward_pre_hooks = {}
        self._backward_hooks = {}
        self.calls = []

    def __call__(
        self,
        h,
        _t_emb,
        _segments,
        _rope,
        *,
        transformer_options,
        mod_row_ids,
        adaln_nvfp4_token=None,
        crossblock_pending=None,
        crossblock_previous=None,
        crossblock_next=None,
        crossblock_token=None,
        crossblock_defer=False,
        blocks_replace=None,
    ):
        self.calls.append(
            {
                "pending": crossblock_pending,
                "previous": crossblock_previous,
                "next": crossblock_next,
                "defer": crossblock_defer,
                "token": crossblock_token,
                "row_token": adaln_nvfp4_token,
                "blocks_replace": blocks_replace,
            }
        )
        if crossblock_pending is not None:
            FakeBlock.transitions.append((crossblock_previous.index, self.index))
        if crossblock_defer:
            return FakePending(h)
        return h


class FakeAttention:
    def __init__(self, order):
        self.order = order
        self.qkv_proj = object()

    def __call__(self, _x, **_kwargs):
        self.order.append("attention-stock")
        return "attention"

    def _forward_projected(self, qkv, **_kwargs):
        self.order.append(("attention-projected", qkv))
        return "attention"


class FakeMLP:
    def __init__(self, order):
        self.order = order
        self.fc1 = object()

    def __call__(self, _x, **_kwargs):
        self.order.append("mlp")
        return "mlp"


class ModelLoopTests(unittest.TestCase):
    def setUp(self):
        FakeBlock.transitions = []
        self.blocks = [FakeBlock(i) for i in range(50)]
        self.h = object()
        self.segments = object()
        self.rows = object()
        self.prefetch_pops = []

    def test_deployable_adapter_matches_frozen_lab_source(self):
        deployed = Path(adapter_spec.origin)
        self.assertEqual(deployed.read_bytes(), (LAB / "comfy_adapter.py").read_bytes())

    def run_loop(self, options):
        with mock.patch.object(model, "DiTBlock", FakeBlock), mock.patch.object(
            model, "is_crossblock_pending", side_effect=lambda value: isinstance(value, FakePending)
        ), mock.patch.object(
            model,
            "crossblock_pending_residual",
            side_effect=lambda pending, *_args: pending.x,
        ), mock.patch.object(
            model.comfy.model_prefetch,
            "make_prefetch_queue",
            return_value="queue",
        ), mock.patch.object(
            model.comfy.model_prefetch,
            "prefetch_queue_pop",
            side_effect=lambda _queue, _device, block: self.prefetch_pops.append(block),
        ), mock.patch.object(adapter.torch, "is_grad_enabled", return_value=False), mock.patch.object(
            adapter.comfy.model_management, "in_training", False
        ):
            return model._run_dit_blocks(
                self.blocks,
                self.h,
                object(),
                self.segments,
                object(),
                options,
                self.rows,
                torch.device("cuda:0"),
            )

    def test_first_last_and_exactly_49_transitions_preserve_prefetch(self):
        result = self.run_loop(
            {
                "minimax_h3_crossblock_gate_qkv_fusion": "auto",
                "minimax_h3_modulation_fusion": "exact",
            }
        )
        self.assertIs(result, self.h)
        self.assertEqual(FakeBlock.transitions, [(i, i + 1) for i in range(49)])
        self.assertIsNone(self.blocks[0].calls[0]["pending"])
        self.assertFalse(self.blocks[-1].calls[0]["defer"])
        self.assertIs(self.blocks[-1].calls[0]["previous"], self.blocks[-2])
        self.assertEqual(self.prefetch_pops, self.blocks + [None])

    def test_stock_mode_has_no_carriers(self):
        result = self.run_loop(
            {
                "minimax_h3_crossblock_gate_qkv_fusion": "stock",
                "minimax_h3_modulation_fusion": "exact",
            }
        )
        self.assertIs(result, self.h)
        self.assertEqual(FakeBlock.transitions, [])
        self.assertTrue(all(not block.calls[0]["defer"] for block in self.blocks))

    def test_local_block_hook_breaks_both_adjacent_transitions(self):
        self.blocks[12]._forward_hooks[1] = object()
        self.run_loop(
            {
                "minimax_h3_crossblock_gate_qkv_fusion": "auto",
                "minimax_h3_modulation_fusion": "exact",
            }
        )
        expected = [(i, i + 1) for i in range(49) if i not in (11, 12)]
        self.assertEqual(FakeBlock.transitions, expected)

    def test_any_block_replacement_disables_crossblock_and_preserves_wrapper(self):
        replacement_calls = []

        def replacement(args, wrappers):
            replacement_calls.append(args["img"])
            return wrappers["original_block"](args)

        result = self.run_loop(
            {
                "minimax_h3_crossblock_gate_qkv_fusion": "auto",
                "minimax_h3_modulation_fusion": "exact",
                "patches_replace": {"dit": {("double_block", 7): replacement}},
            }
        )
        self.assertIs(result, self.h)
        self.assertEqual(FakeBlock.transitions, [])
        self.assertEqual(replacement_calls, [self.h])
        self.assertEqual(self.prefetch_pops, self.blocks + [None])

    def test_missing_dit_get_allocates_a_fresh_empty_dict(self):
        options = {}
        first = options.get("patches_replace", {}).get("dit", {})
        second = options.get("patches_replace", {}).get("dit", {})
        self.assertIsNot(first, second)

    def test_loop_passes_one_owned_mapping_when_no_replacement_is_configured(self):
        options = {
            "minimax_h3_crossblock_gate_qkv_fusion": "auto",
            "minimax_h3_modulation_fusion": "exact",
        }
        self.run_loop(options)
        owned = self.blocks[0].calls[0]["blocks_replace"]
        self.assertIsInstance(owned, dict)
        self.assertEqual(owned, {})
        self.assertTrue(
            all(block.calls[0]["blocks_replace"] is owned for block in self.blocks)
        )

    def test_configured_dit_mapping_keeps_its_original_identity(self):
        mapping = {}
        options = {
            "minimax_h3_crossblock_gate_qkv_fusion": "auto",
            "minimax_h3_modulation_fusion": "exact",
            "patches_replace": {"dit": mapping},
        }
        self.run_loop(options)
        self.assertTrue(
            all(block.calls[0]["blocks_replace"] is mapping for block in self.blocks)
        )


class DiTBlockTransactionTests(unittest.TestCase):
    def make_block(self, order):
        block = SimpleNamespace()

        def adaln(_t_emb):
            order.append("adaln")
            return (
                "shift-msa",
                "scale-msa",
                "gate-msa",
                "shift-mlp",
                "scale-mlp",
                "gate-mlp",
            )

        block.adaln_proj = adaln
        block.norm1 = object()
        block.norm2 = object()
        block.attn = FakeAttention(order)
        block.mlp = FakeMLP(order)
        return block

    def test_materialized_carrier_is_byte_exact_old_mod_gate(self):
        previous = SimpleNamespace()
        current = SimpleNamespace()
        segments = [(0, 2, 0)]
        x_reference = torch.randn(2, 8, dtype=torch.bfloat16)
        x_candidate = x_reference.clone()
        mlp = torch.randn_like(x_reference)
        gate = torch.randn(1, 8, dtype=torch.float32)
        reference = model._mod_gate(
            x_reference, gate, mlp, segments, row_ids=None
        )
        pending = adapter.H3CrossBlockPending(
            adapter._CARRIER_SEAL,
            previous,
            current,
            x_candidate,
            mlp,
            gate,
            segments,
            None,
            {},
            adapter.CROSSBLOCK_TOKEN,
            object(),
            object(),
            "dynamic",
            None,
        )
        candidate = adapter.materialize_pending(
            pending,
            previous,
            current,
            x_candidate,
            segments,
            None,
            adapter.CROSSBLOCK_TOKEN,
            model._mod_gate,
        )
        self.assertTrue(torch.equal(candidate, reference))
        self.assertEqual(candidate.untyped_storage().data_ptr(), x_candidate.untyped_storage().data_ptr())

    def forward(self, block, pending, **kwargs):
        return REAL_DIT_BLOCK.forward(
            block,
            "x",
            object(),
            "segments",
            object(),
            transformer_options={
                "minimax_h3_crossblock_gate_qkv_fusion": "auto",
                "minimax_h3_rms_adaln_nvfp4_fusion": "stock",
                "minimax_h3_modulation_fusion": "exact",
            },
            mod_row_ids="rows",
            adaln_nvfp4_token=model._ADALN_NVFP4_TOKEN,
            crossblock_pending=pending,
            crossblock_previous="previous",
            crossblock_next=kwargs.get("next_block"),
            crossblock_token=model._CROSSBLOCK_TOKEN,
            crossblock_defer=kwargs.get("defer", False),
            blocks_replace=kwargs.get("blocks_replace", {}),
        )

    @staticmethod
    def fallback_patches(order):
        def norm_mod(*_args, **_kwargs):
            order.append("norm")
            return "normed"

        def gate(x, gate_value, _other, _segments, _rows):
            order.append(("gate", gate_value))
            return x

        return (
            mock.patch.object(model, "adaln_parent_eligible", return_value=False),
            mock.patch.object(model, "_norm_mod", side_effect=norm_mod),
            mock.patch.object(model, "_mod_gate", side_effect=gate),
        )

    def test_pre_mutation_none_materializes_before_unchanged_attention(self):
        order = []
        block = self.make_block(order)
        pending = object()

        def materialize(*args):
            order.append("materialize")
            callback = args[-1]
            return callback(args[3], "previous-gate", "previous-mlp", args[4], args[5])

        patches = self.fallback_patches(order)
        with patches[0], patches[1], patches[2], mock.patch.object(
            model,
            "try_crossblock_qkv_transition",
            side_effect=lambda *_args: order.append("try-crossblock") or None,
        ), mock.patch.object(
            model, "materialize_crossblock_pending", side_effect=materialize
        ) as materialize_mock:
            self.assertEqual(self.forward(block, pending), "x")
            self.assertIs(materialize_mock.call_args.args[-1], model._mod_gate)
        self.assertEqual(
            order,
            [
                "adaln",
                "try-crossblock",
                "materialize",
                ("gate", "previous-gate"),
                "norm",
                "attention-stock",
                ("gate", "gate-msa"),
                "norm",
                "mlp",
                ("gate", "gate-mlp"),
            ],
        )

    def test_success_enters_private_projected_attention(self):
        order = []
        block = self.make_block(order)
        patches = self.fallback_patches(order)
        with patches[0], patches[1], patches[2], mock.patch.object(
            model, "try_crossblock_qkv_transition", return_value="qkv"
        ), mock.patch.object(
            model,
            "materialize_crossblock_pending",
            side_effect=AssertionError("materialize"),
        ):
            self.assertEqual(self.forward(block, object()), "x")
        self.assertIn(("attention-projected", "qkv"), order)
        self.assertNotIn("attention-stock", order)

    def test_postmutation_failure_propagates_without_old_gate(self):
        order = []
        block = self.make_block(order)
        patches = self.fallback_patches(order)
        with patches[0], patches[1], patches[2], mock.patch.object(
            model,
            "try_crossblock_qkv_transition",
            side_effect=RuntimeError("post-mutation"),
        ), mock.patch.object(
            model, "materialize_crossblock_pending"
        ) as materialize_mock:
            with self.assertRaisesRegex(RuntimeError, "post-mutation"):
                self.forward(block, object())
        materialize_mock.assert_not_called()
        self.assertEqual(order, ["adaln"])

    def test_deferred_block_skips_only_its_final_gate(self):
        order = []
        block = self.make_block(order)
        next_block = object()
        deferred = object()
        patches = self.fallback_patches(order)
        with patches[0], patches[1], patches[2], mock.patch.object(
            model, "defer_crossblock_qkv_transition", return_value=deferred
        ) as defer_mock:
            self.assertIs(
                self.forward(block, None, next_block=next_block, defer=True), deferred
            )
        defer_mock.assert_called_once()
        self.assertIn(("gate", "gate-msa"), order)
        self.assertNotIn(("gate", "gate-mlp"), order)


if __name__ == "__main__":
    unittest.main()
