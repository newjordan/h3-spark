"""CPU-only contract and layout tests for the H3 AdaLN/NVFP4 prototype."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

try:
    from .adaln_nvfp4 import (
        WIDTH,
        _check_entrypoints,
        adaln_nvfp4_dynamic,
        eager_adaln_reference,
    )
except ImportError:
    from adaln_nvfp4 import (
        WIDTH,
        _check_entrypoints,
        adaln_nvfp4_dynamic,
        eager_adaln_reference,
    )


ROWS = 20423
LAYERS = 50
STEPS = 20
FUSED_SITES_PER_BLOCK = 2  # qkv_proj and fc1
BYTES_PER_BF16_ACTIVATION = ROWS * WIDTH * 2


def roundup(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def scale_offset(row: int, col: int, col_length: int = WIDTH // 16) -> int:
    row_block = row // 128
    rem = row % 128
    d4 = rem // 32
    d3 = rem % 32
    col_block_group = col // 4
    d5 = col % 4
    col_block_group_count = (col_length + 3) // 4
    return (
        ((row_block * col_block_group_count + col_block_group) * 32 + d3) * 16
        + d4 * 4
        + d5
    )


class ContractTests(unittest.TestCase):
    def test_eager_reference_preserves_all_bf16_boundaries(self) -> None:
        normed = torch.full((1, WIDTH), 1.234375, dtype=torch.bfloat16)
        scale = torch.full((1, WIDTH), -0.9, dtype=torch.float32)
        shift = torch.full((1, WIDTH), 0.00391, dtype=torch.float32)
        row_ids = torch.zeros(1, dtype=torch.int32)

        actual = eager_adaln_reference(normed, scale, shift, row_ids)
        scale_bf16 = scale.to(torch.bfloat16)
        shift_bf16 = shift.to(torch.bfloat16)
        expected = normed.clone()
        expected.mul_(1.0 + scale_bf16).add_(shift_bf16)
        collapsed = (
            normed.to(torch.float32) * (1.0 + scale) + shift
        ).to(torch.bfloat16)

        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(torch.equal(actual, collapsed))

    def test_reference_accepts_chunk_stride_and_arbitrary_row_map(self) -> None:
        torch.manual_seed(260812)
        rows = 5
        table = torch.randn(3, WIDTH * 6, dtype=torch.float32)
        scale = table[:, WIDTH:2 * WIDTH]
        shift = table[:, 4 * WIDTH:5 * WIDTH]
        row_ids = torch.tensor([2, 2, 0, 1, 0], dtype=torch.int32)
        normed = torch.randn(rows, WIDTH, dtype=torch.bfloat16)

        self.assertEqual(scale.stride(), (WIDTH * 6, 1))
        fused_reference = eager_adaln_reference(normed, scale, shift, row_ids)
        segmented = normed.clone()
        for row, mod_row in enumerate(row_ids.tolist()):
            segmented[row].mul_(
                1.0 + scale[mod_row].to(torch.bfloat16)
            ).add_(shift[mod_row].to(torch.bfloat16))
        self.assertTrue(torch.equal(fused_reference, segmented))

    def test_dynamic_scale_rounding_order(self) -> None:
        bf16_amax = torch.tensor(12.625, dtype=torch.bfloat16)
        exact = (bf16_amax / 2688.0).to(torch.float32)
        widened_first = bf16_amax.to(torch.float32) / 2688.0
        self.assertEqual(exact.item(), 0.00469970703125)
        self.assertNotEqual(exact.item(), widened_first.item())

    def test_launch_geometry_and_output_layout(self) -> None:
        self.assertEqual(WIDTH % 16, 0)
        self.assertEqual((WIDTH // 8) % 32, 0)
        self.assertEqual((WIDTH // 16) % 4, 0)
        self.assertEqual(roundup(ROWS, 16), 20432)
        self.assertEqual(roundup(roundup(ROWS, 16), 128), 20480)
        self.assertEqual((20432, WIDTH // 2), (20432, 2688))
        self.assertEqual((20480, WIDTH // 16), (20480, 336))

    def test_scale_swizzle_is_bijective(self) -> None:
        scale_cols = WIDTH // 16
        offsets = {
            scale_offset(row, col)
            for row in range(256)
            for col in range(scale_cols)
        }
        self.assertEqual(len(offsets), 256 * scale_cols)
        self.assertEqual(min(offsets), 0)
        self.assertEqual(max(offsets), 256 * scale_cols - 1)

    def test_traffic_savings_at_target_shape(self) -> None:
        # After stock RMSNorm, exact dynamic currently performs AdaLN read/write,
        # then reads its result once for amax and once for quantization.
        baseline_bytes = 4 * BYTES_PER_BF16_ACTIVATION
        fused_bytes = 2 * BYTES_PER_BF16_ACTIVATION
        saved_per_site = baseline_bytes - fused_bytes
        sites_per_render = FUSED_SITES_PER_BLOCK * LAYERS * STEPS
        self.assertEqual(BYTES_PER_BF16_ACTIVATION, 219_588_096)
        self.assertEqual(saved_per_site, 439_176_192)
        self.assertEqual(sites_per_render, 2000)
        self.assertEqual(saved_per_site * sites_per_render, 878_352_384_000)

    def test_cpu_input_fails_before_extension_load(self) -> None:
        normed = torch.zeros((1, WIDTH), dtype=torch.bfloat16)
        scale = torch.zeros((1, WIDTH), dtype=torch.bfloat16)
        shift = torch.zeros_like(scale)
        row_ids = torch.zeros(1, dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "CUDA tensor"):
            adaln_nvfp4_dynamic(normed, scale, shift, row_ids)

    def test_loader_requires_both_entrypoints(self) -> None:
        complete = SimpleNamespace(
            adaln_nvfp4=lambda: None,
            adaln_nvfp4_dynamic=lambda: None,
        )
        _check_entrypoints(complete, Path("complete.so"))
        del complete.adaln_nvfp4_dynamic
        with self.assertRaisesRegex(ImportError, "adaln_nvfp4_dynamic"):
            _check_entrypoints(complete, Path("stale.so"))


if __name__ == "__main__":
    unittest.main()
