"""CPU-only structural checks for the fixed H3 block-scale layout."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from swiglu_nvfp4 import (
    H3_NVFP4_WIDTHS,
    _check_entrypoints,
    bf16_nvfp4_dynamic,
    bf16_nvfp4_from_amax_bits,
)


WIDTH = 14336
SCALE_COLS = WIDTH // 16
DIRECT_VALUES_PER_THREAD = 8


def roundup(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def scale_offset(row: int, col: int, col_length: int = SCALE_COLS) -> int:
    rb = row // 128
    rem = row % 128
    d4 = rem // 32
    d3 = rem % 32
    cbg = col // 4
    d5 = col % 4
    cbg_count = (col_length + 3) // 4
    return ((rb * cbg_count + cbg) * 32 + d3) * 16 + d4 * 4 + d5


class ScaleLayoutTests(unittest.TestCase):
    def test_h3_fixed_dimensions(self) -> None:
        self.assertEqual(WIDTH // 2, 7168)
        self.assertEqual(SCALE_COLS, 896)
        self.assertEqual(SCALE_COLS % 4, 0)

    def test_swizzle_is_bijection_over_full_tiles(self) -> None:
        rows = 256
        offsets = {
            scale_offset(row, col)
            for row in range(rows)
            for col in range(SCALE_COLS)
        }
        self.assertEqual(len(offsets), rows * SCALE_COLS)
        self.assertEqual(min(offsets), 0)
        self.assertEqual(max(offsets), rows * SCALE_COLS - 1)

    def test_dynamic_scale_rounding_order(self) -> None:
        bf16_amax = torch.tensor(3920.0, dtype=torch.bfloat16)
        exact_order = (bf16_amax / 2688.0).to(torch.float32)
        widened_first = bf16_amax.to(torch.float32) / 2688.0
        self.assertEqual(exact_order.item(), 1.4609375)
        self.assertNotEqual(exact_order.item(), widened_first.item())

    def test_padding_shapes(self) -> None:
        for rows, q_rows, scale_rows in (
            (1, 16, 128),
            (16, 16, 128),
            (17, 32, 128),
            (127, 128, 128),
            (129, 144, 256),
            (20423, 20432, 20480),
        ):
            self.assertEqual(roundup(rows, 16), q_rows)
            self.assertEqual(roundup(q_rows, 128), scale_rows)

    def test_all_h3_widths_preserve_lane_group_boundaries(self) -> None:
        # The direct kernel assigns eight values to a lane and two adjacent
        # lanes to one 16-value scale. Every H3 row must therefore end on both
        # a complete lane group and a warp boundary, even when a CUDA block
        # straddles two rows (the 5376-wide case).
        self.assertEqual(H3_NVFP4_WIDTHS, frozenset((5376, 7168, 14336)))
        for width in H3_NVFP4_WIDTHS:
            threads_per_row = width // DIRECT_VALUES_PER_THREAD
            self.assertEqual(width % 16, 0)
            self.assertEqual(threads_per_row % 2, 0)
            self.assertEqual(threads_per_row % 32, 0)
            self.assertEqual((width // 16) % 4, 0)

    def test_direct_output_shapes(self) -> None:
        expected = {
            5376: ((144, 2688), (256, 336)),
            7168: ((144, 3584), (256, 448)),
            14336: ((144, 7168), (256, 896)),
        }
        rows = 129
        for width, (q_shape, scale_shape) in expected.items():
            self.assertEqual((roundup(rows, 16), width // 2), q_shape)
            self.assertEqual(
                (roundup(roundup(rows, 16), 128), width // 16),
                scale_shape,
            )
        self.assertEqual(roundup(20423, 16), 20432)
        self.assertEqual(roundup(roundup(20423, 16), 128), 20480)

    def test_python_guard_rejects_non_cuda_before_loading_extension(self) -> None:
        x = torch.zeros((2, 5376), dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "CUDA tensor"):
            bf16_nvfp4_dynamic(x)

    def test_python_guard_rejects_non_h3_width(self) -> None:
        x = torch.zeros((2, 4096), dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "H3 width"):
            bf16_nvfp4_dynamic(x)

    def test_from_amax_guard_rejects_non_cuda_before_loading_extension(self) -> None:
        x = torch.zeros((2, 5376), dtype=torch.bfloat16)
        max_bits = torch.zeros(1, dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "CUDA tensor"):
            bf16_nvfp4_from_amax_bits(x, max_bits)

    def test_loader_requires_the_full_additive_abi(self) -> None:
        complete = SimpleNamespace(
            bf16_nvfp4=lambda: None,
            bf16_nvfp4_dynamic=lambda: None,
            bf16_nvfp4_from_amax_bits=lambda: None,
            swiglu_nvfp4=lambda: None,
            swiglu_nvfp4_dynamic=lambda: None,
        )
        _check_entrypoints(complete, Path("complete.so"))
        del complete.bf16_nvfp4_from_amax_bits
        with self.assertRaisesRegex(ImportError, "bf16_nvfp4_from_amax_bits"):
            _check_entrypoints(complete, Path("stale.so"))


if __name__ == "__main__":
    unittest.main()
