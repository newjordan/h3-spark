from __future__ import annotations

from pathlib import Path

import pytest

import contract


HERE = Path(__file__).resolve().parent
SOURCE = (HERE / "rms_adaln_nvfp4_static_cuda.cu").read_text()


def test_register_schedule_covers_every_feature_once():
    columns = [
        col
        for tid in range(contract.THREADS)
        for col in contract.owned_columns(tid)
    ]
    assert len(columns) == contract.WIDTH
    assert sorted(columns) == list(range(contract.WIDTH))
    assert contract.VECTORS_PER_THREAD == 11
    assert {len(contract.owned_vectors(tid)) for tid in range(128)} == {10, 11}


def test_every_nvfp4_scale_block_has_one_aligned_leader():
    writes = contract.scale_leaders()
    cols = [scale_col for _iteration, _tid, scale_col in writes]
    assert len(cols) == contract.SCALE_COLS
    assert sorted(cols) == list(range(contract.SCALE_COLS))
    assert all(tid % 4 == 0 for _iteration, tid, _scale_col in writes)


@pytest.mark.parametrize("rows", [1, 15, 16, 17, 127, 128, 129, 20423])
def test_swizzle_is_injective_and_inside_allocation(rows):
    allocated = contract.scale_rows(rows) * contract.SCALE_COLS
    offsets = {
        contract.swizzled_scale_offset(row, col)
        for row in range(contract.padded_rows(rows))
        for col in range(contract.SCALE_COLS)
    }
    assert len(offsets) == contract.padded_rows(rows) * contract.SCALE_COLS
    assert min(offsets) == 0
    assert max(offsets) < allocated


def test_exact_chef_shape_traffic_case_is_material():
    traffic = contract.traffic_bytes(20423)
    assert traffic == {
        "dynamic_v2": 713_767_196,
        "static_register": 274_509_312,
        "saved": 439_257_884,
    }
    # Two full 20,423 x 5,376 BF16 reads disappear on each QKV/FC1 path.
    assert traffic["saved"] > 100 * 1024 * 1024


def test_source_has_production_safety_and_exactness_markers():
    required = (
        "CUDAGuard",
        "getCurrentCUDAStream",
        "C10_CUDA_KERNEL_LAUNCH_CHECK",
        "__float2bfloat16_rn",
        "__NV_E2M1",
        "cudaRoundNearest",
        "selected_mod_row >= 0 && selected_mod_row < modulation_rows",
        "warp_scratch[0] + warp_scratch[2]",
        "warp_scratch[1] + warp_scratch[3]",
        "if constexpr (EmitRstd)",
    )
    assert all(marker in SOURCE for marker in required)
    forbidden = (".item<", "cudaDeviceSynchronize", "getDefaultCUDAStream")
    assert all(marker not in SOURCE for marker in forbidden)
