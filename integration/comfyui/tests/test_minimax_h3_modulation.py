import torch
import pytest

from comfy.ldm.minimax import kernels
from comfy.ldm.minimax.model import (
    _coalesce_mod_segments,
    _mod_gate,
    _mod_scale_shift,
)
from comfy.ldm.minimax.kernels import (
    segmented_adaln_bf16_,
    segmented_gate_bf16_,
    segmented_rms_adaln_bf16,
)


def test_coalesce_mod_segments_preserves_modulation_math():
    segments = [
        (0, 2, 0),
        (2, 5, 1),
        (5, 9, 1),
        (9, 12, 2),
        (12, 13, 1),
    ]
    merged = _coalesce_mod_segments(segments)
    assert merged == [(0, 2, 0), (2, 9, 1), (9, 12, 2), (12, 13, 1)]

    generator = torch.Generator().manual_seed(260812)
    x = torch.randn(13, 16, generator=generator)
    other = torch.randn(13, 16, generator=generator)
    shift = torch.randn(3, 16, generator=generator)
    scale = torch.randn(3, 16, generator=generator)
    gate = torch.randn(3, 16, generator=generator)

    old_scale_shift = _mod_scale_shift(x.clone(), shift, scale, segments)
    new_scale_shift = _mod_scale_shift(x.clone(), shift, scale, merged)
    assert torch.equal(old_scale_shift, new_scale_shift)

    old_gate = _mod_gate(x.clone(), gate, other, segments)
    new_gate = _mod_gate(x.clone(), gate, other, merged)
    assert torch.equal(old_gate, new_gate)


def test_production_like_direct_refs_coalesce_21_to_16_segments():
    # Thirteen text tag runs, six adjacent visual references sharing one
    # modulation row, then distinct target audio/video spans.
    segments = []
    cursor = 0
    for index in range(13):
        segments.append((cursor, cursor + 1, index % 2))
        cursor += 1
    for _ in range(6):
        segments.append((cursor, cursor + 4, 7))
        cursor += 4
    segments.extend([(cursor, cursor + 2, 8), (cursor + 2, cursor + 8, 9)])

    assert len(segments) == 21
    assert len(_coalesce_mod_segments(segments)) == 16


@pytest.mark.parametrize("kind", ("adaln", "rms_adaln", "gate"))
@pytest.mark.parametrize("shape", ((8,), (0, 8), (2, 7)))
def test_segmented_kernels_reject_invalid_modulation_contracts(monkeypatch, kind, shape):
    monkeypatch.setattr(kernels, "_can_fuse", lambda _tensor, _row_ids: True)
    x = torch.empty(3, 8, dtype=torch.bfloat16)
    row_ids = torch.zeros(3, dtype=torch.int32)
    invalid = torch.empty(shape, dtype=torch.bfloat16)

    if kind == "adaln":
        assert not segmented_adaln_bf16_(x, invalid, invalid, row_ids)
    elif kind == "rms_adaln":
        weight = torch.empty(8, dtype=torch.bfloat16)
        assert segmented_rms_adaln_bf16(
            x, weight, invalid, invalid, row_ids, 1e-5
        ) is None
    else:
        assert not segmented_gate_bf16_(x, x.clone(), invalid, row_ids)


@pytest.mark.parametrize("kind", ("adaln", "rms_adaln", "gate"))
def test_segmented_kernels_reject_grad_bearing_modulation(monkeypatch, kind):
    monkeypatch.setattr(kernels, "_can_fuse", lambda _tensor, _row_ids: True)
    x = torch.empty(3, 8, dtype=torch.bfloat16)
    row_ids = torch.zeros(3, dtype=torch.int32)
    modulation = torch.empty(
        2, 8, dtype=torch.bfloat16, requires_grad=True
    )

    if kind == "adaln":
        assert not segmented_adaln_bf16_(x, modulation, modulation, row_ids)
    elif kind == "rms_adaln":
        weight = torch.empty(8, dtype=torch.bfloat16)
        assert segmented_rms_adaln_bf16(
            x, weight, modulation, modulation, row_ids, 1e-5
        ) is None
    else:
        assert not segmented_gate_bf16_(x, x.clone(), modulation, row_ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_segmented_kernels_match_strided_live_modulation_views():
    generator = torch.Generator(device="cuda").manual_seed(260812)
    seq, hidden, rows = 41, 96, 4
    segments = [(0, 7, 0), (7, 19, 2), (19, 31, 2), (31, 41, 3)]
    row_ids = torch.empty(seq, dtype=torch.int32)
    for a, b, row in segments:
        row_ids[a:b] = row
    row_ids = row_ids.cuda()

    # AdaLN projection chunks are views with a 6*hidden row stride in the live
    # model. Exercise that layout rather than a friendlier contiguous tensor.
    projected = torch.randn(rows, hidden * 6, generator=generator, device="cuda")
    shift, scale, gate = projected.chunk(6, dim=-1)[:3]
    assert not shift.is_contiguous()
    x = torch.randn(seq, hidden, generator=generator, device="cuda", dtype=torch.bfloat16)
    other = torch.randn_like(x)

    expected_h = _mod_scale_shift(x.clone(), shift, scale, segments)
    actual_h = x.clone()
    assert segmented_adaln_bf16_(actual_h, scale, shift, row_ids)
    assert torch.equal(actual_h, expected_h)

    expected_x = _mod_gate(x.clone(), gate, other, segments)
    actual_x = x.clone()
    assert segmented_gate_bf16_(actual_x, other, gate, row_ids)
    assert torch.equal(actual_x, expected_x)

    weight = torch.randn(hidden, generator=generator, device="cuda", dtype=torch.bfloat16)
    expected_fused = _mod_scale_shift(
        torch.nn.functional.rms_norm(x, (hidden,), weight, 1e-5),
        shift,
        scale,
        segments,
    )
    actual_fused = segmented_rms_adaln_bf16(
        x, weight, scale, shift, row_ids, 1e-5
    )
    assert actual_fused is not None
    assert torch.isfinite(actual_fused).all()
    assert (actual_fused.float() - expected_fused.float()).abs().max() <= 0.03125
