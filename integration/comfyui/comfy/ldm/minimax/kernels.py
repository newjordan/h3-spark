"""Optional inference kernels for MiniMax H3's segmented modulation."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _bf16_round_to_fp32(x):
        """Round FP32 to BF16 RNE while retaining an FP32 SSA value."""
        bits = tl.cast(x, tl.uint32, bitcast=True)
        bits += 0x7FFF + ((bits >> 16) & 1)
        bits &= 0xFFFF0000
        return tl.cast(bits, tl.float32, bitcast=True)


    @triton.jit
    def _segmented_adaln_bf16_kernel(
        h_ptr,
        scale_ptr,
        shift_ptr,
        row_ids_ptr,
        n_elements,
        scale_row_stride,
        shift_row_stride,
        d: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < n_elements
        rows = offsets // d
        cols = offsets - rows * d
        mod_rows = tl.load(row_ids_ptr + rows, mask=mask, other=0)
        h = tl.load(h_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(
            scale_ptr + mod_rows * scale_row_stride + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        shift = tl.load(
            shift_ptr + mod_rows * shift_row_stride + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        # Match the three BF16 round-to-nearest-even boundaries in:
        # h.mul_(1 + scale.to(bf16)).add_(shift.to(bf16)).
        multiplier = _bf16_round_to_fp32(1.0 + _bf16_round_to_fp32(scale))
        out = _bf16_round_to_fp32(h * multiplier)
        out = _bf16_round_to_fp32(out + _bf16_round_to_fp32(shift))
        tl.store(h_ptr + offsets, out, mask=mask)


    @triton.jit
    def _segmented_rms_adaln_bf16_kernel(
        x_ptr,
        weight_ptr,
        scale_ptr,
        shift_ptr,
        row_ids_ptr,
        out_ptr,
        scale_row_stride,
        shift_row_stride,
        d: tl.constexpr,
        eps: tl.constexpr,
        block_d: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_d)
        mask = cols < d
        x = tl.load(x_ptr + row * d + cols, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x * x, axis=0) / d + eps)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        mod_row = tl.load(row_ids_ptr + row)
        scale = tl.load(
            scale_ptr + mod_row * scale_row_stride + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        shift = tl.load(
            shift_ptr + mod_row * shift_row_stride + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        normed = _bf16_round_to_fp32(x * rstd * weight)
        multiplier = _bf16_round_to_fp32(1.0 + _bf16_round_to_fp32(scale))
        out = _bf16_round_to_fp32(normed * multiplier)
        out = _bf16_round_to_fp32(out + _bf16_round_to_fp32(shift))
        tl.store(out_ptr + row * d + cols, out, mask=mask)


    @triton.jit
    def _segmented_gate_bf16_kernel(
        x_ptr,
        other_ptr,
        gate_ptr,
        row_ids_ptr,
        n_elements,
        gate_row_stride,
        d: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < n_elements
        rows = offsets // d
        cols = offsets - rows * d
        mod_rows = tl.load(row_ids_ptr + rows, mask=mask, other=0)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        other = tl.load(other_ptr + offsets, mask=mask, other=0.0)
        gate = tl.load(
            gate_ptr + mod_rows * gate_row_stride + cols,
            mask=mask,
            other=0.0,
        ).to(tl.bfloat16)
        tl.store(x_ptr + offsets, x + other * gate, mask=mask)


def _can_fuse(tensor: torch.Tensor, row_ids: torch.Tensor | None) -> bool:
    return (
        triton is not None
        and tensor.device.type == "cuda"
        and tensor.dtype == torch.bfloat16
        and tensor.ndim == 2
        and tensor.numel() > 0
        and tensor.is_contiguous()
        and not tensor.requires_grad
        and row_ids is not None
        and row_ids.device == tensor.device
        and row_ids.dtype == torch.int32
        and row_ids.ndim == 1
        and row_ids.shape[0] == tensor.shape[0]
        and row_ids.is_contiguous()
    )


def segmented_adaln_bf16_(
    h: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    row_ids: torch.Tensor | None,
) -> bool:
    """Apply H3's segmented AdaLN in place; return whether fusion was used."""
    if (
        not _can_fuse(h, row_ids)
        or scale.device != h.device
        or shift.device != h.device
        or scale.ndim != 2
        or shift.ndim != 2
        or scale.shape != shift.shape
        or scale.shape[0] == 0
        or scale.stride(-1) != 1
        or shift.stride(-1) != 1
        or scale.shape[-1] != h.shape[-1]
        or scale.requires_grad
        or shift.requires_grad
    ):
        return False

    block = 1024
    _segmented_adaln_bf16_kernel[(triton.cdiv(h.numel(), block),)](
        h,
        scale,
        shift,
        row_ids,
        h.numel(),
        scale.stride(0),
        shift.stride(0),
        d=h.shape[1],
        block=block,
        num_warps=8,
        num_stages=1,
    )
    return True


def segmented_rms_adaln_bf16(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    row_ids: torch.Tensor | None,
    eps: float,
) -> torch.Tensor | None:
    """Fuse weighted RMSNorm and segmented AdaLN, or return ``None``."""
    if (
        not _can_fuse(x, row_ids)
        or weight.device != x.device
        or weight.dtype != torch.bfloat16
        or weight.ndim != 1
        or weight.shape[0] != x.shape[1]
        or not weight.is_contiguous()
        or weight.requires_grad
        or scale.device != x.device
        or shift.device != x.device
        or scale.ndim != 2
        or shift.ndim != 2
        or scale.shape != shift.shape
        or scale.shape[0] == 0
        or scale.stride(-1) != 1
        or shift.stride(-1) != 1
        or scale.shape[-1] != x.shape[-1]
        or scale.requires_grad
        or shift.requires_grad
    ):
        return None

    rows, d = x.shape
    out = torch.empty_like(x)
    _segmented_rms_adaln_bf16_kernel[(rows,)](
        x,
        weight,
        scale,
        shift,
        row_ids,
        out,
        scale.stride(0),
        shift.stride(0),
        d=d,
        eps=eps,
        block_d=triton.next_power_of_2(d),
        num_warps=8,
        num_stages=1,
    )
    return out


def segmented_gate_bf16_(
    x: torch.Tensor,
    other: torch.Tensor,
    gate: torch.Tensor,
    row_ids: torch.Tensor | None,
) -> bool:
    """Apply H3's segmented residual gate in place; return whether fused."""
    if (
        not _can_fuse(x, row_ids)
        or other.device != x.device
        or other.dtype != torch.bfloat16
        or other.shape != x.shape
        or not other.is_contiguous()
        or other.requires_grad
        or gate.device != x.device
        or gate.ndim != 2
        or gate.shape[0] == 0
        or gate.stride(-1) != 1
        or gate.shape[-1] != x.shape[-1]
        or gate.requires_grad
    ):
        return False

    block = 1024
    _segmented_gate_bf16_kernel[(triton.cdiv(x.numel(), block),)](
        x,
        other,
        gate,
        row_ids,
        x.numel(),
        gate.stride(0),
        d=x.shape[1],
        block=block,
        num_warps=8,
        num_stages=1,
    )
    return True
