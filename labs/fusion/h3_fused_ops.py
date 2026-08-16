import torch
import triton
import triton.language as tl


@triton.jit
def _bf16_round_to_fp32(x):
    """Round FP32 to BF16 RNE while retaining an FP32 SSA type between operations."""
    bits = tl.cast(x, tl.uint32, bitcast=True)
    bits += 0x7FFF + ((bits >> 16) & 1)
    bits &= 0xFFFF0000
    return tl.cast(bits, tl.float32, bitcast=True)


@triton.jit
def _segmented_rms_adaln_bf16_kernel(
    x_ptr,
    weight_ptr,
    scale_ptr,
    shift_ptr,
    row_ids_ptr,
    out_ptr,
    d: tl.constexpr,
    eps: tl.constexpr,
    block_d: tl.constexpr,
    staged_rounding: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, block_d)
    mask = cols < d
    x = tl.load(x_ptr + row * d + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / d + eps)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    mod_row = tl.load(row_ids_ptr + row)
    scale = tl.load(scale_ptr + mod_row * d + cols, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(shift_ptr + mod_row * d + cols, mask=mask, other=0.0).to(tl.float32)

    if staged_rounding:
        # Mirrors RMSNorm's BF16 output and each BF16 modulation operation.
        normed = _bf16_round_to_fp32(x * rstd * weight)
        multiplier = _bf16_round_to_fp32(1.0 + _bf16_round_to_fp32(scale))
        out = _bf16_round_to_fp32(normed * multiplier)
        out = _bf16_round_to_fp32(out + _bf16_round_to_fp32(shift))
    else:
        out = x * rstd * weight * (1.0 + scale) + shift
    tl.store(out_ptr + row * d + cols, out, mask=mask)


@triton.jit
def _segmented_gate_bf16_kernel(
    x_ptr,
    other_ptr,
    gate_ptr,
    row_ids_ptr,
    n_elements,
    d: tl.constexpr,
    block: tl.constexpr,
    staged_rounding: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < n_elements
    rows = offsets // d
    cols = offsets - rows * d
    mod_rows = tl.load(row_ids_ptr + rows, mask=mask, other=0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    other = tl.load(other_ptr + offsets, mask=mask, other=0.0)
    gate = tl.load(gate_ptr + mod_rows * d + cols, mask=mask, other=0.0)
    if staged_rounding:
        gate = gate.to(tl.bfloat16)
    tl.store(x_ptr + offsets, x + other * gate, mask=mask)


@triton.jit
def _segmented_adaln_bf16_kernel(
    h_ptr,
    scale_ptr,
    shift_ptr,
    row_ids_ptr,
    n_elements,
    d: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < n_elements
    rows = offsets // d
    cols = offsets - rows * d
    mod_rows = tl.load(row_ids_ptr + rows, mask=mask, other=0)
    h = tl.load(h_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + mod_rows * d + cols, mask=mask, other=0.0).to(tl.float32)
    shift = tl.load(shift_ptr + mod_rows * d + cols, mask=mask, other=0.0).to(tl.float32)
    multiplier = _bf16_round_to_fp32(1.0 + _bf16_round_to_fp32(scale))
    out = _bf16_round_to_fp32(h * multiplier)
    out = _bf16_round_to_fp32(out + _bf16_round_to_fp32(shift))
    tl.store(h_ptr + offsets, out, mask=mask)


def segmented_rms_adaln_bf16(x, weight, scale, shift, row_ids, eps=1e-5, *, staged_rounding=False):
    """Fused weighted RMSNorm and arbitrary row-segment AdaLN for H3 BF16 activations."""
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("the prototype kernel requires BF16 x and weight")
    if x.ndim != 2 or not x.is_contiguous():
        raise ValueError("x must be a contiguous [S, D] tensor")
    s, d = x.shape
    out = torch.empty_like(x)
    _segmented_rms_adaln_bf16_kernel[(s,)](
        x,
        weight,
        scale,
        shift,
        row_ids,
        out,
        d=d,
        eps=eps,
        block_d=triton.next_power_of_2(d),
        staged_rounding=staged_rounding,
        num_warps=8,
        num_stages=1,
    )
    return out


def segmented_adaln_bf16_(h, scale, shift, row_ids, *, block=1024):
    """Apply H3's segmented BF16 scale/shift in place, preserving PyTorch rounding."""
    if h.dtype != torch.bfloat16 or h.ndim != 2 or not h.is_contiguous():
        raise ValueError("h must be a contiguous BF16 [S, D] tensor")
    n_elements = h.numel()
    _segmented_adaln_bf16_kernel[(triton.cdiv(n_elements, block),)](
        h,
        scale,
        shift,
        row_ids,
        n_elements,
        d=h.shape[1],
        block=block,
        num_warps=8,
        num_stages=1,
    )
    return h


def segmented_gate_bf16_(x, other, gate, row_ids, *, staged_rounding=True, block=1024):
    """Accumulate an arbitrary row-segment gate into H3's BF16 residual in place."""
    if x.dtype != torch.bfloat16 or other.dtype != torch.bfloat16:
        raise TypeError("the prototype kernel requires BF16 x and other")
    if x.shape != other.shape or x.ndim != 2 or not x.is_contiguous() or not other.is_contiguous():
        raise ValueError("x and other must be matching contiguous [S, D] tensors")
    n_elements = x.numel()
    _segmented_gate_bf16_kernel[(triton.cdiv(n_elements, block),)](
        x,
        other,
        gate,
        row_ids,
        n_elements,
        d=x.shape[1],
        block=block,
        staged_rounding=staged_rounding,
        num_warps=8,
        num_stages=1,
    )
    return x
