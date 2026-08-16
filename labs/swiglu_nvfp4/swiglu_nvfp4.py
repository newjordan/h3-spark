"""Isolated Triton prototype: BF16 SwiGLU directly to cuBLAS NVFP4 layout.

This intentionally has no ComfyUI integration.  The packing and scale swizzle
mirror comfy-kitchen 0.2.27's NVFP4 CUDA path.  A caller supplies the FP32
per-tensor decode scale, so there is no tensor-wide reduction in this kernel.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _swizzled_scale_offset(in_row, in_col, n_col_blocks, padded_scale_cols):
    row_block = in_row // 128
    col_block = in_col // 4
    in_block_row = in_row % 128
    in_block_col = in_col % 4
    sub_block = in_block_row // 32
    fine_row = in_block_row % 32
    combined_block = row_block * n_col_blocks + col_block
    intermediate_col = sub_block * 4 + in_block_col
    linear_idx = combined_block * 512 + fine_row * 16 + intermediate_col
    out_row = linear_idx // padded_scale_cols
    out_col = linear_idx % padded_scale_cols
    return out_row * padded_scale_cols + out_col


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 4, "BLOCK_K": 256}, num_warps=8),
        triton.Config({"BLOCK_M": 8, "BLOCK_K": 128}, num_warps=8),
        triton.Config({"BLOCK_M": 8, "BLOCK_K": 256}, num_warps=8),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 128}, num_warps=8),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 256}, num_warps=8),
    ],
    key=["m", "k"],
)
@triton.jit
def _swiglu_nvfp4_kernel(
    x_ptr,
    q_ptr,
    scales_ptr,
    per_tensor_scale_ptr,
    m: tl.constexpr,
    padded_m: tl.constexpr,
    k: tl.constexpr,
    raw_stride_m: tl.constexpr,
    q_stride_m: tl.constexpr,
    scale_cols: tl.constexpr,
    padded_scale_cols: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    valid = (rows[:, None] < m) & (cols[None, :] < k)

    gate_offs = rows[:, None] * raw_stride_m + cols[None, :]
    up_offs = gate_offs + k
    gate = tl.load(x_ptr + gate_offs, mask=valid, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + up_offs, mask=valid, other=0.0).to(tl.float32)

    # Match eager BF16 `silu(gate).mul_(up)`: SiLU is first rounded to BF16,
    # then the product is rounded to BF16 before block-scale calculation.
    silu_bf16 = (gate * tl.sigmoid(gate)).to(tl.bfloat16)
    act = (silu_bf16.to(tl.float32) * up).to(tl.bfloat16).to(tl.float32)
    act = tl.where(valid, act, 0.0)

    blocks = act.reshape(BLOCK_M, BLOCK_K // 16, 16)
    max_abs = tl.max(tl.abs(blocks), axis=2)
    tensor_scale = tl.load(per_tensor_scale_ptr).to(tl.float32)
    scaled_block = tl.minimum((max_abs / 6.0) / tensor_scale, 448.0)
    block_fp8 = scaled_block.to(tl.float8e4nv)

    block_cols = pid_k * (BLOCK_K // 16) + tl.arange(0, BLOCK_K // 16)
    n_col_blocks = tl.cdiv(scale_cols, 4)
    scale_offs = _swizzled_scale_offset(
        rows[:, None], block_cols[None, :], n_col_blocks, padded_scale_cols
    )
    scale_valid = (rows[:, None] < padded_m) & (block_cols[None, :] < scale_cols)
    tl.store(scales_ptr + scale_offs, block_fp8, mask=scale_valid)

    decoded_block = block_fp8.to(tl.float32) * tensor_scale
    zero_block = decoded_block < 1.0e-10
    denom = tl.where(zero_block, 1.0, decoded_block)
    encoded = blocks / denom[:, :, None]
    encoded = tl.where(zero_block[:, :, None], 0.0, encoded)

    pairs = encoded.reshape(BLOCK_M, BLOCK_K // 2, 2)
    pair_ids = tl.arange(0, BLOCK_K // 2)
    lanes = tl.arange(0, 2)
    even = tl.sum(tl.where(lanes[None, None, :] == 0, pairs, 0.0), axis=2)
    odd = tl.sum(tl.where(lanes[None, None, :] == 1, pairs, 0.0), axis=2)

    packed_u16 = tl.inline_asm_elementwise(
        asm="""
        {
            .reg .b8 fp4_byte;
            .reg .b16 result;
            cvt.rn.satfinite.e2m1x2.f32 fp4_byte, $1, $2;
            mov.b16 result, {fp4_byte, 0};
            mov.u16 $0, result;
        }
        """,
        constraints="=h,f,f",
        args=[even, odd],
        dtype=tl.uint16,
        is_pure=True,
        pack=1,
    )
    packed = (packed_u16 & 0xFF).to(tl.uint8)
    out_cols = pid_k * (BLOCK_K // 2) + pair_ids
    out_valid = (rows[:, None] < padded_m) & (out_cols[None, :] < k // 2)
    out_offs = rows[:, None] * q_stride_m + out_cols[None, :]
    tl.store(q_ptr + out_offs, packed, mask=out_valid)


def fused_swiglu_nvfp4(
    x: torch.Tensor,
    per_tensor_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``silu(x[:, :K]) * x[:, K:]`` without a BF16 intermediate.

    The return shapes exactly match ``ck.quantize_nvfp4(..., pad_16x=True)``.
    H3's K=14336 is already 16-aligned; rows are padded to a multiple of 16.
    """
    if x.ndim != 2 or not x.is_contiguous():
        raise ValueError("x must be a contiguous 2D tensor")
    if x.dtype != torch.bfloat16 or not x.is_cuda:
        raise ValueError("prototype supports CUDA BF16 input only")
    if x.shape[1] % 2:
        raise ValueError("SwiGLU input width must be even")
    k = x.shape[1] // 2
    if k % 256:
        raise ValueError("prototype requires activated width divisible by 256")
    if not isinstance(per_tensor_scale, torch.Tensor) or per_tensor_scale.numel() != 1:
        raise ValueError("per_tensor_scale must be a scalar tensor")
    if per_tensor_scale.dtype != torch.float32 or per_tensor_scale.device != x.device:
        per_tensor_scale = per_tensor_scale.to(device=x.device, dtype=torch.float32)

    m = x.shape[0]
    padded_m = triton.cdiv(m, 16) * 16
    scale_rows = triton.cdiv(padded_m, 128) * 128
    scale_cols = triton.cdiv(k // 16, 4) * 4
    qdata = torch.empty((padded_m, k // 2), dtype=torch.uint8, device=x.device)
    scales = torch.zeros((scale_rows, scale_cols), dtype=torch.float8_e4m3fn, device=x.device)

    def grid(meta):
        return (triton.cdiv(padded_m, meta["BLOCK_M"]), triton.cdiv(k, meta["BLOCK_K"]))

    _swiglu_nvfp4_kernel[grid](
        x,
        qdata,
        scales,
        per_tensor_scale,
        m=m,
        padded_m=padded_m,
        k=k,
        raw_stride_m=x.stride(0),
        q_stride_m=qdata.stride(0),
        scale_cols=scale_cols,
        padded_scale_cols=scale_cols,
    )
    return qdata, scales

