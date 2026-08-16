"""Isolated exact RMSNorm + segmented AdaLN -> dynamic NVFP4 prototype."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

import torch


WIDTH = 5376
HERE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def load_extension():
    library = HERE / "build" / "h3_rms_adaln_nvfp4_native_v2.so"
    if not library.is_file():
        raise FileNotFoundError(f"build the native prototype first: {library}")
    spec = importlib.util.spec_from_file_location(
        "h3_rms_adaln_nvfp4_native_v2", library
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {library}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    row_ids: torch.Tensor,
) -> None:
    if x.ndim != 2 or x.shape[1] != WIDTH or x.shape[0] <= 0:
        raise ValueError(f"x must have nonempty shape [S, {WIDTH}]")
    if x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous():
        raise ValueError("x must be contiguous CUDA BF16")
    if weight.shape != (WIDTH,) or weight.dtype != torch.bfloat16:
        raise ValueError(f"weight must be BF16 [{WIDTH}]")
    if not weight.is_cuda or not weight.is_contiguous():
        raise ValueError("weight must be contiguous CUDA")
    if scale.dtype not in (torch.bfloat16, torch.float32) or shift.dtype != scale.dtype:
        raise ValueError("scale/shift must have matching BF16 or FP32 dtype")
    if scale.ndim != 2 or shift.shape != scale.shape or scale.shape[1] != WIDTH:
        raise ValueError(f"scale/shift must have matching [M, {WIDTH}] shapes")
    if scale.stride(1) != 1 or shift.stride(1) != 1:
        raise ValueError("scale/shift feature dimension must be contiguous")
    if row_ids.shape != (x.shape[0],) or row_ids.dtype != torch.int32:
        raise ValueError("row_ids must be int32 [S]")
    if not row_ids.is_cuda or not row_ids.is_contiguous():
        raise ValueError("row_ids must be contiguous CUDA")
    if not (x.device == weight.device == scale.device == shift.device == row_ids.device):
        raise ValueError("all tensors must share one CUDA device")


def rms_adaln_bf16(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    row_ids: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Debug gate: materialize the exact fused BF16 boundary and row rstd."""
    _check(x, weight, scale, shift, row_ids)
    return tuple(load_extension().rms_adaln_bf16(x, weight, scale, shift, row_ids, eps))


def rms_adaln_nvfp4_dynamic(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    row_ids: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-pass exact fusion producing packed FP4, E4M3 block scales and scale."""
    _check(x, weight, scale, shift, row_ids)
    packed, scale_bytes, global_scale = load_extension().rms_adaln_nvfp4_dynamic(
        x, weight, scale, shift, row_ids, eps
    )
    return packed, scale_bytes.view(torch.float8_e4m3fn), global_scale


__all__ = [
    "WIDTH",
    "load_extension",
    "rms_adaln_bf16",
    "rms_adaln_nvfp4_dynamic",
]
