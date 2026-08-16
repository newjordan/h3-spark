"""Isolated wrapper for the one-pass static RMS+AdaLN NVFP4 prototype."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

import torch


WIDTH = 5376
HERE = Path(__file__).resolve().parent
NATIVE_ABI_VERSION = 1
NATIVE_BUILD_ID = "h3-rms-adaln-nvfp4-static-register-sm12-v1"


@lru_cache(maxsize=1)
def load_extension():
    library = HERE / "build" / "h3_rms_adaln_nvfp4_static_native_v1.so"
    if not library.is_file():
        raise FileNotFoundError(f"build the native prototype first: {library}")
    spec = importlib.util.spec_from_file_location(
        "h3_rms_adaln_nvfp4_static_native_v1", library
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {library}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if (
        module.abi_version() != NATIVE_ABI_VERSION
        or module.build_id() != NATIVE_BUILD_ID
    ):
        raise ImportError("static prototype ABI/build identity mismatch")
    return module


def _check(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    row_ids: torch.Tensor,
    global_scale: torch.Tensor,
) -> None:
    if x.ndim != 2 or x.shape[0] <= 0 or x.shape[1] != WIDTH:
        raise ValueError(f"x must have nonempty shape [S, {WIDTH}]")
    if x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous():
        raise ValueError("x must be contiguous CUDA BF16")
    if (
        weight.shape != (WIDTH,)
        or weight.dtype != torch.bfloat16
        or not weight.is_cuda
        or not weight.is_contiguous()
    ):
        raise ValueError(f"weight must be contiguous CUDA BF16 [{WIDTH}]")
    if (
        scale.ndim != 2
        or scale.shape[0] <= 0
        or scale.shape[1] != WIDTH
        or shift.shape != scale.shape
        or scale.dtype not in (torch.bfloat16, torch.float32)
        or shift.dtype != scale.dtype
        or scale.stride(1) != 1
        or shift.stride(1) != 1
    ):
        raise ValueError(f"scale/shift must be matching BF16/FP32 [M, {WIDTH}]")
    if (
        row_ids.shape != (x.shape[0],)
        or row_ids.dtype != torch.int32
        or not row_ids.is_cuda
        or not row_ids.is_contiguous()
    ):
        raise ValueError("row_ids must be contiguous CUDA int32 [S]")
    if (
        global_scale.numel() != 1
        or global_scale.dtype != torch.float32
        or not global_scale.is_cuda
        or not global_scale.is_contiguous()
    ):
        raise ValueError("global_scale must be one contiguous CUDA FP32 value")
    if not (
        x.device
        == weight.device
        == scale.device
        == shift.device
        == row_ids.device
        == global_scale.device
    ):
        raise ValueError("all tensors must share one CUDA device")


def rms_adaln_nvfp4_static(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    row_ids: torch.Tensor,
    global_scale: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return packed FP4 and byte-backed E4M3 block scales."""
    _check(x, weight, scale, shift, row_ids, global_scale)
    qdata, block_scale = load_extension().rms_adaln_nvfp4_static(
        x, weight, scale, shift, row_ids, global_scale, eps
    )
    return qdata, block_scale.view(torch.float8_e4m3fn)


def rms_adaln_nvfp4_static_debug(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    row_ids: torch.Tensor,
    global_scale: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gate-only variant that additionally exposes each row's FP32 rstd."""
    _check(x, weight, scale, shift, row_ids, global_scale)
    qdata, block_scale, rstd = load_extension().rms_adaln_nvfp4_static_debug(
        x, weight, scale, shift, row_ids, global_scale, eps
    )
    return qdata, block_scale.view(torch.float8_e4m3fn), rstd


__all__ = [
    "NATIVE_ABI_VERSION",
    "NATIVE_BUILD_ID",
    "WIDTH",
    "load_extension",
    "rms_adaln_nvfp4_static",
    "rms_adaln_nvfp4_static_debug",
]
