"""Fail-closed Python facade for the isolated H3 fused Q preprocessing lab."""

from __future__ import annotations

import math
import importlib.util
import os
from pathlib import Path

import torch


_MODULE_NAME = "h3_q_rms_rope_int8_native_v1"
_HERE = Path(__file__).resolve().parent
_DEFAULT_LIBRARY = _HERE / "build" / f"{_MODULE_NAME}.so"
_LIBRARY_ENV = "H3_Q_RMS_ROPE_INT8_LIBRARY"
_BUILD_ID = "h3-q-rms-rope96-qint8-v3-torch2.13-sm120"
_TORCH_VERSION = "2.13.0+cu130"
_TORCH_GIT = "cf30153c4c131c8164ee7798e5022d810682e2cb"
_CUDA_VERSION = "13.0"
_DEVICE_CAPABILITY = (12, 1)


def _runtime_supported(device=None) -> bool:
    if (
        torch.__version__ != _TORCH_VERSION
        or torch.version.git_version != _TORCH_GIT
        or torch.version.cuda != _CUDA_VERSION
        or not torch.cuda.is_available()
    ):
        return False
    try:
        target = torch.device(device) if device is not None else torch.device("cuda")
        index = torch.cuda.current_device() if target.index is None else target.index
        return torch.cuda.get_device_capability(index) == _DEVICE_CAPABILITY
    except (AssertionError, RuntimeError):
        return False


def _load_native():
    if not _runtime_supported():
        return None
    configured = os.environ.get(_LIBRARY_ENV)
    path = Path(configured).expanduser() if configured else _DEFAULT_LIBRARY
    if not path.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except (ImportError, OSError, RuntimeError):
        return None
    try:
        build_id = module.build_id()
    except (AttributeError, RuntimeError):
        return None
    if build_id != _BUILD_ID:
        return None
    return module


_NATIVE = _load_native()


def available(device=None) -> bool:
    return _runtime_supported(device) and callable(
        getattr(_NATIVE, "qk_rms_rope_q_int8", None)
    )


def eligible(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    epsilon: float,
) -> bool:
    tensors = (q, k, freqs, q_weight, k_weight)
    return bool(
        all(isinstance(t, torch.Tensor) for t in tensors)
        and available(q.device)
        and not torch.is_grad_enabled()
        and all(t.is_cuda and t.dtype is torch.bfloat16 for t in tensors)
        and all(not t.requires_grad for t in tensors)
        and q.shape == k.shape
        and q.ndim == 4
        and q.shape[0] == 1
        and q.shape[1] > 1024
        and q.shape[2:] == (56, 128)
        and q.stride(-1) == k.stride(-1) == 1
        and q.stride(-2) == k.stride(-2) == 128
        and q.stride(1) == k.stride(1) == 21504
        and q.stride(0) == k.stride(0) == q.shape[1] * 21504
        and k.storage_offset() == q.storage_offset() + 7168
        and q.untyped_storage().data_ptr() == k.untyped_storage().data_ptr()
        and freqs.shape == (1, q.shape[1], 1, 48, 2, 2)
        and freqs.is_contiguous()
        and q_weight.shape == k_weight.shape == (128,)
        and q_weight.is_contiguous()
        and k_weight.is_contiguous()
        and all(t.device == q.device for t in tensors)
        and float(epsilon) == 1e-5
        and math.isfinite(float(epsilon))
        and float(epsilon) > 0.0
    )


def try_qk_rms_rope_q_int8(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    epsilon: float,
    warps_per_block: int = 16,
):
    """Return ``(q_int8, q_scale, k)`` or ``None`` before any launch."""
    if not eligible(q, k, freqs, q_weight, k_weight, epsilon):
        return None
    if warps_per_block not in (8, 16, 32):
        return None
    return _NATIVE.qk_rms_rope_q_int8(
        q, k, freqs, q_weight, k_weight, float(epsilon), warps_per_block
    )
