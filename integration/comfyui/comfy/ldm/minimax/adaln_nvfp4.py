"""Optional MiniMax H3 exact AdaLN to NVFP4 linear fast path."""

from __future__ import annotations

import importlib.util
import logging
import os
from functools import lru_cache
from pathlib import Path

import torch

import comfy.model_management
from comfy.ldm.minimax.linear_nvfp4 import _linear_eligible
from comfy.ldm.minimax.swiglu_nvfp4 import (
    HOOK_ATTRS,
    NVFP4_LAYOUT,
    _cuda_device_supported,
    _dynamic_eligible,
    _has_global_forward_hooks,
)
from comfy.ops import cast_bias_weight, run_every_op, uncast_bias_weight
from comfy.quant_ops import QuantizedTensor, TensorCoreNVFP4Layout


LIBRARY_ENV = "COMFY_MINIMAX_H3_ADALN_NVFP4_LIBRARY"
ROW_IDS_TOKEN = object()
H3_ADALN_LINEAR_SHAPES = frozenset(
    {
        (5376, 21504),  # QKV
        (5376, 28672),  # MLP gate/up
    }
)


@lru_cache(maxsize=1)
def _load_native():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        return None
    path = Path(os.environ.get(LIBRARY_ENV, Path(__file__).with_name("adaln_nvfp4_native_v1.so"))).expanduser()
    if not path.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location("h3_adaln_nvfp4_native_v1", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not callable(getattr(module, "adaln_nvfp4", None)) or not callable(
            getattr(module, "adaln_nvfp4_dynamic", None)
        ):
            raise ImportError("native module is missing required entrypoints")
        return module
    except (ImportError, OSError, RuntimeError) as error:
        logging.warning("MiniMax H3 AdaLN NVFP4 fusion disabled: %s", error)
        return None


def _inputs_eligible(linear, normed, scale, shift, row_ids):
    return (
        isinstance(normed, torch.Tensor)
        and normed.ndim == 2
        and normed.shape[0] > 0
        and normed.shape[1] == 5376
        and normed.is_cuda
        and _cuda_device_supported(normed.device)
        and normed.dtype == torch.bfloat16
        and normed.is_contiguous()
        and not normed.requires_grad
        and isinstance(scale, torch.Tensor)
        and isinstance(shift, torch.Tensor)
        and scale.ndim == 2
        and scale.shape == shift.shape
        and scale.shape[0] > 0
        and scale.shape[1] == 5376
        and scale.dtype in (torch.bfloat16, torch.float32)
        and shift.dtype == scale.dtype
        and scale.device == normed.device
        and shift.device == normed.device
        and scale.stride(-1) == 1
        and shift.stride(-1) == 1
        and not scale.requires_grad
        and not shift.requires_grad
        and isinstance(row_ids, torch.Tensor)
        and row_ids.ndim == 1
        and row_ids.shape[0] == normed.shape[0]
        and row_ids.dtype == torch.int32
        and row_ids.device == normed.device
        and row_ids.is_contiguous()
        and not row_ids.requires_grad
        and (getattr(linear, "in_features", None), getattr(linear, "out_features", None)) in H3_ADALN_LINEAR_SHAPES
    )


def parent_eligible(module):
    """Whether an owning Attention/MLP can use its internal producer fusion."""
    return (
        all(not bool(getattr(module, name, None)) for name in HOOK_ATTRS)
        and not _has_global_forward_hooks()
    )


def try_linear(linear, normed, scale, shift, row_ids, row_ids_token, mode, modulation_mode):
    """Return an exact modulated QKV/FC1 output, or ``None`` for stock dispatch."""
    if (
        modulation_mode != "exact"
        or row_ids_token is not ROW_IDS_TOKEN
        or mode != "auto"
        or not _inputs_eligible(linear, normed, scale, shift, row_ids)
        or not _linear_eligible(linear, normed)
    ):
        return None

    if not _dynamic_eligible(linear):
        return None

    native = _load_native()
    quantize = getattr(native, "adaln_nvfp4_dynamic", None)
    if not callable(quantize):
        return None

    run_every_op()
    weight, bias, offload_stream = cast_bias_weight(
        linear, normed, offloadable=True, compute_dtype=normed.dtype, want_requant=True
    )
    try:
        if (
            not isinstance(weight, QuantizedTensor)
            or weight._layout_cls != NVFP4_LAYOUT
            or getattr(weight._params, "transposed", False)
        ):
            return None

        qdata, block_scale, input_scale = quantize(normed, scale, shift, row_ids)
        params = TensorCoreNVFP4Layout.Params(
            scale=input_scale,
            orig_dtype=normed.dtype,
            orig_shape=tuple(normed.shape),
            block_scale=block_scale.view(torch.float8_e4m3fn),
        )
        qinput = QuantizedTensor(qdata, NVFP4_LAYOUT, params)
        return torch.nn.functional.linear(qinput, weight, bias)
    finally:
        uncast_bias_weight(linear, weight, bias, offload_stream)


__all__ = ["ROW_IDS_TOKEN", "parent_eligible", "try_linear"]
