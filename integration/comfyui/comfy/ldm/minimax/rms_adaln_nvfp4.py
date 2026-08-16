"""Guarded exact RMSNorm + AdaLN -> dynamic NVFP4 for MiniMax H3."""

from __future__ import annotations

import importlib.util
import logging
import os
from functools import lru_cache
from pathlib import Path

import torch

import comfy.model_management
import comfy.ops
from comfy.ldm.minimax.adaln_nvfp4 import H3_ADALN_LINEAR_SHAPES, ROW_IDS_TOKEN
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


LIBRARY_ENV = "COMFY_MINIMAX_H3_RMS_ADALN_NVFP4_LIBRARY"
WIDTH = 5376
CERTIFIED_TORCH_VERSION = "2.13.0+cu130"
CERTIFIED_TORCH_GIT_VERSION = "cf30153c4c131c8164ee7798e5022d810682e2cb"
NATIVE_ABI_VERSION = 2
NATIVE_BUILD_ID = "h3-rms-adaln-nvfp4-sm12-safe-row-v2"
CERTIFIED_H3_RMS_EPS = 1e-5


def _native_abi_eligible(module) -> bool:
    abi_version = getattr(module, "abi_version", None)
    build_id = getattr(module, "build_id", None)
    return (
        callable(abi_version)
        and abi_version() == NATIVE_ABI_VERSION
        and callable(build_id)
        and build_id() == NATIVE_BUILD_ID
        and callable(getattr(module, "rms_adaln_nvfp4_dynamic", None))
    )


@lru_cache(maxsize=1)
def _load_native():
    """Load the prebuilt SM12 binary only; never compile on a request."""
    if (
        torch.__version__ != CERTIFIED_TORCH_VERSION
        or torch.version.git_version != CERTIFIED_TORCH_GIT_VERSION
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability()[0] != 12
    ):
        return None
    path = Path(
        os.environ.get(
            LIBRARY_ENV,
            Path(__file__).with_name("rms_adaln_nvfp4_native_v2.so"),
        )
    ).expanduser()
    if not path.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location(
            "h3_rms_adaln_nvfp4_native_v2", path
        )
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not _native_abi_eligible(module):
            raise ImportError("native module has an incompatible RMS+AdaLN ABI")
        return module
    except (ImportError, OSError, RuntimeError) as error:
        logging.warning("MiniMax H3 RMS+AdaLN NVFP4 fusion disabled: %s", error)
        return None


def _tensor_inputs_eligible(linear, x, scale, shift, row_ids):
    return (
        isinstance(x, torch.Tensor)
        and x.ndim == 2
        and x.shape[0] > 0
        and x.shape[1] == WIDTH
        and x.is_cuda
        and _cuda_device_supported(x.device)
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and not x.requires_grad
        and isinstance(scale, torch.Tensor)
        and isinstance(shift, torch.Tensor)
        and scale.ndim == 2
        and scale.shape == shift.shape
        and scale.shape[0] > 0
        and scale.shape[1] == WIDTH
        and scale.dtype in (torch.bfloat16, torch.float32)
        and shift.dtype == scale.dtype
        and scale.device == x.device
        and shift.device == x.device
        and scale.stride(-1) == 1
        and shift.stride(-1) == 1
        and not scale.requires_grad
        and not shift.requires_grad
        and isinstance(row_ids, torch.Tensor)
        and row_ids.ndim == 1
        and row_ids.shape[0] == x.shape[0]
        and row_ids.dtype == torch.int32
        and row_ids.device == x.device
        and row_ids.is_contiguous()
        and not row_ids.requires_grad
        and (getattr(linear, "in_features", None), getattr(linear, "out_features", None))
        in H3_ADALN_LINEAR_SHAPES
    )


def _norm_eligible(norm, x):
    """Require an unmodified weighted H3 RMSNorm whose call may be bypassed."""
    weight = getattr(norm, "weight", None)
    normalized_shape = tuple(getattr(norm, "normalized_shape", ()))
    eps = getattr(norm, "eps", None)
    if eps is None:
        eps = torch.finfo(x.dtype).eps
    forward_impl = getattr(type(norm), "forward", None)
    allowed_norm_types = {
        torch.nn.RMSNorm,
        comfy.ops.disable_weight_init.RMSNorm,
        comfy.ops.manual_cast.RMSNorm,
    }
    return (
        type(norm) in allowed_norm_types
        and forward_impl is type(norm).forward
        and "forward" not in getattr(norm, "__dict__", {})
        and normalized_shape == (WIDTH,)
        and isinstance(eps, (float, int))
        and float(eps) == CERTIFIED_H3_RMS_EPS
        and bool(getattr(norm, "elementwise_affine", False))
        and isinstance(weight, torch.Tensor)
        and tuple(weight.shape) == (WIDTH,)
        and not weight.requires_grad
        and getattr(norm, "bias", None) is None
        and not comfy.model_management.in_training
        and not torch.is_grad_enabled()
        and all(not bool(getattr(norm, name, None)) for name in HOOK_ATTRS)
        and not _has_global_forward_hooks()
        and not getattr(norm, "weight_function", ())
        and not getattr(norm, "bias_function", ())
        and getattr(norm, "weight_lowvram_function", None) is None
        and getattr(norm, "bias_lowvram_function", None) is None
        and x.dtype == torch.bfloat16
    )


def _cast_norm_weight_eligible(weight, bias, x):
    return (
        isinstance(weight, torch.Tensor)
        and weight.shape == (WIDTH,)
        and weight.dtype == torch.bfloat16
        and weight.device == x.device
        and weight.is_contiguous()
        and not weight.requires_grad
        and bias is None
    )


def try_linear(linear, norm, x, scale, shift, row_ids, row_ids_token, mode, modulation_mode):
    """Return a fused QKV/FC1 result, or ``None`` for the exact stock path."""
    # Lazy import avoids a module cycle: the isolated static adapter reuses the
    # guards in this module. Unbridged layers return NO_STATIC_ROUTE and retain
    # this exact dynamic-v2 implementation.
    from comfy.ldm.minimax.rms_adaln_nvfp4_static import (
        NO_STATIC_ROUTE,
        try_static_route,
    )

    static_result = try_static_route(
        linear, norm, x, scale, shift, row_ids, row_ids_token, mode, modulation_mode
    )
    if static_result is not NO_STATIC_ROUTE:
        return static_result

    if (
        mode != "auto"
        or modulation_mode != "exact"
        or row_ids_token is not ROW_IDS_TOKEN
        or not _tensor_inputs_eligible(linear, x, scale, shift, row_ids)
        or not _norm_eligible(norm, x)
        or not _linear_eligible(linear, x)
        or not _dynamic_eligible(linear)
    ):
        return None
    native = _load_native()
    quantize = getattr(native, "rms_adaln_nvfp4_dynamic", None)
    if not callable(quantize):
        return None
    eps = getattr(norm, "eps", None)
    if eps is None:
        eps = torch.finfo(x.dtype).eps

    run_every_op()
    norm_weight, norm_bias, norm_offload_stream = cast_bias_weight(
        norm, x, offloadable=True, compute_dtype=x.dtype
    )
    try:
        if not _cast_norm_weight_eligible(norm_weight, norm_bias, x):
            return None
        # Finish quantization before casting the linear: DynamicVRAM may reuse
        # the same asynchronous cast buffer for the next module.
        qdata, block_scale, input_scale = quantize(
            x, norm_weight, scale, shift, row_ids, float(eps)
        )
    finally:
        uncast_bias_weight(norm, norm_weight, norm_bias, norm_offload_stream)

    weight, bias, offload_stream = cast_bias_weight(
        linear, x, offloadable=True, compute_dtype=x.dtype, want_requant=True
    )
    try:
        if (
            not isinstance(weight, QuantizedTensor)
            or weight._layout_cls != NVFP4_LAYOUT
            or getattr(weight._params, "transposed", False)
        ):
            return None
        params = TensorCoreNVFP4Layout.Params(
            scale=input_scale,
            orig_dtype=x.dtype,
            orig_shape=tuple(x.shape),
            block_scale=block_scale.view(torch.float8_e4m3fn),
        )
        qinput = QuantizedTensor(qdata, NVFP4_LAYOUT, params)
        return torch.nn.functional.linear(qinput, weight, bias)
    finally:
        uncast_bias_weight(linear, weight, bias, offload_stream)


__all__ = ["try_linear"]
