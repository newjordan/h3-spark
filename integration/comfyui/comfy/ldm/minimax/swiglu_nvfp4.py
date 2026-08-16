"""Optional MiniMax H3 SwiGLU to NVFP4 CUDA fusion."""

from __future__ import annotations

import importlib.util
import logging
import os
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.modules.module as torch_module

import comfy.model_management
from comfy.ops import cast_bias_weight, uncast_bias_weight
from comfy.quant_ops import QuantizedTensor, TensorCoreNVFP4Layout


NVFP4_LAYOUT = "TensorCoreNVFP4Layout"
BF16_DTYPE = str(torch.bfloat16)
LIBRARY_ENV = "COMFY_MINIMAX_H3_SWIGLU_NVFP4_LIBRARY"
BRIDGE_ATTRS = (
    "_h3_nvfp4_fusion_enabled",
    "_h3_nvfp4_allowed_shapes",
    "_h3_nvfp4_allowed_dtypes",
    "_h3_nvfp4_validated_input_scale",
    "_h3_nvfp4_plan_hook_ids",
)
HOOK_ATTRS = (
    "_forward_pre_hooks",
    "_forward_hooks",
    "_backward_pre_hooks",
    "_backward_hooks",
)
GLOBAL_FORWARD_HOOK_ATTRS = (
    "_global_forward_pre_hooks",
    "_global_forward_hooks",
)


@lru_cache(maxsize=1)
def _load_native():
    # The bundled AOT cubin is built for Blackwell SM 12.x. Do not defer an
    # architecture mismatch to the first kernel launch.
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        return None
    path = Path(os.environ.get(LIBRARY_ENV, Path(__file__).with_name("swiglu_nvfp4_native_v1.so"))).expanduser()
    if not path.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location("h3_swiglu_nvfp4_native_v1", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not callable(getattr(module, "swiglu_nvfp4", None)) or not callable(
            getattr(module, "swiglu_nvfp4_dynamic", None)
        ):
            raise ImportError("native module is missing required entrypoints")
        return module
    except (ImportError, OSError, RuntimeError) as error:
        logging.warning("MiniMax H3 SwiGLU NVFP4 fusion disabled: %s", error)
        return None


@lru_cache(maxsize=8)
def _cuda_device_supported(device):
    try:
        return torch.cuda.is_available() and torch.cuda.get_device_capability(device)[0] == 12
    except (AssertionError, RuntimeError):
        return False


def _raw_eligible(x):
    return (
        isinstance(x, torch.Tensor)
        and x.ndim == 2
        and x.is_cuda
        and _cuda_device_supported(x.device)
        and x.dtype == torch.bfloat16
        and not x.requires_grad
        and x.is_contiguous()
        and x.shape[0] > 0
        and x.shape[1] == 28672
    )


def _linear_eligible(linear):
    weight = getattr(linear, "weight", None)
    return (
        not comfy.model_management.in_training
        and getattr(linear, "in_features", None) == 14336
        and getattr(linear, "quant_format", None) == "nvfp4"
        and getattr(linear, "layout_type", None) == NVFP4_LAYOUT
        and isinstance(weight, QuantizedTensor)
        and weight._layout_cls == NVFP4_LAYOUT
        and not getattr(weight._params, "transposed", False)
        and getattr(linear, "pre_quant_scale", None) is None
        and not getattr(linear, "_full_precision_mm", False)
        and not getattr(linear, "comfy_force_cast_weights", False)
        and not getattr(linear, "weight_function", ())
        and not getattr(linear, "bias_function", ())
        and getattr(linear, "weight_lowvram_function", None) is None
        and getattr(linear, "bias_lowvram_function", None) is None
    )


def _static_scale(linear, x):
    shapes = getattr(linear, BRIDGE_ATTRS[1], None)
    dtypes = getattr(linear, BRIDGE_ATTRS[2], None)
    validated_scale = getattr(linear, BRIDGE_ATTRS[3], None)
    hook_ids = getattr(linear, BRIDGE_ATTRS[4], None)
    scale = getattr(linear, "input_scale", None)
    if not (
        getattr(linear, BRIDGE_ATTRS[0], None) is True
        and isinstance(shapes, frozenset)
        and (int(x.shape[0]), 14336) in shapes
        and isinstance(dtypes, frozenset)
        and x.dtype == torch.bfloat16
        and BF16_DTYPE in dtypes
        and isinstance(scale, torch.Tensor)
        and scale.ndim == 0
        and scale.dtype == torch.float32
        and scale.device == x.device
        and not scale.requires_grad
        and validated_scale is scale
        and _plan_hooks_match(linear, hook_ids)
        and not _has_global_forward_hooks()
    ):
        return None
    return scale


def _has_global_forward_hooks():
    # A direct fused call does not enter Module.__call__, so it cannot preserve
    # process-wide hooks registered through register_module_forward_*_hook.
    return any(bool(getattr(torch_module, name, None)) for name in GLOBAL_FORWARD_HOOK_ATTRS)


def _plan_hooks_match(linear, hook_ids):
    return (
        isinstance(hook_ids, tuple)
        and len(hook_ids) == 2
        and all(type(value) is int for value in hook_ids)
        and hook_ids[0] != hook_ids[1]
        and set(getattr(linear, "_forward_pre_hooks", ())) == {hook_ids[0]}
        and set(getattr(linear, "_forward_hooks", ())) == {hook_ids[1]}
        and not bool(getattr(linear, "_forward_pre_hooks_with_kwargs", None))
        and not bool(getattr(linear, "_forward_hooks_with_kwargs", None))
        and set(getattr(linear, "_forward_hooks_always_called", ())) == {hook_ids[1]}
        and not bool(getattr(linear, "_backward_pre_hooks", None))
        and not bool(getattr(linear, "_backward_hooks", None))
    )


def _dynamic_eligible(linear):
    return (
        getattr(linear, "input_scale", None) is None
        and all(not hasattr(linear, name) for name in BRIDGE_ATTRS)
        and all(not bool(getattr(linear, name, None)) for name in HOOK_ATTRS)
        and not _has_global_forward_hooks()
    )


def _eager_swiglu(x):
    gate, up = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate).mul_(up)


def try_linear(linear, x, mode):
    """Return fused FC2 output, or ``None`` to run the stock H3 MLP path."""
    if mode not in ("static", "auto") or not _raw_eligible(x):
        return None

    scale = _static_scale(linear, x)
    route = "static" if scale is not None else None
    if route is None and mode == "auto" and _dynamic_eligible(linear):
        route = "dynamic"
    if route is None or not _linear_eligible(linear):
        return None
    native = _load_native()
    if native is None:
        return None

    weight, bias, offload_stream = cast_bias_weight(
        linear, x, offloadable=True, compute_dtype=x.dtype, want_requant=True
    )
    try:
        if route == "static" and _static_scale(linear, x) is not scale:
            return None
        if (
            not isinstance(weight, QuantizedTensor)
            or weight._layout_cls != NVFP4_LAYOUT
            or getattr(weight._params, "transposed", False)
        ):
            return torch.nn.functional.linear(_eager_swiglu(x), weight, bias)

        if route == "static":
            qdata, block_scale = native.swiglu_nvfp4(x, scale, True)
            if _static_scale(linear, x) is not scale:
                return None
        else:
            qdata, block_scale, scale = native.swiglu_nvfp4_dynamic(x)
        block_scale = block_scale.view(torch.float8_e4m3fn)
        params = TensorCoreNVFP4Layout.Params(
            scale=scale,
            orig_dtype=x.dtype,
            orig_shape=(x.shape[0], 14336),
            block_scale=block_scale,
        )
        qinput = QuantizedTensor(qdata, NVFP4_LAYOUT, params)
        return torch.nn.functional.linear(qinput, weight, bias)
    finally:
        uncast_bias_weight(linear, weight, bias, offload_stream)


__all__ = ["try_linear"]
