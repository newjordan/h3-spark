"""Optional MiniMax H3 BF16 to NVFP4 linear fast path."""

from __future__ import annotations

import torch

import comfy.model_management
from comfy.ldm.minimax.swiglu_nvfp4 import (
    BF16_DTYPE,
    BRIDGE_ATTRS,
    HOOK_ATTRS,
    NVFP4_LAYOUT,
    _cuda_device_supported,
    _has_global_forward_hooks,
    _load_native,
    _plan_hooks_match,
)
from comfy.ops import cast_bias_weight, run_every_op, uncast_bias_weight
from comfy.quant_ops import QuantizedTensor, TensorCoreNVFP4Layout


H3_LINEAR_SHAPES = frozenset(
    {
        (5376, 21504),  # QKV
        (7168, 5376),   # attention output
        (5376, 28672),  # MLP gate/up
    }
)
H3_SUPPLIED_AMAX_BUILD_ID = "h3-nvfp4-supplied-amax-v1-torch2.13-sm120"


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
        and x.shape[1] in (5376, 7168)
    )


def _linear_config_eligible(linear, input_features):
    weight = getattr(linear, "weight", None)
    return (
        not comfy.model_management.in_training
        and (getattr(linear, "in_features", None), getattr(linear, "out_features", None)) in H3_LINEAR_SHAPES
        and getattr(linear, "in_features", None) == input_features
        and getattr(linear, "bias", None) is None
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


def _linear_eligible(linear, x):
    return _linear_config_eligible(linear, x.shape[1])


def _static_scale(linear, x):
    shapes = getattr(linear, BRIDGE_ATTRS[1], None)
    dtypes = getattr(linear, BRIDGE_ATTRS[2], None)
    validated_scale = getattr(linear, BRIDGE_ATTRS[3], None)
    hook_ids = getattr(linear, BRIDGE_ATTRS[4], None)
    scale = getattr(linear, "input_scale", None)
    if not (
        getattr(linear, BRIDGE_ATTRS[0], None) is True
        and isinstance(shapes, frozenset)
        and tuple(x.shape) in shapes
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


def _dynamic_eligible(linear):
    return (
        getattr(linear, "input_scale", None) is None
        and all(not hasattr(linear, name) for name in BRIDGE_ATTRS)
        and all(not bool(getattr(linear, name, None)) for name in HOOK_ATTRS)
        and not _has_global_forward_hooks()
    )


def try_linear(linear, x, mode):
    """Return a direct-quantized linear output, or ``None`` for stock dispatch."""
    if mode not in ("static", "auto") or not _raw_eligible(x):
        return None

    scale = _static_scale(linear, x)
    route = "static" if scale is not None else None
    if route is None and mode == "auto" and _dynamic_eligible(linear):
        route = "dynamic"
    if route is None or not _linear_eligible(linear, x):
        return None

    native = _load_native()
    quantize = getattr(native, "bf16_nvfp4" if route == "static" else "bf16_nvfp4_dynamic", None)
    if not callable(quantize):
        return None

    run_every_op()
    if route == "static" and _static_scale(linear, x) is not scale:
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
            return torch.nn.functional.linear(x, weight, bias)

        if route == "static":
            qdata, block_scale = quantize(x, scale)
            if _static_scale(linear, x) is not scale:
                return None
        else:
            qdata, block_scale, scale = quantize(x)
        params = TensorCoreNVFP4Layout.Params(
            scale=scale,
            orig_dtype=x.dtype,
            orig_shape=tuple(x.shape),
            block_scale=block_scale.view(torch.float8_e4m3fn),
        )
        qinput = QuantizedTensor(qdata, NVFP4_LAYOUT, params)
        return torch.nn.functional.linear(qinput, weight, bias)
    finally:
        uncast_bias_weight(linear, weight, bias, offload_stream)


def _from_amax_eligible(linear, x, mode, max_bits=None):
    if (
        mode != "auto"
        or torch.is_grad_enabled()
        or not _raw_eligible(x)
        or tuple(x.shape[1:]) != (7168,)
        or (getattr(linear, "in_features", None), getattr(linear, "out_features", None)) != (7168, 5376)
        or not _dynamic_eligible(linear)
        or not _linear_eligible(linear, x)
    ):
        return False
    return max_bits is None or (
        isinstance(max_bits, torch.Tensor)
        and max_bits.dtype == torch.int32
        and max_bits.numel() == 1
        and max_bits.is_cuda
        and max_bits.device == x.device
        and max_bits.is_contiguous()
        and not max_bits.requires_grad
    )


def _supplied_amax_native():
    """Return only the exact additive supplied-amax ABI.

    Keep this check out of the shared loader so an older extension can still
    serve the already-qualified dynamic/direct/SwiGLU routes.  Capability
    mismatch fails closed before the attention producer mutates Q/K.
    """
    native = _load_native()
    build_id = getattr(native, "h3_supplied_amax_build_id", None)
    quantize = getattr(native, "bf16_nvfp4_from_amax_bits", None)
    if not callable(build_id) or not callable(quantize):
        return None
    try:
        return native if build_id() == H3_SUPPLIED_AMAX_BUILD_ID else None
    except (OSError, RuntimeError):
        return None


def can_linear_from_amax_bits(linear, x, mode):
    if not _from_amax_eligible(linear, x, mode):
        return False
    return _supplied_amax_native() is not None


def can_static_linear_for_output(linear, rows, device, dtype, mode):
    """Preflight a static H3 attention output before its producer launches."""
    device = torch.device(device)
    shapes = getattr(linear, BRIDGE_ATTRS[1], None)
    dtypes = getattr(linear, BRIDGE_ATTRS[2], None)
    validated_scale = getattr(linear, BRIDGE_ATTRS[3], None)
    hook_ids = getattr(linear, BRIDGE_ATTRS[4], None)
    scale = getattr(linear, "input_scale", None)
    return (
        mode in ("static", "auto")
        and not torch.is_grad_enabled()
        and isinstance(rows, int)
        and rows > 0
        and device.type == "cuda"
        and _cuda_device_supported(device)
        and dtype == torch.bfloat16
        and getattr(linear, BRIDGE_ATTRS[0], None) is True
        and isinstance(shapes, frozenset)
        and (rows, 7168) in shapes
        and isinstance(dtypes, frozenset)
        and BF16_DTYPE in dtypes
        and isinstance(scale, torch.Tensor)
        and scale.ndim == 0
        and scale.dtype == torch.float32
        and scale.device == device
        and not scale.requires_grad
        and validated_scale is scale
        and _plan_hooks_match(linear, hook_ids)
        and (getattr(linear, "in_features", None), getattr(linear, "out_features", None))
        == (7168, 5376)
        and _linear_config_eligible(linear, 7168)
        and not _has_global_forward_hooks()
    )


def can_linear_from_amax_bits_for_output(linear, rows, device, dtype, mode):
    """Preflight H3's freshly allocated contiguous attention output."""
    device = torch.device(device)
    if not (
        mode == "auto"
        and not torch.is_grad_enabled()
        and isinstance(rows, int)
        and rows > 0
        and device.type == "cuda"
        and _cuda_device_supported(device)
        and dtype == torch.bfloat16
        and _dynamic_eligible(linear)
        and _linear_config_eligible(linear, 7168)
        and (getattr(linear, "in_features", None), getattr(linear, "out_features", None))
        == (7168, 5376)
    ):
        return False
    return _supplied_amax_native() is not None


def try_linear_from_amax_bits(linear, x, max_bits, mode):
    """Use an exact producer amax for H3's attention output projection."""
    if not _from_amax_eligible(linear, x, mode, max_bits):
        return None

    native = _supplied_amax_native()
    if native is None:
        return None
    quantize = getattr(native, "bf16_nvfp4_from_amax_bits", None)

    run_every_op()
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
        qdata, block_scale, scale = quantize(x, max_bits)
        params = TensorCoreNVFP4Layout.Params(
            scale=scale,
            orig_dtype=x.dtype,
            orig_shape=tuple(x.shape),
            block_scale=block_scale.view(torch.float8_e4m3fn),
        )
        qinput = QuantizedTensor(qdata, NVFP4_LAYOUT, params)
        return torch.nn.functional.linear(qinput, weight, bias)
    finally:
        uncast_bias_weight(linear, weight, bias, offload_stream)


__all__ = [
    "H3_SUPPLIED_AMAX_BUILD_ID",
    "can_linear_from_amax_bits",
    "can_linear_from_amax_bits_for_output",
    "can_static_linear_for_output",
    "try_linear",
    "try_linear_from_amax_bits",
]
