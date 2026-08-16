"""Non-installed reference adapter for ComfyUI's H3 linear_input_act.

This file documents and exercises the narrow production contract for the
isolated native CUDA kernels. It is deliberately not imported by ComfyUI and
does not monkey-patch anything.
"""

from __future__ import annotations

from typing import Literal

import torch

try:
    from .native_cuda import swiglu_nvfp4, swiglu_nvfp4_dynamic
except ImportError:  # Allow direct execution with swiglu_nvfp4_lab on sys.path.
    from native_cuda import swiglu_nvfp4, swiglu_nvfp4_dynamic


NVFP4_LAYOUT = "TensorCoreNVFP4Layout"
_BRIDGE_ATTRS = (
    "_h3_nvfp4_fusion_enabled",
    "_h3_nvfp4_allowed_shapes",
    "_h3_nvfp4_allowed_dtypes",
    "_h3_nvfp4_validated_input_scale",
)
_LINEAR_HOOK_ATTRS = (
    "_forward_pre_hooks",
    "_forward_hooks",
    "_backward_pre_hooks",
    "_backward_hooks",
)
Route = Literal["static", "dynamic"]


def _route_from_mode_state(
    mode: object,
    *,
    static_bridge_valid: bool,
    input_scale_is_none: bool,
    bridge_attrs_absent: bool,
    hooks_absent: bool,
) -> Route | None:
    """Pure, fail-closed rollout decision used by CPU-only routing tests."""
    if not isinstance(mode, str):
        return None
    if mode == "static":
        return "static" if static_bridge_valid else None
    if mode == "auto":
        if static_bridge_valid:
            return "static"
        if input_scale_is_none and bridge_attrs_absent and hooks_absent:
            return "dynamic"
    return None  # Includes the default stock mode and unknown values.


def _raw_input_is_eligible(x: object) -> bool:
    # Validate structure before inspecting shape. This may be called from a
    # generic activation site, so malformed values must fail closed.
    return (
        isinstance(x, torch.Tensor)
        and x.ndim == 2
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and not x.requires_grad
        and x.is_contiguous()
        and x.shape[1] == 28672
    )


def _common_linear_is_eligible(linear, x: torch.Tensor, quantized_tensor_type) -> bool:
    weight = getattr(linear, "weight", None)
    return (
        getattr(linear, "in_features", None) == 14336
        and getattr(linear, "quant_format", None) == "nvfp4"
        and getattr(linear, "layout_type", None) == NVFP4_LAYOUT
        and isinstance(weight, quantized_tensor_type)
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


def _static_bridge_is_valid(linear, x: torch.Tensor) -> bool:
    # These immutable certificates exist only during a validated Apply scope.
    marker = getattr(linear, _BRIDGE_ATTRS[0], None)
    allowed_shapes = getattr(linear, _BRIDGE_ATTRS[1], None)
    allowed_dtypes = getattr(linear, _BRIDGE_ATTRS[2], None)
    validated_scale = getattr(linear, _BRIDGE_ATTRS[3], None)
    scale = getattr(linear, "input_scale", None)
    return (
        marker is True
        and isinstance(allowed_shapes, frozenset)
        and isinstance(allowed_dtypes, frozenset)
        and (int(x.shape[0]), 14336) in allowed_shapes
        and str(x.dtype) in allowed_dtypes
        and isinstance(scale, torch.Tensor)
        and scale.ndim == 0
        and scale.numel() == 1
        and scale.dtype == torch.float32
        and scale.device == x.device
        and validated_scale is scale
    )


def _bridge_attrs_absent(linear) -> bool:
    return all(not hasattr(linear, name) for name in _BRIDGE_ATTRS)


def _linear_hooks_absent(linear) -> bool:
    return all(not bool(getattr(linear, name, None)) for name in _LINEAR_HOOK_ATTRS)


def _eligible_route(linear, x: object, mode: object) -> Route | None:
    """Select a native route only after every production invariant holds."""
    if (
        not isinstance(mode, str)
        or mode not in ("static", "auto")
        or not _raw_input_is_eligible(x)
    ):
        return None

    import comfy.model_management
    from comfy.quant_ops import QuantizedTensor

    if (
        comfy.model_management.in_training
        or not _common_linear_is_eligible(linear, x, QuantizedTensor)
    ):
        return None

    return _route_from_mode_state(
        mode,
        static_bridge_valid=_static_bridge_is_valid(linear, x),
        input_scale_is_none=getattr(linear, "input_scale", None) is None,
        bridge_attrs_absent=_bridge_attrs_absent(linear),
        hooks_absent=_linear_hooks_absent(linear),
    )


def _eager_swiglu(x: torch.Tensor) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate).mul_(up)


def _make_qinput(
    x: torch.Tensor,
    qdata: torch.Tensor,
    block_scale: torch.Tensor,
    scale: torch.Tensor,
    quantized_tensor_type,
    layout_type,
):
    params = layout_type.Params(
        scale=scale,
        orig_dtype=x.dtype,
        orig_shape=(x.shape[0], x.shape[1] // 2),
        block_scale=block_scale,
    )
    return quantized_tensor_type(qdata, NVFP4_LAYOUT, params)


def try_h3_swiglu_nvfp4_linear(linear, x: torch.Tensor, mode: object):
    """Return a native fused FC2 output, or None for Comfy's stock path.

    stock (the rollout default) always delegates. static requires the exact
    Apply bridge certificate. auto first honors that certificate, then permits
    exact two-pass dynamic quantization only when no scale or bridge state
    exists and the Linear has no registered hooks.
    """
    route = _eligible_route(linear, x, mode)
    if route is None:
        return None

    from comfy.ops import cast_bias_weight, uncast_bias_weight
    from comfy.quant_ops import QuantizedTensor, TensorCoreNVFP4Layout

    # Both native routes share exactly one cast/offload lifecycle.
    weight, bias, offload_stream = cast_bias_weight(
        linear,
        x,
        offloadable=True,
        compute_dtype=x.dtype,
        want_requant=True,
    )
    try:
        # A late patch or dtype decision can return full precision. Consume the
        # already-staged result here instead of triggering a second cast.
        if (
            not isinstance(weight, QuantizedTensor)
            or weight._layout_cls != NVFP4_LAYOUT
            or getattr(weight._params, "transposed", False)
        ):
            return torch.nn.functional.linear(_eager_swiglu(x), weight, bias)

        if route == "static":
            scale = linear.input_scale
            qdata, block_scale = swiglu_nvfp4(x, scale)
        else:
            qdata, block_scale, scale = swiglu_nvfp4_dynamic(x)

        qinput = _make_qinput(
            x,
            qdata,
            block_scale,
            scale,
            QuantizedTensor,
            TensorCoreNVFP4Layout,
        )
        return torch.nn.functional.linear(qinput, weight, bias)
    finally:
        uncast_bias_weight(linear, weight, bias, offload_stream)


def _eligible_static_swiglu_nvfp4(linear, x: torch.Tensor, input_act: str) -> bool:
    """Compatibility predicate for the earlier static-only reference."""
    return input_act == "swiglu" and _eligible_route(linear, x, "static") == "static"


def try_static_swiglu_nvfp4_linear(linear, x: torch.Tensor, input_act: str):
    """Compatibility wrapper for an exact static Apply bridge."""
    if input_act != "swiglu":
        return None
    return try_h3_swiglu_nvfp4_linear(linear, x, "static")


__all__ = [
    "try_h3_swiglu_nvfp4_linear",
    "try_static_swiglu_nvfp4_linear",
]
