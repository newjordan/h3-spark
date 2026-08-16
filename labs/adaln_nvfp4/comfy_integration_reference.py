"""Non-installed, fail-closed ComfyUI adapter for the AdaLN/NVFP4 lab."""

from __future__ import annotations

from typing import Literal

import torch

try:
    from .native_cuda import adaln_nvfp4, adaln_nvfp4_dynamic
except ImportError:
    from native_cuda import adaln_nvfp4, adaln_nvfp4_dynamic


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
    if mode == "static":
        return "static" if static_bridge_valid else None
    if mode == "auto":
        if static_bridge_valid:
            return "static"
        if input_scale_is_none and bridge_attrs_absent and hooks_absent:
            return "dynamic"
    return None


def _tensor_inputs_are_eligible(normed, scale, shift, row_ids) -> bool:
    return (
        isinstance(normed, torch.Tensor)
        and normed.ndim == 2
        and normed.shape[1] == 5376
        and normed.shape[0] > 0
        and normed.is_cuda
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
    )


def _linear_is_eligible(linear, quantized_tensor_type) -> bool:
    weight = getattr(linear, "weight", None)
    return (
        getattr(linear, "in_features", None) == 5376
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


def _static_bridge_is_valid(linear, normed: torch.Tensor) -> bool:
    marker = getattr(linear, _BRIDGE_ATTRS[0], None)
    allowed_shapes = getattr(linear, _BRIDGE_ATTRS[1], None)
    allowed_dtypes = getattr(linear, _BRIDGE_ATTRS[2], None)
    validated_scale = getattr(linear, _BRIDGE_ATTRS[3], None)
    input_scale = getattr(linear, "input_scale", None)
    return (
        marker is True
        and isinstance(allowed_shapes, frozenset)
        and (int(normed.shape[0]), 5376) in allowed_shapes
        and isinstance(allowed_dtypes, frozenset)
        and str(normed.dtype) in allowed_dtypes
        and isinstance(input_scale, torch.Tensor)
        and input_scale.ndim == 0
        and input_scale.numel() == 1
        and input_scale.dtype == torch.float32
        and input_scale.device == normed.device
        and validated_scale is input_scale
    )


def _eligible_route(
    linear,
    normed,
    scale,
    shift,
    row_ids,
    mode: object,
) -> Route | None:
    if mode not in ("static", "auto") or not _tensor_inputs_are_eligible(
        normed, scale, shift, row_ids
    ):
        return None

    import comfy.model_management
    from comfy.quant_ops import QuantizedTensor

    if comfy.model_management.in_training or not _linear_is_eligible(
        linear, QuantizedTensor
    ):
        return None

    static_bridge_valid = _static_bridge_is_valid(linear, normed)
    return _route_from_mode_state(
        mode,
        static_bridge_valid=static_bridge_valid,
        input_scale_is_none=getattr(linear, "input_scale", None) is None,
        bridge_attrs_absent=all(
            not hasattr(linear, name) for name in _BRIDGE_ATTRS
        ),
        hooks_absent=all(
            not bool(getattr(linear, name, None))
            for name in _LINEAR_HOOK_ATTRS
        ),
    )


def _make_qinput(normed, qdata, block_scale, scale, tensor_type, layout_type):
    params = layout_type.Params(
        scale=scale,
        orig_dtype=normed.dtype,
        orig_shape=tuple(normed.shape),
        block_scale=block_scale,
    )
    return tensor_type(qdata, NVFP4_LAYOUT, params)


def try_h3_adaln_nvfp4_linear(
    linear,
    normed: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    row_ids: torch.Tensor,
    mode: object,
):
    """Return fused QKV/FC1 output, or ``None`` for the exact stock path."""
    route = _eligible_route(linear, normed, scale, shift, row_ids, mode)
    if route is None:
        return None

    from comfy.ops import cast_bias_weight, uncast_bias_weight
    from comfy.quant_ops import QuantizedTensor, TensorCoreNVFP4Layout

    weight, bias, offload_stream = cast_bias_weight(
        linear,
        normed,
        offloadable=True,
        compute_dtype=normed.dtype,
        want_requant=True,
    )
    try:
        # Do not perform the native op if staging produced an unexpected weight.
        # Returning None after uncast asks the caller to execute its untouched
        # norm/mod/linear path; this branch is exceptional and correctness-first.
        if (
            not isinstance(weight, QuantizedTensor)
            or weight._layout_cls != NVFP4_LAYOUT
            or getattr(weight._params, "transposed", False)
        ):
            return None

        if route == "static":
            qdata, block_scale = adaln_nvfp4(
                normed, scale, shift, row_ids, linear.input_scale
            )
            input_scale = linear.input_scale
        else:
            qdata, block_scale, input_scale = adaln_nvfp4_dynamic(
                normed, scale, shift, row_ids
            )
        qinput = _make_qinput(
            normed,
            qdata,
            block_scale,
            input_scale,
            QuantizedTensor,
            TensorCoreNVFP4Layout,
        )
        return torch.nn.functional.linear(qinput, weight, bias)
    finally:
        uncast_bias_weight(linear, weight, bias, offload_stream)


__all__ = ["try_h3_adaln_nvfp4_linear"]
