"""Fail-closed H3 previous-MLP gate -> next-block QKV adapter."""

from __future__ import annotations

import torch
import torch.nn.modules.module as torch_module

import comfy.model_management
import comfy.ldm.minimax.swiglu_nvfp4 as swiglu_nvfp4
from comfy.ldm.minimax.adaln_nvfp4 import ROW_IDS_TOKEN, parent_eligible
from comfy.ldm.minimax.gate_rms_adaln_nvfp4 import (
    _cast_norm_weight_eligible,
    _gate_inputs_eligible,
    _load_native,
    _norm_eligible,
    _spark_device_eligible,
    _storage_disjoint,
    _tensor_inputs_eligible,
)
from comfy.ldm.minimax.linear_nvfp4 import _linear_eligible
from comfy.ldm.minimax.swiglu_nvfp4 import HOOK_ATTRS, NVFP4_LAYOUT, _dynamic_eligible
from comfy.ops import cast_bias_weight, run_every_op, uncast_bias_weight
from comfy.quant_ops import QuantizedTensor, TensorCoreNVFP4Layout

try:
    import comfy.ldm.minimax.gate_rms_adaln_nvfp4_static as static_gate
except ImportError:
    static_gate = None


QKV_PAIR = (5376, 21504)
STATIC_BRIDGE_ATTRS = (
    "_h3_nvfp4_fusion_enabled",
    "_h3_nvfp4_allowed_shapes",
    "_h3_nvfp4_allowed_dtypes",
    "_h3_nvfp4_validated_input_scale",
    "_h3_nvfp4_plan_hook_ids",
)
CROSSBLOCK_TOKEN = object()
_CARRIER_SEAL = object()
_PENDING = object()
_CLAIMED = object()
_MATERIALIZED = object()
_GLOBAL_HOOK_ATTRS = (
    "_global_forward_pre_hooks",
    "_global_forward_hooks",
    "_global_backward_pre_hooks",
    "_global_backward_hooks",
)


class H3CrossBlockPending:
    """One-use provenance carrier owned by the normal H3 block loop."""

    __slots__ = (
        "_seal",
        "_state",
        "_previous_block",
        "_current_block",
        "_x",
        "_mlp",
        "_gate",
        "_mod_segments",
        "_row_ids",
        "_blocks_replace",
        "_loop_token",
        "_native",
        "_producer",
        "_route",
        "_input_scale",
    )

    def __init__(
        self,
        seal,
        previous_block,
        current_block,
        x,
        mlp,
        gate,
        mod_segments,
        row_ids,
        blocks_replace,
        loop_token,
        native,
        producer,
        route,
        input_scale,
    ):
        if seal is not _CARRIER_SEAL:
            raise TypeError("H3 cross-block carriers are internal")
        self._seal = seal
        self._state = _PENDING
        self._previous_block = previous_block
        self._current_block = current_block
        self._x = x
        self._mlp = mlp
        self._gate = gate
        self._mod_segments = mod_segments
        self._row_ids = row_ids
        self._blocks_replace = blocks_replace
        self._loop_token = loop_token
        self._native = native
        self._producer = producer
        self._route = route
        self._input_scale = input_scale


def is_pending(value) -> bool:
    return (
        isinstance(value, H3CrossBlockPending)
        and value._seal is _CARRIER_SEAL
        and value._state is _PENDING
    )


def _global_hooks_absent() -> bool:
    return all(not bool(getattr(torch_module, name, None)) for name in _GLOBAL_HOOK_ATTRS)


def _block_hooks_absent(block) -> bool:
    return all(not bool(getattr(block, name, None)) for name in HOOK_ATTRS)


def _blocks_replace_empty(blocks_replace) -> bool:
    return isinstance(blocks_replace, dict) and not blocks_replace


def _bridge_state(linear) -> str:
    present = tuple(hasattr(linear, name) for name in STATIC_BRIDGE_ATTRS)
    if not any(present):
        return "absent"
    return "complete" if all(present) else "partial"


def _certified_static_scale(linear, x):
    plan_hooks_match = getattr(swiglu_nvfp4, "_plan_hooks_match", None)
    scale = getattr(linear, "input_scale", None)
    shapes = getattr(linear, STATIC_BRIDGE_ATTRS[1], None)
    dtypes = getattr(linear, STATIC_BRIDGE_ATTRS[2], None)
    validated_scale = getattr(linear, STATIC_BRIDGE_ATTRS[3], None)
    hook_ids = getattr(linear, STATIC_BRIDGE_ATTRS[4], None)
    if not (
        _bridge_state(linear) == "complete"
        and getattr(linear, STATIC_BRIDGE_ATTRS[0], None) is True
        and isinstance(shapes, frozenset)
        and tuple(x.shape) in shapes
        and isinstance(dtypes, frozenset)
        and str(torch.bfloat16) in dtypes
        and x.dtype == torch.bfloat16
        and isinstance(scale, torch.Tensor)
        and scale.ndim == 0
        and scale.dtype == torch.float32
        and scale.device == x.device
        and not scale.requires_grad
        and validated_scale is scale
        and callable(plan_hooks_match)
        and plan_hooks_match(linear, hook_ids)
        and _global_hooks_absent()
    ):
        return None
    return scale


def _producer_capability(linear, x):
    state = _bridge_state(linear)
    if state == "complete":
        input_scale = _certified_static_scale(linear, x)
        load_native = getattr(static_gate, "_load_native", None)
        if input_scale is None or not callable(load_native):
            return None, None, None, None
        native = load_native()
        producer = getattr(native, "gate_rms_adaln_nvfp4_static", None)
        if not callable(producer):
            return None, None, None, None
        return "static", native, producer, input_scale
    if state != "absent":
        return None, None, None, None
    native = _load_native()
    producer = getattr(native, "gate_rms_adaln_nvfp4_dynamic", None)
    if not callable(producer):
        return None, None, None, None
    return "dynamic", native, producer, None


def block_pair_eligible(
    previous_block,
    current_block,
    crossblock_token,
    mode,
    modulation_mode,
    blocks_replace,
) -> bool:
    """Cheap loop-level gate; the tensor/native contract is checked later."""
    return (
        crossblock_token is CROSSBLOCK_TOKEN
        and mode == "auto"
        and modulation_mode == "exact"
        and _blocks_replace_empty(blocks_replace)
        and previous_block is not None
        and current_block is not None
        and type(previous_block) is type(current_block)
        and getattr(previous_block, "mlp", None) is not None
        and getattr(current_block, "attn", None) is not None
        and getattr(current_block, "norm1", None) is not None
        and _block_hooks_absent(previous_block)
        and _block_hooks_absent(current_block)
        and _global_hooks_absent()
        and not comfy.model_management.in_training
        and not torch.is_grad_enabled()
    )


def _pending_inputs_eligible(x, mlp, gate, row_ids) -> bool:
    return (
        isinstance(x, torch.Tensor)
        and x.ndim == 2
        and x.shape[0] > 0
        and x.shape[1] == QKV_PAIR[0]
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and not x.requires_grad
        and isinstance(mlp, torch.Tensor)
        and mlp.shape == x.shape
        and mlp.device == x.device
        and mlp.dtype == x.dtype
        and mlp.is_contiguous()
        and not mlp.requires_grad
        and isinstance(gate, torch.Tensor)
        and gate.ndim == 2
        and gate.shape[0] > 0
        and gate.shape[1] == QKV_PAIR[0]
        and gate.device == x.device
        and gate.dtype == torch.float32
        and gate.stride(-1) == 1
        and not gate.requires_grad
        and isinstance(row_ids, torch.Tensor)
        and row_ids.ndim == 1
        and row_ids.shape[0] == x.shape[0]
        and row_ids.device == x.device
        and row_ids.dtype == torch.int32
        and row_ids.is_contiguous()
        and not row_ids.requires_grad
        and _storage_disjoint(x, mlp, gate, row_ids)
    )


def _pending_identity_eligible(
    pending,
    previous_block,
    current_block,
    x,
    mod_segments,
    row_ids,
    crossblock_token,
) -> bool:
    return (
        is_pending(pending)
        and pending._previous_block is previous_block
        and pending._current_block is current_block
        and pending._x is x
        and pending._mod_segments is mod_segments
        and pending._row_ids is row_ids
        and pending._loop_token is crossblock_token
        and crossblock_token is CROSSBLOCK_TOKEN
    )


def _require_pending_identity(
    pending,
    previous_block,
    current_block,
    x,
    mod_segments,
    row_ids,
    crossblock_token,
):
    if not _pending_identity_eligible(
        pending,
        previous_block,
        current_block,
        x,
        mod_segments,
        row_ids,
        crossblock_token,
    ):
        raise RuntimeError("invalid or already-consumed H3 cross-block carrier")


def defer_qkv_transition(
    previous_block,
    current_block,
    x,
    mlp,
    gate,
    mod_segments,
    row_ids,
    crossblock_token,
    mode,
    modulation_mode,
    blocks_replace,
):
    """Return a sealed carrier, or ``None`` without changing ``x``."""
    if not block_pair_eligible(
        previous_block,
        current_block,
        crossblock_token,
        mode,
        modulation_mode,
        blocks_replace,
    ) or not _pending_inputs_eligible(x, mlp, gate, row_ids):
        return None
    owner = getattr(current_block, "attn", None)
    linear = getattr(owner, "qkv_proj", None)
    route, native, producer, input_scale = _producer_capability(linear, x)
    if producer is None:
        return None
    return H3CrossBlockPending(
        _CARRIER_SEAL,
        previous_block,
        current_block,
        x,
        mlp,
        gate,
        mod_segments,
        row_ids,
        blocks_replace,
        crossblock_token,
        native,
        producer,
        route,
        input_scale,
    )


def pending_residual(
    pending,
    previous_block,
    current_block,
    mod_segments,
    row_ids,
    crossblock_token,
):
    """Return the bound residual without exposing an unvalidated carrier."""
    x = getattr(pending, "_x", None)
    _require_pending_identity(
        pending,
        previous_block,
        current_block,
        x,
        mod_segments,
        row_ids,
        crossblock_token,
    )
    return x


def materialize_pending(
    pending,
    previous_block,
    current_block,
    x,
    mod_segments,
    row_ids,
    crossblock_token,
    materialize,
):
    """Claim a carrier for the exact old residual-gate implementation."""
    _require_pending_identity(
        pending,
        previous_block,
        current_block,
        x,
        mod_segments,
        row_ids,
        crossblock_token,
    )
    pending._state = _MATERIALIZED
    return materialize(x, pending._gate, pending._mlp, mod_segments, row_ids)


def _preflight_eligible(
    current_block,
    linear,
    norm,
    route,
    input_scale,
    x,
    other,
    gate,
    scale,
    shift,
    row_ids,
):
    owner = getattr(current_block, "attn", None)
    return (
        getattr(current_block, "norm1", None) is norm
        and getattr(owner, "qkv_proj", None) is linear
        and parent_eligible(owner)
        and (getattr(linear, "in_features", None), getattr(linear, "out_features", None))
        == QKV_PAIR
        and _tensor_inputs_eligible(linear, x, scale, shift, row_ids)
        and _spark_device_eligible(x.device)
        and _gate_inputs_eligible(x, other, gate, scale, shift, row_ids)
        and _norm_eligible(norm, x)
        and _linear_eligible(linear, x)
        and (
            (route == "dynamic" and input_scale is None and _dynamic_eligible(linear))
            or (
                route == "static"
                and input_scale is not None
                and _certified_static_scale(linear, x) is input_scale
                and _storage_disjoint(x, input_scale)
            )
        )
    )


def _consumer_still_eligible(
    pending,
    previous_mlp,
    previous_gate,
    previous_block,
    current_block,
    x,
    mod_segments,
    row_ids,
    crossblock_token,
    mode,
    modulation_mode,
    blocks_replace,
) -> bool:
    _require_pending_identity(
        pending,
        previous_block,
        current_block,
        x,
        mod_segments,
        row_ids,
        crossblock_token,
    )
    if pending._mlp is not previous_mlp or pending._gate is not previous_gate:
        raise RuntimeError("H3 cross-block carrier inputs changed")
    if not (
        block_pair_eligible(
            previous_block,
            current_block,
            crossblock_token,
            mode,
            modulation_mode,
            blocks_replace,
        )
        and pending._blocks_replace is blocks_replace
    ):
        return False
    owner = getattr(current_block, "attn", None)
    linear = getattr(owner, "qkv_proj", None)
    route, native, producer, input_scale = _producer_capability(linear, x)
    return (
        pending._route == route
        and pending._native is native
        and pending._producer is producer
        and pending._input_scale is input_scale
    )


def _quantized_outputs_eligible(
    route, qdata, block_scale, input_scale, rstd, x
) -> bool:
    padded_rows = (int(x.shape[0]) + 15) // 16 * 16
    scale_rows = (padded_rows + 127) // 128 * 128
    return (
        isinstance(qdata, torch.Tensor)
        and qdata.shape == (padded_rows, QKV_PAIR[0] // 2)
        and qdata.dtype == torch.uint8
        and qdata.device == x.device
        and qdata.is_contiguous()
        and not qdata.requires_grad
        and isinstance(block_scale, torch.Tensor)
        and block_scale.shape == (scale_rows, QKV_PAIR[0] // 16)
        and block_scale.dtype == torch.uint8
        and block_scale.device == x.device
        and block_scale.is_contiguous()
        and not block_scale.requires_grad
        and (
            (
                route == "dynamic"
                and isinstance(input_scale, torch.Tensor)
                and input_scale.shape == (1,)
                and input_scale.dtype == torch.float32
                and input_scale.device == x.device
                and input_scale.is_contiguous()
                and not input_scale.requires_grad
                and isinstance(rstd, torch.Tensor)
                and rstd.shape == (int(x.shape[0]),)
                and rstd.dtype == torch.float32
                and rstd.device == x.device
                and rstd.is_contiguous()
                and not rstd.requires_grad
                and _storage_disjoint(
                    x, qdata, block_scale, input_scale, rstd
                )
            )
            or (
                route == "static"
                and isinstance(input_scale, torch.Tensor)
                and input_scale.ndim == 0
                and input_scale.dtype == torch.float32
                and input_scale.device == x.device
                and not input_scale.requires_grad
                and rstd is None
                and _storage_disjoint(x, qdata, block_scale, input_scale)
            )
        )
    )


def _postclaim_eligible(
    pending,
    previous_mlp,
    previous_gate,
    previous_block,
    current_block,
    x,
    next_scale,
    next_shift,
    mod_segments,
    row_ids,
    crossblock_token,
    mode,
    modulation_mode,
    blocks_replace,
) -> bool:
    if not (
        isinstance(pending, H3CrossBlockPending)
        and pending._seal is _CARRIER_SEAL
        and pending._state is _CLAIMED
        and pending._previous_block is previous_block
        and pending._current_block is current_block
        and pending._x is x
        and pending._mlp is previous_mlp
        and pending._gate is previous_gate
        and pending._mod_segments is mod_segments
        and pending._row_ids is row_ids
        and pending._blocks_replace is blocks_replace
        and pending._loop_token is crossblock_token
        and pending._route in ("dynamic", "static")
        and block_pair_eligible(
            previous_block,
            current_block,
            crossblock_token,
            mode,
            modulation_mode,
            blocks_replace,
        )
    ):
        return False
    owner = getattr(current_block, "attn", None)
    linear = getattr(owner, "qkv_proj", None)
    norm = getattr(current_block, "norm1", None)
    route, native, producer, input_scale = _producer_capability(linear, x)
    return (
        pending._route == route
        and pending._native is native
        and pending._producer is producer
        and pending._input_scale is input_scale
        and _preflight_eligible(
            current_block,
            linear,
            norm,
            route,
            input_scale,
            x,
            previous_mlp,
            previous_gate,
            next_scale,
            next_shift,
            row_ids,
        )
    )


def try_qkv_transition(
    previous_block,
    current_block,
    x,
    pending,
    next_scale,
    next_shift,
    mod_segments,
    row_ids,
    row_ids_token,
    crossblock_token,
    mode,
    modulation_mode,
    blocks_replace,
):
    """Return projected QKV or ``None`` only before the carrier is claimed."""
    _require_pending_identity(
        pending,
        previous_block,
        current_block,
        x,
        mod_segments,
        row_ids,
        crossblock_token,
    )
    previous_mlp = pending._mlp
    previous_gate = pending._gate
    route = pending._route
    route_scale = pending._input_scale
    if row_ids_token is not ROW_IDS_TOKEN or not _consumer_still_eligible(
        pending,
        previous_mlp,
        previous_gate,
        previous_block,
        current_block,
        x,
        mod_segments,
        row_ids,
        crossblock_token,
        mode,
        modulation_mode,
        blocks_replace,
    ):
        return None

    owner = current_block.attn
    linear = owner.qkv_proj
    norm = current_block.norm1
    if not _preflight_eligible(
        current_block,
        linear,
        norm,
        route,
        route_scale,
        x,
        previous_mlp,
        previous_gate,
        next_scale,
        next_shift,
        row_ids,
    ):
        return None
    eps = getattr(norm, "eps", None)
    if eps is None:
        eps = torch.finfo(x.dtype).eps

    run_every_op()
    if not _consumer_still_eligible(
        pending,
        previous_mlp,
        previous_gate,
        previous_block,
        current_block,
        x,
        mod_segments,
        row_ids,
        crossblock_token,
        mode,
        modulation_mode,
        blocks_replace,
    ) or not _preflight_eligible(
        current_block,
        linear,
        norm,
        route,
        route_scale,
        x,
        previous_mlp,
        previous_gate,
        next_scale,
        next_shift,
        row_ids,
    ):
        return None

    norm_weight, norm_bias, norm_offload_stream = cast_bias_weight(
        norm, x, offloadable=True, compute_dtype=x.dtype
    )
    try:
        if (
            not _cast_norm_weight_eligible(norm_weight, norm_bias, x)
            or not _storage_disjoint(
                x,
                norm_weight,
                previous_mlp,
                previous_gate,
                next_scale,
                next_shift,
                row_ids,
            )
            or not _consumer_still_eligible(
                pending,
                previous_mlp,
                previous_gate,
                previous_block,
                current_block,
                x,
                mod_segments,
                row_ids,
                crossblock_token,
                mode,
                modulation_mode,
                blocks_replace,
            )
            or not _preflight_eligible(
                current_block,
                linear,
                norm,
                route,
                route_scale,
                x,
                previous_mlp,
                previous_gate,
                next_scale,
                next_shift,
                row_ids,
            )
        ):
            return None
        pending._state = _CLAIMED
        if route == "static":
            qdata, block_scale = pending._producer(
                x,
                previous_mlp,
                previous_gate,
                norm_weight,
                next_scale,
                next_shift,
                row_ids,
                route_scale,
                float(eps),
            )
            input_scale = route_scale
            rstd = None
        else:
            qdata, block_scale, input_scale, rstd = pending._producer(
                x,
                previous_mlp,
                previous_gate,
                norm_weight,
                next_scale,
                next_shift,
                row_ids,
                float(eps),
            )
    finally:
        uncast_bias_weight(norm, norm_weight, norm_bias, norm_offload_stream)

    if not _postclaim_eligible(
        pending,
        previous_mlp,
        previous_gate,
        previous_block,
        current_block,
        x,
        next_scale,
        next_shift,
        mod_segments,
        row_ids,
        crossblock_token,
        mode,
        modulation_mode,
        blocks_replace,
    ):
        raise RuntimeError("H3 cross-block contract changed after mutation")
    if not _quantized_outputs_eligible(
        route, qdata, block_scale, input_scale, rstd, x
    ):
        raise RuntimeError("cross-block producer returned invalid NVFP4 outputs")

    weight, bias, offload_stream = cast_bias_weight(
        linear, x, offloadable=True, compute_dtype=x.dtype, want_requant=True
    )
    try:
        if (
            not _postclaim_eligible(
                pending,
                previous_mlp,
                previous_gate,
                previous_block,
                current_block,
                x,
                next_scale,
                next_shift,
                mod_segments,
                row_ids,
                crossblock_token,
                mode,
                modulation_mode,
                blocks_replace,
            )
            or not isinstance(weight, QuantizedTensor)
            or weight._layout_cls != NVFP4_LAYOUT
            or getattr(weight._params, "transposed", False)
        ):
            raise RuntimeError("QKV owner/hook/weight changed after cross-block mutation")
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


__all__ = [
    "CROSSBLOCK_TOKEN",
    "H3CrossBlockPending",
    "block_pair_eligible",
    "defer_qkv_transition",
    "is_pending",
    "materialize_pending",
    "pending_residual",
    "try_qkv_transition",
]
