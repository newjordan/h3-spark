"""Fail-closed H3 attention-gate -> static RMS/AdaLN -> FC1 adapter.

Routing is intentionally tri-state. A complete, exact v3 bridge claims the
call for the supplied-scale candidate. No bridge delegates verbatim to the
qualified dynamic gate-v2 adapter. Any partial, stale, or malformed bridge
returns ``None`` before the residual can be mutated, so the model executes its
stock gate/RMS/AdaLN/FC1 path instead of crossing scale contracts.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
from functools import lru_cache
from pathlib import Path

import torch

import comfy.ldm.minimax.gate_rms_adaln_nvfp4 as dynamic_gate
from comfy.ldm.minimax.adaln_nvfp4 import ROW_IDS_TOKEN, parent_eligible
from comfy.ldm.minimax.linear_nvfp4 import _linear_eligible, _static_scale
from comfy.ldm.minimax.rms_adaln_nvfp4 import (
    CERTIFIED_TORCH_GIT_VERSION,
    CERTIFIED_TORCH_VERSION,
    WIDTH,
    _cast_norm_weight_eligible,
    _norm_eligible,
    _tensor_inputs_eligible,
)
from comfy.ldm.minimax.swiglu_nvfp4 import BRIDGE_ATTRS, NVFP4_LAYOUT
from comfy.ops import cast_bias_weight, run_every_op, uncast_bias_weight
from comfy.quant_ops import QuantizedTensor, TensorCoreNVFP4Layout


LIBRARY_ENV = "COMFY_MINIMAX_H3_GATE_RMS_ADALN_NVFP4_STATIC_LIBRARY"
NATIVE_MODULE_NAME = "h3_gate_rms_adaln_nvfp4_static_native_v1"
NATIVE_ABI_VERSION = 1
NATIVE_BUILD_ID = (
    "h3-gate-rms-adaln-nvfp4-static-torch2.13-cf30153-sm120-gb10-safe-v1"
)
NATIVE_SHA256 = "96b514d1a8ef3007809f41ed565c6d94d270faae34c826981b795e5d0cb21fe9"
CERTIFIED_CUDA_VERSION = "13.0"
CERTIFIED_DEVICE_CAPABILITY = (12, 1)
FC1_PAIR = (5376, 28672)
NO_STATIC_BRIDGE = object()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _native_abi_eligible(module) -> bool:
    abi = getattr(module, "abi_version", None)
    build_id = getattr(module, "build_id", None)
    return (
        callable(abi)
        and abi() == NATIVE_ABI_VERSION
        and callable(build_id)
        and build_id() == NATIVE_BUILD_ID
        and callable(getattr(module, "gate_rms_adaln_nvfp4_static", None))
    )


def _spark_device_eligible(device=None) -> bool:
    try:
        return (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(device)
            == CERTIFIED_DEVICE_CAPABILITY
        )
    except (AssertionError, RuntimeError, TypeError, ValueError):
        return False


@lru_cache(maxsize=1)
def _load_native():
    """Load one hash-pinned AOT cubin; never compile on a request."""
    if (
        torch.__version__ != CERTIFIED_TORCH_VERSION
        or torch.version.git_version != CERTIFIED_TORCH_GIT_VERSION
        or torch.version.cuda != CERTIFIED_CUDA_VERSION
        or not _spark_device_eligible()
    ):
        return None
    path = Path(
        os.environ.get(
            LIBRARY_ENV,
            # Production installation keeps the hash-pinned AOT module beside
            # this adapter.  The isolated seconds gate loads the lab build by
            # its explicit path, so no request can accidentally JIT or select
            # an unqualified build-directory artifact.
            Path(__file__).with_name(f"{NATIVE_MODULE_NAME}.so"),
        )
    ).expanduser()
    if not path.is_file():
        return None
    try:
        if _file_sha256(path) != NATIVE_SHA256:
            raise ImportError("native module SHA-256 is not certified")
        spec = importlib.util.spec_from_file_location(NATIVE_MODULE_NAME, path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not _native_abi_eligible(module):
            raise ImportError("native module has an incompatible static gate ABI")
        return module
    except (ImportError, OSError, RuntimeError) as error:
        logging.warning("MiniMax H3 static gate/RMS fusion disabled: %s", error)
        return None


def _storage_disjoint(x, *others) -> bool:
    if not isinstance(x, torch.Tensor) or any(
        not isinstance(other, torch.Tensor) for other in others
    ):
        return False
    try:
        return all(not torch._C._overlaps(x, other) for other in others)
    except (AttributeError, RuntimeError, TypeError):
        return False


def _gate_inputs_eligible(x, other, gate, scale, shift, row_ids) -> bool:
    """Reuse gate-v2 semantics and retain a local exact identity for tests."""
    return dynamic_gate._gate_inputs_eligible(
        x, other, gate, scale, shift, row_ids
    )


def _bridge_state(linear) -> str:
    present = tuple(hasattr(linear, name) for name in BRIDGE_ATTRS)
    if not any(present):
        return "absent"
    return "complete" if all(present) else "partial"


def _preflight_eligible(
    owner, linear, norm, x, other, gate, scale, shift, row_ids
) -> bool:
    return (
        getattr(owner, "fc1", None) is linear
        and parent_eligible(owner)
        and (
            getattr(linear, "in_features", None),
            getattr(linear, "out_features", None),
        )
        == FC1_PAIR
        and _tensor_inputs_eligible(linear, x, scale, shift, row_ids)
        and _spark_device_eligible(getattr(x, "device", None))
        and _gate_inputs_eligible(x, other, gate, scale, shift, row_ids)
        and _norm_eligible(norm, x)
        and _linear_eligible(linear, x)
    )


def _quantized_outputs_eligible(qdata, block_scale, x) -> bool:
    padded_rows = (int(x.shape[0]) + 15) // 16 * 16
    scale_rows = (padded_rows + 127) // 128 * 128
    return (
        isinstance(qdata, torch.Tensor)
        and qdata.shape == (padded_rows, WIDTH // 2)
        and qdata.dtype == torch.uint8
        and qdata.device == x.device
        and qdata.is_contiguous()
        and not qdata.requires_grad
        and isinstance(block_scale, torch.Tensor)
        and block_scale.shape == (scale_rows, WIDTH // 16)
        and block_scale.dtype == torch.uint8
        and block_scale.device == x.device
        and block_scale.is_contiguous()
        and not block_scale.requires_grad
    )


def try_static_linear(
    owner,
    linear,
    norm,
    x,
    other,
    gate,
    scale,
    shift,
    row_ids,
    row_ids_token,
    mode,
    modulation_mode,
):
    """Return static result/None, or ``NO_STATIC_BRIDGE`` for dynamic-v2."""
    state = _bridge_state(linear)
    if state == "absent":
        return NO_STATIC_BRIDGE
    if state != "complete":
        return None
    if (
        mode != "auto"
        or modulation_mode != "exact"
        or row_ids_token is not ROW_IDS_TOKEN
        or not _preflight_eligible(
            owner, linear, norm, x, other, gate, scale, shift, row_ids
        )
    ):
        return None
    input_scale = _static_scale(linear, x)
    if input_scale is None or not _storage_disjoint(x, input_scale):
        return None

    native = _load_native()
    quantize = getattr(native, "gate_rms_adaln_nvfp4_static", None)
    if not callable(quantize):
        return None
    eps = getattr(norm, "eps", None)
    if eps is None:
        eps = torch.finfo(x.dtype).eps

    run_every_op()
    # Deferred model-management callbacks may alter hooks, owners, inputs, or
    # bridge state. Recheck the complete contract at the last safe boundary.
    if (
        _bridge_state(linear) != "complete"
        or not _preflight_eligible(
            owner, linear, norm, x, other, gate, scale, shift, row_ids
        )
        or _static_scale(linear, x) is not input_scale
        or not _storage_disjoint(x, input_scale)
    ):
        return None

    norm_weight, norm_bias, norm_offload_stream = cast_bias_weight(
        norm, x, offloadable=True, compute_dtype=x.dtype
    )
    mutation_started = False
    try:
        if (
            not _preflight_eligible(
                owner, linear, norm, x, other, gate, scale, shift, row_ids
            )
            or not _cast_norm_weight_eligible(norm_weight, norm_bias, x)
            or not _storage_disjoint(x, norm_weight, input_scale)
            or _bridge_state(linear) != "complete"
            or _static_scale(linear, x) is not input_scale
        ):
            return None
        mutation_started = True
        qdata, block_scale = quantize(
            x,
            other,
            gate,
            norm_weight,
            scale,
            shift,
            row_ids,
            input_scale,
            float(eps),
        )
    finally:
        uncast_bias_weight(norm, norm_weight, norm_bias, norm_offload_stream)

    # Native success has changed x and incremented its version. Every failure
    # from here must surface; returning None would apply the residual twice.
    if not mutation_started:
        raise RuntimeError("static gate mutation state was not entered")
    if not _quantized_outputs_eligible(qdata, block_scale, x):
        raise RuntimeError("static gate producer returned invalid NVFP4 outputs")

    weight, bias, offload_stream = cast_bias_weight(
        linear, x, offloadable=True, compute_dtype=x.dtype, want_requant=True
    )
    try:
        if (
            _bridge_state(linear) != "complete"
            or getattr(owner, "fc1", None) is not linear
            or not parent_eligible(owner)
            or not _norm_eligible(norm, x)
            or not _linear_eligible(linear, x)
            or _static_scale(linear, x) is not input_scale
            or not isinstance(weight, QuantizedTensor)
            or weight._layout_cls != NVFP4_LAYOUT
            or getattr(weight._params, "transposed", False)
        ):
            raise RuntimeError(
                "FC1 eligibility/hook/owner/weight changed after static gate mutation"
            )
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


def try_linear(
    owner,
    linear,
    norm,
    x,
    other,
    gate,
    scale,
    shift,
    row_ids,
    row_ids_token,
    mode,
    modulation_mode,
):
    """Prefer static bridge; delegate dynamic-v2 only when bridge is absent."""
    result = try_static_linear(
        owner,
        linear,
        norm,
        x,
        other,
        gate,
        scale,
        shift,
        row_ids,
        row_ids_token,
        mode,
        modulation_mode,
    )
    if result is not NO_STATIC_BRIDGE:
        return result
    return dynamic_gate.try_linear(
        owner,
        linear,
        norm,
        x,
        other,
        gate,
        scale,
        shift,
        row_ids,
        row_ids_token,
        mode,
        modulation_mode,
    )


def try_mlp_transition(
    owner,
    norm,
    x,
    attention_output,
    attention_gate,
    mlp_scale,
    mlp_shift,
    row_ids,
    row_ids_token,
    mode,
    modulation_mode,
    transformer_options,
):
    """Seal the mutating producer and private MLP continuation transaction."""
    raw = try_linear(
        owner,
        owner.fc1,
        norm,
        x,
        attention_output,
        attention_gate,
        mlp_scale,
        mlp_shift,
        row_ids,
        row_ids_token,
        mode,
        modulation_mode,
    )
    if raw is None:
        return None
    return owner._forward_raw(raw, transformer_options=transformer_options)


__all__ = ["try_linear", "try_mlp_transition"]
