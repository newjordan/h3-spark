"""Fail-closed H3 attention-gate -> RMS/AdaLN -> FC1 NVFP4 adapter."""

from __future__ import annotations

import importlib.util
import logging
import os
from functools import lru_cache
from pathlib import Path

import torch

from comfy.ldm.minimax.adaln_nvfp4 import ROW_IDS_TOKEN, parent_eligible
from comfy.ldm.minimax.linear_nvfp4 import _linear_eligible
from comfy.ldm.minimax.rms_adaln_nvfp4 import (
    CERTIFIED_TORCH_GIT_VERSION,
    CERTIFIED_TORCH_VERSION,
    _cast_norm_weight_eligible,
    _norm_eligible,
    _tensor_inputs_eligible,
)
from comfy.ldm.minimax.swiglu_nvfp4 import NVFP4_LAYOUT, _dynamic_eligible
from comfy.ops import cast_bias_weight, run_every_op, uncast_bias_weight
from comfy.quant_ops import QuantizedTensor, TensorCoreNVFP4Layout


LIBRARY_ENV = "COMFY_MINIMAX_H3_GATE_RMS_ADALN_NVFP4_LIBRARY"
NATIVE_ABI_VERSION = 2
NATIVE_BUILD_ID = (
    "h3-gate-rms-adaln-nvfp4-torch2.13-cf30153-sm120-gb10-safe-v2"
)
NATIVE_MODULE_NAME = "h3_gate_rms_adaln_nvfp4_native_v2"
CERTIFIED_CUDA_VERSION = "13.0"
CERTIFIED_DEVICE_CAPABILITY = (12, 1)
FC1_PAIR = (5376, 28672)


def _native_abi_eligible(module) -> bool:
    abi = getattr(module, "abi_version", None)
    build_id = getattr(module, "build_id", None)
    return (
        callable(abi)
        and abi() == NATIVE_ABI_VERSION
        and callable(build_id)
        and build_id() == NATIVE_BUILD_ID
        and callable(getattr(module, "gate_rms_adaln_nvfp4_dynamic", None))
    )


@lru_cache(maxsize=1)
def _load_native():
    if (
        torch.__version__ != CERTIFIED_TORCH_VERSION
        or torch.version.git_version != CERTIFIED_TORCH_GIT_VERSION
        or torch.version.cuda != CERTIFIED_CUDA_VERSION
        or not torch.cuda.is_available()
    ):
        return None
    path = Path(
        os.environ.get(
            LIBRARY_ENV,
            Path(__file__).with_name(f"{NATIVE_MODULE_NAME}.so"),
        )
    ).expanduser()
    if not path.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location(NATIVE_MODULE_NAME, path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not _native_abi_eligible(module):
            raise ImportError("incompatible gate/RMS native ABI")
        return module
    except (ImportError, OSError, RuntimeError) as error:
        logging.warning("MiniMax H3 gate/RMS fusion disabled: %s", error)
        return None


def _spark_device_eligible(device) -> bool:
    """Restrict this qualified build to the GB10 device used by the gate."""
    try:
        return (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(device)
            == CERTIFIED_DEVICE_CAPABILITY
        )
    except (AssertionError, RuntimeError, TypeError, ValueError):
        return False


def _storage_disjoint(x, *others) -> bool:
    """Fail closed when the writable residual may share storage."""
    if not isinstance(x, torch.Tensor) or any(
        not isinstance(other, torch.Tensor) for other in others
    ):
        return False
    try:
        return all(not torch._C._overlaps(x, other) for other in others)
    except (AttributeError, RuntimeError, TypeError):
        return False


def _gate_inputs_eligible(x, other, gate, scale, shift, row_ids) -> bool:
    return (
        isinstance(x, torch.Tensor)
        and isinstance(other, torch.Tensor)
        and other.shape == x.shape
        and other.device == x.device
        and other.dtype == torch.bfloat16
        and other.is_contiguous()
        and not other.requires_grad
        and isinstance(gate, torch.Tensor)
        and isinstance(scale, torch.Tensor)
        and isinstance(shift, torch.Tensor)
        and isinstance(row_ids, torch.Tensor)
        and gate.ndim == 2
        and gate.shape == scale.shape
        and gate.device == x.device
        # The production AdaLN curve path is FP32.  Keep this producer fusion
        # narrower than the reusable RMS/AdaLN kernel until BF16 modulation is
        # independently qualified at this exact gate boundary.
        and scale.dtype == torch.float32
        and gate.dtype == scale.dtype
        and gate.stride(-1) == 1
        and not gate.requires_grad
        and _storage_disjoint(x, other, gate, scale, shift, row_ids)
    )


def _preflight_eligible(owner, linear, norm, x, other, gate, scale, shift, row_ids):
    """Re-runnable guard set for the last safe fallback boundary."""
    return (
        getattr(owner, "fc1", None) is linear
        and parent_eligible(owner)
        and (
            getattr(linear, "in_features", None),
            getattr(linear, "out_features", None),
        )
        == FC1_PAIR
        and _tensor_inputs_eligible(linear, x, scale, shift, row_ids)
        and _spark_device_eligible(x.device)
        and _gate_inputs_eligible(x, other, gate, scale, shift, row_ids)
        and _norm_eligible(norm, x)
        and _linear_eligible(linear, x)
        and _dynamic_eligible(linear)
    )


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
    """Return fused FC1 output/None; after x mutation all failures raise."""
    if (
        mode != "auto"
        or modulation_mode != "exact"
        or row_ids_token is not ROW_IDS_TOKEN
        or not _preflight_eligible(
            owner, linear, norm, x, other, gate, scale, shift, row_ids
        )
    ):
        return None
    native = _load_native()
    quantize = getattr(native, "gate_rms_adaln_nvfp4_dynamic", None)
    if not callable(quantize):
        return None
    eps = getattr(norm, "eps", None)
    if eps is None:
        eps = torch.finfo(x.dtype).eps

    run_every_op()
    # run_every_op() may execute deferred model-management callbacks. Recheck
    # every owner/module/input guard before the final safe fallback boundary.
    if not _preflight_eligible(
        owner, linear, norm, x, other, gate, scale, shift, row_ids
    ):
        return None
    norm_weight, norm_bias, norm_offload_stream = cast_bias_weight(
        norm, x, offloadable=True, compute_dtype=x.dtype
    )
    mutation_started = False
    try:
        if (
            not _cast_norm_weight_eligible(norm_weight, norm_bias, x)
            or not _storage_disjoint(x, norm_weight)
        ):
            return None
        mutation_started = True
        qdata, block_scale, input_scale, _rstd = quantize(
            x, other, gate, norm_weight, scale, shift, row_ids, float(eps)
        )
    finally:
        uncast_bias_weight(norm, norm_weight, norm_bias, norm_offload_stream)

    # x now contains the exact gated residual. Never report a stock fallback
    # after this point: any unexpected runtime state must surface as an error.
    if not mutation_started:
        raise RuntimeError("gate/RMS mutation state was not entered")
    weight, bias, offload_stream = cast_bias_weight(
        linear, x, offloadable=True, compute_dtype=x.dtype, want_requant=True
    )
    try:
        if (
            not isinstance(weight, QuantizedTensor)
            or weight._layout_cls != NVFP4_LAYOUT
            or getattr(weight._params, "transposed", False)
        ):
            raise RuntimeError("FC1 weight changed after gate/RMS preflight")
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
    """Fuse the in-block attention residual into the FC1 producer.

    ``None`` is returned only before ``x`` can be mutated.  Once the native
    producer succeeds, the private MLP continuation is mandatory and all
    errors propagate; falling back would apply the residual gate twice.
    """
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
