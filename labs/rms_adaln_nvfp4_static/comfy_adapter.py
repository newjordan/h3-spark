"""Isolated H3 static RMSNorm + AdaLN -> NVFP4 routing adapter.

The certified one-pass static kernel is selected only on the existing H3
QKV/FC1 producer call sites and a live v3 ``StaticScalePlan`` bridge. Unbridged
layers delegate to the already-integrated dynamic-v2 helper. Partial, stale, or
shape-mismatched bridge state returns ``None`` for the caller's exact staged
fallback; it is never reinterpreted as dynamic eligibility.

This module is an integration candidate outside the live ComfyUI package.  Its
native loader is AOT-only and accepts one hash/ABI/build/device combination.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
from functools import lru_cache
from pathlib import Path

import torch

from comfy.ldm.minimax.adaln_nvfp4 import ROW_IDS_TOKEN
from comfy.ldm.minimax.linear_nvfp4 import _linear_eligible, _static_scale
from comfy.ldm.minimax.rms_adaln_nvfp4 import (
    CERTIFIED_H3_RMS_EPS,
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


LIBRARY_ENV = "COMFY_MINIMAX_H3_RMS_ADALN_NVFP4_STATIC_LIBRARY"
NATIVE_MODULE_NAME = "h3_rms_adaln_nvfp4_static_native_v1"
NATIVE_ABI_VERSION = 1
NATIVE_BUILD_ID = "h3-rms-adaln-nvfp4-static-register-sm12-v1"
NATIVE_SHA256 = "6cd7c9714bef48ab80dd9afd83f1cbc611403326ad59b0d49cf4d53d55b507b3"
CERTIFIED_CUDA_VERSION = "13.0"
CERTIFIED_DEVICE_CAPABILITY = (12, 1)

# Only the two producer edges whose RMSNorm/AdaLN bytes were certified. The
# unchanged model.py call sites separately bind these pairs to the exact
# Attention.qkv_proj / MLP.fc1 owners before passing ROW_IDS_TOKEN.
CERTIFIED_LINEAR_SHAPES = frozenset({
    (5376, 21504),
    (5376, 28672),
})
NO_STATIC_ROUTE = object()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _native_abi_eligible(module) -> bool:
    abi = getattr(module, "abi_version", None)
    build = getattr(module, "build_id", None)
    return (
        callable(abi)
        and abi() == NATIVE_ABI_VERSION
        and callable(build)
        and build() == NATIVE_BUILD_ID
        and callable(getattr(module, "rms_adaln_nvfp4_static", None))
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
    """Load one prebuilt, hash-pinned Spark binary; never compile or JIT."""
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
            raise ImportError("native module has an incompatible static RMS ABI")
        return module
    except (ImportError, OSError, RuntimeError) as error:
        logging.warning("MiniMax H3 static RMS+AdaLN NVFP4 fusion disabled: %s", error)
        return None


def _bridge_state(linear) -> str:
    """Return absent, partial, or complete without reading bridge values."""
    present = tuple(hasattr(linear, name) for name in BRIDGE_ATTRS)
    if not any(present):
        return "absent"
    return "complete" if all(present) else "partial"


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


def _try_static_linear(
    linear,
    norm,
    x,
    scale,
    shift,
    row_ids,
    input_scale,
):
    native = _load_native()
    quantize = getattr(native, "rms_adaln_nvfp4_static", None)
    if not callable(quantize):
        return None
    eps = getattr(norm, "eps", None)
    if eps is None:
        eps = torch.finfo(x.dtype).eps

    run_every_op()
    if (
        _bridge_state(linear) != "complete"
        or not _norm_eligible(norm, x)
        or not _linear_eligible(linear, x)
        or _static_scale(linear, x) is not input_scale
    ):
        return None
    norm_weight, norm_bias, norm_offload_stream = cast_bias_weight(
        norm,
        x,
        offloadable=True,
        compute_dtype=x.dtype,
    )
    try:
        if (
            not _cast_norm_weight_eligible(norm_weight, norm_bias, x)
            or _bridge_state(linear) != "complete"
            or not _norm_eligible(norm, x)
            or not _linear_eligible(linear, x)
            or _static_scale(linear, x) is not input_scale
        ):
            return None
        # Complete quantization before casting the Linear. DynamicVRAM/AIMDO
        # may reuse the same asynchronous cast buffer for the next module.
        qdata, block_scale = quantize(
            x,
            norm_weight,
            scale,
            shift,
            row_ids,
            input_scale,
            float(eps),
        )
    finally:
        uncast_bias_weight(
            norm,
            norm_weight,
            norm_bias,
            norm_offload_stream,
        )

    # A native launch never mutates x.  Unexpected outputs fail to the caller's
    # staged path, but launch/runtime exceptions propagate rather than risk
    # replay after an asynchronous CUDA failure.
    if not _quantized_outputs_eligible(qdata, block_scale, x):
        return None

    weight, bias, offload_stream = cast_bias_weight(
        linear,
        x,
        offloadable=True,
        compute_dtype=x.dtype,
        want_requant=True,
    )
    try:
        if (
            _bridge_state(linear) != "complete"
            or not _norm_eligible(norm, x)
            or not _linear_eligible(linear, x)
            or _static_scale(linear, x) is not input_scale
            or not isinstance(weight, QuantizedTensor)
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


def try_static_route(
    linear,
    norm,
    x,
    scale,
    shift,
    row_ids,
    row_ids_token,
    mode,
    modulation_mode,
):
    """Return a static result, ``None`` fallback, or ``NO_STATIC_ROUTE``.

    ``NO_STATIC_ROUTE`` means no v3 bridge exists and dynamic-v2 retains the
    call. Any bridge state claims the call: malformed metadata, guard failure,
    unavailable AOT code, or a post-cast mismatch returns ``None`` for the
    hook-observable staged path instead of crossing scaling contracts.
    """
    bridge_state = _bridge_state(linear)
    if bridge_state == "absent":
        return NO_STATIC_ROUTE
    if bridge_state != "complete":
        return None
    if (
        mode != "auto"
        or modulation_mode != "exact"
        or row_ids_token is not ROW_IDS_TOKEN
        or (
            getattr(linear, "in_features", None),
            getattr(linear, "out_features", None),
        ) not in CERTIFIED_LINEAR_SHAPES
        or not _tensor_inputs_eligible(linear, x, scale, shift, row_ids)
        or not _spark_device_eligible(getattr(x, "device", None))
        or not _norm_eligible(norm, x)
        or not _linear_eligible(linear, x)
    ):
        return None

    input_scale = _static_scale(linear, x)
    if input_scale is None:
        return None
    return _try_static_linear(
        linear,
        norm,
        x,
        scale,
        shift,
        row_ids,
        input_scale,
    )


__all__ = ["NO_STATIC_ROUTE", "try_static_route"]
