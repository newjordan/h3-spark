"""Lazy loader and Python facade for the experimental native CUDA kernel."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch


_HERE = Path(__file__).resolve().parent
_SOURCE = _HERE / "swiglu_nvfp4_cuda.cu"
_BUILD = _HERE / "build"
_EXTENSION_NAME = "h3_swiglu_nvfp4_native_v1"
_LIBRARY_ENV = "H3_SWIGLU_NVFP4_LIBRARY"
_REQUIRED_ENTRYPOINTS = (
    "bf16_nvfp4",
    "bf16_nvfp4_dynamic",
    "bf16_nvfp4_from_amax_bits",
    "swiglu_nvfp4",
    "swiglu_nvfp4_dynamic",
)
H3_NVFP4_WIDTHS = frozenset((5376, 7168, 14336))


@lru_cache(maxsize=1)
def load_extension(*, verbose: bool = False):
    """Compile/load the extension on first use, keeping artifacts local."""
    import importlib.util

    configured_library = os.environ.get(_LIBRARY_ENV)
    cmake_library = (
        Path(configured_library).expanduser()
        if configured_library
        else _HERE / "cmake-build-120" / (_EXTENSION_NAME + ".so")
    )
    if cmake_library.is_file():
        spec = importlib.util.spec_from_file_location(
            _EXTENSION_NAME, cmake_library
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {cmake_library}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _check_entrypoints(module, cmake_library)
        return module
    if configured_library:
        raise FileNotFoundError(
            f"{_LIBRARY_ENV} does not name a file: {cmake_library}"
        )

    from torch.utils.cpp_extension import load

    _BUILD.mkdir(parents=True, exist_ok=True)
    # CUDA 13 identifies GB10 as compute capability 12.1. The PyTorch 2.13
    # wheel currently advertises sm_120 as its newest named target; 12.0 PTX is
    # forward-compatible on GB10. Respect an explicit caller override.
    prior_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if prior_arch_list is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0+PTX"
    try:
        module = load(
            name=_EXTENSION_NAME,
            sources=[str(_SOURCE)],
            build_directory=str(_BUILD),
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "--expt-relaxed-constexpr",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            ],
            verbose=verbose,
        )
    finally:
        if prior_arch_list is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
    _check_entrypoints(module, _SOURCE)
    return module


def _check_entrypoints(module, source: Path) -> None:
    missing = [
        name for name in _REQUIRED_ENTRYPOINTS
        if not callable(getattr(module, name, None))
    ]
    if missing:
        raise ImportError(
            f"native H3 extension {source} is missing entrypoints: {missing}"
        )


def swiglu_nvfp4(
    raw: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    eager_bf16_rounding: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse H3 SwiGLU and NVFP4 quantization.

    Args:
        raw: Contiguous BF16 CUDA tensor shaped ``[S, 28672]``. Columns
            ``[:14336]`` are the gate and columns ``[14336:]`` are the up arm.
        global_scale: Contiguous one-element FP32 CUDA tensor using
            comfy-kitchen's NVFP4 global decode-scale convention.
        eager_bf16_rounding: Preserve the two BF16 roundings made by
            ``F.silu(gate).mul_(up)`` before quantization. Disable only to
            investigate direct FP32 activation-to-NVFP4 behavior.

    Returns:
        ``(packed, block_scales)``. Packed data is uint8 with shape
        ``[ceil(S/16)*16, 7168]``. Scales are float8_e4m3fn in the cuBLAS
        swizzled layout with shape ``[ceil(S/128)*128, 896]``.
    """
    extension = load_extension()
    packed, scale_bytes = extension.swiglu_nvfp4(
        raw, global_scale, eager_bf16_rounding
    )
    return packed, scale_bytes.view(torch.float8_e4m3fn)


def swiglu_nvfp4_dynamic(
    raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse eager-equivalent SwiGLU, dynamic scale reduction, and NVFP4.

    The first CUDA pass computes the BF16-rounded activation amax and preserves
    ComfyUI ordering: (BF16 amax / 2688.0).to(float32). The second pass
    recomputes SwiGLU and emits packed data plus swizzled E4M3 block scales.

    Returns:
        A tuple of packed data, block scales, and the FP32 global scale.
    """
    extension = load_extension()
    packed, scale_bytes, global_scale = extension.swiglu_nvfp4_dynamic(raw)
    return packed, scale_bytes.view(torch.float8_e4m3fn), global_scale


def bf16_nvfp4(
    input: torch.Tensor,
    global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize one H3 BF16 activation using a supplied NVFP4 scale.

    This direct path is deliberately width-gated to H3's QKV/FC1 input
    (5376), attention-output input (7168), and FC2 input (14336). It emits the
    exact packed representation consumed by comfy-kitchen's NVFP4 GEMM.
    """
    _check_direct_input(input)
    _check_direct_scale(input, global_scale)
    extension = load_extension()
    packed, scale_bytes = extension.bf16_nvfp4(input, global_scale)
    return packed, scale_bytes.view(torch.float8_e4m3fn)


def bf16_nvfp4_dynamic(
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dynamically quantize one H3 BF16 activation in two CUDA passes.

    The reduction reads the source directly and stores only one atomic BF16
    maximum. The quantization pass derives the stock BF16-rounded global scale
    while emitting packed FP4 and block scales, avoiding ``input.abs()``'s
    full-size temporary and a separate scale-finalization launch.
    """
    _check_direct_input(input)
    extension = load_extension()
    packed, scale_bytes, global_scale = extension.bf16_nvfp4_dynamic(input)
    return packed, scale_bytes.view(torch.float8_e4m3fn), global_scale


def bf16_nvfp4_from_amax_bits(
    input: torch.Tensor,
    max_bits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize using exact BF16 magnitude bits reduced by the producer.

    ``max_bits`` must be a one-element int32 CUDA tensor containing
    ``max(abs(input).view(bfloat16_bits))``. The packing and global-scale
    rounding are otherwise identical to :func:`bf16_nvfp4_dynamic`.
    """
    _check_direct_input(input)
    if not isinstance(max_bits, torch.Tensor):
        raise TypeError("max_bits must be a torch.Tensor")
    if max_bits.dtype != torch.int32:
        raise TypeError("max_bits must have dtype torch.int32")
    if max_bits.numel() != 1:
        raise ValueError("max_bits must contain exactly one value")
    if not max_bits.is_cuda or max_bits.device != input.device:
        raise ValueError("max_bits must be on the input CUDA device")
    if not max_bits.is_contiguous():
        raise ValueError("max_bits must be contiguous")
    extension = load_extension()
    packed, scale_bytes, global_scale = extension.bf16_nvfp4_from_amax_bits(
        input, max_bits
    )
    return packed, scale_bytes.view(torch.float8_e4m3fn), global_scale


def _check_direct_input(input: torch.Tensor) -> None:
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")
    if input.ndim != 2 or input.shape[1] not in H3_NVFP4_WIDTHS:
        raise ValueError(
            "input must have shape [S, K] with H3 width "
            f"{sorted(H3_NVFP4_WIDTHS)}, got {tuple(input.shape)}"
        )
    if input.shape[0] <= 0:
        raise ValueError("input must contain at least one row")
    if input.dtype != torch.bfloat16:
        raise TypeError(f"input must have dtype torch.bfloat16, got {input.dtype}")
    if not input.is_cuda:
        raise ValueError("input must be a CUDA tensor")
    if not input.is_contiguous():
        raise ValueError("input must be contiguous")


def _check_direct_scale(
    input: torch.Tensor,
    global_scale: torch.Tensor,
) -> None:
    """Mirror the native supplied-scale contract before loading the extension."""
    if not isinstance(global_scale, torch.Tensor):
        raise TypeError("global_scale must be a torch.Tensor")
    if global_scale.dtype != torch.float32:
        raise TypeError(
            "global_scale must have dtype torch.float32, "
            f"got {global_scale.dtype}"
        )
    if global_scale.numel() != 1:
        raise ValueError("global_scale must contain exactly one value")
    if not global_scale.is_cuda:
        raise ValueError("global_scale must be a CUDA tensor")
    if not global_scale.is_contiguous():
        raise ValueError("global_scale must be contiguous")
    if global_scale.device != input.device:
        raise ValueError("input and global_scale must be on the same CUDA device")


__all__ = [
    "H3_NVFP4_WIDTHS",
    "bf16_nvfp4",
    "bf16_nvfp4_dynamic",
    "bf16_nvfp4_from_amax_bits",
    "load_extension",
    "swiglu_nvfp4",
    "swiglu_nvfp4_dynamic",
]
