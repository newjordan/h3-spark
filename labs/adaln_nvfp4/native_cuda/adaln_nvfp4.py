"""Python facade for the isolated H3 AdaLN-to-NVFP4 CUDA prototype."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch


WIDTH = 5376
_HERE = Path(__file__).resolve().parent
_SOURCE = _HERE / "adaln_nvfp4_cuda.cu"
_EXTENSION_NAME = "h3_adaln_nvfp4_native_v1"
_LIBRARY_ENV = "H3_ADALN_NVFP4_LIBRARY"
_REQUIRED_ENTRYPOINTS = ("adaln_nvfp4", "adaln_nvfp4_dynamic")


@lru_cache(maxsize=1)
def load_extension(*, verbose: bool = False):
    """Load the CMake binary, falling back to a local JIT research build."""
    import importlib.util

    configured = os.environ.get(_LIBRARY_ENV)
    library = (
        Path(configured).expanduser()
        if configured
        else _HERE / "cmake-build-120" / (_EXTENSION_NAME + ".so")
    )
    if library.is_file():
        spec = importlib.util.spec_from_file_location(_EXTENSION_NAME, library)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {library}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _check_entrypoints(module, library)
        return module
    if configured:
        raise FileNotFoundError(f"{_LIBRARY_ENV} does not name a file: {library}")

    from torch.utils.cpp_extension import load

    build = _HERE / "build"
    build.mkdir(parents=True, exist_ok=True)
    prior_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if prior_arch_list is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0+PTX"
    try:
        module = load(
            name=_EXTENSION_NAME,
            sources=[str(_SOURCE)],
            build_directory=str(build),
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
        raise ImportError(f"native H3 extension {source} is missing: {missing}")


def _check_inputs(
    normed: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    row_ids: torch.Tensor,
) -> None:
    if not isinstance(normed, torch.Tensor):
        raise TypeError("normed must be a torch.Tensor")
    if normed.ndim != 2 or normed.shape[1] != WIDTH or normed.shape[0] <= 0:
        raise ValueError(f"normed must have nonempty shape [S, {WIDTH}]")
    if normed.dtype != torch.bfloat16:
        raise TypeError("normed must have dtype torch.bfloat16")
    if not normed.is_cuda:
        raise ValueError("normed must be a CUDA tensor")
    if not normed.is_contiguous():
        raise ValueError("normed must be contiguous")

    if not isinstance(scale, torch.Tensor) or not isinstance(shift, torch.Tensor):
        raise TypeError("scale and shift must be torch.Tensor instances")
    if scale.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("scale must have dtype torch.bfloat16 or torch.float32")
    if shift.dtype != scale.dtype:
        raise TypeError("scale and shift must have the same dtype")
    if scale.ndim != 2 or shift.shape != scale.shape:
        raise ValueError("scale and shift must have matching 2D shapes")
    if scale.shape[0] <= 0 or scale.shape[1] != WIDTH:
        raise ValueError(f"scale and shift must have shape [M, {WIDTH}]")
    if scale.stride(-1) != 1 or shift.stride(-1) != 1:
        raise ValueError("scale and shift must be contiguous in their last dimension")
    if not scale.is_cuda or not shift.is_cuda:
        raise ValueError("scale and shift must be CUDA tensors")

    if not isinstance(row_ids, torch.Tensor):
        raise TypeError("row_ids must be a torch.Tensor")
    if row_ids.dtype != torch.int32:
        raise TypeError("row_ids must have dtype torch.int32")
    if row_ids.ndim != 1 or row_ids.shape[0] != normed.shape[0]:
        raise ValueError("row_ids must have shape [S]")
    if not row_ids.is_cuda:
        raise ValueError("row_ids must be a CUDA tensor")
    if not row_ids.is_contiguous():
        raise ValueError("row_ids must be contiguous")
    if not (normed.device == scale.device == shift.device == row_ids.device):
        raise ValueError("all inputs must be on the same CUDA device")


def _check_global_scale(normed: torch.Tensor, global_scale: torch.Tensor) -> None:
    if not isinstance(global_scale, torch.Tensor):
        raise TypeError("global_scale must be a torch.Tensor")
    if global_scale.dtype != torch.float32:
        raise TypeError("global_scale must have dtype torch.float32")
    if global_scale.numel() != 1:
        raise ValueError("global_scale must contain exactly one value")
    if not global_scale.is_cuda:
        raise ValueError("global_scale must be a CUDA tensor")
    if not global_scale.is_contiguous():
        raise ValueError("global_scale must be contiguous")
    if global_scale.device != normed.device:
        raise ValueError("normed and global_scale must be on the same CUDA device")


def adaln_nvfp4(
    normed: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    row_ids: torch.Tensor,
    global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply exact BF16 segmented AdaLN and quantize with a supplied scale."""
    _check_inputs(normed, scale, shift, row_ids)
    _check_global_scale(normed, global_scale)
    packed, scale_bytes = load_extension().adaln_nvfp4(
        normed, scale, shift, row_ids, global_scale
    )
    return packed, scale_bytes.view(torch.float8_e4m3fn)


def adaln_nvfp4_dynamic(
    normed: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    row_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply exact BF16 segmented AdaLN and dynamically quantize in two passes."""
    _check_inputs(normed, scale, shift, row_ids)
    packed, scale_bytes, global_scale = load_extension().adaln_nvfp4_dynamic(
        normed, scale, shift, row_ids
    )
    return packed, scale_bytes.view(torch.float8_e4m3fn), global_scale


def eager_adaln_reference(
    normed: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    row_ids: torch.Tensor,
) -> torch.Tensor:
    """Materialized reference with the same BF16 boundaries as H3 exact mode."""
    selected_scale = scale.index_select(0, row_ids.to(torch.int64)).to(torch.bfloat16)
    selected_shift = shift.index_select(0, row_ids.to(torch.int64)).to(torch.bfloat16)
    output = normed.clone()
    output.mul_(1.0 + selected_scale).add_(selected_shift)
    return output


__all__ = [
    "WIDTH",
    "adaln_nvfp4",
    "adaln_nvfp4_dynamic",
    "eager_adaln_reference",
    "load_extension",
]
