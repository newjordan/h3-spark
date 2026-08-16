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
    "swiglu_nvfp4",
    "swiglu_nvfp4_dynamic",
    "swiglu_nvfp4_dynamic_oneshot",
    "swiglu_nvfp4_dynamic_fused",
    "swiglu_nvfp4_dynamic_twolevel",
    "swiglu_nvfp4_dynamic_interval",
    "swiglu_winner_lu",
    "swiglu_nvfp4_static_rebind",
    "swiglu_nvfp4_dynamic_coop",
    "swiglu_nvfp4_dynamic_vec",
    "swiglu_nvfp4_dynamic_vec16",
    "swiglu_nvfp4_dynamic_inplace",
    "swiglu_nvfp4_dynamic_inplace_prod",
    "fc1_epilogue_assoc_mismatches",
    "fc1_paired_store",
    "swiglu_nvfp4_dynamic_swizzle",
    "fc1_paired_wmma",
    "fc1_paired_nvfp4",
    "fc1_paired_nvfp4_tiled",
    "fc1_paired_nvfp4_scaled",
    "fc1_paired_nvfp4_scaled_piped",
    "fc1_paired_nvfp4_scaled_tma",
    "fc1_paired_nvfp4_scaled_tma_sf",
    "fc1_paired_nvfp4_scaled_tma256",
    "fc1_paired_nvfp4_scaled_tma256_sw",
    "fc1_paired_nvfp4_scaled_tma256k2",
    "fc1_paired_nvfp4_scaled_tma256k2_amax",
    "fc1_paired_nvfp4_scaled_tma256k2_ldmb",
    "fc1_paired_nvfp4_scaled_tma256k2_pipe",
    "fc1_paired_nvfp4_scaled_tma256k2_leads",
    "fc1_paired_nvfp4_scaled_tma256k2_pipea",
    "fc1_paired_nvfp4_scaled_tma256k2s1",
    "fc1_paired_nvfp4_scaled_tma256k2s1_attrs",
    "fc1_paired_nvfp4_scaled_tma256n32",
    "fc1_paired_nvfp4_scaled_tma256n32_attrs",
    "fc1_paired_nvfp4_scaled_tma128k2",
    "fc1_paired_nvfp4_scaled_tma128k2_attrs",
    "fc1_paired_nvfp4_scaled_tma256k2_sw",
    "fc1_paired_nvfp4_scaled_tma128k4",
    "fc1_paired_nvfp4_scaled_tma128k2n",
    "fc1_paired_nvfp4_scaled_tma256k4",
    "fc1_paired_nvfp4_scaled_tma256k2n2",
    "fc1_paired_nvfp4_scaled_tma256k2ws",
    "fc1_paired_nvfp4_scaled_tma256k2ws4",
    "fc1_paired_nvfp4_scaled_tma256k2ws4_attrs",
    "fc1_paired_nvfp4_scaled_tma256k2p",
    "fc1_nvfp4_scaled_tma128n128k4",
    "fc1_nvfp4_scaled_tma256k4n1",
    "fc1_nvfp4_scaled_tma128n128k4_pipe",
    "fc1_nvfp4_scaled_tma128n128k4_ldm",
    "fc1_nvfp4_scaled_tma128n128k4_sw",
    "fc1_nvfp4_scaled_tma128n128k4ws",
    "fc1_nvfp4_scaled_tma128n128k4ws_attrs",
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
    certify_live_amax(raw, global_scale)
    return packed, scale_bytes.view(torch.float8_e4m3fn), global_scale


def swiglu_nvfp4_dynamic_oneshot(
    raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-SwiGLU dynamic producer: store activation during amax, then pack.

    Algebra is unchanged: BF16 amax bits and ``(amax / 2688).to(float32)``
    still come from the eager-rounded SwiGLU. The pack pass reads the stored
    activation instead of recomputing SiLU.
    """
    extension = load_extension()
    packed, scale_bytes, global_scale = extension.swiglu_nvfp4_dynamic_oneshot(
        raw
    )
    return packed, scale_bytes.view(torch.float8_e4m3fn), global_scale


def swiglu_nvfp4_dynamic_fused(
    raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dynamic producer: bound-filtered amax, then 8-wide pack from amax bits.

    Algebra is unchanged: BF16 amax bits and ``(amax / 2688).to(float32)``
    still come from the eager-rounded SwiGLU. Exact SiLU is skipped only
    when ``ru_bf16(|silu| upper bound * |up|)`` cannot change the bit-max.
    Pack recomputes SwiGLU; no 558 MiB activation store.
    """
    extension = load_extension()
    packed, scale_bytes, global_scale = extension.swiglu_nvfp4_dynamic_fused(
        raw
    )
    return packed, scale_bytes.view(torch.float8_e4m3fn), global_scale


def swiglu_nvfp4_dynamic_twolevel(
    raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dynamic producer: two-level bound-argmax amax, then pack from bits.

    Pass 1 is SiLU-free except one exact eval per CTA at its bound-argmax,
    giving lower bound L. Pass 2 evaluates SiLU only where
    ``ru_bf16(bound) > L``. Pack recomputes SwiGLU; no 558 MiB store.
    """
    extension = load_extension()
    packed, scale_bytes, global_scale, _exact = (
        extension.swiglu_nvfp4_dynamic_twolevel(raw)
    )
    return packed, scale_bytes.view(torch.float8_e4m3fn), global_scale


def swiglu_nvfp4_dynamic_twolevel_stats(
    raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Same as ``swiglu_nvfp4_dynamic_twolevel`` plus sparse exact-SiLU count."""
    extension = load_extension()
    packed, scale_bytes, global_scale, exact = (
        extension.swiglu_nvfp4_dynamic_twolevel(raw)
    )
    return (
        packed,
        scale_bytes.view(torch.float8_e4m3fn),
        global_scale,
        int(exact.item()),
    )


def swiglu_nvfp4_dynamic_interval(
    raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dynamic producer: skip sparse amax scan when G(L)==G(U).

    ``G(x)=(bf16(x)/2688).to(bf16)`` is nondecreasing. If the bound-argmax
    exact L and the tensor-wide bound U share G, every amax in ``[L, U]``
    yields the same stock global scale, so the second HBM scan is a no-op.
    """
    extension = load_extension()
    packed, scale_bytes, global_scale, _exact, _scanned, _l, _u = (
        extension.swiglu_nvfp4_dynamic_interval(raw)
    )
    return packed, scale_bytes.view(torch.float8_e4m3fn), global_scale


def swiglu_winner_lu(raw: torch.Tensor) -> tuple[int, int]:
    """Bound-argmax exact L and tensor-wide bound U abs-bits."""
    extension = load_extension()
    lo, hi = extension.swiglu_winner_lu(raw)
    return int(lo.item()), int(hi.item())


def swiglu_nvfp4_dynamic_interval_stats(
    raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, bool, int, int]:
    """Interval producer plus sparse count, scan flag, and L/U abs-bits."""
    extension = load_extension()
    packed, scale_bytes, global_scale, exact, scanned, lo, hi = (
        extension.swiglu_nvfp4_dynamic_interval(raw)
    )
    return (
        packed,
        scale_bytes.view(torch.float8_e4m3fn),
        global_scale,
        int(exact.item()),
        bool(int(scanned.item())),
        int(lo.item()),
        int(hi.item()),
    )


STATIC_REBIND_ROWS = 20423
STATIC_REBIND_WIDTH = 28672
LIVE_AMAX_SOURCE = "live_amax"

# raw storage key -> id(scale). Shape-only rebind is not a source bind:
# a foreign G with the same [S, 28672] shape still packs. live_amax
# records the scale object produced by the two-pass amax of this storage.
_LIVE_AMAX_CERTS: dict[tuple, int] = {}


def _raw_source_key(raw: torch.Tensor) -> tuple:
    return (
        raw.untyped_storage().data_ptr(),
        int(raw.shape[0]),
        int(raw.shape[1]),
        str(raw.dtype),
        str(raw.device),
    )


def certify_live_amax(raw: torch.Tensor, scale: torch.Tensor) -> None:
    """Bind ``scale`` as the live (amax/2688).to(bf16).to(fp32) of ``raw``."""
    if not isinstance(raw, torch.Tensor) or not isinstance(scale, torch.Tensor):
        raise TypeError("raw and scale must be tensors")
    _LIVE_AMAX_CERTS[_raw_source_key(raw)] = id(scale)


def clear_live_amax_certs() -> None:
    _LIVE_AMAX_CERTS.clear()


def _static_rebind_fingerprint(
    raw: torch.Tensor,
    global_scale: torch.Tensor,
    certified_scale: torch.Tensor | None = None,
    *,
    source: str | None = None,
) -> None:
    """Fail-closed shape/source fingerprint. Not a margin."""
    if not isinstance(raw, torch.Tensor):
        raise TypeError("raw must be a torch.Tensor")
    if raw.ndim != 2 or raw.shape[1] != STATIC_REBIND_WIDTH:
        raise ValueError(
            f"raw must have shape [S, {STATIC_REBIND_WIDTH}], got {tuple(raw.shape)}"
        )
    if raw.shape[0] <= 0:
        raise ValueError("raw must contain at least one row")
    if raw.dtype != torch.bfloat16:
        raise TypeError(f"raw must have dtype torch.bfloat16, got {raw.dtype}")
    if not raw.is_cuda:
        raise ValueError("raw must be a CUDA tensor")
    if not raw.is_contiguous():
        raise ValueError("raw must be contiguous")
    if raw.requires_grad:
        raise ValueError("raw must not require grad")
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
    if global_scale.device != raw.device:
        raise ValueError("raw and global_scale must be on the same CUDA device")
    if global_scale.requires_grad:
        raise ValueError("global_scale must not require grad")
    if certified_scale is not None and certified_scale is not global_scale:
        raise ValueError("global_scale is not the certified scale object")
    if source is None:
        return
    if source != LIVE_AMAX_SOURCE:
        raise ValueError(
            f"unknown static-rebind source {source!r}; "
            f"only {LIVE_AMAX_SOURCE!r} is a byte-exact skip"
        )
    if certified_scale is None:
        raise ValueError("live_amax source requires certified_scale")
    if _LIVE_AMAX_CERTS.get(_raw_source_key(raw)) != id(global_scale):
        raise ValueError(
            "global_scale is not the live_amax certified for this raw"
        )


def swiglu_nvfp4_static_rebind(
    raw: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    certified_scale: torch.Tensor | None = None,
    source: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack-only SwiGLU NVFP4 using a fail-closed supplied scale.

    Skips the live amax pass. Byte-identical to the dynamic two-pass iff
    ``global_scale`` is the stock ``(amax/2688).to(bf16).to(fp32)`` for
    this ``raw``. ``source='live_amax'`` binds that scale object to this
    raw's storage (not shape-only, not a raised overflow margin).
    """
    _static_rebind_fingerprint(
        raw, global_scale, certified_scale, source=source
    )
    extension = load_extension()
    packed, scale_bytes = extension.swiglu_nvfp4_static_rebind(
        raw, global_scale
    )
    return packed, scale_bytes.view(torch.float8_e4m3fn)


def swiglu_nvfp4_dynamic_coop(
    raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-launch cooperative bound-filtered amax + 8-wide pack.

    Same algebra as ``swiglu_nvfp4_dynamic_fused``. No 558 MiB store.
    """
    extension = load_extension()
    packed, scale_bytes, global_scale, _blocks = (
        extension.swiglu_nvfp4_dynamic_coop(raw)
    )
    return packed, scale_bytes.view(torch.float8_e4m3fn), global_scale


def swiglu_nvfp4_dynamic_coop_stats(
    raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    extension = load_extension()
    packed, scale_bytes, global_scale, blocks = (
        extension.swiglu_nvfp4_dynamic_coop(raw)
    )
    return (
        packed,
        scale_bytes.view(torch.float8_e4m3fn),
        global_scale,
        int(blocks.item()),
    )


def swiglu_nvfp4_dynamic_vec(
    raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-pass producer: 8-wide exact amax, then 8-wide pack from bits.

    Every SiLU is evaluated. No bound skip, no 558 MiB store, no 1x1
    finalize launch. Algebra matches ``swiglu_nvfp4_dynamic``.
    """
    extension = load_extension()
    packed, scale_bytes, global_scale = extension.swiglu_nvfp4_dynamic_vec(raw)
    return packed, scale_bytes.view(torch.float8_e4m3fn), global_scale


def swiglu_nvfp4_dynamic_vec16(
    raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-pass: 16-wide, one thread owns one NVFP4 scale block."""
    extension = load_extension()
    packed, scale_bytes, global_scale = extension.swiglu_nvfp4_dynamic_vec16(
        raw
    )
    return packed, scale_bytes.view(torch.float8_e4m3fn), global_scale


def swiglu_nvfp4_dynamic_inplace(
    raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-SiLU producer: overwrite gate with silu(gate), then mul-pack.

    Mutates ``raw``. Lab-only. Algebra matches eager
    ``round(round(silu(g))*u)``.
    """
    extension = load_extension()
    packed, scale_bytes, global_scale = extension.swiglu_nvfp4_dynamic_inplace(
        raw
    )
    return packed, scale_bytes.view(torch.float8_e4m3fn), global_scale


def swiglu_nvfp4_dynamic_inplace_prod(
    raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-SiLU producer: overwrite up with the eager product, then pack.

    Mutates ``raw``. Lab-only. Algebra matches eager
    ``round(round(silu(g))*u)``. Pack reads only the up half.
    """
    extension = load_extension()
    packed, scale_bytes, global_scale = (
        extension.swiglu_nvfp4_dynamic_inplace_prod(raw)
    )
    return packed, scale_bytes.view(torch.float8_e4m3fn), global_scale


def fc1_epilogue_assoc_mismatches(raw: torch.Tensor) -> int:
    """Count products where fused FP32-acc SiLU disagrees with eager BF16."""
    extension = load_extension()
    return int(extension.fc1_epilogue_assoc_mismatches(raw).item())


def fc1_paired_store(
    input: torch.Tensor,
    weight: torch.Tensor,
    *,
    product: bool,
) -> torch.Tensor:
    """Paired-N FC1 MMA. ``product=True`` stores eager SwiGLU, else ``[gate|up]``."""
    extension = load_extension()
    return extension.fc1_paired_store(input, weight, product)


def fc1_paired_wmma(
    input: torch.Tensor,
    weight: torch.Tensor,
    *,
    product: bool,
) -> torch.Tensor:
    """Tensor-core paired-N FC1. Same eager epilogue as ``fc1_paired_store``."""
    extension = load_extension()
    return extension.fc1_paired_wmma(input, weight, product)


def fc1_paired_nvfp4(
    a_packed: torch.Tensor,
    w_packed: torch.Tensor,
    *,
    product: bool,
) -> torch.Tensor:
    """SM120 NVFP4 paired-N. Same eager epilogue as the WMMA path."""
    extension = load_extension()
    return extension.fc1_paired_nvfp4(a_packed, w_packed, product)


def fc1_paired_nvfp4_tiled(
    a_packed: torch.Tensor,
    w_packed: torch.Tensor,
    *,
    product: bool,
) -> torch.Tensor:
    """Persistent tiled SM120 NVFP4 paired-N. Reuses A across the N panel."""
    extension = load_extension()
    return extension.fc1_paired_nvfp4_tiled(a_packed, w_packed, product)


def fc1_paired_nvfp4_scaled(
    a_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    alpha: torch.Tensor,
    *,
    product: bool,
) -> torch.Tensor:
    """Kitchen-scale SM120 NVFP4 paired-N. Same eager epilogue as the unit-scale path."""
    extension = load_extension()
    if a_scales.dtype != torch.uint8:
        a_scales = a_scales.view(torch.uint8)
    if w_scales.dtype != torch.uint8:
        w_scales = w_scales.view(torch.uint8)
    return extension.fc1_paired_nvfp4_scaled(
        a_packed, a_scales, w_packed, w_scales, alpha, product
    )


def fc1_paired_nvfp4_scaled_piped(
    a_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    alpha: torch.Tensor,
    *,
    product: bool,
) -> torch.Tensor:
    """Pipelined kitchen-scale NVFP4 paired-N. Same atom, scale map, and epilogue."""
    extension = load_extension()
    if a_scales.dtype != torch.uint8:
        a_scales = a_scales.view(torch.uint8)
    if w_scales.dtype != torch.uint8:
        w_scales = w_scales.view(torch.uint8)
    return extension.fc1_paired_nvfp4_scaled_piped(
        a_packed, a_scales, w_packed, w_scales, alpha, product
    )


def fc1_paired_nvfp4_scaled_tma(
    a_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    alpha: torch.Tensor,
    *,
    product: bool,
) -> torch.Tensor:
    """TMA 128x128 kitchen-scale NVFP4 paired-N. Same atom, scale map, epilogue."""
    extension = load_extension()
    if a_scales.dtype != torch.uint8:
        a_scales = a_scales.view(torch.uint8)
    if w_scales.dtype != torch.uint8:
        w_scales = w_scales.view(torch.uint8)
    return extension.fc1_paired_nvfp4_scaled_tma(
        a_packed, a_scales, w_packed, w_scales, alpha, product
    )


def fc1_paired_nvfp4_scaled_tma_sf(
    a_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    alpha: torch.Tensor,
    *,
    product: bool,
) -> torch.Tensor:
    """TMA 128x128 paired-N that bulk-loads each cuBLAS 128x4 scale slab."""
    extension = load_extension()
    if a_scales.dtype != torch.uint8:
        a_scales = a_scales.view(torch.uint8)
    if w_scales.dtype != torch.uint8:
        w_scales = w_scales.view(torch.uint8)
    return extension.fc1_paired_nvfp4_scaled_tma_sf(
        a_packed, a_scales, w_packed, w_scales, alpha, product
    )


def fc1_paired_nvfp4_scaled_tma256(
    a_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    alpha: torch.Tensor,
    *,
    product: bool,
) -> torch.Tensor:
    """TMA 256x64 3-stage kitchen-scale NVFP4 paired-N. Same atom and scale map."""
    extension = load_extension()
    if a_scales.dtype != torch.uint8:
        a_scales = a_scales.view(torch.uint8)
    if w_scales.dtype != torch.uint8:
        w_scales = w_scales.view(torch.uint8)
    return extension.fc1_paired_nvfp4_scaled_tma256(
        a_packed, a_scales, w_packed, w_scales, alpha, product
    )


def fc1_paired_nvfp4_scaled_tma256_sw(
    a_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    alpha: torch.Tensor,
    *,
    product: bool,
) -> torch.Tensor:
    """TMA 256x64 3-stage with SWIZZLE_32B remapped to the PTX fragment."""
    extension = load_extension()
    if a_scales.dtype != torch.uint8:
        a_scales = a_scales.view(torch.uint8)
    if w_scales.dtype != torch.uint8:
        w_scales = w_scales.view(torch.uint8)
    return extension.fc1_paired_nvfp4_scaled_tma256_sw(
        a_packed, a_scales, w_packed, w_scales, alpha, product
    )


def fc1_paired_nvfp4_scaled_tma256k2(
    a_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    alpha: torch.Tensor,
    *,
    product: bool,
) -> torch.Tensor:
    """TMA 256x64 with a K=128 box: two m16n8k64 atoms per load."""
    extension = load_extension()
    if a_scales.dtype != torch.uint8:
        a_scales = a_scales.view(torch.uint8)
    if w_scales.dtype != torch.uint8:
        w_scales = w_scales.view(torch.uint8)
    return extension.fc1_paired_nvfp4_scaled_tma256k2(
        a_packed, a_scales, w_packed, w_scales, alpha, product
    )


def fc1_paired_nvfp4_scaled_tma256k2_ldm(
    a_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """TMA 256x64 K=128 A fragment via ldmatrix.x4."""
    extension = load_extension()
    if a_scales.dtype != torch.uint8:
        a_scales = a_scales.view(torch.uint8)
    if w_scales.dtype != torch.uint8:
        w_scales = w_scales.view(torch.uint8)
    return extension.fc1_paired_nvfp4_scaled_tma256k2_ldm(
        a_packed, a_scales, w_packed, w_scales, alpha
    )


def fc1_paired_nvfp4_scaled_tma256k2_ldmb(
    a_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """TMA 256x64 K=128 B fragment via ldmatrix.x4 (two N-subtiles)."""
    extension = load_extension()
    if a_scales.dtype != torch.uint8:
        a_scales = a_scales.view(torch.uint8)
    if w_scales.dtype != torch.uint8:
        w_scales = w_scales.view(torch.uint8)
    return extension.fc1_paired_nvfp4_scaled_tma256k2_ldmb(
        a_packed, a_scales, w_packed, w_scales, alpha
    )


def fc1_paired_nvfp4_scaled_tma256k2_pipe(
    a_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """TMA 256x64 K=128 next-N B/SFB overlapped on m16n8k64."""
    extension = load_extension()
    if a_scales.dtype != torch.uint8:
        a_scales = a_scales.view(torch.uint8)
    if w_scales.dtype != torch.uint8:
        w_scales = w_scales.view(torch.uint8)
    return extension.fc1_paired_nvfp4_scaled_tma256k2_pipe(
        a_packed, a_scales, w_packed, w_scales, alpha
    )


def fc1_paired_nvfp4_scaled_tma256k2_leads(
    a_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """TMA 256x64 K=128 SFA/SFB only on contributing scale-vec lanes."""
    extension = load_extension()
    if a_scales.dtype != torch.uint8:
        a_scales = a_scales.view(torch.uint8)
    if w_scales.dtype != torch.uint8:
        w_scales = w_scales.view(torch.uint8)
    return extension.fc1_paired_nvfp4_scaled_tma256k2_leads(
        a_packed, a_scales, w_packed, w_scales, alpha
    )


def fc1_paired_nvfp4_scaled_tma256k2_pipea(
    a_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """TMA 256x64 K=128 next-K A/SFA overlapped on m16n8k64."""
    extension = load_extension()
    if a_scales.dtype != torch.uint8:
        a_scales = a_scales.view(torch.uint8)
    if w_scales.dtype != torch.uint8:
        w_scales = w_scales.view(torch.uint8)
    return extension.fc1_paired_nvfp4_scaled_tma256k2_pipea(
        a_packed, a_scales, w_packed, w_scales, alpha
    )


def fc1_paired_nvfp4_scaled_tma256k2s1(
    a_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """TMA 256x64 K=128 1-stage paired-N."""
    extension = load_extension()
    if a_scales.dtype != torch.uint8:
        a_scales = a_scales.view(torch.uint8)
    if w_scales.dtype != torch.uint8:
        w_scales = w_scales.view(torch.uint8)
    return extension.fc1_paired_nvfp4_scaled_tma256k2s1(
        a_packed, a_scales, w_packed, w_scales, alpha
    )


def fc1_paired_nvfp4_scaled_tma256k2s1_attrs() -> dict:
    """Occupancy/regs/smem for 1-stage k2 vs 3-stage k2."""
    return dict(load_extension().fc1_paired_nvfp4_scaled_tma256k2s1_attrs())


def fc1_paired_nvfp4_scaled_tma256n32(
    a_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """TMA 256x32 K=128 2-stage paired-N (half acc)."""
    extension = load_extension()
    if a_scales.dtype != torch.uint8:
        a_scales = a_scales.view(torch.uint8)
    if w_scales.dtype != torch.uint8:
        w_scales = w_scales.view(torch.uint8)
    return extension.fc1_paired_nvfp4_scaled_tma256n32(
        a_packed, a_scales, w_packed, w_scales, alpha
    )


def fc1_paired_nvfp4_scaled_tma256n32_attrs() -> dict:
    """Occupancy/regs/smem for 256x32 K=128 2-stage."""
    return dict(load_extension().fc1_paired_nvfp4_scaled_tma256n32_attrs())


def fc1_paired_nvfp4_scaled_tma128k2(
    a_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """TMA 128x64 K=128 2-stage paired-N (64-float acc)."""
    extension = load_extension()
    if a_scales.dtype != torch.uint8:
        a_scales = a_scales.view(torch.uint8)
    if w_scales.dtype != torch.uint8:
        w_scales = w_scales.view(torch.uint8)
    return extension.fc1_paired_nvfp4_scaled_tma128k2(
        a_packed, a_scales, w_packed, w_scales, alpha
    )


def fc1_paired_nvfp4_scaled_tma128k2_attrs() -> dict:
    """Occupancy/regs/smem for 128x64 K=128 2-stage."""
    return dict(load_extension().fc1_paired_nvfp4_scaled_tma128k2_attrs())


def fc1_paired_nvfp4_scaled_tma256k2_amax(
    a_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TMA 256x64 K=128 [gate|up] store + epilogue live amax. Certifies G."""
    extension = load_extension()
    if a_scales.dtype != torch.uint8:
        a_scales = a_scales.view(torch.uint8)
    if w_scales.dtype != torch.uint8:
        w_scales = w_scales.view(torch.uint8)
    raw, scale = extension.fc1_paired_nvfp4_scaled_tma256k2_amax(
        a_packed, a_scales, w_packed, w_scales, alpha
    )
    certify_live_amax(raw, scale)
    return raw, scale


def fc1_paired_nvfp4_scaled_tma256k2_sw(
    a_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    alpha: torch.Tensor,
    *,
    product: bool,
) -> torch.Tensor:
    """TMA 256x64 K=128 box with SWIZZLE_64B remapped to the PTX fragment."""
    extension = load_extension()
    if a_scales.dtype != torch.uint8:
        a_scales = a_scales.view(torch.uint8)
    if w_scales.dtype != torch.uint8:
        w_scales = w_scales.view(torch.uint8)
    return extension.fc1_paired_nvfp4_scaled_tma256k2_sw(
        a_packed, a_scales, w_packed, w_scales, alpha, product
    )


def fc1_paired_nvfp4_scaled_tma128k4(
    a_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    alpha: torch.Tensor,
    *,
    product: bool,
) -> torch.Tensor:
    """TMA 128x64 with a K=256 box: four m16n8k64 atoms per load."""
    extension = load_extension()
    if a_scales.dtype != torch.uint8:
        a_scales = a_scales.view(torch.uint8)
    if w_scales.dtype != torch.uint8:
        w_scales = w_scales.view(torch.uint8)
    return extension.fc1_paired_nvfp4_scaled_tma128k4(
        a_packed, a_scales, w_packed, w_scales, alpha, product
    )


def fc1_paired_nvfp4_scaled_tma128k2n(
    a_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    alpha: torch.Tensor,
    *,
    product: bool,
) -> torch.Tensor:
    """TMA 128x128 K=128 box: 16 n-subtiles, same register budget as k2."""
    extension = load_extension()
    if a_scales.dtype != torch.uint8:
        a_scales = a_scales.view(torch.uint8)
    if w_scales.dtype != torch.uint8:
        w_scales = w_scales.view(torch.uint8)
    return extension.fc1_paired_nvfp4_scaled_tma128k2n(
        a_packed, a_scales, w_packed, w_scales, alpha, product
    )


def fc1_paired_nvfp4_scaled_tma256k4(
    a_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    alpha: torch.Tensor,
    *,
    product: bool,
) -> torch.Tensor:
    """TMA 256x64 with a K=256 box: four m16n8k64 atoms per load."""
    extension = load_extension()
    if a_scales.dtype != torch.uint8:
        a_scales = a_scales.view(torch.uint8)
    if w_scales.dtype != torch.uint8:
        w_scales = w_scales.view(torch.uint8)
    return extension.fc1_paired_nvfp4_scaled_tma256k4(
        a_packed, a_scales, w_packed, w_scales, alpha, product
    )


def fc1_paired_nvfp4_scaled_tma256k2n2(
    a_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    alpha: torch.Tensor,
    *,
    product: bool,
) -> torch.Tensor:
    """TMA 256x128: sequential N-halves, one A stream, k2 register acc."""
    extension = load_extension()
    if a_scales.dtype != torch.uint8:
        a_scales = a_scales.view(torch.uint8)
    if w_scales.dtype != torch.uint8:
        w_scales = w_scales.view(torch.uint8)
    return extension.fc1_paired_nvfp4_scaled_tma256k2n2(
        a_packed, a_scales, w_packed, w_scales, alpha, product
    )


def fc1_paired_nvfp4_scaled_tma256k2ws(
    a_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    alpha: torch.Tensor,
    *,
    product: bool,
) -> torch.Tensor:
    """TMA 256x64 K=128 warp-specialized: producer + 8 MMA warps."""
    extension = load_extension()
    if a_scales.dtype != torch.uint8:
        a_scales = a_scales.view(torch.uint8)
    if w_scales.dtype != torch.uint8:
        w_scales = w_scales.view(torch.uint8)
    return extension.fc1_paired_nvfp4_scaled_tma256k2ws(
        a_packed, a_scales, w_packed, w_scales, alpha, product
    )


def fc1_paired_nvfp4_scaled_tma256k2ws4(
    a_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """TMA 256x64 K=128 12-warp: 4 TMA producers + 8 MMA on the k2 tile."""
    extension = load_extension()
    if a_scales.dtype != torch.uint8:
        a_scales = a_scales.view(torch.uint8)
    if w_scales.dtype != torch.uint8:
        w_scales = w_scales.view(torch.uint8)
    return extension.fc1_paired_nvfp4_scaled_tma256k2ws4(
        a_packed, a_scales, w_packed, w_scales, alpha
    )


def fc1_paired_nvfp4_scaled_tma256k2ws4_attrs() -> dict:
    """Register / smem launch attrs for 12-warp 4-producer k2."""
    return dict(load_extension().fc1_paired_nvfp4_scaled_tma256k2ws4_attrs())


def fc1_paired_nvfp4_scaled_tma256k2p(
    a_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    alpha: torch.Tensor,
    *,
    product: bool,
) -> torch.Tensor:
    """TMA 256x64 K=128 persistent: one CTA walks all N for a 256-row A panel."""
    extension = load_extension()
    if a_scales.dtype != torch.uint8:
        a_scales = a_scales.view(torch.uint8)
    if w_scales.dtype != torch.uint8:
        w_scales = w_scales.view(torch.uint8)
    return extension.fc1_paired_nvfp4_scaled_tma256k2p(
        a_packed, a_scales, w_packed, w_scales, alpha, product
    )


def fc1_nvfp4_scaled_tma128n128k4(
    a_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """TMA 128x128x256 single-N: kitchen tile geometry, one B operand."""
    extension = load_extension()
    if a_scales.dtype != torch.uint8:
        a_scales = a_scales.view(torch.uint8)
    if w_scales.dtype != torch.uint8:
        w_scales = w_scales.view(torch.uint8)
    return extension.fc1_nvfp4_scaled_tma128n128k4(
        a_packed, a_scales, w_packed, w_scales, alpha
    )


def fc1_nvfp4_scaled_tma256k4n1(
    a_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """TMA 256x64 K=256 2-stage single-N. Paired 2-stage is 112 KiB."""
    extension = load_extension()
    if a_scales.dtype != torch.uint8:
        a_scales = a_scales.view(torch.uint8)
    if w_scales.dtype != torch.uint8:
        w_scales = w_scales.view(torch.uint8)
    return extension.fc1_nvfp4_scaled_tma256k4n1(
        a_packed, a_scales, w_packed, w_scales, alpha
    )


def fc1_nvfp4_scaled_tma128n128k4_pipe(
    a_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """TMA 128x128x256 with next-N B fragments overlapped on MMA."""
    extension = load_extension()
    if a_scales.dtype != torch.uint8:
        a_scales = a_scales.view(torch.uint8)
    if w_scales.dtype != torch.uint8:
        w_scales = w_scales.view(torch.uint8)
    return extension.fc1_nvfp4_scaled_tma128n128k4_pipe(
        a_packed, a_scales, w_packed, w_scales, alpha
    )


def fc1_nvfp4_scaled_tma128n128k4_ldm(
    a_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """TMA 128x128x256 A fragment via ldmatrix.x4 (kitchen register map)."""
    extension = load_extension()
    if a_scales.dtype != torch.uint8:
        a_scales = a_scales.view(torch.uint8)
    if w_scales.dtype != torch.uint8:
        w_scales = w_scales.view(torch.uint8)
    return extension.fc1_nvfp4_scaled_tma128n128k4_ldm(
        a_packed, a_scales, w_packed, w_scales, alpha
    )


def fc1_nvfp4_scaled_tma128n128k4_sw(
    a_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """TMA 128x128x256 SWIZZLE_128B remapped to the linear PTX fragment."""
    extension = load_extension()
    if a_scales.dtype != torch.uint8:
        a_scales = a_scales.view(torch.uint8)
    if w_scales.dtype != torch.uint8:
        w_scales = w_scales.view(torch.uint8)
    return extension.fc1_nvfp4_scaled_tma128n128k4_sw(
        a_packed, a_scales, w_packed, w_scales, alpha
    )


def fc1_nvfp4_scaled_tma128n128k4ws(
    a_packed: torch.Tensor,
    a_scales: torch.Tensor,
    w_packed: torch.Tensor,
    w_scales: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """TMA 128x128x256 12-warp: 4 TMA producers + 8 MMA, 86 KiB dynamic."""
    extension = load_extension()
    if a_scales.dtype != torch.uint8:
        a_scales = a_scales.view(torch.uint8)
    if w_scales.dtype != torch.uint8:
        w_scales = w_scales.view(torch.uint8)
    return extension.fc1_nvfp4_scaled_tma128n128k4ws(
        a_packed, a_scales, w_packed, w_scales, alpha
    )


def fc1_nvfp4_scaled_tma128n128k4ws_attrs() -> dict:
    """Register / smem launch attrs for the 12-warp kitchen tile."""
    return dict(load_extension().fc1_nvfp4_scaled_tma128n128k4ws_attrs())


def swiglu_nvfp4_dynamic_swizzle(
    raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-pass producer tiled to one cuBLAS 128x4 E4M3 slab per CTA."""
    extension = load_extension()
    packed, scale_bytes, global_scale = (
        extension.swiglu_nvfp4_dynamic_swizzle(raw)
    )
    return packed, scale_bytes.view(torch.float8_e4m3fn), global_scale


def swiglu_nvfp4_dynamic_from_product(
    activated: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Consumer of an FC1 SwiGLU epilogue: amax+pack on the 14336-wide product.

    Byte-identical to ``swiglu_nvfp4_dynamic(raw)`` iff ``activated`` is the
    eager ``round(round(silu(g))*u)`` that the GEMM would store instead of
    the 28672-wide ``[gate|up]`` pair.
    """
    return bf16_nvfp4_dynamic(activated)


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
    "load_extension",
    "swiglu_nvfp4",
    "swiglu_nvfp4_dynamic",
    "swiglu_nvfp4_dynamic_fused",
    "swiglu_nvfp4_dynamic_twolevel",
    "swiglu_nvfp4_dynamic_interval",
    "swiglu_nvfp4_static_rebind",
    "swiglu_nvfp4_dynamic_coop",
    "swiglu_nvfp4_dynamic_vec",
    "swiglu_nvfp4_dynamic_inplace",
    "swiglu_nvfp4_dynamic_inplace_prod",
    "fc1_epilogue_assoc_mismatches",
    "fc1_paired_store",
    "fc1_paired_wmma",
    "fc1_paired_nvfp4",
    "fc1_paired_nvfp4_tiled",
    "fc1_paired_nvfp4_scaled",
    "fc1_paired_nvfp4_scaled_piped",
    "fc1_paired_nvfp4_scaled_tma",
    "fc1_paired_nvfp4_scaled_tma_sf",
    "fc1_paired_nvfp4_scaled_tma256",
    "fc1_paired_nvfp4_scaled_tma256_sw",
    "fc1_paired_nvfp4_scaled_tma256k2",
    "fc1_paired_nvfp4_scaled_tma256k2_amax",
    "fc1_paired_nvfp4_scaled_tma256k2_sw",
    "fc1_paired_nvfp4_scaled_tma128k4",
    "fc1_paired_nvfp4_scaled_tma128k2n",
    "swiglu_nvfp4_dynamic_swizzle",
    "swiglu_nvfp4_dynamic_from_product",
]
