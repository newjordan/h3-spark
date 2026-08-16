#!/usr/bin/env python3
"""Seconds gate for attention gate -> supplied-scale RMS/AdaLN -> FC1 NVFP4."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
WIDTH = 5376
QUALIFIED_GATE_LAB = HERE.parent / "gate_rms_adaln_nvfp4_lab"
QUALIFIED_STATIC_LAB = HERE.parent / "rms_adaln_nvfp4_static_lab"
ADALN_LAB = HERE.parent / "adaln_nvfp4_lab" / "native_cuda"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def load_candidate(path: Path, adapter):
    module = load_module(adapter.NATIVE_MODULE_NAME, path)
    if (
        module.abi_version() != adapter.NATIVE_ABI_VERSION
        or module.build_id() != adapter.NATIVE_BUILD_ID
        or not callable(getattr(module, "gate_rms_adaln_nvfp4_static", None))
        or not callable(
            getattr(module, "gate_rms_adaln_nvfp4_static_debug", None)
        )
    ):
        raise RuntimeError("candidate ABI/build/entrypoint contract differs")
    return module


def byte_diff(left: torch.Tensor, right: torch.Tensor) -> int:
    return int((left.view(torch.uint8) != right.view(torch.uint8)).sum().item())


def median_ms(fn, warmup: int, reps: int) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main() -> None:
    wall_start = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=20423)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument(
        "--comfy-root",
        type=Path,
        default=os.environ.get("COMFYUI_ROOT"),
        help="ComfyUI checkout (or set COMFYUI_ROOT)",
    )
    parser.add_argument("--library", type=Path)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.comfy_root is None or not args.comfy_root.is_dir():
        parser.error("--comfy-root must name a ComfyUI checkout")
    if args.rows <= 0 or args.warmup < 0 or args.reps <= 0:
        parser.error("rows/reps must be positive and warmup nonnegative")

    sys.path.insert(0, str(args.comfy_root.resolve()))
    sys.path.insert(0, str(ADALN_LAB.resolve()))
    sys.path.insert(0, str(HERE))
    from comfy.ldm.minimax.kernels import segmented_gate_bf16_
    from adaln_nvfp4 import eager_adaln_reference
    import comfy_adapter as adapter

    static_reference = load_module(
        "h3_qualified_static_reference", QUALIFIED_STATIC_LAB / "prototype.py"
    )
    library = (
        args.library
        if args.library is not None
        else HERE / "build" / f"{adapter.NATIVE_MODULE_NAME}.so"
    ).resolve()
    if not library.is_file():
        parser.error(f"candidate library does not exist: {library}")
    if sha256(library) != adapter.NATIVE_SHA256:
        raise RuntimeError("candidate binary SHA-256 differs from adapter")
    if torch.version.cuda != adapter.CERTIFIED_CUDA_VERSION:
        raise RuntimeError("runtime CUDA version is outside the certified contract")

    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    if capability != adapter.CERTIFIED_DEVICE_CAPABILITY:
        raise RuntimeError(
            f"expected capability {adapter.CERTIFIED_DEVICE_CAPABILITY}, got {capability}"
        )
    reference_native = static_reference.load_extension()
    candidate_native = load_candidate(library, adapter)

    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream), torch.no_grad():
        if torch.cuda.current_stream(device).cuda_stream != stream.cuda_stream:
            raise RuntimeError("nondefault stream context did not activate")
        torch.manual_seed(26081226)
        x = torch.randn(args.rows, WIDTH, device=device, dtype=torch.bfloat16)
        other = (torch.randn_like(x) * 0.05).to(torch.bfloat16)
        weight = (torch.randn(WIDTH, device=device) * 0.1 + 1.0).to(
            torch.bfloat16
        )
        table = (
            torch.randn(21, WIDTH * 6, device=device) * 0.1
        ).to(torch.float32)
        scale = table[:, WIDTH : 2 * WIDTH]
        gate = table[:, 2 * WIDTH : 3 * WIDTH]
        shift = table[:, 4 * WIDTH : 5 * WIDTH]
        row_ids = (
            torch.arange(args.rows, device=device, dtype=torch.int64)
            .mul_(21)
            .div_(args.rows, rounding_mode="floor")
            .to(torch.int32)
        )
        eps = 1e-5

        x_reference = x.clone()
        x_candidate = x.clone()
        if not segmented_gate_bf16_(x_reference, other, gate, row_ids):
            raise RuntimeError("stock segmented gate did not route")
        normed, reference_rstd = torch.ops.aten._fused_rms_norm(
            x_reference, [WIDTH], weight, eps
        )
        materialized = eager_adaln_reference(normed, scale, shift, row_ids)
        # Match TensorCoreNVFP4Layout calibration: BF16 division, then widen.
        input_scale = (materialized.abs().max() / 2688.0).to(torch.float32)
        reference_q, reference_s, static_reference_rstd = (
            reference_native.rms_adaln_nvfp4_static_debug(
                x_reference,
                weight,
                scale,
                shift,
                row_ids,
                input_scale,
                eps,
            )
        )
        version_before = x_candidate._version
        candidate_q, candidate_s, candidate_rstd = (
            candidate_native.gate_rms_adaln_nvfp4_static_debug(
                x_candidate,
                other,
                gate,
                weight,
                scale,
                shift,
                row_ids,
                input_scale,
                eps,
            )
        )
        version_delta = x_candidate._version - version_before
        stream.synchronize()

        timing_reference = x.clone()
        timing_candidate = x.clone()

        def reference_call():
            if not segmented_gate_bf16_(
                timing_reference, other, gate, row_ids
            ):
                raise RuntimeError("stock segmented gate did not route")
            return reference_native.rms_adaln_nvfp4_static(
                timing_reference,
                weight,
                scale,
                shift,
                row_ids,
                input_scale,
                eps,
            )

        def candidate_call():
            return candidate_native.gate_rms_adaln_nvfp4_static(
                timing_candidate,
                other,
                gate,
                weight,
                scale,
                shift,
                row_ids,
                input_scale,
                eps,
            )

        staged_ms = median_ms(reference_call, args.warmup, args.reps)
        fused_ms = median_ms(candidate_call, args.warmup, args.reps)

    result = {
        "qualification_status": "candidate_static_v1",
        "rows": args.rows,
        "width": WIDTH,
        "modulation_dtype": str(scale.dtype),
        "stream": "nondefault",
        "candidate_version_delta": version_delta,
        "gate_output_byte_mismatches": byte_diff(x_candidate, x_reference),
        "rstd_byte_mismatches": byte_diff(
            candidate_rstd, reference_rstd.reshape(-1)
        ),
        "qualified_static_rstd_byte_mismatches": byte_diff(
            candidate_rstd, static_reference_rstd
        ),
        "packed_byte_mismatches": byte_diff(candidate_q, reference_q),
        "scale_byte_mismatches": byte_diff(candidate_s, reference_s),
        "stock_gate_plus_one_pass_static_ms": staged_ms,
        "fused_gate_one_pass_static_ms": fused_ms,
        "speedup": staged_ms / fused_ms,
        "saved_ms": staged_ms - fused_ms,
        "activation_reread_saved_bytes": args.rows * WIDTH * 2,
        "projected_calls_per_20_step_render": 50 * 20,
        "projected_saved_ms_per_20_step_render": (staged_ms - fused_ms) * 1000,
        "native_abi_version": candidate_native.abi_version(),
        "native_build_id": candidate_native.build_id(),
        "source_sha256": sha256(HERE / "gate_rms_adaln_nvfp4_static_cuda.cu"),
        "adapter_sha256": sha256(HERE / "comfy_adapter.py"),
        "binary_sha256": sha256(library),
        "qualified_gate_v2_source_sha256": sha256(
            QUALIFIED_GATE_LAB / "gate_rms_adaln_nvfp4_cuda.cu"
        ),
        "qualified_static_source_sha256": sha256(
            QUALIFIED_STATIC_LAB / "rms_adaln_nvfp4_static_cuda.cu"
        ),
        "qualified_static_binary_sha256": sha256(
            QUALIFIED_STATIC_LAB
            / "build"
            / "h3_rms_adaln_nvfp4_static_native_v1.so"
        ),
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "cuda_version": torch.version.cuda,
        "device_capability": list(capability),
        "wall_seconds": time.monotonic() - wall_start,
    }
    result["byte_exact"] = all(
        result[key] == 0
        for key in (
            "gate_output_byte_mismatches",
            "rstd_byte_mismatches",
            "qualified_static_rstd_byte_mismatches",
            "packed_byte_mismatches",
            "scale_byte_mismatches",
        )
    )
    result["contract_exact"] = (
        result["byte_exact"] and result["candidate_version_delta"] == 1
    )
    result["performance_gate"] = result["saved_ms"] > 0.5
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    if not result["contract_exact"]:
        raise SystemExit("static gate fusion exactness/version contract failed")
    if not result["performance_gate"]:
        raise SystemExit("static gate fusion saved <=0.5 ms/path")


if __name__ == "__main__":
    main()
