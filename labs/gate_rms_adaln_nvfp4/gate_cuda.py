#!/usr/bin/env python3
"""Seconds-only exactness/timing gate for gate -> RMS/AdaLN -> NVFP4."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
WIDTH = 5376


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_candidate(path: Path, module_name: str, abi: int, build_id: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.abi_version() != abi or module.build_id() != build_id:
        raise RuntimeError("candidate ABI/build ID does not match the lab adapter")
    if not callable(getattr(module, "gate_rms_adaln_nvfp4_dynamic", None)):
        raise RuntimeError("candidate is missing the gate producer entrypoint")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=129)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument(
        "--comfy-root",
        type=Path,
        default=os.environ.get("COMFYUI_ROOT"),
        help="ComfyUI checkout (or set COMFYUI_ROOT)",
    )
    parser.add_argument(
        "--rms-lab",
        type=Path,
        default=HERE.parent / "rms_adaln_nvfp4_lab",
    )
    parser.add_argument("--library", type=Path)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.comfy_root is None or not args.comfy_root.is_dir():
        parser.error("--comfy-root must name a ComfyUI checkout")
    if not args.rms_lab.is_dir():
        parser.error("--rms-lab must name the certified RMS/AdaLN lab")

    sys.path.insert(0, str(args.comfy_root.resolve()))
    sys.path.insert(0, str(args.rms_lab.resolve()))
    # Both labs intentionally use a module named ``comfy_adapter``. Keep this
    # gate's adapter first while importing the RMS reference by its unique
    # native-wrapper module name below.
    sys.path.insert(0, str(HERE))
    from comfy.ldm.minimax.kernels import segmented_gate_bf16_
    from rms_adaln_nvfp4 import load_extension as load_rms_extension
    from comfy_adapter import (
        CERTIFIED_CUDA_VERSION,
        CERTIFIED_DEVICE_CAPABILITY,
        NATIVE_ABI_VERSION,
        NATIVE_BUILD_ID,
        NATIVE_MODULE_NAME,
    )

    library = (
        args.library
        if args.library is not None
        else HERE / "build" / f"{NATIVE_MODULE_NAME}.so"
    ).resolve()
    if not library.is_file():
        parser.error(f"candidate library does not exist: {library}")
    source_path = HERE / "gate_rms_adaln_nvfp4_cuda.cu"
    adapter_path = HERE / "comfy_adapter.py"

    if torch.version.cuda != CERTIFIED_CUDA_VERSION:
        raise RuntimeError("runtime CUDA version is outside the certified contract")
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    if capability != CERTIFIED_DEVICE_CAPABILITY:
        raise RuntimeError(
            f"expected certified capability {CERTIFIED_DEVICE_CAPABILITY}, got {capability}"
        )

    reference_native = load_rms_extension()
    candidate_native = load_candidate(
        library, NATIVE_MODULE_NAME, NATIVE_ABI_VERSION, NATIVE_BUILD_ID
    )
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream), torch.no_grad():
        if torch.cuda.current_stream(device).cuda_stream != stream.cuda_stream:
            raise RuntimeError("nondefault stream context did not activate")
        torch.manual_seed(26081225)
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
        reference_q, reference_s, reference_global = (
            reference_native.rms_adaln_nvfp4_dynamic(
                x_reference, weight, scale, shift, row_ids, eps
            )
        )
        _, reference_rstd = torch.ops.aten._fused_rms_norm(
            x_reference, [WIDTH], weight, eps
        )
        version_before = x_candidate._version
        candidate_q, candidate_s, candidate_global, candidate_rstd = (
            candidate_native.gate_rms_adaln_nvfp4_dynamic(
                x_candidate, other, gate, weight, scale, shift, row_ids, eps
            )
        )
        candidate_version_delta = x_candidate._version - version_before
        stream.synchronize()

        timing_reference = x.clone()
        timing_candidate = x.clone()

        def reference_call():
            if not segmented_gate_bf16_(
                timing_reference, other, gate, row_ids
            ):
                raise RuntimeError("stock segmented gate did not route")
            return reference_native.rms_adaln_nvfp4_dynamic(
                timing_reference, weight, scale, shift, row_ids, eps
            )

        def candidate_call():
            return candidate_native.gate_rms_adaln_nvfp4_dynamic(
                timing_candidate, other, gate, weight, scale, shift, row_ids, eps
            )

        stock_ms = median_ms(reference_call, args.warmup, args.reps)
        fused_ms = median_ms(candidate_call, args.warmup, args.reps)

    result = {
        "qualification_status": "candidate_v2",
        "rows": args.rows,
        "width": WIDTH,
        "modulation_dtype": str(scale.dtype),
        "stream": "nondefault",
        "candidate_version_delta": candidate_version_delta,
        "gate_output_byte_mismatches": byte_diff(x_candidate, x_reference),
        "rstd_byte_mismatches": byte_diff(
            candidate_rstd, reference_rstd.reshape(-1)
        ),
        "packed_byte_mismatches": byte_diff(candidate_q, reference_q),
        "scale_byte_mismatches": byte_diff(candidate_s, reference_s),
        "global_scale_byte_mismatches": byte_diff(
            candidate_global, reference_global
        ),
        "stock_gate_plus_rms_adaln_nvfp4_ms": stock_ms,
        "fused_gate_rms_adaln_nvfp4_ms": fused_ms,
        "speedup": stock_ms / fused_ms,
        "saved_ms": stock_ms - fused_ms,
        "native_abi_version": candidate_native.abi_version(),
        "native_build_id": candidate_native.build_id(),
        "source_sha256": sha256(source_path),
        "adapter_sha256": sha256(adapter_path),
        "binary_sha256": sha256(library),
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "cuda_version": torch.version.cuda,
        "device_capability": list(capability),
    }
    result["byte_exact"] = all(
        result[key] == 0
        for key in (
            "gate_output_byte_mismatches",
            "rstd_byte_mismatches",
            "packed_byte_mismatches",
            "scale_byte_mismatches",
            "global_scale_byte_mismatches",
        )
    )
    result["contract_exact"] = (
        result["byte_exact"] and result["candidate_version_delta"] == 1
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    if not result["contract_exact"]:
        raise SystemExit("gate fusion exactness/version contract failed")


if __name__ == "__main__":
    main()
