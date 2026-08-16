"""One-shot production-shape parity gate for the frozen H3 Q bundle.

This intentionally performs no warmup or timing loop.  It compares the rebuilt
Sage generic path, rebuilt output-amax path, Q-v3 prequantized path, and the
supplied-amax NVFP4 consumer on the default and one non-default stream.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys

import torch


Q_ROOT = Path(__file__).resolve().parent
COMFY = Path(os.environ["COMFYUI_ROOT"])
SAGE_ROOT = Path(os.environ["H3_SAGE_ROOT"])
CONSUMER_ROOT = Path(os.environ["H3_NVFP4_CONSUMER_ROOT"])
Q_BINARY = Path(
    os.environ.get(
        "H3_Q_RMS_ROPE_INT8_LIBRARY",
        Q_ROOT / "build-release" / "h3_q_rms_rope_int8_native_v1.so",
    )
)
CONSUMER_BINARY = Path(os.environ["H3_SWIGLU_NVFP4_LIBRARY"])
SAGE_FUSED = next((SAGE_ROOT / "sageattention").glob("_fused*.so"))
SAGE_QATTN = next((SAGE_ROOT / "sageattention").glob("_qattn_sm89*.so"))

EXPECTED_SHA256 = {
    Q_BINARY: "b97b30c30222276d490d2a3d4cf2fff1081d7ff3a66d8e01395589feccd2da1f",
    CONSUMER_BINARY: "45b21fe1957e535a860fcd28014b6b194666a29b5b98f4672c7e3b99230edb8f",
    SAGE_FUSED: "30137fab72eb9571d911202f7e72b0c9a633b5eafd2dbadb758080a281651a12",
    SAGE_QATTN: "e4861cdd94cdd2bbbe6c96e16663be620c221454048756ddd2e6d5379460d2db",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


for artifact, expected in EXPECTED_SHA256.items():
    actual = sha256(artifact)
    if actual != expected:
        raise SystemExit(f"artifact SHA mismatch: {artifact}: {actual}")

os.environ["H3_Q_RMS_ROPE_INT8_LIBRARY"] = str(Q_BINARY)
os.environ["H3_SWIGLU_NVFP4_LIBRARY"] = str(CONSUMER_BINARY)
sys.path[:0] = [str(Q_ROOT), str(SAGE_ROOT), str(CONSUMER_ROOT), str(COMFY)]

import comfy_kitchen  # noqa: E402
import h3_q_rms_rope_int8 as q_fusion  # noqa: E402
from comfy.ldm.minimax.model import rope_rotation_table  # noqa: E402
from sageattention import (  # noqa: E402
    _fused,
    sageattn_h3_prequantized_q_pv_fp8_cuda,
    sageattn_qk_int8_pv_fp8_cuda,
    sageattn_qk_int8_pv_fp8_cuda_with_output_amax,
    sm89_compile,
)
from swiglu_nvfp4 import (  # noqa: E402
    bf16_nvfp4_dynamic,
    bf16_nvfp4_from_amax_bits,
    load_extension,
)


HEADS = 56
DIM = 128
ROT = 96
INNER = HEADS * DIM
QKV = INNER * 3
SEQ = 20423


def qkv_views(packed: torch.Tensor):
    q, k, v = packed.split(INNER, dim=-1)
    return (
        q.view(1, SEQ, HEADS, DIM),
        k.view(1, SEQ, HEADS, DIM),
        v.view(1, SEQ, HEADS, DIM),
    )


def marker_fields(marker: torch.Tensor):
    raw = int(marker.item()) & 0xFFFFFFFF
    return raw & 0xFFFF, bool(raw & 0x80000000)


def output_amax_bits(output: torch.Tensor) -> int:
    return int(output.view(torch.int16).bitwise_and(0x7FFF).max().item())


def triplet_equal(left, right) -> bool:
    return all(
        torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        for a, b in zip(left, right, strict=True)
    )


def reference(packed, freqs, q_weight, k_weight):
    q, k, v = qkv_views(packed)
    comfy_kitchen.rms_rope_split_half_(
        q, k, freqs, q_weight, k_weight, epsilon=1e-5, rot_dim=ROT
    )
    stock = sageattn_qk_int8_pv_fp8_cuda(
        q,
        k,
        v,
        tensor_layout="NHD",
        is_causal=False,
        qk_quant_gran="per_warp",
        sm_scale=1.0 / math.sqrt(DIM),
        pv_accum_dtype="fp32+fp32",
        smooth_k=True,
    )
    tracked, marker = sageattn_qk_int8_pv_fp8_cuda_with_output_amax(
        q,
        k,
        v,
        tensor_layout="NHD",
        is_causal=False,
        qk_quant_gran="per_warp",
        sm_scale=1.0 / math.sqrt(DIM),
        pv_accum_dtype="fp32+fp32",
        smooth_k=True,
    )
    q_int8 = torch.empty_like(q, dtype=torch.int8)
    q_scale = torch.empty(
        (1, HEADS, ((SEQ + 127) // 128) * 4),
        dtype=torch.float32,
        device=q.device,
    )
    _fused.quant_per_warp_int8_cuda(q, q_int8, q_scale, 128, 32, 0)
    return stock, tracked, marker, q_int8, q_scale, k


def candidate(packed, freqs, q_weight, k_weight):
    q, k, v = qkv_views(packed)
    before = (packed._version, q._version, k._version)
    prequantized = q_fusion.try_qk_rms_rope_q_int8(
        q, k, freqs, q_weight, k_weight, 1e-5, 16
    )
    if prequantized is None:
        raise RuntimeError("Q-v3 producer unexpectedly failed its preflight")
    q_int8, q_scale, transformed_k = prequantized
    after = (packed._version, q._version, k._version)
    output, marker = sageattn_h3_prequantized_q_pv_fp8_cuda(
        q_int8, q_scale, transformed_k, v, return_output_amax=True
    )
    flat = output.reshape(SEQ, INNER)
    dynamic = bf16_nvfp4_dynamic(flat)
    supplied = bf16_nvfp4_from_amax_bits(flat, marker)
    return (
        output,
        marker,
        q_int8,
        q_scale,
        transformed_k,
        tuple(a - b for a, b in zip(after, before, strict=True)),
        dynamic,
        supplied,
    )


def run_once(source, freqs, q_weight, k_weight):
    ref = reference(source.clone(), freqs, q_weight, k_weight)
    got = candidate(source.clone(), freqs, q_weight, k_weight)
    ref_stock, ref_tracked, ref_marker, ref_q, ref_scale, ref_k = ref
    output, marker, q_int8, q_scale, k, versions, dynamic, supplied = got
    ref_bits, ref_nonfinite = marker_fields(ref_marker)
    bits, nonfinite = marker_fields(marker)
    return {
        "generic_vs_output_amax_bf16_exact": torch.equal(ref_stock, ref_tracked),
        "prequant_output_bf16_exact": torch.equal(output, ref_stock),
        "q_int8_exact": torch.equal(q_int8, ref_q),
        "q_scale_exact": torch.equal(q_scale, ref_scale),
        "k_bf16_exact": torch.equal(k, ref_k),
        "marker_exact": torch.equal(marker, ref_marker),
        "marker_matches_output": bits == output_amax_bits(output),
        "reference_marker_matches_output": ref_bits == output_amax_bits(ref_tracked),
        "marker_finite": not nonfinite and not ref_nonfinite,
        "packed_q_k_version_delta_exact": versions == (1, 1, 1),
        "version_deltas": list(versions),
        "consumer_triplet_exact": triplet_equal(dynamic, supplied),
    }


def main() -> None:
    if (
        _fused.h3_build_id()
        != "h3-sage-fused-current-stream-v1-torch2.13-sm120"
        or sm89_compile._qattn_sm89.h3_build_id()
        != "h3-sage-qattn-output-amax-v2-torch2.13-sm120"
        or q_fusion._NATIVE.build_id()
        != "h3-q-rms-rope96-qint8-v3-torch2.13-sm120"
        or load_extension().h3_supplied_amax_build_id()
        != "h3-nvfp4-supplied-amax-v1-torch2.13-sm120"
    ):
        raise SystemExit("native build-ID mismatch")

    torch.manual_seed(20260812)
    device = torch.device("cuda")
    source = torch.randn((SEQ, QKV), dtype=torch.bfloat16, device=device)
    angles = torch.randn((SEQ, ROT), dtype=torch.float32, device=device)
    angles[:, ROT // 2 :] = angles[:, : ROT // 2]
    freqs = rope_rotation_table(angles, torch.bfloat16)
    q_weight = (torch.randn(DIM, device=device) * 0.1 + 1).to(torch.bfloat16)
    k_weight = (torch.randn(DIM, device=device) * 0.1 + 1).to(torch.bfloat16)

    with torch.no_grad():
        default = run_once(source, freqs, q_weight, k_weight)
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(5_000_000)
            nondefault = run_once(source, freqs, q_weight, k_weight)
        stream.synchronize()

    result = {
        "artifacts": {str(path): digest for path, digest in EXPECTED_SHA256.items()},
        "shape": [1, SEQ, HEADS, DIM],
        "default_stream": default,
        "nondefault_stream": nondefault,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    checks = [
        value
        for stream_result in (default, nondefault)
        for key, value in stream_result.items()
        if key != "version_deltas"
    ]
    if not all(checks):
        raise SystemExit("H3 release parity gate failed")


if __name__ == "__main__":
    main()
