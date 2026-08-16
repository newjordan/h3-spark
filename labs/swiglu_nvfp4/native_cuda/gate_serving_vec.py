#!/usr/bin/env python3
"""Algebra gate: serving two-pass == lab vec; gap is the missing 1x1.

Serving auto still calls swiglu_nvfp4_dynamic = amax + <<<1,1>>>
finalize_dynamic_scale + pack. Lab swiglu_nvfp4_dynamic_vec is the same
two-pass algebra with G inlined in the pack (no 1x1). That 1x1 is a
named CBF barrier (50/eval).

The two natives export the same PyInit name, so each role runs in its
own process. Does not patch serving. Does not promote.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROWS = 20423
WIDTH = 28672
REPEATS = 3
SERVING_SO = Path(
    "$HOME/comfyui-h3-current/comfy/ldm/minimax/"
    "swiglu_nvfp4_native_v1.so"
)
# QUALIFICATION_RECEIPT.json consumer_sha256
SERVING_SHA256 = (
    "45b21fe1957e535a860fcd28014b6b194666a29b5b98f4672c7e3b99230edb8f"
)
MODULE_NAME = "h3_swiglu_nvfp4_native_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_so(path: Path):
    spec = importlib.util.spec_from_file_location(MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def byte_diff(left: torch.Tensor, right: torch.Tensor) -> int:
    return int((left.view(torch.uint8) != right.view(torch.uint8)).sum().item())


def bits_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(
        left.view(torch.uint8).equal(right.contiguous().view(torch.uint8))
    )


def time_ms(fn, raw: torch.Tensor) -> float:
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    for _ in range(2):
        fn(raw)
    torch.cuda.synchronize()
    samples = []
    for _ in range(REPEATS):
        starter.record()
        fn(raw)
        ender.record()
        torch.cuda.synchronize()
        samples.append(starter.elapsed_time(ender))
    return min(samples)


def profile_kernels(fn, raw: torch.Tensor) -> dict:
    fn(raw)
    torch.cuda.synchronize()
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        fn(raw)
        torch.cuda.synchronize()
    names: list[str] = []
    for evt in prof.events():
        if evt.device_type == torch.profiler.DeviceType.CUDA and evt.name:
            names.append(evt.name)
    counts = Counter(names)
    compute = [
        name
        for name in names
        if any(
            key in name.lower()
            for key in ("swiglu", "finalize", "amax", "nvfp4")
        )
    ]
    return {
        "all_cuda": dict(sorted(counts.items())),
        "compute_unique": sorted(set(compute)),
        "has_finalize_1x1": any("finalize_dynamic_scale" in name for name in names),
        "cuda_event_count": len(names),
    }


def source_launches() -> dict:
    serving_cu = (
        REPO
        / "labs"
        / "sage_output_amax"
        / "nvfp4_consumer"
        / "swiglu_nvfp4_cuda.cu"
    )
    lab_cu = HERE / "swiglu_nvfp4_cuda.cu"
    serving_text = serving_cu.read_text()
    lab_text = lab_cu.read_text()
    vec_body = lab_text.split(
        "std::vector<torch::Tensor> swiglu_nvfp4_dynamic_vec", 1
    )[1].split(
        "std::vector<torch::Tensor> swiglu_nvfp4_dynamic_vec16", 1
    )[0]
    dynamic_body = lab_text.split(
        "std::vector<torch::Tensor> swiglu_nvfp4_dynamic(", 1
    )[1].split(
        "std::vector<torch::Tensor> swiglu_nvfp4_dynamic_oneshot", 1
    )[0]
    return {
        "serving_source": str(serving_cu.relative_to(REPO)),
        "lab_source": str(lab_cu.relative_to(REPO)),
        "serving_has_finalize_1x1": (
            "finalize_dynamic_scale_kernel<<<1, 1, 0, stream>>>" in serving_text
        ),
        "lab_dynamic_has_finalize_1x1": (
            "finalize_dynamic_scale_kernel<<<1, 1, 0, stream>>>" in dynamic_body
        ),
        "lab_vec_has_finalize_1x1": (
            "finalize_dynamic_scale_kernel" in vec_body
        ),
        "serving_named": [
            "swiglu_amax_bf16_bits_kernel",
            "finalize_dynamic_scale_kernel<<<1,1>>>",
            "swiglu_nvfp4_kernel<true>",
        ],
        "vec_named": [
            "swiglu_amax_vec_kernel",
            "swiglu_nvfp4_pack_from_amax_bits_kernel (G inlined)",
        ],
        "missing_on_vec": "finalize_dynamic_scale_kernel<<<1,1>>>",
    }


def role_serving(raw_path: Path, out_path: Path) -> None:
    native = load_so(SERVING_SO)
    if not callable(getattr(native, "swiglu_nvfp4_dynamic", None)):
        raise ImportError("serving .so missing swiglu_nvfp4_dynamic")
    has_vec = callable(getattr(native, "swiglu_nvfp4_dynamic_vec", None))

    def fn(raw: torch.Tensor):
        packed, scale_bytes, global_scale = native.swiglu_nvfp4_dynamic(raw)
        return packed, scale_bytes.view(torch.float8_e4m3fn), global_scale

    raw = torch.load(raw_path, map_location="cuda", weights_only=True)
    q, s, g = fn(raw)
    profile = profile_kernels(fn, raw)
    ms = time_ms(fn, raw)
    torch.save(
        {
            "q": q.detach().cpu(),
            "s": s.detach().cpu().view(torch.uint8),
            "g": g.detach().cpu(),
            "has_vec": has_vec,
            "min_ms": ms,
            "profile": profile,
        },
        out_path,
    )


def role_vec(raw_path: Path, out_path: Path) -> None:
    sys.path.insert(0, str(HERE))
    from swiglu_nvfp4 import load_extension, swiglu_nvfp4_dynamic_vec

    load_extension(verbose=False)
    raw = torch.load(raw_path, map_location="cuda", weights_only=True)
    q, s, g = swiglu_nvfp4_dynamic_vec(raw)
    profile = profile_kernels(swiglu_nvfp4_dynamic_vec, raw)
    ms = time_ms(swiglu_nvfp4_dynamic_vec, raw)
    torch.save(
        {
            "q": q.detach().cpu(),
            "s": s.detach().cpu().view(torch.uint8),
            "g": g.detach().cpu(),
            "min_ms": ms,
            "profile": profile,
        },
        out_path,
    )


def role_eager(raw_path: Path, out_path: Path) -> None:
    import torch.nn.functional as F
    import comfy_kitchen as ck

    raw = torch.load(raw_path, map_location="cuda", weights_only=True)
    gate, up = raw.chunk(2, dim=-1)
    activated = F.silu(gate).mul_(up)
    scale = (activated.abs().amax() / (6.0 * 448.0)).to(torch.float32).reshape(1)
    packed, block_scales = ck.quantize_nvfp4(
        activated, scale, pad_16x=True, hi_first=True
    )
    torch.save(
        {
            "q": packed.detach().cpu(),
            "s": block_scales.detach().cpu().view(torch.uint8),
            "g": scale.detach().cpu(),
        },
        out_path,
    )


def run_role(role: str, raw_path: Path, out_path: Path) -> None:
    env = os.environ.copy()
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--role",
        role,
        "--raw",
        str(raw_path),
        "--out",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, cwd=str(HERE), env=env)


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--role":
        role = sys.argv[2]
        raw_path = Path(sys.argv[sys.argv.index("--raw") + 1])
        out_path = Path(sys.argv[sys.argv.index("--out") + 1])
        if role == "serving":
            role_serving(raw_path, out_path)
        elif role == "vec":
            role_vec(raw_path, out_path)
        elif role == "eager":
            role_eager(raw_path, out_path)
        else:
            raise SystemExit(f"unknown role {role}")
        return 0

    if not SERVING_SO.is_file():
        raise FileNotFoundError(SERVING_SO)
    serving_sha = sha256_file(SERVING_SO)
    device = torch.device("cuda")
    torch.manual_seed(26081313)
    raw = torch.randn(ROWS, WIDTH, device=device, dtype=torch.bfloat16)

    with tempfile.TemporaryDirectory(prefix="h3_serving_vec_") as tmp:
        tmp_path = Path(tmp)
        raw_path = tmp_path / "raw.pt"
        serving_path = tmp_path / "serving.pt"
        vec_path = tmp_path / "vec.pt"
        eager_path = tmp_path / "eager.pt"
        torch.save(raw.cpu(), raw_path)
        del raw
        torch.cuda.empty_cache()
        run_role("serving", raw_path, serving_path)
        run_role("vec", raw_path, vec_path)
        run_role("eager", raw_path, eager_path)
        serving = torch.load(serving_path, map_location="cpu", weights_only=False)
        vec = torch.load(vec_path, map_location="cpu", weights_only=False)
        eager = torch.load(eager_path, map_location="cpu", weights_only=False)

    src = source_launches()
    q_diff = byte_diff(serving["q"], vec["q"])
    s_diff = byte_diff(serving["s"], vec["s"])
    g_bits_equal = bits_equal(serving["g"], vec["g"])
    launch_ok = bool(
        serving["profile"]["has_finalize_1x1"]
        and not vec["profile"]["has_finalize_1x1"]
        and src["serving_has_finalize_1x1"]
        and src["lab_dynamic_has_finalize_1x1"]
        and not src["lab_vec_has_finalize_1x1"]
    )
    payload = {
        "identity": (
            "Serving swiglu_nvfp4_dynamic == lab swiglu_nvfp4_dynamic_vec "
            "in packed bytes, E4M3, and G bits at S=20423. Same two-pass "
            "algebra; launch difference is exactly the missing "
            "finalize_dynamic_scale<<<1,1>>> (G inlined in the vec pack). "
            "That 1x1 is a named CBF barrier (50/eval). Not patched. "
            "Not promoted."
        ),
        "device": torch.cuda.get_device_name(device),
        "rows": ROWS,
        "width": WIDTH,
        "serving_so": str(SERVING_SO),
        "serving_sha256": serving_sha,
        "serving_sha256_expected": SERVING_SHA256,
        "serving_sha256_match": serving_sha == SERVING_SHA256,
        "serving_has_vec_entrypoint": bool(serving["has_vec"]),
        "serving_vs_vec_q": q_diff,
        "serving_vs_vec_s": s_diff,
        "serving_vs_vec_g_bits": g_bits_equal,
        "serving_vs_eager_q": byte_diff(serving["q"], eager["q"]),
        "serving_vs_eager_s": byte_diff(serving["s"], eager["s"]),
        "serving_vs_eager_g_bits": bits_equal(serving["g"], eager["g"]),
        "vec_vs_eager_q": byte_diff(vec["q"], eager["q"]),
        "vec_vs_eager_s": byte_diff(vec["s"], eager["s"]),
        "vec_vs_eager_g_bits": bits_equal(vec["g"], eager["g"]),
        "serving_g": float(serving["g"].reshape(-1)[0].item()),
        "vec_g": float(vec["g"].reshape(-1)[0].item()),
        "serving_min_ms": round(float(serving["min_ms"]), 4),
        "vec_min_ms": round(float(vec["min_ms"]), 4),
        "launches": {
            "serving_source": src["serving_named"],
            "vec_source": src["vec_named"],
            "missing_on_vec": src["missing_on_vec"],
            "serving_source_has_1x1": src["serving_has_finalize_1x1"],
            "lab_dynamic_source_has_1x1": src["lab_dynamic_has_finalize_1x1"],
            "lab_vec_source_has_1x1": src["lab_vec_has_finalize_1x1"],
            "serving_profile_has_1x1": serving["profile"]["has_finalize_1x1"],
            "vec_profile_has_1x1": vec["profile"]["has_finalize_1x1"],
            "serving_compute": serving["profile"]["compute_unique"],
            "vec_compute": vec["profile"]["compute_unique"],
            "serving_cuda": serving["profile"]["all_cuda"],
            "vec_cuda": vec["profile"]["all_cuda"],
            "delta_is_missing_1x1": launch_ok,
        },
        "promoted": False,
        "serving_default": "stock",
        "patched_serving": False,
    }
    payload["pass"] = bool(
        payload["serving_sha256_match"]
        and not payload["serving_has_vec_entrypoint"]
        and payload["serving_vs_vec_q"] == 0
        and payload["serving_vs_vec_s"] == 0
        and payload["serving_vs_vec_g_bits"]
        and payload["serving_vs_eager_q"] == 0
        and payload["serving_vs_eager_s"] == 0
        and payload["serving_vs_eager_g_bits"]
        and payload["vec_vs_eager_q"] == 0
        and payload["vec_vs_eager_s"] == 0
        and payload["vec_vs_eager_g_bits"]
        and payload["launches"]["delta_is_missing_1x1"]
        and not payload["promoted"]
        and not payload["patched_serving"]
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    out = HERE / "gate_serving_vec_20423.json"
    out.write_text(text + "\n")
    print(text)
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
