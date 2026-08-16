#!/usr/bin/env python3
"""Byte-exact gate: new swiglu_nvfp4_dynamic (no <<<1,1>>>) vs old serving .so.

Same entrypoint, G inlined in the 4-wide pack. Must match old packed q/s/G
and eager kitchen. Does not start Comfy.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import comfy_kitchen as ck


HERE = Path(__file__).resolve().parent
NEW_SO = HERE / "cmake-build-120" / "h3_swiglu_nvfp4_native_v1.so"
OLD_SO = Path(
    "$HOME/comfyui-h3-current/comfy/ldm/minimax/"
    "swiglu_nvfp4_native_v1.so.bak-pre-vec"
)
ROWS = 20423
WIDTH = 28672
REPEATS = 3


def load_so(path: Path):
    spec = importlib.util.spec_from_file_location(
        "h3_swiglu_nvfp4_native_v1", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def byte_diff(left: torch.Tensor, right: torch.Tensor) -> int:
    return int((left.view(torch.uint8) != right.view(torch.uint8)).sum().item())


def time_ms(fn) -> float:
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(REPEATS):
        starter.record()
        fn()
        ender.record()
        torch.cuda.synchronize()
        samples.append(starter.elapsed_time(ender))
    return min(samples)


def eager_kitchen(raw: torch.Tensor):
    gate, up = raw.chunk(2, dim=-1)
    activated = F.silu(gate).mul_(up)
    scale = (activated.abs().amax() / (6.0 * 448.0)).to(torch.float32).reshape(1)
    packed, block_scales = ck.quantize_nvfp4(
        activated, scale, pad_16x=True, hi_first=True
    )
    return packed, block_scales, scale


def dump_old() -> Path:
    import os
    import subprocess
    import sys

    dump = Path("/tmp/h3_swiglu_old_dump.pt")
    env = os.environ.copy()
    env["H3_DUMP_OLD_SWIGLU"] = str(dump)
    subprocess.check_call([sys.executable, str(Path(__file__).resolve())], env=env)
    return dump


def main() -> int:
    import os

    dump_path = os.environ.get("H3_DUMP_OLD_SWIGLU")
    if dump_path:
        import shutil

        old_so = Path("/tmp/h3_swiglu_nvfp4_native_v1.so")
        shutil.copy2(OLD_SO, old_so)
        old = load_so(old_so)
        device = torch.device("cuda")
        torch.manual_seed(26081390)
        raw = torch.randn(ROWS, WIDTH, device=device, dtype=torch.bfloat16)
        old_q, old_s, old_g = old.swiglu_nvfp4_dynamic(raw)
        old_ms = time_ms(lambda: old.swiglu_nvfp4_dynamic(raw))
        torch.save(
            {"q": old_q.cpu(), "s": old_s.cpu(), "g": old_g.cpu(), "ms": old_ms},
            dump_path,
        )
        return 0

    if not NEW_SO.is_file():
        raise FileNotFoundError(NEW_SO)
    if not OLD_SO.is_file():
        raise FileNotFoundError(OLD_SO)
    dumped = torch.load(dump_old(), map_location="cpu", weights_only=False)
    new = load_so(NEW_SO)
    device = torch.device("cuda")
    torch.manual_seed(26081390)
    raw = torch.randn(ROWS, WIDTH, device=device, dtype=torch.bfloat16)

    old_q = dumped["q"].to(device)
    old_s = dumped["s"].to(device)
    old_g = dumped["g"].to(device)
    new_q, new_s, new_g = new.swiglu_nvfp4_dynamic(raw)
    kit_q, kit_s, kit_g = eager_kitchen(raw)

    payload = {
        "identity": (
            "Serving swiglu_nvfp4_dynamic now inlines G from amax bits in "
            "the 4-wide pack. No finalize_dynamic_scale<<<1,1>>>. Same "
            "entrypoint. Bytes match the pre-change serving .so and eager "
            "kitchen."
        ),
        "device": torch.cuda.get_device_name(0),
        "rows": ROWS,
        "new_vs_old_q": byte_diff(new_q, old_q),
        "new_vs_old_s": byte_diff(new_s, old_s),
        "new_vs_old_g_bits": bool(new_g.view(torch.int32).eq(old_g.view(torch.int32)).all().item()),
        "new_vs_kitchen_q": byte_diff(new_q[:ROWS], kit_q[:ROWS]),
        "new_vs_kitchen_s": byte_diff(new_s[: kit_s.shape[0]], kit_s),
        "new_vs_kitchen_g_bits": bool(
            new_g.view(torch.int32).eq(kit_g.view(torch.int32)).all().item()
        ),
        "old_min_ms": round(float(dumped["ms"]), 4),
        "new_min_ms": round(time_ms(lambda: new.swiglu_nvfp4_dynamic(raw)), 4),
        "old_g": float(old_g.item()),
        "new_g": float(new_g.item()),
    }
    payload["pass"] = (
        payload["new_vs_old_q"] == 0
        and payload["new_vs_old_s"] == 0
        and payload["new_vs_old_g_bits"]
        and payload["new_vs_kitchen_q"] == 0
        and payload["new_vs_kitchen_s"] == 0
        and payload["new_vs_kitchen_g_bits"]
    )
    out = HERE / "gate_serving_inline_g_20423.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"pass": payload["pass"], "receipt": str(out)}, indent=2))
    print(json.dumps({k: payload[k] for k in (
        "new_vs_old_q", "new_vs_old_s", "new_vs_old_g_bits",
        "new_vs_kitchen_q", "old_min_ms", "new_min_ms"
    )}, indent=2))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
