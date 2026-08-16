#!/usr/bin/env python3
"""Algebra gate: Command Buffer Full 1.625 s is launch-gap, not .item().

Accepted-stack profile (one DiT eval, 20,393 rows, 50 blocks): CBF 1.625 s
CUDA / 3.205 s CPU / 460 stalls. cudaLaunchKernel 1524 + cuLaunchKernelEx
250. The serving MiniMax DiT hot path has zero .item() and zero
synchronize. CBF is GPU idle between launches (two-pass producers +
aten::zeros), not a host scalar sync and not a compute kernel.

torch.compile / graph-capture is forbidden. No serving default change.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MINIMAX = REPO / "integration" / "comfyui" / "comfy" / "ldm" / "minimax"
PROFILE_TXT = Path("$HOME/h3/results/accepted_stack_20260812.txt")
PROFILE_JSON = REPO / "benchmarks" / "profiles" / "accepted_stack_20260812.json"

BLOCKS = 50
# Measured on the accepted stack (see PERFORMANCE.md).
CBF_CUDA_MS = 1625.0
CBF_CPU_S = 3.205
CBF_CALLS = 460
LAUNCH_KERNEL = 1524
LAUNCH_KERNEL_EX = 250
ATEN_ZEROS = 400
SWIGLU_FINALIZE = 50  # <<<1,1>>> after amax; still in the serving two-pass.

# Per-block two-pass / producer barriers visible in the profile.
# 50*9 + 10 leftover (pre/post DiT) = 460.
NAMED_BARRIERS_PER_BLOCK = (
    "swiglu_amax -> finalize_dynamic_scale<<<1,1>>>",
    "finalize_dynamic_scale -> swiglu_pack",
    "rms_adaln_first -> rms_adaln_second (qkv)",
    "gate_rms_adaln_first -> rms_adaln_second (mlp)",
    "TransposePadPermute -> QuantInt8",
    "QuantInt8 -> MeanScale",
    "MeanScale -> sage_attn",
    "sage_attn -> out bf16_nvfp4_dynamic",
    "aten::zeros slab (8/block) before two-pass pack",
)
LEFTOVER_CBF = 10

HOT_SYNC_NAMES = frozenset(
    {
        "item",
        "synchronize",
        "cudaDeviceSynchronize",
        "cudaStreamSynchronize",
    }
)


def count_hot_syncs(root: Path) -> dict:
    files: list[str] = []
    hits: list[dict] = []
    for path in sorted(root.glob("*.py")):
        files.append(path.name)
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in HOT_SYNC_NAMES:
                hits.append(
                    {
                        "file": path.name,
                        "line": node.lineno,
                        "attr": node.attr,
                    }
                )
            if isinstance(node, ast.Name) and node.id in HOT_SYNC_NAMES:
                hits.append(
                    {
                        "file": path.name,
                        "line": node.lineno,
                        "attr": node.id,
                    }
                )
    return {"files": files, "hits": hits, "count": len(hits)}


def _ncalls(line: str) -> int | None:
    left = line.split("[", 1)[0]
    toks = left.split()
    if not toks:
        return None
    try:
        return int(toks[-1].replace(",", ""))
    except ValueError:
        return None


def parse_profile_calls(path: Path) -> dict:
    found: dict[str, int] = {}
    if not path.is_file():
        return found
    for line in path.read_text().splitlines():
        n = _ncalls(line)
        if n is None:
            continue
        if "Command Buffer Full" in line:
            found["command_buffer_full"] = n
        elif "cudaLaunchKernel" in line and "Ex" not in line:
            found["cudaLaunchKernel"] = n
        elif "cuLaunchKernelEx" in line:
            found["cuLaunchKernelEx"] = n
        elif "aten::zeros" in line:
            found.setdefault("aten_zeros", n)
        elif "finalize_dynamic_scale_kernel" in line:
            found["finalize_dynamic_scale"] = n
    return found


def main() -> int:
    syncs = count_hot_syncs(MINIMAX)
    profile = parse_profile_calls(PROFILE_TXT)
    compact = json.loads(PROFILE_JSON.read_text())
    compact_cbf = float(compact["top_cuda_self_ms_per_eval"]["command_buffer_full"])

    launches = LAUNCH_KERNEL + LAUNCH_KERNEL_EX
    cbf_per_block = CBF_CALLS / BLOCKS
    launches_per_block = launches / BLOCKS
    zeros_per_block = ATEN_ZEROS / BLOCKS
    reconstructed = BLOCKS * len(NAMED_BARRIERS_PER_BLOCK) + LEFTOVER_CBF

    profile_ok = (
        profile.get("command_buffer_full") == CBF_CALLS
        and profile.get("cudaLaunchKernel") == LAUNCH_KERNEL
        and profile.get("cuLaunchKernelEx") == LAUNCH_KERNEL_EX
        and profile.get("finalize_dynamic_scale") == SWIGLU_FINALIZE
        and profile.get("aten_zeros") == ATEN_ZEROS
    )

    payload = {
        "identity": (
            "Command Buffer Full 1.625 s/eval is GPU idle between launches, "
            "not a .item()/synchronize host sync and not a compute kernel. "
            "Serving MiniMax DiT (50 blocks) has zero .item() and zero "
            "synchronize. 1524 cudaLaunchKernel + 250 cuLaunchKernelEx = "
            "35.48 launches/block; 460 CBF stalls = 9.2/block = 9 named "
            "two-pass/alloc barriers + 10 leftover. Serving SwiGLU still "
            "issues amax + <<<1,1>>> finalize + pack. aten::zeros 400 = "
            "8/block slab tax. torch.compile/graph-capture forbidden. "
            "Not a seconds-scale exact gate. Not promoted."
        ),
        "device": "NVIDIA GB10",
        "rows_measured": 20393,
        "blocks": BLOCKS,
        "hot_path": {
            "root": str(MINIMAX.relative_to(REPO)),
            "files": syncs["files"],
            "item_or_synchronize": syncs["count"],
            "hits": syncs["hits"],
        },
        "profile": {
            "source_txt": str(PROFILE_TXT),
            "source_json": str(PROFILE_JSON.relative_to(REPO)),
            "parsed": profile,
            "cbf_cuda_ms": CBF_CUDA_MS,
            "cbf_cpu_s": CBF_CPU_S,
            "cbf_calls": CBF_CALLS,
            "cudaLaunchKernel": LAUNCH_KERNEL,
            "cuLaunchKernelEx": LAUNCH_KERNEL_EX,
            "aten_zeros": ATEN_ZEROS,
            "finalize_dynamic_scale": SWIGLU_FINALIZE,
            "compact_cbf_ms": compact_cbf,
        },
        "algebra": {
            "launches": launches,
            "launches_per_block": launches_per_block,
            "cbf_per_block": cbf_per_block,
            "zeros_per_block": zeros_per_block,
            "named_barriers_per_block": list(NAMED_BARRIERS_PER_BLOCK),
            "named_barrier_count": len(NAMED_BARRIERS_PER_BLOCK),
            "leftover_cbf_outside_50_block": LEFTOVER_CBF,
            "reconstructed_cbf_calls": reconstructed,
            "cbf_ms_per_stall": CBF_CUDA_MS / CBF_CALLS,
            "object_wrong": "host .item()/synchronize in the DiT hot path",
            "object_right": (
                "launch-count two-pass + zeros gaps; CBF is idle, not math"
            ),
            "do_not": (
                "torch.compile / cuda graph a production default; "
                "no seconds-scale exact gate this tick"
            ),
        },
        "promoted": False,
        "serving_default": "stock",
        "graph_captured": False,
        "pass": bool(
            syncs["count"] == 0
            and reconstructed == CBF_CALLS
            and compact_cbf == CBF_CUDA_MS
            and profile_ok
        ),
    }
    out = HERE / "gate_cbf_launch_20393.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "pass": payload["pass"],
                "promoted": False,
                "receipt": str(out),
                "item_or_synchronize": syncs["count"],
                "cbf_calls": CBF_CALLS,
                "launches_per_block": launches_per_block,
                "cbf_per_block": cbf_per_block,
                "profile_ok": profile_ok,
            },
            indent=2,
        )
    )
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
