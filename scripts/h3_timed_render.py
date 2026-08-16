#!/usr/bin/env python3
"""Timed single-render H3 runner for controlled serving comparisons.

Loads a caller-supplied module exposing workflow(), submits one job to an idle
ComfyUI API, polls history until terminal, and prints one JSON result.

    python3 h3_timed_render.py --tag baseline_cold [--api http://127.0.0.1:18188]
                               --prompt TEXT --refs INPUTS [--seed 26081201]
                               [--steps 20] [--length 124]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request


def load_workflow_builder(path):
    spec = importlib.util.spec_from_file_location("h3_workflow_builder", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load workflow builder: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, "workflow", None)):
        raise AttributeError(f"workflow builder has no callable workflow(): {path}")
    return mod.workflow


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--workflow-builder",
        default=os.environ.get("H3_WORKFLOW_BUILDER"),
        help="path to a Python module exposing workflow() (or H3_WORKFLOW_BUILDER)",
    )
    ap.add_argument("--api", default="http://127.0.0.1:18188")
    ap.add_argument("--seed", type=int, default=26081201)
    ap.add_argument("--tag", required=True, help="run label; also output filename prefix")
    ap.add_argument("--prompt", required=True)
    ap.add_argument(
        "--refs",
        required=True,
        help="comma-separated ComfyUI input-relative reference paths",
    )
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--length", type=int, default=124)
    ap.add_argument("--ref-image-size", choices=["match", "half", "max"], default="match")
    ap.add_argument("--timeout", type=int, default=10800, help="max seconds to wait")
    ap.add_argument("--poll", type=int, default=10)
    ap.add_argument("--compile", choices=["inductor", "cudagraphs"], default=None,
                    help="wrap the unet in TorchCompileModel with this backend")
    ap.add_argument("--attention", choices=["stock", "sage2-quality", "sage2-fast"],
                    default="stock", help="H3-scoped attention backend")
    ap.add_argument("--fusion", choices=["stock", "exact", "aggressive"], default="exact",
                    help="H3 segmented modulation kernel mode")
    ap.add_argument(
        "--swiglu-nvfp4-fusion",
        choices=["stock", "static", "auto"],
        default="stock",
        help="H3 FC2 SwiGLU-to-NVFP4 fusion mode",
    )
    ap.add_argument(
        "--rms-adaln-nvfp4-fusion",
        choices=["stock", "auto"],
        default="stock",
        help="H3 RMSNorm+AdaLN-to-NVFP4 fusion mode (independent A/B switch)",
    )
    ap.add_argument(
        "--q-rms-rope-int8-fusion",
        choices=["stock", "auto"],
        default="auto",
        help="H3 Q RMSNorm+RoPE-to-Sage-INT8 fusion mode",
    )
    ap.add_argument(
        "--crossblock-gate-qkv-fusion",
        choices=["stock", "auto"],
        default="stock",
        help="H3 cross-block final-gate -> next-QKV fusion (HOLD; default stock)",
    )
    ap.add_argument(
        "--nvfp4-scales",
        choices=["dynamic", "calibrate", "validate", "static"],
        default="dynamic",
        help="NVFP4 activation-scale mode",
    )
    ap.add_argument(
        "--nvfp4-prefix",
        default="",
        help="calibration artifact prefix",
    )
    ap.add_argument("--nvfp4-margin", type=float, default=1.20)
    ap.add_argument(
        "--nvfp4-excluded-layers",
        default="",
        help="comma/newline-separated layers that must retain dynamic scaling",
    )
    ap.add_argument(
        "--nvfp4-concept",
        default="",
        help="concept path required for calibrate/validate modes",
    )
    ap.add_argument("--profile", action="store_true",
                    help="wrap the unet in H3ProfilerModel (kernel-time table + chrome trace)")
    ap.add_argument("--profile-out", default="h3_prof",
                    help="output prefix for profiler table/trace")
    ap.add_argument("--profile-wait", type=int, default=2,
                    help="diffusion calls to warm before profiling")
    ap.add_argument("--profile-active", type=int, default=1,
                    help="diffusion calls to capture")
    ap.add_argument(
        "--sampler-only",
        action="store_true",
        help=(
            "stop at the sampler and preview its latent metadata; skips both "
            "VAEs, audio/video assembly, and MP4 encoding for short kernel smokes"
        ),
    )
    ap.add_argument(
        "--skip-attention-calibration",
        action="store_true",
        help="skip the one-time Sage-vs-SDPA quality calibration in short smokes",
    )
    args = ap.parse_args()
    if not args.workflow_builder:
        ap.error("--workflow-builder or H3_WORKFLOW_BUILDER is required")
    if args.nvfp4_scales != "dynamic" and not args.nvfp4_prefix:
        ap.error("--nvfp4-prefix is required outside dynamic scale mode")
    if args.nvfp4_scales in ("calibrate", "validate") and not args.nvfp4_concept:
        ap.error("--nvfp4-concept is required for calibrate/validate")

    workflow = load_workflow_builder(args.workflow_builder)
    refs = [r for r in args.refs.split(",") if r]
    job = workflow(
        args.seed,
        f"h3_ladder/{args.tag}_seed{args.seed}",
        args.prompt,
        refs,
        length=args.length,
        attention="stock",
        ref_image_size=args.ref_image_size,
        modulation_fusion=args.fusion,
        swiglu_nvfp4_fusion=args.swiglu_nvfp4_fusion,
        rms_adaln_nvfp4_fusion=args.rms_adaln_nvfp4_fusion,
        q_rms_rope_int8_fusion=args.q_rms_rope_int8_fusion,
        nvfp4_static_artifact="",
    )
    job["client_id"] = f"h3-ladder-{args.tag}"
    if args.steps != 20:
        job["prompt"]["124"]["inputs"]["steps"] = args.steps
    job["prompt"]["136"]["inputs"]["ref_image_size"] = args.ref_image_size
    # Rebuild the serving wrapper deterministically below. The production
    # builder has environment-backed defaults, which must not leak into A/Bs.
    job["prompt"].pop("202", None)
    model_ref = ["127", 0]
    if args.compile:
        job["prompt"]["200"] = {
            "class_type": "TorchCompileModel",
            "inputs": {"model": model_ref, "backend": args.compile},
        }
        model_ref = ["200", 0]
    if (
        args.attention != "stock"
        or args.fusion != "exact"
        or args.swiglu_nvfp4_fusion != "stock"
        or args.rms_adaln_nvfp4_fusion != "stock"
        or args.q_rms_rope_int8_fusion != "stock"
        or args.crossblock_gate_qkv_fusion != "stock"
    ):
        job["prompt"]["202"] = {
            "class_type": "H3SageAttentionModel",
            "inputs": {
                "model": model_ref,
                "mode": ({"sage2-quality": "quality", "sage2-fast": "fast"}
                         .get(args.attention, "stock")),
                "calibrate_first_call": not args.skip_attention_calibration,
                "modulation_fusion": args.fusion,
                "swiglu_nvfp4_fusion": args.swiglu_nvfp4_fusion,
                "rms_adaln_nvfp4_fusion": args.rms_adaln_nvfp4_fusion,
                "q_rms_rope_int8_fusion": args.q_rms_rope_int8_fusion,
                "crossblock_gate_qkv_fusion": args.crossblock_gate_qkv_fusion,
            },
        }
        model_ref = ["202", 0]
    if args.nvfp4_scales != "dynamic":
        common = {
            "model": model_ref,
            "artifact_prefix": args.nvfp4_prefix,
        }
        if args.nvfp4_scales == "calibrate":
            class_type = "H3CalibrateNVFP4InputScales"
            inputs = {
                **common,
                "margin": args.nvfp4_margin,
                "concept_path": args.nvfp4_concept,
                "model_id": "minimax_h3_ref2va_pruned_nvfp4.safetensors",
            }
        elif args.nvfp4_scales == "validate":
            class_type = "H3ValidateNVFP4InputScales"
            inputs = {
                **common,
                "validation_concept_path": args.nvfp4_concept,
                "expected_model_id": "minimax_h3_ref2va_pruned_nvfp4.safetensors",
                "on_mismatch": "error",
            }
        else:
            class_type = "H3ApplyNVFP4InputScales"
            inputs = {
                **common,
                "on_mismatch": "error",
                "expected_model_id": "minimax_h3_ref2va_pruned_nvfp4.safetensors",
            }
        if args.nvfp4_scales in ("validate", "static") and args.nvfp4_excluded_layers:
            inputs["excluded_layers"] = args.nvfp4_excluded_layers
        job["prompt"]["203"] = {"class_type": class_type, "inputs": inputs}
        model_ref = ["203", 0]
    if args.profile:
        job["prompt"]["201"] = {
            "class_type": "H3ProfilerModel",
            "inputs": {"model": model_ref, "wait_calls": args.profile_wait,
                       "active_calls": args.profile_active,
                       "out_prefix": args.profile_out},
        }
        model_ref = ["201", 0]
    job["prompt"]["124"]["inputs"]["model"] = model_ref
    job["prompt"]["126"]["inputs"]["model"] = model_ref
    if args.sampler_only:
        # PreviewAny is an output node accepting any Comfy type. Pointing it at
        # the sampler keeps the exact model/conditioning/latent shape while
        # pruning the video VAE, audio VAE, mux, and encoder from execution.
        job["prompt"]["92"] = {
            "class_type": "PreviewAny",
            "inputs": {"source": ["125", 0]},
        }

    # refuse to time on a busy box -- the number would be noise
    with urllib.request.urlopen(f"{args.api}/queue", timeout=10) as r:
        q = json.load(r)
    if q.get("queue_running") or q.get("queue_pending"):
        print(json.dumps({"tag": args.tag, "error": "ABORT: queue not empty"}))
        return 1

    t0 = time.time()
    req = urllib.request.Request(
        f"{args.api}/prompt",
        data=json.dumps(job).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            receipt = json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:2000]
        print(json.dumps({"tag": args.tag, "error": f"SUBMIT_REJECTED {e.code}", "body": body}))
        return 1
    pid = receipt["prompt_id"]
    print(json.dumps({"tag": args.tag, "submitted": pid, "seed": args.seed,
                      "host": socket.gethostname(), "refs": len(refs),
                      "steps": args.steps, "length": args.length}), flush=True)

    while time.time() - t0 < args.timeout:
        time.sleep(args.poll)
        try:
            with urllib.request.urlopen(f"{args.api}/history/{pid}", timeout=10) as r:
                hist = json.load(r)
        except Exception as e:  # transient poll failure: keep waiting
            print(json.dumps({"tag": args.tag, "poll_error": str(e)}), flush=True)
            continue
        if pid not in hist:
            continue
        entry = hist[pid]
        status = entry.get("status", {})
        if not status.get("completed") and status.get("status_str") != "error":
            continue
        wall = time.time() - t0
        stamps = {}
        for name, payload in status.get("messages", []):
            if isinstance(payload, dict) and "timestamp" in payload:
                stamps[name] = payload["timestamp"]
        exec_s = None
        if "execution_start" in stamps and "execution_success" in stamps:
            exec_s = round((stamps["execution_success"] - stamps["execution_start"]) / 1000, 1)
        outputs = []
        for node_out in entry.get("outputs", {}).values():
            for kind in ("images", "video", "gifs", "audio"):
                for item in node_out.get(kind, []):
                    outputs.append(item.get("filename"))
        print(json.dumps({
            "tag": args.tag,
            "RESULT": status.get("status_str"),
            "wall_seconds": round(wall, 1),
            "executor_seconds": exec_s,
            "host": socket.gethostname(),
            "seed": args.seed,
            "steps": args.steps,
            "ref_image_size": args.ref_image_size,
            "attention": args.attention,
            "fusion": args.fusion,
            "swiglu_nvfp4_fusion": args.swiglu_nvfp4_fusion,
            "rms_adaln_nvfp4_fusion": args.rms_adaln_nvfp4_fusion,
            "nvfp4_scales": args.nvfp4_scales,
            "sampler_only": args.sampler_only,
            "outputs": outputs,
        }), flush=True)
        return 0 if status.get("status_str") == "success" else 2

    print(json.dumps({"tag": args.tag, "error": f"TIMEOUT {args.timeout}s"}), flush=True)
    return 3


if __name__ == "__main__":
    sys.exit(main())
