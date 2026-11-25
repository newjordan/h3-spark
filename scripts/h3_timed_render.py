#!/usr/bin/env python3
"""Submit one H3 workflow to a local ComfyUI and print one JSON timing record.

``--preset lab`` (the default) is an A/B baseline: stock attention, exact
modulation, stock SwiGLU and RMS fusions, and Q fusion auto. It does not
match ``config/h3_spark_serving.env``. ``--preset serving`` does.

The workflow builder must return a prompt that uses the fixed H3 node ids:
127 (model), 124 and 126 (model consumers), 136 (reference size), and 125
when ``--sampler-only`` is set.

    python3 scripts/h3_timed_render.py --preset serving --tag warm \\
        --workflow-builder path/to/builder.py \\
        --prompt "..." --refs a.png,b.png
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


_LAB_PRESET = {
    "attention": "stock",
    "fusion": "exact",
    "swiglu_nvfp4_fusion": "stock",
    "rms_adaln_nvfp4_fusion": "stock",
    "q_rms_rope_int8_fusion": "auto",
    "crossblock_gate_qkv_fusion": "stock",
}
_SERVING_PRESET = {
    "attention": "sage2-quality",
    "fusion": "exact",
    "swiglu_nvfp4_fusion": "auto",
    "rms_adaln_nvfp4_fusion": "auto",
    "q_rms_rope_int8_fusion": "auto",
    "crossblock_gate_qkv_fusion": "stock",
}
_PROMPT_NODES = ("127", "124", "126", "136")


def load_workflow_builder(path):
    spec = importlib.util.spec_from_file_location("h3_workflow_builder", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load workflow builder: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, "workflow", None)):
        raise AttributeError(f"workflow builder has no callable workflow(): {path}")
    return mod.workflow


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Submit one H3 workflow to a local ComfyUI and print one JSON "
            "timing record. --preset lab is an A/B baseline and does not "
            "match config/h3_spark_serving.env; --preset serving does."
        ),
        epilog=(
            "Stdout is one JSON object per event. A busy queue, a rejected "
            "submit, or an unreachable API is an error record and a non-zero "
            "status."
        ),
    )
    ap.add_argument(
        "--preset",
        choices=["lab", "serving"],
        default="lab",
        help="lab = stock attention and stock NVFP4 fusions; serving = config/h3_spark_serving.env",
    )
    ap.add_argument(
        "--workflow-builder",
        default=os.environ.get("H3_WORKFLOW_BUILDER"),
        help="Python module exposing workflow() (or H3_WORKFLOW_BUILDER)",
    )
    ap.add_argument("--api", default="http://127.0.0.1:18188", help="ComfyUI base URL")
    ap.add_argument("--seed", type=int, default=26081201)
    ap.add_argument("--tag", required=True, help="run label stored in the JSON record")
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
    ap.add_argument("--poll", type=int, default=10, help="seconds between history polls")
    ap.add_argument(
        "--compile",
        choices=["inductor", "cudagraphs"],
        default=None,
        help="wrap the model in TorchCompileModel with this backend",
    )
    ap.add_argument(
        "--attention",
        choices=["stock", "sage2-quality", "sage2-fast"],
        default=None,
        help="override the preset attention backend",
    )
    ap.add_argument(
        "--fusion",
        choices=["stock", "exact", "aggressive"],
        default=None,
        help="override the preset segmented modulation mode",
    )
    ap.add_argument(
        "--swiglu-nvfp4-fusion",
        choices=["stock", "static", "auto"],
        default=None,
        help="override the preset FC2 SwiGLU-to-NVFP4 fusion mode",
    )
    ap.add_argument(
        "--rms-adaln-nvfp4-fusion",
        choices=["stock", "auto"],
        default=None,
        help="override the preset RMSNorm+AdaLN-to-NVFP4 fusion mode",
    )
    ap.add_argument(
        "--q-rms-rope-int8-fusion",
        choices=["stock", "auto"],
        default=None,
        help="override the preset Q RMSNorm+RoPE-to-Sage-INT8 fusion mode",
    )
    ap.add_argument(
        "--crossblock-gate-qkv-fusion",
        choices=["stock", "auto"],
        default=None,
        help="override the preset cross-block gate->QKV mode (HOLD; serving default is stock)",
    )
    ap.add_argument(
        "--nvfp4-scales",
        choices=["dynamic", "calibrate", "validate", "static"],
        default="dynamic",
        help="NVFP4 activation-scale mode",
    )
    ap.add_argument("--nvfp4-prefix", default="", help="calibration artifact prefix")
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
    ap.add_argument(
        "--profile",
        action="store_true",
        help="wrap the model in H3ProfilerModel",
    )
    ap.add_argument("--profile-out", default="h3_prof", help="profiler output prefix")
    ap.add_argument("--profile-wait", type=int, default=2)
    ap.add_argument("--profile-active", type=int, default=1)
    ap.add_argument(
        "--sampler-only",
        action="store_true",
        help="stop at the sampler preview; skips VAE, mux, and encode",
    )
    ap.add_argument(
        "--skip-attention-calibration",
        action="store_true",
        help="skip the one-time Sage-vs-SDPA quality calibration",
    )
    args = ap.parse_args(argv)
    preset = _SERVING_PRESET if args.preset == "serving" else _LAB_PRESET
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if not args.workflow_builder:
        ap.error("--workflow-builder or H3_WORKFLOW_BUILDER is required")
    if args.nvfp4_scales != "dynamic" and not args.nvfp4_prefix:
        ap.error("--nvfp4-prefix is required outside dynamic scale mode")
    if args.nvfp4_scales in ("calibrate", "validate") and not args.nvfp4_concept:
        ap.error("--nvfp4-concept is required for calibrate/validate")
    return args


def _split_refs(raw: str) -> list[str]:
    refs = [part.strip() for part in raw.split(",") if part.strip()]
    if not refs:
        raise ValueError("refs is empty after trimming whitespace")
    return refs


def _require_nodes(prompt: dict, *, sampler_only: bool) -> None:
    needed = list(_PROMPT_NODES)
    if sampler_only:
        needed.append("125")
    missing = [node for node in needed if node not in prompt]
    if missing:
        raise ValueError(
            "workflow() prompt is missing node id(s) "
            + ", ".join(missing)
            + ". Expected 127 (model), 124 and 126 (model consumers), "
            "136 (reference size)"
            + (", 125 (sampler output)" if sampler_only else "")
            + "."
        )


def prepare_job(args: argparse.Namespace) -> dict:
    """Build the ComfyUI prompt. Does not touch the network."""
    workflow = load_workflow_builder(args.workflow_builder)
    refs = _split_refs(args.refs)
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
    prompt = job.setdefault("prompt", {})
    _require_nodes(prompt, sampler_only=args.sampler_only)
    job["client_id"] = f"h3-ladder-{args.tag}"
    if args.steps != 20:
        prompt["124"]["inputs"]["steps"] = args.steps
    prompt["136"]["inputs"]["ref_image_size"] = args.ref_image_size
    # Rebuild the serving wrapper deterministically. The builder's
    # environment-backed defaults must not leak into an A/B.
    prompt.pop("202", None)
    model_ref = ["127", 0]
    if args.compile:
        prompt["200"] = {
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
        prompt["202"] = {
            "class_type": "H3SageAttentionModel",
            "inputs": {
                "model": model_ref,
                "mode": (
                    {"sage2-quality": "quality", "sage2-fast": "fast"}.get(
                        args.attention, "stock"
                    )
                ),
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
        prompt["203"] = {"class_type": class_type, "inputs": inputs}
        model_ref = ["203", 0]
    if args.profile:
        prompt["201"] = {
            "class_type": "H3ProfilerModel",
            "inputs": {
                "model": model_ref,
                "wait_calls": args.profile_wait,
                "active_calls": args.profile_active,
                "out_prefix": args.profile_out,
            },
        }
        model_ref = ["201", 0]
    prompt["124"]["inputs"]["model"] = model_ref
    prompt["126"]["inputs"]["model"] = model_ref
    if args.sampler_only:
        prompt["92"] = {
            "class_type": "PreviewAny",
            "inputs": {"source": ["125", 0]},
        }
    return job


def result_record(
    args: argparse.Namespace,
    *,
    status: str | None,
    wall_seconds: float,
    executor_seconds: float | None,
    outputs: list,
    host: str,
) -> dict:
    return {
        "tag": args.tag,
        "RESULT": status,
        "wall_seconds": round(wall_seconds, 1),
        "executor_seconds": executor_seconds,
        "host": host,
        "seed": args.seed,
        "steps": args.steps,
        "length": args.length,
        "preset": args.preset,
        "ref_image_size": args.ref_image_size,
        "attention": args.attention,
        "fusion": args.fusion,
        "swiglu_nvfp4_fusion": args.swiglu_nvfp4_fusion,
        "rms_adaln_nvfp4_fusion": args.rms_adaln_nvfp4_fusion,
        "q_rms_rope_int8_fusion": args.q_rms_rope_int8_fusion,
        "crossblock_gate_qkv_fusion": args.crossblock_gate_qkv_fusion,
        "nvfp4_scales": args.nvfp4_scales,
        "sampler_only": args.sampler_only,
        "outputs": outputs,
    }


def _emit(payload: dict) -> None:
    print(json.dumps(payload), flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        job = prepare_job(args)
    except (OSError, ValueError, ImportError, AttributeError, KeyError) as exc:
        _emit({"tag": args.tag, "error": str(exc)})
        return 2

    try:
        with urllib.request.urlopen(f"{args.api}/queue", timeout=10) as response:
            queue = json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:2000]
        _emit({"tag": args.tag, "error": f"QUEUE_REJECTED {exc.code}", "body": body})
        return 1
    except urllib.error.URLError as exc:
        _emit({"tag": args.tag, "error": f"API_UNREACHABLE {exc.reason}"})
        return 1
    if queue.get("queue_running") or queue.get("queue_pending"):
        _emit({"tag": args.tag, "error": "ABORT: queue not empty"})
        return 1

    started = time.time()
    request = urllib.request.Request(
        f"{args.api}/prompt",
        data=json.dumps(job).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            receipt = json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:2000]
        _emit({"tag": args.tag, "error": f"SUBMIT_REJECTED {exc.code}", "body": body})
        return 1
    except urllib.error.URLError as exc:
        _emit({"tag": args.tag, "error": f"API_UNREACHABLE {exc.reason}"})
        return 1
    prompt_id = receipt["prompt_id"]
    _emit(
        {
            "tag": args.tag,
            "submitted": prompt_id,
            "seed": args.seed,
            "host": socket.gethostname(),
            "refs": len(_split_refs(args.refs)),
            "steps": args.steps,
            "length": args.length,
            "preset": args.preset,
            "attention": args.attention,
        }
    )

    while time.time() - started < args.timeout:
        time.sleep(args.poll)
        try:
            with urllib.request.urlopen(
                f"{args.api}/history/{prompt_id}", timeout=10
            ) as response:
                history = json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            _emit({"tag": args.tag, "poll_error": str(exc)})
            continue
        if prompt_id not in history:
            continue
        entry = history[prompt_id]
        status = entry.get("status", {})
        if not status.get("completed") and status.get("status_str") != "error":
            continue
        wall = time.time() - started
        stamps = {}
        for name, payload in status.get("messages", []):
            if isinstance(payload, dict) and "timestamp" in payload:
                stamps[name] = payload["timestamp"]
        executor_seconds = None
        if "execution_start" in stamps and "execution_success" in stamps:
            executor_seconds = round(
                (stamps["execution_success"] - stamps["execution_start"]) / 1000, 1
            )
        outputs = []
        for node_out in entry.get("outputs", {}).values():
            for kind in ("images", "video", "gifs", "audio"):
                for item in node_out.get(kind, []):
                    outputs.append(item.get("filename"))
        _emit(
            result_record(
                args,
                status=status.get("status_str"),
                wall_seconds=wall,
                executor_seconds=executor_seconds,
                outputs=outputs,
                host=socket.gethostname(),
            )
        )
        return 0 if status.get("status_str") == "success" else 2

    _emit({"tag": args.tag, "error": f"TIMEOUT {args.timeout}s"})
    return 3


if __name__ == "__main__":
    sys.exit(main())
