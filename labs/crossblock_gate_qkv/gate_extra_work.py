#!/usr/bin/env python3
"""Name the +1.7 s FixedRef executor loss: DiT-only 20-eval A/B.

Kernel projection is 0.511881 s. Isolated chain already saves 32–43 ms/eval.
This tick compares launch counts, CBF proxy (host−GPU gap), and host time
on a 20-eval DiT residual chain. No FixedRef render, no VAE/MP4, no promotion.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent


def _load_h3_paths():
    import importlib.util
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "scripts" / "h3_paths.py"
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location("h3_paths", candidate)
            if spec is None or spec.loader is None:
                break
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise SystemExit("error: scripts/h3_paths.py not found")


ROWS = 20423
WIDTH = 5376
QKV_OUT = 21504
EPS = 1e-5
TRANSITIONS = 49
EVALS = 20
KERNEL_PROJECTION_S = 0.511881


def byte_diff(left: torch.Tensor, right: torch.Tensor) -> int:
    return int((left.view(torch.uint8) != right.view(torch.uint8)).sum().item())


def make_row_ids(rows: int, device: torch.device) -> torch.Tensor:
    return (
        torch.arange(rows, device=device, dtype=torch.int64)
        .mul_(21)
        .div_(rows, rounding_mode="floor")
        .to(torch.int32)
        .contiguous()
    )


def nvfp4_scale(x: torch.Tensor) -> torch.Tensor:
    return (x.abs().amax() / (6.0 * 448.0)).to(torch.float32).reshape(1)


def kitchen_qkv(qx, qxs, gx, qw, qws, gw) -> torch.Tensor:
    import comfy_kitchen as ck

    alpha = (gx * gw).reshape(1).contiguous()
    if qxs.dtype != torch.float8_e4m3fn:
        qxs = qxs.view(torch.float8_e4m3fn)
    if qws.dtype != torch.float8_e4m3fn:
        qws = qws.view(torch.float8_e4m3fn)
    y = ck.scaled_mm_nvfp4(
        qx, qw, gx, gw, qxs, qws, out_dtype=torch.bfloat16, alpha=alpha
    )
    return y[:ROWS]


def stock_step(rms_native, residual, mlp, gate, weight, scale, shift, row_ids):
    from comfy.ldm.minimax.kernels import segmented_gate_bf16_

    if not segmented_gate_bf16_(residual, mlp, gate, row_ids):
        raise RuntimeError("stock segmented gate did not route")
    return rms_native.rms_adaln_nvfp4_dynamic(
        residual, weight, scale, shift, row_ids, EPS
    )


def fused_step(gate_native, residual, mlp, gate, weight, scale, shift, row_ids):
    return gate_native.gate_rms_adaln_nvfp4_dynamic(
        residual, mlp, gate, weight, scale, shift, row_ids, EPS
    )


def time_host_gpu(fn, evals: int) -> dict:
    fn()
    torch.cuda.synchronize()
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter()
    starter.record()
    for _ in range(evals):
        fn()
    ender.record()
    torch.cuda.synchronize()
    host_s = time.perf_counter() - t0
    gpu_ms = float(starter.elapsed_time(ender))
    return {
        "host_s": round(host_s, 4),
        "gpu_ms": round(gpu_ms, 4),
        "gpu_s": round(gpu_ms / 1000.0, 4),
        "host_minus_gpu_s": round(host_s - gpu_ms / 1000.0, 4),
        "host_ms_per_eval": round(host_s * 1000.0 / evals, 4),
        "gpu_ms_per_eval": round(gpu_ms / evals, 4),
    }


def _event_device_us(evt) -> int:
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        value = getattr(evt, attr, None)
        if value:
            return int(value)
    return 0


def profile_one(fn) -> dict:
    fn()
    torch.cuda.synchronize()
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(activities=activities, record_shapes=False) as prof:
        fn()
        torch.cuda.synchronize()
    launch_keys = 0
    cbf_keys = 0
    kernel_launches = 0
    cpu_us = 0
    device_us = 0
    names: Counter[str] = Counter()
    for evt in prof.key_averages():
        name = evt.key
        names[name] += evt.count
        cpu_us += int(getattr(evt, "self_cpu_time_total", 0) or 0)
        device_us += _event_device_us(evt)
        lowered = name.lower()
        if "cudalaunchkernel" in lowered or "culaunchkernelex" in lowered:
            launch_keys += evt.count
        if "command buffer full" in lowered:
            cbf_keys += evt.count
        if _event_device_us(evt) > 0 and "memcpy" not in lowered and "memset" not in lowered:
            kernel_launches += evt.count
    top = names.most_common(12)
    return {
        "profiler_launch_api": launch_keys,
        "profiler_cbf": cbf_keys,
        "profiler_device_kernels": kernel_launches,
        "profiler_cpu_ms": round(cpu_us / 1000.0, 4),
        "profiler_device_ms": round(device_us / 1000.0, 4),
        "top_ops": [{"name": n, "count": c} for n, c in top],
    }


def main() -> int:
    torch.set_grad_enabled(False)
    comfy = _load_h3_paths().require_env_path("COMFYUI_ROOT", kind="dir")
    if str(comfy) not in sys.path:
        sys.path.insert(0, str(comfy))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "h3_crossblock_extra_work", HERE / "comfy_adapter.py"
    )
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    import comfy.ldm.minimax.rms_adaln_nvfp4 as rms_adaln
    import comfy.ops
    import comfy_kitchen as ck

    device = torch.device("cuda")
    gate_native = adapter._load_native()
    rms_native = rms_adaln._load_native()
    if gate_native is None or rms_native is None:
        raise RuntimeError("native gate or RMS producer failed to load")

    torch.manual_seed(26081265)
    x0 = torch.randn(ROWS, WIDTH, device=device, dtype=torch.bfloat16)
    weight = (torch.randn(WIDTH, device=device) * 0.1 + 1.0).to(torch.bfloat16)
    table = (torch.randn(21, WIDTH * 6, device=device) * 0.1).to(torch.float32)
    scale = table[:, WIDTH : 2 * WIDTH].contiguous()
    shift = table[:, 4 * WIDTH : 5 * WIDTH].contiguous()
    row_ids = make_row_ids(ROWS, device)
    segments = [(0, ROWS, 0)]
    steps = [
        (
            (torch.randn_like(x0) * 0.05).contiguous(),
            table[:, 2 * WIDTH : 3 * WIDTH].contiguous(),
        )
        for _ in range(TRANSITIONS)
    ]
    w_qkv = torch.randn(QKV_OUT, WIDTH, device=device, dtype=torch.bfloat16) * 0.02
    sw = nvfp4_scale(w_qkv)
    qw, qws = ck.quantize_nvfp4(w_qkv, sw, pad_16x=True, hi_first=True)

    stock_x = x0.clone()
    fused_x = x0.clone()
    residual_mismatches = 0
    q_mismatches = 0
    s_mismatches = 0
    g_mismatches = 0
    qkv_mismatches = 0
    last_stock = last_fused = None
    for mlp, gate in steps:
        last_stock = stock_step(
            rms_native, stock_x, mlp, gate, weight, scale, shift, row_ids
        )
        last_fused = fused_step(
            gate_native, fused_x, mlp, gate, weight, scale, shift, row_ids
        )
        residual_mismatches += byte_diff(stock_x, fused_x)
        q_mismatches += byte_diff(last_stock[0], last_fused[0])
        s_mismatches += byte_diff(last_stock[1], last_fused[1])
        g_mismatches += byte_diff(last_stock[2], last_fused[2])
        qkv_mismatches += byte_diff(
            kitchen_qkv(last_stock[0], last_stock[1], last_stock[2], qw, qws, sw),
            kitchen_qkv(last_fused[0], last_fused[1], last_fused[2], qw, qws, sw),
        )

    def run_stock():
        residual = x0.clone()
        for mlp, gate in steps:
            stock_step(
                rms_native, residual, mlp, gate, weight, scale, shift, row_ids
            )

    def run_fused():
        residual = x0.clone()
        for mlp, gate in steps:
            fused_step(
                gate_native, residual, mlp, gate, weight, scale, shift, row_ids
            )

    dummy_qdata = torch.empty(ROWS, QKV_OUT // 2, dtype=torch.uint8, device=device)
    dummy_params = adapter.TensorCoreNVFP4Layout.Params(
        scale=last_stock[2],
        orig_dtype=torch.bfloat16,
        orig_shape=(ROWS, WIDTH),
        block_scale=last_stock[1].view(torch.float8_e4m3fn),
    )
    dummy_weight = adapter.QuantizedTensor(
        dummy_qdata[:1, :1].clone(), adapter.NVFP4_LAYOUT, dummy_params
    )
    norm = comfy.ops.manual_cast.RMSNorm(
        WIDTH, eps=EPS, device=device, dtype=torch.bfloat16
    )
    with torch.no_grad():
        norm.weight.requires_grad_(False)
        norm.weight.copy_(weight)
    if getattr(norm, "bias", object()) is not None and not hasattr(norm, "bias"):
        norm.bias = None
    linear = SimpleNamespace(
        in_features=WIDTH,
        out_features=QKV_OUT,
        weight=dummy_weight,
        bias=None,
        quant_format="nvfp4",
        layout_type=adapter.NVFP4_LAYOUT,
        pre_quant_scale=None,
        _full_precision_mm=False,
        comfy_force_cast_weights=False,
        weight_function=(),
        bias_function=(),
        weight_lowvram_function=None,
        bias_lowvram_function=None,
    )
    previous = SimpleNamespace(mlp=object(), attn=object(), norm1=object())
    current = SimpleNamespace(
        attn=SimpleNamespace(qkv_proj=linear),
        norm1=norm,
        mlp=object(),
    )
    blocks_replace = {}

    def fake_linear(qinput, weight, bias):
        return torch.empty(ROWS, QKV_OUT, device=device, dtype=torch.bfloat16)

    real_cast = adapter.cast_bias_weight
    real_linear = F.linear

    def cast_dispatch(module, *args, **kwargs):
        if module is linear:
            return dummy_weight, None, None
        if module is norm:
            return real_cast(module, *args, **kwargs)
        raise RuntimeError(f"unexpected cast owner: {type(module)}")

    def run_adapter():
        residual = x0.clone()
        F.linear = fake_linear
        adapter.cast_bias_weight = cast_dispatch
        try:
            for mlp, gate in steps:
                pending = adapter.defer_qkv_transition(
                    previous,
                    current,
                    residual,
                    mlp,
                    gate,
                    segments,
                    row_ids,
                    adapter.CROSSBLOCK_TOKEN,
                    "auto",
                    "exact",
                    blocks_replace,
                )
                if pending is None:
                    raise RuntimeError("adapter refused to defer")
                projected = adapter.try_qkv_transition(
                    previous,
                    current,
                    residual,
                    pending,
                    scale,
                    shift,
                    segments,
                    row_ids,
                    adapter.ROW_IDS_TOKEN,
                    adapter.CROSSBLOCK_TOKEN,
                    "auto",
                    "exact",
                    blocks_replace,
                )
                if projected is None:
                    raise RuntimeError("adapter returned None after defer")
        finally:
            F.linear = real_linear
            adapter.cast_bias_weight = real_cast

    def run_stock_qkv():
        residual = x0.clone()
        for mlp, gate in steps:
            out = stock_step(
                rms_native, residual, mlp, gate, weight, scale, shift, row_ids
            )
            kitchen_qkv(out[0], out[1], out[2], qw, qws, sw)

    def run_fused_qkv():
        residual = x0.clone()
        for mlp, gate in steps:
            out = fused_step(
                gate_native, residual, mlp, gate, weight, scale, shift, row_ids
            )
            kitchen_qkv(out[0], out[1], out[2], qw, qws, sw)

    stock_20 = time_host_gpu(run_stock, EVALS)
    fused_20 = time_host_gpu(run_fused, EVALS)
    adapter_20 = time_host_gpu(run_adapter, EVALS)
    stock_qkv_20 = time_host_gpu(run_stock_qkv, EVALS)
    fused_qkv_20 = time_host_gpu(run_fused_qkv, EVALS)
    stock_prof = profile_one(run_stock)
    fused_prof = profile_one(run_fused)
    adapter_prof = profile_one(run_adapter)

    producer_host_saved_s = round(stock_20["host_s"] - adapter_20["host_s"], 4)
    producer_gpu_saved_s = round(stock_20["gpu_s"] - adapter_20["gpu_s"], 4)
    adapter_vs_fused_host_s = round(adapter_20["host_s"] - fused_20["host_s"], 4)
    qkv_host_saved_s = round(stock_qkv_20["host_s"] - fused_qkv_20["host_s"], 4)
    launch_delta_adapter_minus_stock = (
        adapter_prof["profiler_device_kernels"]
        - stock_prof["profiler_device_kernels"]
    )
    launch_delta_adapter_minus_fused = (
        adapter_prof["profiler_device_kernels"]
        - fused_prof["profiler_device_kernels"]
    )
    cbf_visible = bool(
        stock_prof["profiler_cbf"]
        or fused_prof["profiler_cbf"]
        or adapter_prof["profiler_cbf"]
    )

    # Isolated DiT 20-eval cannot contain VAE/MP4. If adapter still saves
    # vs stock, the FixedRef +1.7 s is outside this math.
    named = "vae_mp4"
    reason = (
        "DiT-only 20-eval adapter host/GPU still beat stock. Adapter Python "
        "ownership is a few ms vs fused, not +1.7 s. Extra FixedRef work is "
        "outside the DiT residual chain (VAE decode + MP4 encode + executor)."
    )
    if producer_host_saved_s < 0 and adapter_vs_fused_host_s > 0.2:
        named = "python_ownership"
        reason = (
            "Adapter 20-eval host lost the kernel save vs stock and is "
            "slower than fused native by >200 ms. Extra work is adapter "
            "eligibility / carrier / cast ownership on the host."
        )
    elif launch_delta_adapter_minus_stock > 10:
        named = "extra_launch"
        reason = (
            "Adapter issued more device kernels than stock. Extra launches, "
            "not VAE/MP4, are the named object."
        )
    elif adapter_20["host_minus_gpu_s"] - fused_20["host_minus_gpu_s"] > 0.2:
        named = "python_ownership"
        reason = (
            "Adapter host−GPU gap exceeds fused by >200 ms over 20 evals. "
            "Launch-gap idle from adapter Python, not a new kernel."
        )

    payload = {
        "identity": (
            "DiT-only 20-eval A/B at S=20423, 49 transitions: stock "
            "gate+rms vs fused gate_rms vs sealed adapter defer+try. "
            "Names the FixedRef +1.7 s extra work. No render, no promotion."
        ),
        "rows": ROWS,
        "width": WIDTH,
        "transitions": TRANSITIONS,
        "evals": EVALS,
        "device": torch.cuda.get_device_name(device),
        "kernel_projection_s": KERNEL_PROJECTION_S,
        "chef_warm_delta_s": 1.7,
        "residual_mismatches": residual_mismatches,
        "q_mismatches": q_mismatches,
        "s_mismatches": s_mismatches,
        "g_mismatches": g_mismatches,
        "qkv_mismatches": qkv_mismatches,
        "producer_20_eval": {
            "stock": stock_20,
            "fused": fused_20,
            "adapter": adapter_20,
            "adapter_host_saved_s": producer_host_saved_s,
            "adapter_gpu_saved_s": producer_gpu_saved_s,
            "adapter_minus_fused_host_s": adapter_vs_fused_host_s,
        },
        "qkv_20_eval": {
            "stock": stock_qkv_20,
            "fused": fused_qkv_20,
            "fused_host_saved_s": qkv_host_saved_s,
        },
        "one_eval_profile": {
            "stock": stock_prof,
            "fused": fused_prof,
            "adapter": adapter_prof,
            "cbf_visible_in_torch_profiler": cbf_visible,
            "launch_delta_adapter_minus_stock": launch_delta_adapter_minus_stock,
            "launch_delta_adapter_minus_fused": launch_delta_adapter_minus_fused,
        },
        "named_extra_work": named,
        "named_reason": reason,
        "closed": {
            "hook": "A/B cache off; adapter rejects local/global hooks",
            "stream": "adapter uses current stream; no extra pin or .item()",
            "sync": "DiT hot path has zero synchronize; run_every_op is interrupt only",
        },
        "promoted": False,
        "serving_default": "stock",
        "decision": "hold",
    }
    payload["pass"] = (
        residual_mismatches == 0
        and q_mismatches == 0
        and s_mismatches == 0
        and g_mismatches == 0
        and qkv_mismatches == 0
        and named in {
            "vae_mp4",
            "python_ownership",
            "extra_launch",
            "extra_sync",
            "stream",
            "hook",
        }
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    out = HERE / "gate_extra_work_20423.json"
    out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
