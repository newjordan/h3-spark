#!/usr/bin/env python3
"""Isolate DiT-only cross-block math: 49 chained residual transitions.

Compares stock (segmented_gate + rms_adaln_dynamic) against the fused
gate_rms producer and against the sealed adapter (defer+try). No render,
no flag retune, no serving change.
"""

from __future__ import annotations

import json
import sys
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
REPEATS = 3


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


def stock_step(rms_native, residual, mlp, gate, weight, scale, shift, row_ids):
    from comfy.ldm.minimax.kernels import segmented_gate_bf16_

    if not segmented_gate_bf16_(residual, mlp, gate, row_ids):
        raise RuntimeError("stock segmented gate did not route")
    q, s, g = rms_native.rms_adaln_nvfp4_dynamic(
        residual, weight, scale, shift, row_ids, EPS
    )
    return q, s, g


def fused_step(gate_native, residual, mlp, gate, weight, scale, shift, row_ids):
    return gate_native.gate_rms_adaln_nvfp4_dynamic(
        residual, mlp, gate, weight, scale, shift, row_ids, EPS
    )


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


def main() -> int:
    torch.set_grad_enabled(False)
    comfy = _load_h3_paths().require_env_path("COMFYUI_ROOT", kind="dir")
    if str(comfy) not in sys.path:
        sys.path.insert(0, str(comfy))
    spec_path = HERE / "comfy_adapter.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("h3_crossblock_dit_chain", spec_path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)

    import comfy.ldm.minimax.rms_adaln_nvfp4 as rms_adaln
    import comfy.ops

    device = torch.device("cuda")
    gate_native = adapter._load_native()
    rms_native = rms_adaln._load_native()
    if gate_native is None or rms_native is None:
        raise RuntimeError("native gate or RMS producer failed to load")

    torch.manual_seed(26081226)
    if torch.is_grad_enabled():
        raise RuntimeError("DiT-chain gate must run under inference")
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

    stock_x = x0.clone()
    fused_x = x0.clone()
    last_stock = last_fused = None
    residual_mismatches = 0
    q_mismatches = 0
    s_mismatches = 0
    g_mismatches = 0
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

    stock_ms = time_ms(run_stock)
    fused_ms = time_ms(run_fused)
    adapter_ms = time_ms(run_adapter)
    payload = {
        "identity": (
            "49-transition DiT residual chain at S=20423: stock gate+rms "
            "vs fused gate_rms vs sealed adapter defer+try"
        ),
        "rows": ROWS,
        "width": WIDTH,
        "transitions": TRANSITIONS,
        "device": torch.cuda.get_device_name(device),
        "residual_mismatches": residual_mismatches,
        "q_mismatches": q_mismatches,
        "s_mismatches": s_mismatches,
        "g_mismatches": g_mismatches,
        "stock_chain_min_ms": round(stock_ms, 4),
        "fused_native_chain_min_ms": round(fused_ms, 4),
        "adapter_chain_min_ms": round(adapter_ms, 4),
        "native_saved_ms": round(stock_ms - fused_ms, 4),
        "adapter_minus_stock_ms": round(adapter_ms - stock_ms, 4),
        "native_projected_s_per_20_eval": round((stock_ms - fused_ms) * 20 / 1000.0, 4),
        "adapter_projected_s_per_20_eval": round((adapter_ms - stock_ms) * 20 / 1000.0, 4),
        "kernel_projection_s": 0.511881,
    }
    payload["pass"] = (
        residual_mismatches == 0
        and q_mismatches == 0
        and s_mismatches == 0
        and g_mismatches == 0
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    out = HERE / "gate_dit_chain_20423.json"
    out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
