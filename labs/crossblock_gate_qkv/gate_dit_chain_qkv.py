#!/usr/bin/env python3
"""DiT-only isolation: 49-transition gate/RMS chain plus kitchen QKV GEMM.

QKV is N-independent of residual math. If fused (q,s,g) == stock (q,s,g),
kitchen QKV output must match. Times whether the 0.51 s kernel save
survives including the GEMM. No flag retune, no render, no promotion.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

import comfy_kitchen as ck


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
REPEATS = 2


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


def time_ms(fn) -> float:
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    fn()
    torch.cuda.synchronize()
    starter.record()
    fn()
    ender.record()
    torch.cuda.synchronize()
    return starter.elapsed_time(ender)


def main() -> int:
    torch.set_grad_enabled(False)
    comfy = _load_h3_paths().require_env_path("COMFYUI_ROOT", kind="dir")
    if str(comfy) not in sys.path:
        sys.path.insert(0, str(comfy))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "h3_crossblock_dit_qkv", HERE / "comfy_adapter.py"
    )
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    import comfy.ldm.minimax.rms_adaln_nvfp4 as rms_adaln

    device = torch.device("cuda")
    gate_native = adapter._load_native()
    rms_native = rms_adaln._load_native()
    if gate_native is None or rms_native is None:
        raise RuntimeError("native gate or RMS producer failed to load")

    torch.manual_seed(26081264)
    x0 = torch.randn(ROWS, WIDTH, device=device, dtype=torch.bfloat16)
    weight = (torch.randn(WIDTH, device=device) * 0.1 + 1.0).to(torch.bfloat16)
    table = (torch.randn(21, WIDTH * 6, device=device) * 0.1).to(torch.float32)
    scale = table[:, WIDTH : 2 * WIDTH].contiguous()
    shift = table[:, 4 * WIDTH : 5 * WIDTH].contiguous()
    row_ids = make_row_ids(ROWS, device)
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
    last_qkv_stock = last_qkv_fused = None
    for mlp, gate in steps:
        qs = stock_step(
            rms_native, stock_x, mlp, gate, weight, scale, shift, row_ids
        )
        qf = fused_step(
            gate_native, fused_x, mlp, gate, weight, scale, shift, row_ids
        )
        residual_mismatches += byte_diff(stock_x, fused_x)
        q_mismatches += byte_diff(qs[0], qf[0])
        s_mismatches += byte_diff(qs[1], qf[1])
        g_mismatches += byte_diff(qs[2], qf[2])
        last_qkv_stock = kitchen_qkv(qs[0], qs[1], qs[2], qw, qws, sw)
        last_qkv_fused = kitchen_qkv(qf[0], qf[1], qf[2], qw, qws, sw)
        qkv_mismatches += byte_diff(last_qkv_stock, last_qkv_fused)

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

    def run_stock_only():
        residual = x0.clone()
        for mlp, gate in steps:
            stock_step(
                rms_native, residual, mlp, gate, weight, scale, shift, row_ids
            )

    def run_fused_only():
        residual = x0.clone()
        for mlp, gate in steps:
            fused_step(
                gate_native, residual, mlp, gate, weight, scale, shift, row_ids
            )

    payload = {
        "identity": (
            "49-transition DiT chain + kitchen QKV GEMM (5376x21504). "
            "QKV is a function of (q,s,g,qw) only. Fused (q,s,g) == stock "
            "implies QKV bytes match. Times whether the gate/RMS kernel "
            "save survives including the GEMM."
        ),
        "rows": ROWS,
        "width": WIDTH,
        "qkv_out": QKV_OUT,
        "transitions": TRANSITIONS,
        "device": torch.cuda.get_device_name(device),
        "residual_mismatches": residual_mismatches,
        "q_mismatches": q_mismatches,
        "s_mismatches": s_mismatches,
        "g_mismatches": g_mismatches,
        "qkv_mismatches": qkv_mismatches,
        "qkv_finite": bool(
            last_qkv_fused is not None
            and torch.isfinite(last_qkv_fused.float()).all().item()
        ),
        "stock_only_min_ms": round(time_ms(run_stock_only), 4),
        "fused_only_min_ms": round(time_ms(run_fused_only), 4),
        "stock_qkv_min_ms": round(time_ms(run_stock_qkv), 4),
        "fused_qkv_min_ms": round(time_ms(run_fused_qkv), 4),
    }
    payload["producer_saved_ms"] = round(
        payload["stock_only_min_ms"] - payload["fused_only_min_ms"], 4
    )
    payload["with_qkv_saved_ms"] = round(
        payload["stock_qkv_min_ms"] - payload["fused_qkv_min_ms"], 4
    )
    payload["qkv_work_ms"] = round(
        payload["fused_qkv_min_ms"] - payload["fused_only_min_ms"], 4
    )
    payload["projected_s_per_20_eval"] = round(
        payload["with_qkv_saved_ms"] * 20 / 1000.0, 4
    )
    payload["kernel_projection_s"] = 0.511881
    payload["pass"] = (
        residual_mismatches == 0
        and q_mismatches == 0
        and s_mismatches == 0
        and g_mismatches == 0
        and qkv_mismatches == 0
        and payload["qkv_finite"]
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    (HERE / "gate_dit_chain_qkv_20423.json").write_text(text + "\n")
    print(text)
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
