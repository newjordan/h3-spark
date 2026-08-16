#!/usr/bin/env python3
"""Byte-exact gate: keep first-pass AdaLN vs stock dynamic gate_rms pack.

Does not patch serving. Loads the local lab cubin only.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROWS = 20423
WIDTH = 5376
EPS = 1e-5
REPEATS = 3
SO = HERE / "cmake-build-120" / "h3_gate_rms_adaln_nvfp4_native_v2.so"


def byte_diff(left: torch.Tensor, right: torch.Tensor) -> int:
    return int((left.view(torch.uint8) != right.view(torch.uint8)).sum().item())


def load_native():
    spec = importlib.util.spec_from_file_location(
        "h3_gate_rms_adaln_nvfp4_native_v2", SO
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_row_ids(rows: int, device: torch.device) -> torch.Tensor:
    return (
        torch.arange(rows, device=device, dtype=torch.int64)
        .mul_(21)
        .div_(rows, rounding_mode="floor")
        .to(torch.int32)
        .contiguous()
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
    native = load_native()
    device = torch.device("cuda")
    torch.manual_seed(26081226)
    x0 = torch.randn(ROWS, WIDTH, device=device, dtype=torch.bfloat16)
    other = (torch.randn_like(x0) * 0.05).contiguous()
    weight = (torch.randn(WIDTH, device=device) * 0.1 + 1.0).to(torch.bfloat16)
    table = (torch.randn(21, WIDTH * 6, device=device) * 0.1).to(torch.float32)
    scale = table[:, WIDTH : 2 * WIDTH].contiguous()
    gate = table[:, 2 * WIDTH : 3 * WIDTH].contiguous()
    shift = table[:, 4 * WIDTH : 5 * WIDTH].contiguous()
    row_ids = make_row_ids(ROWS, device)

    stock_x = x0.clone()
    keep_x = x0.clone()
    stock = native.gate_rms_adaln_nvfp4_dynamic(
        stock_x, other, gate, weight, scale, shift, row_ids, EPS
    )
    keep = native.gate_rms_adaln_nvfp4_dynamic_keep(
        keep_x, other, gate, weight, scale, shift, row_ids, EPS
    )
    payload = {
        "identity": (
            "keep first-pass AdaLN BF16 and pack from it; residual mutation "
            "unchanged; second pass does not recompute RMS/AdaLN"
        ),
        "rows": ROWS,
        "width": WIDTH,
        "device": torch.cuda.get_device_name(device),
        "residual_mismatches": byte_diff(stock_x, keep_x),
        "q_mismatches": byte_diff(stock[0], keep[0]),
        "s_mismatches": byte_diff(stock[1], keep[1]),
        "g_mismatches": byte_diff(stock[2], keep[2]),
        "rstd_mismatches": byte_diff(stock[3], keep[3]),
    }

    def run_stock():
        residual = x0.clone()
        native.gate_rms_adaln_nvfp4_dynamic(
            residual, other, gate, weight, scale, shift, row_ids, EPS
        )

    def run_keep():
        residual = x0.clone()
        native.gate_rms_adaln_nvfp4_dynamic_keep(
            residual, other, gate, weight, scale, shift, row_ids, EPS
        )

    payload["stock_min_ms"] = round(time_ms(run_stock), 4)
    payload["keep_min_ms"] = round(time_ms(run_keep), 4)
    payload["saved_ms"] = round(
        payload["stock_min_ms"] - payload["keep_min_ms"], 4
    )
    payload["projected_s_per_20_step"] = round(
        payload["saved_ms"] * 50 * 20 / 1000.0, 4
    )
    payload["pass"] = (
        payload["residual_mismatches"] == 0
        and payload["q_mismatches"] == 0
        and payload["s_mismatches"] == 0
        and payload["g_mismatches"] == 0
        and payload["rstd_mismatches"] == 0
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    (HERE / "gate_keep_20423.json").write_text(text + "\n")
    print(text)
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
