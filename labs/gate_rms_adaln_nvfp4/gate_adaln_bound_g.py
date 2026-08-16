#!/usr/bin/env python3
"""Fail-closed gate: cheap AdaLN amax bound vs live first-pass G.

AdaLN is staged bf16(bf16(w*rstd*x) * bf16(1+s) + bf16(sh)).
A cheap U from residual/table maxima would skip the first-pass AdaLN
scan. G(U)==G(T) is required for pack-from-bound to be two-pass.
Does not patch serving.
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
SO = HERE / "cmake-build-120" / "h3_gate_rms_adaln_nvfp4_native_v2.so"


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


def g_from_absmax_bf16(absmax: torch.Tensor) -> torch.Tensor:
    # Native: scale_bf16 = bf16(float(absmax) / 2688); return float(scale_bf16).
    return (absmax.float() / 2688.0).to(torch.bfloat16).float().reshape(1)


def main() -> int:
    torch.set_grad_enabled(False)
    native = load_native()
    device = torch.device("cuda")
    torch.manual_seed(26081270)
    x0 = torch.randn(ROWS, WIDTH, device=device, dtype=torch.bfloat16)
    other = (torch.randn_like(x0) * 0.05).contiguous()
    weight = (torch.randn(WIDTH, device=device) * 0.1 + 1.0).to(torch.bfloat16)
    table = (torch.randn(21, WIDTH * 6, device=device) * 0.1).to(torch.float32)
    scale = table[:, WIDTH : 2 * WIDTH].contiguous()
    gate = table[:, 2 * WIDTH : 3 * WIDTH].contiguous()
    shift = table[:, 4 * WIDTH : 5 * WIDTH].contiguous()
    row_ids = make_row_ids(ROWS, device)

    residual = x0.clone()
    packed, scales, live_g, rstd = native.gate_rms_adaln_nvfp4_dynamic(
        residual, other, gate, weight, scale, shift, row_ids, EPS
    )

    # Cheap global bound: max|x| * max(rstd) * max|w| * max|1+s| + max|sh|.
    x_abs = residual.abs().amax()
    rstd_max = rstd.amax()
    w_abs = weight.abs().amax()
    scale_bf16 = scale.to(torch.bfloat16)
    mult_bf16 = (1.0 + scale_bf16.float()).to(torch.bfloat16)
    mult_abs = mult_bf16.abs().amax()
    shift_bf16 = shift.to(torch.bfloat16)
    shift_abs = shift_bf16.abs().amax()
    u_global = (
        x_abs.float() * rstd_max * w_abs.float() * mult_abs.float()
        + shift_abs.float()
    )
    g_global = g_from_absmax_bf16(u_global.to(torch.bfloat16))

    # Per-row residual amax + per-mod-row table max. Still no AdaLN.
    x_row = residual.abs().amax(dim=1).float()
    mult_row = mult_bf16.abs().amax(dim=1).float()
    shift_row = shift_bf16.abs().amax(dim=1).float()
    u_row = (
        x_row * rstd * w_abs.float() * mult_row[row_ids] + shift_row[row_ids]
    )
    g_row = g_from_absmax_bf16(u_row.amax().to(torch.bfloat16))

    # Reconstruct live T from G: not needed. Compare G objects.
    payload = {
        "identity": (
            "cheap AdaLN U (residual/table maxima) G(U)==G(T) live "
            "first-pass; per-row residual amax + per-mod table max too"
        ),
        "rows": ROWS,
        "width": WIDTH,
        "device": torch.cuda.get_device_name(device),
        "live_g": float(live_g.item()),
        "g_global": float(g_global.item()),
        "g_row": float(g_row.item()),
        "u_global": float(u_global.item()),
        "u_row": float(u_row.amax().item()),
        "g_global_eq_live": bool(torch.equal(g_global, live_g)),
        "g_row_eq_live": bool(torch.equal(g_row, live_g)),
        "g_global_vs_live_ulps": int(
            (
                g_global.view(torch.int32) - live_g.view(torch.int32)
            ).abs().item()
        ),
        "g_row_vs_live_ulps": int(
            (g_row.view(torch.int32) - live_g.view(torch.int32)).abs().item()
        ),
        "full_finite": bool(torch.isfinite(residual.float()).all().item()),
    }
    # Closed negative is a pass of the investigation. Equality would
    # unlock a first-pass AdaLN skip; report either outcome.
    payload["pass"] = payload["full_finite"] and float(live_g.item()) > 0.0
    payload["bound_is_live_g"] = (
        payload["g_global_eq_live"] or payload["g_row_eq_live"]
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    (HERE / "gate_adaln_bound_g_20423.json").write_text(text + "\n")
    print(text)
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
