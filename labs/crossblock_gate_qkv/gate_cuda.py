#!/usr/bin/env python3
"""Seconds-scale exactness gate for cross-block gate -> next QKV.

Compares the stock staged residual-gate + RMS/AdaLN NVFP4 producer against
the sealed adapter at the FixedRef production shape. No model load, no render.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
WIDTH = 5376
QKV_OUT = 21504
EPS = 1e-5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def byte_diff(left: torch.Tensor, right: torch.Tensor) -> int:
    return int((left.view(torch.uint8) != right.view(torch.uint8)).sum().item())


def load_adapter(comfy_root: Path):
    if str(comfy_root) not in sys.path:
        sys.path.insert(0, str(comfy_root))
    spec = importlib.util.spec_from_file_location(
        "h3_crossblock_gate_cuda", HERE / "comfy_adapter.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
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


def stock_producer(rms_native, x, mlp, gate, weight, scale, shift, row_ids):
    from comfy.ldm.minimax.kernels import segmented_gate_bf16_

    residual = x.clone()
    if not segmented_gate_bf16_(residual, mlp, gate, row_ids):
        raise RuntimeError("stock segmented gate did not route")
    qdata, block_scale, global_scale = rms_native.rms_adaln_nvfp4_dynamic(
        residual, weight, scale, shift, row_ids, EPS
    )
    _, rstd = torch.ops.aten._fused_rms_norm(residual, [WIDTH], weight, EPS)
    return residual, qdata, block_scale, global_scale, rstd


def run_stream(adapter, rms_native, device, rows: int, stream) -> dict:
    with torch.cuda.stream(stream), torch.no_grad():
        if torch.cuda.current_stream(device).cuda_stream != stream.cuda_stream:
            raise RuntimeError("requested CUDA stream did not activate")
        torch.manual_seed(26081226)
        x = torch.randn(rows, WIDTH, device=device, dtype=torch.bfloat16)
        mlp = (torch.randn_like(x) * 0.05).contiguous()
        weight = (torch.randn(WIDTH, device=device) * 0.1 + 1.0).to(torch.bfloat16)
        table = (torch.randn(21, WIDTH * 6, device=device) * 0.1).to(torch.float32)
        scale = table[:, WIDTH : 2 * WIDTH].contiguous()
        gate = table[:, 2 * WIDTH : 3 * WIDTH].contiguous()
        shift = table[:, 4 * WIDTH : 5 * WIDTH].contiguous()
        row_ids = make_row_ids(rows, device)
        segments = [(0, rows, 0)]

        stock_residual, stock_q, stock_s, stock_g, stock_rstd = stock_producer(
            rms_native, x, mlp, gate, weight, scale, shift, row_ids
        )

        fused_x = x.clone()
        version_before = fused_x._version
        fused_q, fused_s, fused_g, fused_rstd = (
            adapter._load_native().gate_rms_adaln_nvfp4_dynamic(
                fused_x, mlp, gate, weight, scale, shift, row_ids, EPS
            )
        )
        fused_delta = fused_x._version - version_before

        dummy_qdata = torch.empty(rows, QKV_OUT // 2, dtype=torch.uint8, device=device)
        dummy_params = adapter.TensorCoreNVFP4Layout.Params(
            scale=stock_g,
            orig_dtype=torch.bfloat16,
            orig_shape=(rows, WIDTH),
            block_scale=stock_s.view(torch.float8_e4m3fn),
        )
        dummy_weight = adapter.QuantizedTensor(
            dummy_qdata[:1, :1].clone(), adapter.NVFP4_LAYOUT, dummy_params
        )
        captured = {}

        def fake_linear(qinput, weight, bias):
            captured["qdata"] = qinput._qdata
            captured["block_scale"] = qinput._params.block_scale
            captured["input_scale"] = qinput._params.scale
            captured["weight"] = weight
            return torch.empty(rows, QKV_OUT, device=device, dtype=torch.bfloat16)

        import comfy.ops
        norm = comfy.ops.manual_cast.RMSNorm(WIDTH, eps=EPS, device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            if norm.weight is None:
                raise RuntimeError("manual_cast RMSNorm did not allocate a weight")
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
        adapter_x = x.clone()
        adapter_version = adapter_x._version

        real_cast = adapter.cast_bias_weight

        def cast_dispatch(module, *args, **kwargs):
            if module is linear:
                return dummy_weight, None, None
            if module is norm:
                return real_cast(module, *args, **kwargs)
            raise RuntimeError(f"unexpected cast_bias_weight owner: {type(module)}")

        pending = adapter.defer_qkv_transition(
            previous,
            current,
            adapter_x,
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
            raise RuntimeError("adapter refused to defer a qualified dynamic pair")

        with torch.no_grad():
            original_linear = F.linear
            F.linear = fake_linear
            adapter.cast_bias_weight = cast_dispatch
            try:
                projected = adapter.try_qkv_transition(
                    previous,
                    current,
                    adapter_x,
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
            finally:
                F.linear = original_linear
                adapter.cast_bias_weight = real_cast

        if projected is None:
            raise RuntimeError("adapter returned None after a sealed defer")
        if pending._state is not adapter._CLAIMED:
            raise RuntimeError("adapter did not claim the carrier")

        material_x = x.clone()
        material_pending = adapter.defer_qkv_transition(
            previous,
            current,
            material_x,
            mlp,
            gate,
            segments,
            row_ids,
            adapter.CROSSBLOCK_TOKEN,
            "auto",
            "exact",
            blocks_replace,
        )
        from comfy.ldm.minimax.kernels import segmented_gate_bf16_

        def materialize(residual, gate_value, mlp_value, _segments, rows):
            if not segmented_gate_bf16_(residual, mlp_value, gate_value, rows):
                raise RuntimeError("materialize gate did not route")
            return residual

        materialized = adapter.materialize_pending(
            material_pending,
            previous,
            current,
            material_x,
            segments,
            row_ids,
            adapter.CROSSBLOCK_TOKEN,
            materialize,
        )
        stream.synchronize()

        result = {
            "stream": int(stream.cuda_stream),
            "producer_residual_mismatches": byte_diff(fused_x, stock_residual),
            "producer_q_mismatches": byte_diff(fused_q, stock_q),
            "producer_scale_mismatches": byte_diff(fused_s, stock_s),
            "producer_global_mismatches": byte_diff(fused_g, stock_g),
            "producer_rstd_mismatches": byte_diff(
                fused_rstd.reshape(-1), stock_rstd.reshape(-1)
            ),
            "producer_version_delta": int(fused_delta),
            "adapter_residual_mismatches": byte_diff(adapter_x, stock_residual),
            "adapter_q_mismatches": byte_diff(captured["qdata"], stock_q),
            "adapter_scale_mismatches": byte_diff(
                captured["block_scale"].view(stock_s.dtype), stock_s
            ),
            "adapter_global_mismatches": byte_diff(captured["input_scale"], stock_g),
            "adapter_version_delta": int(adapter_x._version - adapter_version),
            "materialize_mismatches": byte_diff(materialized, stock_residual),
            "materialize_state": (
                "materialized"
                if material_pending._state is adapter._MATERIALIZED
                else "unexpected"
            ),
        }
        failures = [
            name
            for name, value in result.items()
            if name.endswith("mismatches") and value != 0
        ]
        if result["producer_version_delta"] != 1:
            failures.append("producer_version_delta")
        if result["adapter_version_delta"] != 1:
            failures.append("adapter_version_delta")
        if material_pending._state is not adapter._MATERIALIZED:
            failures.append("materialize_state")
        result["failures"] = failures
        result["pass"] = not failures
        return result


def main() -> int:
    wall0 = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=20423)
    parser.add_argument(
        "--comfy-root",
        type=Path,
        default=os.environ.get("COMFYUI_ROOT", "$HOME/comfyui-h3-current"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rows <= 0:
        parser.error("rows must be positive")

    adapter = load_adapter(args.comfy_root.resolve())
    import comfy.ldm.minimax.gate_rms_adaln_nvfp4 as dynamic_gate
    import comfy.ldm.minimax.rms_adaln_nvfp4 as rms_adaln

    device = torch.device("cuda")
    if torch.version.cuda != dynamic_gate.CERTIFIED_CUDA_VERSION:
        raise RuntimeError("runtime CUDA is outside the certified contract")
    if torch.cuda.get_device_capability(device) != dynamic_gate.CERTIFIED_DEVICE_CAPABILITY:
        raise RuntimeError("device capability is outside the certified contract")
    native = adapter._load_native()
    rms_native = rms_adaln._load_native()
    if native is None or rms_native is None:
        raise RuntimeError("native gate or RMS producer failed to load")

    default_stream = torch.cuda.default_stream(device)
    other_stream = torch.cuda.Stream(device=device)
    default = run_stream(adapter, rms_native, device, args.rows, default_stream)
    other = run_stream(adapter, rms_native, device, args.rows, other_stream)
    gate_so = Path(args.comfy_root) / "comfy/ldm/minimax/h3_gate_rms_adaln_nvfp4_native_v2.so"
    rms_so = Path(args.comfy_root) / "comfy/ldm/minimax/rms_adaln_nvfp4_native_v2.so"
    payload = {
        "rows": args.rows,
        "width": WIDTH,
        "wall_seconds": round(time.monotonic() - wall0, 3),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "gate_native_sha256": sha256_file(gate_so) if gate_so.is_file() else None,
        "rms_native_sha256": sha256_file(rms_so) if rms_so.is_file() else None,
        "default_stream": default,
        "nondefault_stream": other,
        "pass": bool(default["pass"] and other["pass"]),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
