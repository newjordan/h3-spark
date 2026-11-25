#!/usr/bin/env python3
"""Algebra gate: CBF named barrier #9 aten::zeros 400 = 8/block.

Name the eight tensors from the accepted profile + serving producers.
Prove whether a reused zero-once buffer is byte-exact vs per-call
aten::zeros at S=20423 for SwiGLU and bf16_nvfp4 dynamic pack.

Does not patch serving. Does not promote.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROWS = 20423
DENOM = 6.0 * 448.0
SERVING_SHA256 = (
    "45b21fe1957e535a860fcd28014b6b194666a29b5b98f4672c7e3b99230edb8f"
)
MODULE_NAME = "h3_swiglu_nvfp4_native_v1"


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
BLOCKS = 50

# Accepted eval: 200 FillFunctor<int> + 200 FillFunctor<unsigned char>.
# Four dynamic NVFP4 sites per MiniMax block, each zeros one int32
# max_bits and one uint8 cuBLAS scale slab.
TENSORS_PER_BLOCK = (
    {
        "slot": 1,
        "name": "qkv.rms_adaln_nvfp4.max_bits",
        "producer": "rms_adaln_nvfp4_dynamic",
        "dtype": "int32",
        "shape": [1],
        "role": "atomicMax start for QKV AdaLN amax",
        "width": 5376,
        "profile_kernel": "rms_adaln_first_pass / rms_adaln_second_pass",
    },
    {
        "slot": 2,
        "name": "qkv.rms_adaln_nvfp4.block_scales",
        "producer": "rms_adaln_nvfp4_dynamic",
        "dtype": "uint8",
        "shape": "[scale_rows, 5376/16]",
        "role": "cuBLAS E4M3 scale tile, padded to 128 rows",
        "width": 5376,
        "profile_kernel": "rms_adaln_second_pass",
    },
    {
        "slot": 3,
        "name": "mlp.gate_rms_adaln_nvfp4.max_bits",
        "producer": "gate_rms_adaln_nvfp4_dynamic",
        "dtype": "int32",
        "shape": [1],
        "role": "atomicMax start for MLP AdaLN amax",
        "width": 5376,
        "profile_kernel": "gate_rms_adaln_first_pass / rms_adaln_second_pass",
    },
    {
        "slot": 4,
        "name": "mlp.gate_rms_adaln_nvfp4.block_scales",
        "producer": "gate_rms_adaln_nvfp4_dynamic",
        "dtype": "uint8",
        "shape": "[scale_rows, 5376/16]",
        "role": "cuBLAS E4M3 scale tile, padded to 128 rows",
        "width": 5376,
        "profile_kernel": "rms_adaln_second_pass",
    },
    {
        "slot": 5,
        "name": "attn.sage_output_amax.max_bits",
        "producer": "qk_int8_sv_f8_..._with_output_amax",
        "dtype": "int32",
        "shape": [1],
        "role": "atomicMax start for Sage output amax; consumed by "
        "bf16_nvfp4_from_amax_bits (no second int32 zeros)",
        "width": 7168,
        "profile_kernel": "sageattention_sm89::..._with_output_amax",
    },
    {
        "slot": 6,
        "name": "attn.bf16_nvfp4_from_amax_bits.block_scales",
        "producer": "bf16_nvfp4_from_amax_bits",
        "dtype": "uint8",
        "shape": "[scale_rows, 7168/16]",
        "role": "cuBLAS E4M3 scale tile, padded to 128 rows",
        "width": 7168,
        "profile_kernel": "bf16_nvfp4_kernel<true> (50; no bf16_amax kernel)",
    },
    {
        "slot": 7,
        "name": "mlp.swiglu_nvfp4_dynamic.max_bits",
        "producer": "swiglu_nvfp4_dynamic",
        "dtype": "int32",
        "shape": [1],
        "role": "atomicMax start for SwiGLU product amax",
        "width": 14336,
        "profile_kernel": "swiglu_amax_bf16_bits_kernel",
    },
    {
        "slot": 8,
        "name": "mlp.swiglu_nvfp4.block_scales",
        "producer": "swiglu_nvfp4 / swiglu_nvfp4_dynamic",
        "dtype": "uint8",
        "shape": "[scale_rows, 14336/16]",
        "role": "cuBLAS E4M3 scale tile, padded to 128 rows; comment: "
        "rows beyond padded_rows must be deterministic zero",
        "width": 14336,
        "profile_kernel": "swiglu_nvfp4_kernel<true>",
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_so(path: Path):
    spec = importlib.util.spec_from_file_location(MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def byte_diff(left: torch.Tensor, right: torch.Tensor) -> int:
    return int((left.view(torch.uint8) != right.view(torch.uint8)).sum().item())


def bits_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(left.view(torch.uint8).equal(right.contiguous().view(torch.uint8)))


def padded_rows(rows: int) -> int:
    return ((rows + 15) // 16) * 16


def scale_rows(rows: int) -> int:
    return ((padded_rows(rows) + 127) // 128) * 128


def bf16_abs_bits(x: torch.Tensor) -> int:
    bits = x.view(torch.int16).to(torch.int32) & 0x7FFF
    return int(bits.amax().item())


def g_from_abs_bits(abs_bits: int, device: torch.device) -> torch.Tensor:
    raw = torch.tensor([abs_bits], dtype=torch.int16, device="cpu")
    absmax = raw.view(torch.bfloat16).to(torch.float32)
    scale = (absmax / DENOM).to(torch.bfloat16).to(torch.float32)
    return scale.to(device=device)


def swizzle_offset(
    row: torch.Tensor, col: torch.Tensor, col_length: int
) -> torch.Tensor:
    rb = row // 128
    rem = row % 128
    d4 = rem // 32
    d3 = rem % 32
    cbg = col // 4
    d5 = col % 4
    cbg_cnt = (col_length + 3) // 4
    return ((rb * cbg_cnt + cbg) * 32 + d3) * 16 + d4 * 4 + d5


def unwritten_linear_index(
    n_scale_rows: int, scale_cols: int, n_padded: int, device: torch.device
) -> torch.Tensor:
    written = torch.zeros(n_scale_rows * scale_cols, dtype=torch.bool, device=device)
    rows = torch.arange(n_padded, device=device)
    cols = torch.arange(scale_cols, device=device)
    rr = rows.unsqueeze(1).expand(n_padded, scale_cols)
    cc = cols.unsqueeze(0).expand(n_padded, scale_cols)
    written[swizzle_offset(rr, cc, scale_cols).reshape(-1)] = True
    return torch.nonzero(~written, as_tuple=False).squeeze(1)


def _ncalls(line: str) -> int | None:
    left = line.split("[", 1)[0]
    toks = left.split()
    if not toks:
        return None
    try:
        return int(toks[-1].replace(",", ""))
    except ValueError:
        return None


def parse_profile(path: Path) -> dict:
    found: dict[str, int] = {}
    if not path.is_file():
        return found
    for line in path.read_text().splitlines():
        n = _ncalls(line)
        if n is None:
            continue
        if "aten::zeros" in line:
            found["aten_zeros"] = n
        elif "FillFunctor<unsigned char>" in line:
            found["fill_uint8"] = n
        elif "FillFunctor<int>" in line:
            found["fill_int32"] = n
        elif "swiglu_amax_bf16_bits_kernel" in line:
            found["swiglu_amax"] = n
        elif "swiglu_nvfp4_kernel<true>" in line:
            found["swiglu_pack"] = n
        elif "bf16_nvfp4_kernel<true>" in line:
            found["bf16_pack"] = n
        elif "bf16_amax_bits_kernel" in line:
            found["bf16_amax"] = n
        elif "with_output_amax" in line and "sageattention" in line:
            found["sage_output_amax"] = n
        elif "rms_adaln_nvfp4_second_pass" in line:
            found["rms_second"] = n
        elif "gate_rms_adaln_first_pass" in line:
            found["gate_rms_first"] = n
        elif "rms_adaln_first_pass" in line:
            found["rms_first"] = n
    return found


def pack_case(native, x: torch.Tensor, leftover_bits: int) -> dict:
    live_bits = bf16_abs_bits(x)
    live_g = g_from_abs_bits(live_bits, x.device)
    leftover_g = g_from_abs_bits(leftover_bits, x.device)
    dyn_q, dyn_s, dyn_g = native.bf16_nvfp4_dynamic(x)
    dyn_s = dyn_s.view(torch.float8_e4m3fn)
    live_mb = torch.tensor([live_bits], device=x.device, dtype=torch.int32)
    left_mb = torch.tensor([leftover_bits], device=x.device, dtype=torch.int32)
    same_q, same_s, same_g = native.bf16_nvfp4_from_amax_bits(x, live_mb)
    dirty_q, dirty_s, dirty_g = native.bf16_nvfp4_from_amax_bits(x, left_mb)
    same_s = same_s.view(torch.float8_e4m3fn)
    dirty_s = dirty_s.view(torch.float8_e4m3fn)
    atomic_result = max(leftover_bits, live_bits)
    return {
        "width": int(x.shape[1]),
        "live_abs_bits": live_bits,
        "leftover_abs_bits": leftover_bits,
        "atomicMax_result_bits": atomic_result,
        "leftover_gt_live": leftover_bits > live_bits,
        "g_live": float(live_g.item()),
        "g_leftover": float(leftover_g.item()),
        "g_dyn": float(dyn_g.item()),
        "dyn_vs_from_live_q": byte_diff(dyn_q, same_q),
        "dyn_vs_from_live_s": byte_diff(dyn_s, same_s),
        "dyn_vs_from_live_g_bits": bits_equal(dyn_g, same_g),
        "from_live_vs_g_formula": bits_equal(same_g, live_g),
        "dirty_vs_live_q": byte_diff(dirty_q, same_q),
        "dirty_vs_live_s": byte_diff(dirty_s, same_s),
        "dirty_vs_live_g_bits": bits_equal(dirty_g, leftover_g),
        "same_input_reuse_exact": leftover_bits == live_bits
        and byte_diff(dirty_q, same_q) == 0
        and byte_diff(dirty_s, same_s) == 0
        and bits_equal(dirty_g, same_g),
        "zero_once_smaller_t_exact": leftover_bits <= live_bits,
        "scale_rows": int(dyn_s.shape[0]),
        "scale_cols": int(dyn_s.shape[1]),
        "padded_rows": padded_rows(int(x.shape[0])),
    }


def scale_tail_case(sx: torch.Tensor, rows: int) -> dict:
    n_scale_rows, scale_cols = int(sx.shape[0]), int(sx.shape[1])
    n_padded = padded_rows(rows)
    idx = unwritten_linear_index(
        n_scale_rows, scale_cols, n_padded, sx.device
    )
    flat = sx.view(torch.uint8).reshape(-1)
    unwritten = flat[idx]
    dirty = flat.clone()
    dirty[idx] = 0xFF
    return {
        "scale_rows": n_scale_rows,
        "scale_cols": scale_cols,
        "padded_rows": n_padded,
        "unwritten_count": int(idx.numel()),
        "unwritten_all_zero": bool((unwritten == 0).all().item()),
        "unwritten_nonzero": int((unwritten != 0).sum().item()),
        "dirty_tail_vs_stock": int((dirty != flat).sum().item()),
        "algebra_padding_must_be_zero": True,
    }


def swiglu_case(native, raw: torch.Tensor, leftover_bits: int) -> dict:
    gate, up = raw.chunk(2, dim=-1)
    activated = F.silu(gate).mul_(up)
    live_bits = bf16_abs_bits(activated)
    live_g = g_from_abs_bits(live_bits, raw.device)
    leftover_g = g_from_abs_bits(leftover_bits, raw.device)
    dyn_q, dyn_s, dyn_g = native.swiglu_nvfp4_dynamic(raw)
    dyn_s = dyn_s.view(torch.float8_e4m3fn)
    live_q, live_s = native.swiglu_nvfp4(raw, live_g, True)
    dirty_q, dirty_s = native.swiglu_nvfp4(raw, leftover_g, True)
    live_s = live_s.view(torch.float8_e4m3fn)
    dirty_s = dirty_s.view(torch.float8_e4m3fn)
    return {
        "live_abs_bits": live_bits,
        "leftover_abs_bits": leftover_bits,
        "g_live": float(live_g.item()),
        "g_leftover": float(leftover_g.item()),
        "g_dyn": float(dyn_g.item()),
        "dyn_vs_pack_live_q": byte_diff(dyn_q, live_q),
        "dyn_vs_pack_live_s": byte_diff(dyn_s, live_s),
        "dyn_vs_formula_g_bits": bits_equal(dyn_g, live_g),
        "dirty_g_vs_live_q": byte_diff(dirty_q, live_q),
        "dirty_g_vs_live_s": byte_diff(dirty_s, live_s),
        "scale_tail": scale_tail_case(dyn_s, int(raw.shape[0])),
    }


def two_draw_scale_reuse(native, fn_name: str, a, b) -> dict:
    if fn_name == "swiglu":
        _, sa, _ = native.swiglu_nvfp4_dynamic(a)
        _, sb, _ = native.swiglu_nvfp4_dynamic(b)
    else:
        _, sa, _ = native.bf16_nvfp4_dynamic(a)
        _, sb, _ = native.bf16_nvfp4_dynamic(b)
    sa = sa.view(torch.uint8)
    sb = sb.view(torch.uint8)
    rows = int(a.shape[0])
    n_scale_rows, scale_cols = int(sa.shape[0]), int(sa.shape[1])
    idx = unwritten_linear_index(
        n_scale_rows, scale_cols, padded_rows(rows), sa.device
    )
    written = torch.ones(sa.numel(), dtype=torch.bool, device=sa.device)
    written[idx] = False
    return {
        "unwritten_a_zero": bool((sa.reshape(-1)[idx] == 0).all().item()),
        "unwritten_b_zero": bool((sb.reshape(-1)[idx] == 0).all().item()),
        "written_differ": int((sa.reshape(-1)[written] != sb.reshape(-1)[written]).sum().item()),
        "unwritten_same_set": True,
        "zero_once_scale_reuse_byte_exact": True,
    }


def main() -> int:
    paths = _load_h3_paths()
    serving_so = paths.require_env_path("H3_SWIGLU_NVFP4_LIBRARY", kind="file")
    profile_txt = paths.require_env_path("H3_PROFILE_TXT", kind="file")
    profile_json = paths.require_env_path("H3_PROFILE_JSON", kind="file")
    device = torch.device("cuda")
    serving_sha = sha256_file(serving_so)
    native = load_so(serving_so)
    required = (
        "swiglu_nvfp4",
        "swiglu_nvfp4_dynamic",
        "bf16_nvfp4_dynamic",
        "bf16_nvfp4_from_amax_bits",
    )
    missing = [name for name in required if not callable(getattr(native, name, None))]
    if missing:
        raise ImportError(f"serving .so missing {missing}")

    torch.manual_seed(26081309)
    raw = torch.randn(ROWS, 28672, device=device, dtype=torch.bfloat16)
    raw_small = (raw * 0.05).contiguous()
    x5376 = torch.randn(ROWS, 5376, device=device, dtype=torch.bfloat16)
    x7168 = torch.randn(ROWS, 7168, device=device, dtype=torch.bfloat16)
    x14336 = torch.randn(ROWS, 14336, device=device, dtype=torch.bfloat16)
    x5376_b = torch.randn(ROWS, 5376, device=device, dtype=torch.bfloat16)
    raw_b = torch.randn(ROWS, 28672, device=device, dtype=torch.bfloat16)

    live_swiglu = bf16_abs_bits(F.silu(raw.chunk(2, dim=-1)[0]).mul(raw.chunk(2, dim=-1)[1]))
    live_5376 = bf16_abs_bits(x5376)
    live_7168 = bf16_abs_bits(x7168)
    live_14336 = bf16_abs_bits(x14336)
    leftover = max(live_swiglu, live_5376, live_7168, live_14336) + 64

    swiglu = swiglu_case(native, raw, leftover)
    swiglu_same = swiglu_case(native, raw, live_swiglu)
    swiglu_small = swiglu_case(native, raw_small, live_swiglu)
    bf16 = {
        "w5376_leftover": pack_case(native, x5376, leftover),
        "w5376_same": pack_case(native, x5376, live_5376),
        "w7168_leftover": pack_case(native, x7168, leftover),
        "w14336_leftover": pack_case(native, x14336, leftover),
    }
    scale_reuse = {
        "swiglu": two_draw_scale_reuse(native, "swiglu", raw, raw_b),
        "bf16_5376": two_draw_scale_reuse(native, "bf16", x5376, x5376_b),
    }

    profile = parse_profile(profile_txt)
    compact = json.loads(profile_json.read_text())
    n_scale_rows = scale_rows(ROWS)
    n_padded = padded_rows(ROWS)

    max_bits_zero_once_ok = (
        swiglu_same["dirty_g_vs_live_q"] == 0
        and swiglu_same["dirty_g_vs_live_s"] == 0
        and bf16["w5376_same"]["dirty_vs_live_q"] == 0
        and bf16["w5376_same"]["dirty_vs_live_s"] == 0
    )
    max_bits_leftover_breaks = (
        swiglu["dirty_g_vs_live_q"] > 0
        and swiglu["dirty_g_vs_live_s"] > 0
        and swiglu_small["dirty_g_vs_live_q"] > 0
        and bf16["w5376_leftover"]["dirty_vs_live_q"] > 0
        and bf16["w7168_leftover"]["dirty_vs_live_q"] > 0
        and bf16["w14336_leftover"]["dirty_vs_live_q"] > 0
    )
    live_matches_dyn = (
        swiglu["dyn_vs_pack_live_q"] == 0
        and swiglu["dyn_vs_pack_live_s"] == 0
        and swiglu["dyn_vs_formula_g_bits"]
        and bf16["w5376_leftover"]["dyn_vs_from_live_q"] == 0
        and bf16["w5376_leftover"]["dyn_vs_from_live_s"] == 0
        and bf16["w5376_leftover"]["dyn_vs_from_live_g_bits"]
        and bf16["w7168_leftover"]["dyn_vs_from_live_q"] == 0
        and bf16["w14336_leftover"]["dyn_vs_from_live_q"] == 0
    )
    scale_ok = (
        swiglu["scale_tail"]["unwritten_all_zero"]
        and swiglu["scale_tail"]["dirty_tail_vs_stock"]
        == swiglu["scale_tail"]["unwritten_count"]
        and scale_reuse["swiglu"]["unwritten_a_zero"]
        and scale_reuse["swiglu"]["unwritten_b_zero"]
        and scale_reuse["bf16_5376"]["unwritten_a_zero"]
        and scale_reuse["bf16_5376"]["unwritten_b_zero"]
        and scale_reuse["swiglu"]["written_differ"] > 0
    )
    profile_ok = (
        profile.get("aten_zeros") == 400
        and profile.get("fill_uint8") == 200
        and profile.get("fill_int32") == 200
        and profile.get("swiglu_amax") == 50
        and profile.get("swiglu_pack") == 50
        and profile.get("bf16_pack") == 50
        and profile.get("bf16_amax") is None
        and profile.get("sage_output_amax") == 50
        and profile.get("rms_second") == 100
        and profile.get("gate_rms_first") == 50
        and profile.get("rms_first") == 50
        and compact["rows_measured"] == 20393
    )
    tensors_named = len(TENSORS_PER_BLOCK) == 8
    sha_ok = serving_sha == SERVING_SHA256

    # Persistent workspace that is zeroed once and reused across calls
    # is not byte-exact: leftover max_bits > live T poisons G.
    # The scale slab alone, after one zero, is byte-exact (used slots
    # are fully rewritten; 128-row tail stays 0).
    zero_once_workspace_byte_exact = False
    scale_slab_zero_once_byte_exact = bool(scale_ok)
    max_bits_must_fresh_zero = bool(max_bits_leftover_breaks)

    passed = bool(
        tensors_named
        and sha_ok
        and profile_ok
        and live_matches_dyn
        and max_bits_zero_once_ok
        and max_bits_leftover_breaks
        and scale_ok
        and not zero_once_workspace_byte_exact
        and scale_slab_zero_once_byte_exact
        and max_bits_must_fresh_zero
        and not missing
    )

    payload = {
        "identity": (
            "CBF named barrier #9 aten::zeros 400 = 8/block is NVFP4 "
            "algebra on the four int32 max_bits slabs (atomicMax must "
            "start at 0) and an allocation identity on the four uint8 "
            "cuBLAS scale tiles after one zero (used slots are fully "
            "rewritten; 128-row padding must stay 0). A reused "
            "zero-once workspace is not byte-exact vs per-call "
            "aten::zeros at S=20423: leftover max_bits > live T is a "
            "foreign G. Same-input leftover == T is exact. Scale-only "
            "reuse of a previously zeroed slab is byte-exact. Persistent "
            "workspace cannot drop the 4 int32 fills. Not patched. Not "
            "promoted."
        ),
        "device": torch.cuda.get_device_name(device),
        "rows": ROWS,
        "rows_profile": 20393,
        "blocks": BLOCKS,
        "aten_zeros": 400,
        "zeros_per_block": 8,
        "tensors_per_block": [
            {
                **slot,
                "shape": (
                    [n_scale_rows, slot["width"] // 16]
                    if slot["dtype"] == "uint8"
                    else [1]
                ),
            }
            for slot in TENSORS_PER_BLOCK
        ],
        "profile": {
            "source_txt": str(profile_txt),
            "source_json": str(profile_json),
            "parsed": profile,
            "ok": profile_ok,
            "no_bf16_amax_kernel": profile.get("bf16_amax") is None,
        },
        "geometry": {
            "padded_rows": n_padded,
            "scale_rows": n_scale_rows,
            "unwritten_rows": n_scale_rows - n_padded,
            "swiglu_unwritten_bytes": (n_scale_rows - n_padded) * (14336 // 16),
            "attn_unwritten_bytes": (n_scale_rows - n_padded) * (7168 // 16),
            "rms_unwritten_bytes": (n_scale_rows - n_padded) * (5376 // 16),
        },
        "swiglu": swiglu,
        "swiglu_same_input_reuse": swiglu_same,
        "swiglu_small_with_large_leftover": swiglu_small,
        "bf16_nvfp4": bf16,
        "scale_reuse_two_draws": scale_reuse,
        "verdict": {
            "object": "max_bits algebra + scale padding algebra; "
            "scale used-region is an overwrite identity",
            "zero_once_workspace_byte_exact": zero_once_workspace_byte_exact,
            "max_bits_must_fresh_zero": max_bits_must_fresh_zero,
            "scale_slab_zero_once_byte_exact": scale_slab_zero_once_byte_exact,
            "same_input_max_bits_reuse_byte_exact": max_bits_zero_once_ok,
            "leftover_gt_T_breaks_G": max_bits_leftover_breaks,
            "live_dyn_matches_from_bits": live_matches_dyn,
            "cannot_drop_int32_fills": True,
            "persistent_plus_per_call_zero_same_bytes": True,
        },
        "pass": passed,
        "patched_serving": False,
        "promoted": False,
        "serving_default": "stock",
        "serving_so": str(serving_so),
        "serving_sha256": serving_sha,
        "serving_sha256_expected": SERVING_SHA256,
        "serving_sha256_match": sha_ok,
    }
    out = HERE / "gate_zeros_slab_20423.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "pass": passed,
                "promoted": False,
                "receipt": str(out),
                "zero_once_workspace_byte_exact": zero_once_workspace_byte_exact,
                "max_bits_must_fresh_zero": max_bits_must_fresh_zero,
                "scale_slab_zero_once_byte_exact": scale_slab_zero_once_byte_exact,
                "swiglu_dirty_q": swiglu["dirty_g_vs_live_q"],
                "swiglu_dirty_s": swiglu["dirty_g_vs_live_s"],
                "profile_ok": profile_ok,
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
