#!/usr/bin/env python3
"""Algebra gate: H3 block-0 residual skip identity.

Ask: is residual_0(step t) a function of residual_0(step t-1) plus a cheap
exact predicate (timestep, modulation, or residual delta) that is byte-equal
when the cache hits and fail-closed when it misses?

Lab-only. Does not patch serving. Does not flip cache default.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
COMFY = Path(os.environ.get("COMFYUI_ROOT", "$HOME/comfyui-h3-current"))
CACHE_SRC = COMFY / "comfy/ldm/minimax/first_block_cache.py"
SERVING_ENV = REPO / "config" / "h3_spark_serving.env"
LIVE_NODE = COMFY / "custom_nodes/h3_sage_attention.py"

SEED = 26081343
EPS = 1e-5
class TorchOperations:
    Linear = torch.nn.Linear
    RMSNorm = torch.nn.RMSNorm


def byte_diff(left: torch.Tensor, right: torch.Tensor) -> int:
    if left.shape != right.shape or left.dtype != right.dtype:
        raise RuntimeError("byte_diff shape/dtype mismatch")
    return int((left.view(torch.uint8) != right.view(torch.uint8)).sum().item())


def bits_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return byte_diff(left, right) == 0


def rel_mean_abs(current: torch.Tensor, previous: torch.Tensor) -> float:
    numerator = (current.float() - previous.float()).abs().mean()
    denominator = previous.float().abs().mean().clamp(min=1e-8)
    return float((numerator / denominator).item())


def make_row_ids(rows: int, n_mod: int, device: torch.device) -> torch.Tensor:
    return (
        torch.arange(rows, device=device, dtype=torch.int64)
        .mul_(n_mod)
        .div_(rows, rounding_mode="floor")
        .to(torch.int32)
        .contiguous()
    )


def segments_from_rows(row_ids: torch.Tensor) -> list[tuple[int, int, int]]:
    rows = row_ids.tolist()
    out: list[tuple[int, int, int]] = []
    start = 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or rows[i] != rows[start]:
            out.append((start, i, int(rows[start])))
            start = i
    return out


def serving_cache_default() -> tuple[str, str]:
    env_mode = "missing"
    for line in SERVING_ENV.read_text().splitlines():
        if line.startswith("H3_CROSS_STEP_CACHE="):
            env_mode = line.split("=", 1)[1].strip()
    node_text = LIVE_NODE.read_text()
    node_default = "missing"
    if '"default": "off"' in node_text and "cross_step_cache" in node_text:
        node_default = "off"
    return env_mode, node_default


def build_stack(hidden, heads, head_dim, ffn, t_dim, device, dtype, n_blocks):
    import comfy.ldm.minimax.model as minimax_model

    blocks = torch.nn.ModuleList()
    for _ in range(n_blocks):
        block = minimax_model.DiTBlock(
            hidden=hidden,
            heads=heads,
            head_dim=head_dim,
            ffn=ffn,
            t_dim=t_dim,
            eps=EPS,
            qk_eps=EPS,
            dtype=dtype,
            device=device,
            operations=TorchOperations,
        )
        blocks.append(block)
    blocks.eval()
    for parameter in blocks.parameters():
        parameter.requires_grad_(False)
    embedder = minimax_model.TimeEmbedder(
        freq_dim=t_dim,
        hidden=hidden,
        out=t_dim,
        dtype=torch.float32,
        device=device,
        operations=TorchOperations,
    )
    embedder.eval()
    for parameter in embedder.parameters():
        parameter.requires_grad_(False)
    return blocks, embedder


def modulation(block, t_emb):
    parts = block.adaln_proj(t_emb)
    return {
        "shift_msa": parts[0],
        "scale_msa": parts[1],
        "gate_msa": parts[2],
        "shift_mlp": parts[3],
        "scale_mlp": parts[4],
        "gate_mlp": parts[5],
    }


def run_one(block, hidden, t_emb, segments, row_ids):
    # DiTBlock _mod_gate is in-place on the residual stream.
    x = hidden.clone()
    y = block(
        x,
        t_emb,
        segments,
        None,
        transformer_options={"minimax_h3_modulation_fusion": "stock"},
        mod_row_ids=row_ids,
    )
    return y, y - hidden


def run_stack(blocks, hidden, t_emb, segments, row_ids):
    h = hidden.clone()
    first_out = None
    first_res = None
    for index, block in enumerate(blocks):
        out, res = run_one(block, h, t_emb, segments, row_ids)
        if index == 0:
            first_out = out
            first_res = res
        h = out
    return first_out, first_res, h, h - first_out


def cheap_scale_copy(prev, scale):
    return (prev.float() * float(scale)).to(prev.dtype)


def per_row_gate_scale(prev, gate_now, gate_prev, row_ids):
    g0 = gate_prev[row_ids].to(torch.float32)
    g1 = gate_now[row_ids].to(torch.float32)
    scale = g1 / g0.clamp(min=1e-8)
    return (prev.float() * scale).to(prev.dtype)


def evaluate_pair(name, blocks, x0, t0, x1, t1, segments, row_ids):
    first0, res0, full0, tail0 = run_stack(blocks, x0, t0, segments, row_ids)
    first1, res1, full1, tail1 = run_stack(blocks, x1, t1, segments, row_ids)
    mod0 = modulation(blocks[0], t0)
    mod1 = modulation(blocks[0], t1)

    pred = {
        "input_equal": bits_equal(x0, x1),
        "t_emb_equal": bits_equal(t0, t1),
        "modulation_equal": all(bits_equal(mod0[k], mod1[k]) for k in mod0),
        "residual_delta_zero": bits_equal(res0, res1),
        "first_out_equal": bits_equal(first0, first1),
        "state_equal": bits_equal(x0, x1) and bits_equal(t0, t1),
    }
    pred["cache_threshold"] = rel_mean_abs(res1, res0) <= 0.08

    gate_mean0 = float(mod0["gate_msa"].float().abs().mean().item())
    gate_mean1 = float(mod1["gate_msa"].float().abs().mean().item())
    gate_ratio = gate_mean1 / max(gate_mean0, 1e-8)

    recon_res = {
        "copy_prev": byte_diff(res1, res0),
        "gate_mean_scale": byte_diff(res1, cheap_scale_copy(res0, gate_ratio)),
        "per_row_gate_msa": byte_diff(
            res1, per_row_gate_scale(res0, mod1["gate_msa"], mod0["gate_msa"], row_ids)
        ),
        "add_input_delta": byte_diff(res1, (res0.float() + (x1 - x0).float()).to(res1.dtype)),
    }
    cache_tail = first1 + tail0
    # Serving cache stores tail in activation dtype (bf16).
    # a + (b - a) is not an involution there.
    fp32_tail = first1.float() + (full0.float() - first0.float())
    recon_full = {
        "cache_tail_add": byte_diff(full1, cache_tail),
        "cache_tail_add_fp32": byte_diff(full1, fp32_tail.to(full1.dtype)),
        "copy_prev_full": byte_diff(full1, full0),
        "first_out_only": byte_diff(full1, first1),
        "bf16_involution": byte_diff(full0, first0 + (full0 - first0)),
        "fp32_involution": byte_diff(full0, (first0.float() + (full0.float() - first0.float())).to(full0.dtype)),
    }

    # Sound hit: predicate true implies the named reconstruction is byte-exact.
    sound = {
        "state_equal_copy_residual": (not pred["state_equal"]) or recon_res["copy_prev"] == 0,
        "state_equal_cache_tail": (not pred["state_equal"]) or recon_full["cache_tail_add"] == 0,
        "residual_delta_zero_copy": (not pred["residual_delta_zero"]) or recon_res["copy_prev"] == 0,
        "t_emb_only_copy": (not pred["t_emb_equal"]) or recon_res["copy_prev"] == 0,
        "modulation_only_copy": (not pred["modulation_equal"]) or recon_res["copy_prev"] == 0,
        "input_only_copy": (not pred["input_equal"]) or recon_res["copy_prev"] == 0,
        "residual_delta_zero_cache_tail": (
            (not pred["residual_delta_zero"]) or recon_full["cache_tail_add"] == 0
        ),
        "first_out_and_t_cache_tail": (
            (not (pred["first_out_equal"] and pred["t_emb_equal"]))
            or recon_full["cache_tail_add"] == 0
        ),
        "first_out_and_t_fp32_tail": (
            (not (pred["first_out_equal"] and pred["t_emb_equal"]))
            or recon_full["cache_tail_add_fp32"] == 0
        ),
        "cache_threshold_cache_tail": (
            (not pred["cache_threshold"]) or recon_full["cache_tail_add"] == 0
        ),
    }

    return {
        "pair": name,
        "predicates": pred,
        "rel_mean_abs": rel_mean_abs(res1, res0),
        "rel_mean_abs_full": rel_mean_abs(full1, full0),
        "gate_msa_mean_ratio": gate_ratio,
        "residual_recon_byte_diff": recon_res,
        "full_recon_byte_diff": recon_full,
        "sound": sound,
        "residual_elems": int(res1.numel()),
        "full_elems": int(full1.numel()),
    }


def run_geometry(label, rows, hidden, heads, head_dim, ffn, t_dim, n_mod, device):
    dtype = torch.bfloat16
    blocks, embedder = build_stack(hidden, heads, head_dim, ffn, t_dim, device, dtype, 2)
    row_ids = make_row_ids(rows, n_mod, device)
    segments = segments_from_rows(row_ids)
    n_t = max(1, n_mod // 3)

    def t_emb_from_sigma(video_sigma: float) -> torch.Tensor:
        # Seven packed timestep classes: cond pinned, audio shifted, video live.
        vals = []
        for i in range(n_t):
            if i == 0:
                vals.append(0.999)
            elif i == 1:
                vals.append(min(1.0, video_sigma * 0.4 + 0.05))
            else:
                vals.append(video_sigma)
        t = torch.tensor(vals, dtype=torch.float32, device=device)
        return embedder(t).to(dtype).contiguous()

    x0 = torch.randn(rows, hidden, device=device, dtype=dtype)
    t0 = t_emb_from_sigma(0.85)
    t1 = t_emb_from_sigma(0.70)
    x_ulp = x0.clone()
    x_ulp[0, 0] = torch.nextafter(
        x0[0, 0], torch.tensor(float("inf"), device=device, dtype=x0.dtype)
    )
    pairs = {
        "replay": (x0, t0, x0.clone(), t0.clone()),
        "same_t_moved_x": (x0, t0, (x0 + 0.05 * torch.randn_like(x0)).contiguous(), t0.clone()),
        "same_x_moved_t": (x0, t0, x0.clone(), t1),
        "moved_both": (x0, t0, (x0 + 0.05 * torch.randn_like(x0)).contiguous(), t1),
        "tiny_x_same_t": (x0, t0, x_ulp, t0.clone()),
        "tiny_x_moved_t": (x0, t0, x_ulp.clone(), t1),
    }
    reports = [
        evaluate_pair(name, blocks, a, ta, b, tb, segments, row_ids)
        for name, (a, ta, b, tb) in pairs.items()
    ]

    # Cheap predicates that never inspect residual(t).
    cheap_false_hits = []
    for report in reports:
        if report["pair"] == "replay":
            continue
        if report["predicates"]["t_emb_equal"] and report["residual_recon_byte_diff"]["copy_prev"] != 0:
            cheap_false_hits.append((report["pair"], "t_emb_equal"))
        if report["predicates"]["modulation_equal"] and report["residual_recon_byte_diff"]["copy_prev"] != 0:
            cheap_false_hits.append((report["pair"], "modulation_equal"))
        if report["predicates"]["cache_threshold"] and report["full_recon_byte_diff"]["cache_tail_add"] != 0:
            cheap_false_hits.append((report["pair"], "cache_threshold"))

    replay = next(r for r in reports if r["pair"] == "replay")
    moved = next(r for r in reports if r["pair"] == "moved_both")
    same_t = next(r for r in reports if r["pair"] == "same_t_moved_x")
    same_x = next(r for r in reports if r["pair"] == "same_x_moved_t")
    tiny = next(r for r in reports if r["pair"] == "tiny_x_same_t")

    return {
        "label": label,
        "rows": rows,
        "hidden": hidden,
        "heads": heads,
        "blocks": 2,
        "mod_rows": n_mod,
        "pairs": reports,
        "replay_residual_exact": replay["residual_recon_byte_diff"]["copy_prev"] == 0,
        "replay_full_copy_exact": replay["full_recon_byte_diff"]["copy_prev_full"] == 0,
        "replay_bf16_tail_exact": replay["full_recon_byte_diff"]["cache_tail_add"] == 0,
        "replay_fp32_tail_exact": replay["full_recon_byte_diff"]["cache_tail_add_fp32"] == 0,
        "replay_bf16_involution_diff": replay["full_recon_byte_diff"]["bf16_involution"],
        "replay_fp32_involution_diff": replay["full_recon_byte_diff"]["fp32_involution"],
        "cheap_predicate_false_hits": [
            {"pair": pair, "predicate": pred} for pair, pred in cheap_false_hits
        ],
        "nontrivial_residual_delta_zero": any(
            r["predicates"]["residual_delta_zero"] and r["pair"] != "replay" for r in reports
        ),
        "tiny_x_actually_moved": not tiny["predicates"]["input_equal"],
        "moved_both_copy_diff": moved["residual_recon_byte_diff"]["copy_prev"],
        "same_t_moved_x_copy_diff": same_t["residual_recon_byte_diff"]["copy_prev"],
        "same_x_moved_t_copy_diff": same_x["residual_recon_byte_diff"]["copy_prev"],
        "threshold_is_exact": all(r["sound"]["cache_threshold_cache_tail"] for r in reports)
        and any(r["predicates"]["cache_threshold"] and r["pair"] != "replay" for r in reports),
    }


def main() -> int:
    torch.set_grad_enabled(False)
    if str(COMFY) not in sys.path:
        sys.path.insert(0, str(COMFY))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    env_mode, node_default = serving_cache_default()
    geometries = [
        run_geometry("small_s129", 129, 64, 4, 16, 128, 64, 21, device),
        run_geometry("small_s20423", 20423, 64, 4, 16, 128, 64, 21, device),
        run_geometry("prod_width_s129", 129, 5376, 56, 128, 14336, 2688, 21, device),
    ]

    all_replay_res = all(g["replay_residual_exact"] and g["replay_full_copy_exact"] for g in geometries)
    all_bf16_broken = all(not g["replay_bf16_tail_exact"] and g["replay_bf16_involution_diff"] > 0 for g in geometries)
    all_fp32_involution = all(g["replay_fp32_involution_diff"] == 0 for g in geometries)
    any_cheap_false = any(
        any(hit["predicate"] in ("t_emb_equal", "modulation_equal", "cache_threshold")
            for hit in g["cheap_predicate_false_hits"])
        for g in geometries
    )
    any_moved_copy_zero = any(g["moved_both_copy_diff"] == 0 for g in geometries)

    identity = (
        "H3 block-0 residual(t) is F(x_t, t_emb_t), not a function of "
        "residual(t-1) plus timestep, modulation, or a cheap residual delta. "
        "Copy / gate-scale / input-delta reconstructions are not byte-exact "
        "once x or t_emb moves. residual_delta==0 is tautological and requires "
        "residual(t). Serving cache tail add first+(full-first) is not "
        "byte-exact even on replay: bf16 a+(b-a) is not an involution. "
        "fp32 tail add is involutive on replay and still misses once x or t "
        "moves. The 0.08 threshold is not an exact predicate."
    )
    verdict = (
        all_replay_res
        and all_bf16_broken
        and all_fp32_involution
        and any_cheap_false
        and not any_moved_copy_zero
        and env_mode == "off"
        and node_default == "off"
    )
    receipt = {
        "identity": identity,
        "device": torch.cuda.get_device_name(0),
        "seed": SEED,
        "dtype": "bfloat16",
        "cache_source": str(CACHE_SRC),
        "serving_default": {
            "H3_CROSS_STEP_CACHE": env_mode,
            "node_cross_step_cache": node_default,
            "unchanged": env_mode == "off" and node_default == "off",
        },
        "geometries": [
            {
                "label": g["label"],
                "rows": g["rows"],
                "hidden": g["hidden"],
                "replay_residual_exact": g["replay_residual_exact"],
                "replay_full_copy_exact": g["replay_full_copy_exact"],
                "replay_bf16_tail_exact": g["replay_bf16_tail_exact"],
                "replay_fp32_tail_exact": g["replay_fp32_tail_exact"],
                "replay_bf16_involution_diff": g["replay_bf16_involution_diff"],
                "replay_fp32_involution_diff": g["replay_fp32_involution_diff"],
                "tiny_x_actually_moved": g["tiny_x_actually_moved"],
                "moved_both_copy_diff": g["moved_both_copy_diff"],
                "same_t_moved_x_copy_diff": g["same_t_moved_x_copy_diff"],
                "same_x_moved_t_copy_diff": g["same_x_moved_t_copy_diff"],
                "cheap_predicate_false_hits": g["cheap_predicate_false_hits"],
                "nontrivial_residual_delta_zero": g["nontrivial_residual_delta_zero"],
                "pairs": g["pairs"],
            }
            for g in geometries
        ],
        "algebra": {
            "block0": "residual = gate_msa*Attn(AdaLN(x,t)) + gate_mlp*MLP(AdaLN(x+attn_res,t))",
            "depends_on": ["x", "t_emb"],
            "not_a_function_of": [
                "residual(t-1) + timestep",
                "residual(t-1) + modulation",
                "residual(t-1) * gate_ratio",
                "residual(t-1) + (x_t - x_{t-1})",
            ],
            "exact_fail_closed_predicate": "x bits equal AND t_emb bits equal (duplicate forward); then copy the previous residual/full, do not add a bf16 tail",
            "residual_delta_zero": "tautology; requires residual(t)",
            "serving_bf16_tail": "first+(full-first) is not involutive; not byte-exact on replay",
            "fp32_tail": "involutive on replay; still not a skip once x or t_emb moves",
            "serving_threshold_0p08": "approximation; not byte-exact",
        },
        "decision": "no_exact_skip",
        "promoted": False,
        "serving_patched": False,
        "default_changed": False,
        "pass": bool(verdict),
    }
    out = HERE / "gate_skip_algebra.json"
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"pass": receipt["pass"], "receipt": str(out), "decision": receipt["decision"]}, indent=2))
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
