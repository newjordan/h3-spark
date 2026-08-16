#!/usr/bin/env python3
"""Merge tonight's live identity model.py with HOLD cross-block wiring.

Source of truth: live identities + first-block cache (default off) plus the
next-sprint cross-block path (default stock). Cache and cross-block auto
are not mixed.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LIVE = Path(
    os.environ.get(
        "H3_LIVE_MODEL",
        str(REPO.parent / "comfyui-h3-current/comfy/ldm/minimax/model.py"),
    )
)
OVERLAY = REPO / "integration/comfyui/comfy/ldm/minimax/model.py"

CROSSBLOCK_IMPORT = '''from comfy.ldm.minimax.first_block_cache import resolve_cache, run_block_stack
from comfy.ldm.minimax.crossblock_gate_qkv import (
    CROSSBLOCK_TOKEN as _CROSSBLOCK_TOKEN,
    block_pair_eligible as crossblock_pair_eligible,
    defer_qkv_transition as defer_crossblock_qkv_transition,
    is_pending as is_crossblock_pending,
    materialize_pending as materialize_crossblock_pending,
    pending_residual as crossblock_pending_residual,
    try_qkv_transition as try_crossblock_qkv_transition,
)
'''

HELPERS = '''
def _h3_crossblock_mode(transformer_options):
    # HOLD: explicit only. Do not inherit RMS/SwiGLU auto.
    return transformer_options.get("minimax_h3_crossblock_gate_qkv_fusion", "stock")


def _loop_owned_blocks_replace(transformer_options):
    """One mapping object for the whole DiT loop.

    ``dict.get(key, {})`` allocates a fresh empty dict on every miss. The
    earlier candidate re-derived that inside each block, so a sealed empty
    ``blocks_replace`` failed ``is`` checks against the next block's empty
    dict. The loop owns one mapping and passes that identity through.
    """
    patches_replace = transformer_options.get("patches_replace")
    if not isinstance(patches_replace, dict):
        return {}
    blocks_replace = patches_replace.get("dit")
    if not isinstance(blocks_replace, dict):
        return {}
    return blocks_replace


'''

FORWARD = '''    def forward(
        self,
        x,
        t_emb,
        mod_segments,
        rope_freqs,
        transformer_options={},
        mod_row_ids=None,
        adaln_nvfp4_token=None,
        crossblock_pending=None,
        crossblock_previous=None,
        crossblock_next=None,
        crossblock_token=None,
        crossblock_defer=False,
        blocks_replace=None,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
        modulation_mode = transformer_options.get("minimax_h3_modulation_fusion", "exact")
        aggressive = modulation_mode == "aggressive"
        trusted_row_map = adaln_nvfp4_token is _ADALN_NVFP4_TOKEN
        rms_adaln_mode = _h3_rms_adaln_mode(transformer_options)
        crossblock_mode = _h3_crossblock_mode(transformer_options)
        if blocks_replace is None:
            blocks_replace = ()
        qkv = None
        if crossblock_pending is not None:
            qkv = try_crossblock_qkv_transition(
                crossblock_previous,
                self,
                x,
                crossblock_pending,
                scale_msa,
                shift_msa,
                mod_segments,
                mod_row_ids,
                adaln_nvfp4_token,
                crossblock_token,
                crossblock_mode,
                modulation_mode,
                blocks_replace,
            )
            if qkv is None:
                x = materialize_crossblock_pending(
                    crossblock_pending,
                    crossblock_previous,
                    self,
                    x,
                    mod_segments,
                    mod_row_ids,
                    crossblock_token,
                    _mod_gate,
                )
        if qkv is not None:
            attn = self.attn._forward_projected(
                qkv, rope_freqs=rope_freqs, transformer_options=transformer_options
            )
        elif modulation_mode == "exact" and trusted_row_map and adaln_parent_eligible(self.attn):
            qkv = _h3_rms_adaln_linear(
                self.attn.qkv_proj, self.norm1, x, shift_msa, scale_msa,
                mod_row_ids, transformer_options
            )
            if qkv is not None:
                attn = self.attn._forward_projected(qkv, rope_freqs=rope_freqs, transformer_options=transformer_options)
            else:
                normed = self.norm1(x)
                qkv = _h3_adaln_linear(
                    self.attn.qkv_proj, normed, shift_msa, scale_msa,
                    mod_row_ids, transformer_options
                )
                if qkv is not None:
                    attn = self.attn._forward_projected(
                        qkv, rope_freqs=rope_freqs, transformer_options=transformer_options)
                else:
                    attn = self.attn(
                        _mod_scale_shift(normed, shift_msa, scale_msa, mod_segments, mod_row_ids),
                        rope_freqs=rope_freqs, transformer_options=transformer_options)
        else:
            h = _norm_mod(self.norm1, x, shift_msa, scale_msa, mod_segments, mod_row_ids, aggressive)
            attn = self.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options)
        mlp = None
        if (
            modulation_mode == "exact"
            and trusted_row_map
            and rms_adaln_mode == "auto"
            and adaln_parent_eligible(self.mlp)
        ):
            mlp = try_gate_rms_adaln_nvfp4_mlp(
                self.mlp, self.norm2, x, attn, gate_msa, scale_mlp, shift_mlp,
                mod_row_ids, _ADALN_NVFP4_TOKEN, rms_adaln_mode,
                modulation_mode, transformer_options,
            )
        if mlp is None:
            x = _mod_gate(x, gate_msa, attn, mod_segments, mod_row_ids)
        if mlp is None and modulation_mode == "exact" and trusted_row_map and adaln_parent_eligible(self.mlp):
            raw = _h3_rms_adaln_linear(
                self.mlp.fc1, self.norm2, x, shift_mlp, scale_mlp,
                mod_row_ids, transformer_options
            )
            if raw is not None:
                mlp = self.mlp._forward_raw(raw, transformer_options=transformer_options)
            else:
                normed = self.norm2(x)
                raw = _h3_adaln_linear(
                    self.mlp.fc1, normed, shift_mlp, scale_mlp,
                    mod_row_ids, transformer_options
                )
                if raw is not None:
                    mlp = self.mlp._forward_raw(raw, transformer_options=transformer_options)
                else:
                    mlp = self.mlp(
                        _mod_scale_shift(normed, shift_mlp, scale_mlp, mod_segments, mod_row_ids),
                        transformer_options=transformer_options)
        elif mlp is None:
            h = _norm_mod(self.norm2, x, shift_mlp, scale_mlp, mod_segments, mod_row_ids, aggressive)
            mlp = self.mlp(h, transformer_options=transformer_options)
        if crossblock_defer and crossblock_next is not None:
            pending = defer_crossblock_qkv_transition(
                self,
                crossblock_next,
                x,
                mlp,
                gate_mlp,
                mod_segments,
                mod_row_ids,
                crossblock_token,
                crossblock_mode,
                modulation_mode,
                blocks_replace,
            )
            if pending is not None:
                return pending
        return _mod_gate(x, gate_mlp, mlp, mod_segments, mod_row_ids)
'''

RUN_BLOCKS = '''
def _run_dit_blocks(
    blocks,
    h,
    t_emb,
    mod_segments,
    rope_freqs,
    transformer_options,
    mod_row_ids,
    device,
):
    blocks_replace = _loop_owned_blocks_replace(transformer_options)
    pending = None
    previous_block = None
    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(
        list(blocks), device, transformer_options
    )
    for i, block in enumerate(blocks):
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block)
        replacement_key = ("double_block", i)
        if replacement_key in blocks_replace:
            replacement = blocks_replace[replacement_key]
            if pending is not None:
                h = materialize_crossblock_pending(
                    pending,
                    previous_block,
                    block,
                    h,
                    mod_segments,
                    mod_row_ids,
                    _CROSSBLOCK_TOKEN,
                    _mod_gate,
                )
                pending = None
                previous_block = None

            def block_wrap(args):
                return {
                    "img": block(
                        args["img"],
                        args["t_emb"],
                        args["mod_segments"],
                        args["rope_freqs"],
                        transformer_options=args["transformer_options"],
                        mod_row_ids=args.get("mod_row_ids"),
                    )
                }

            h = replacement(
                {
                    "img": h,
                    "t_emb": t_emb,
                    "mod_segments": mod_segments,
                    "rope_freqs": rope_freqs,
                    "transformer_options": transformer_options,
                    "mod_row_ids": mod_row_ids,
                },
                {"original_block": block_wrap},
            )["img"]
            continue

        next_block = blocks[i + 1] if i + 1 < len(blocks) else None
        crossblock_defer = (
            type(block) is DiTBlock
            and type(next_block) is DiTBlock
            and crossblock_pair_eligible(
                block,
                next_block,
                _CROSSBLOCK_TOKEN,
                _h3_crossblock_mode(transformer_options),
                transformer_options.get("minimax_h3_modulation_fusion", "exact"),
                blocks_replace,
            )
        )
        result = block(
            h,
            t_emb,
            mod_segments,
            rope_freqs,
            transformer_options=transformer_options,
            mod_row_ids=mod_row_ids,
            adaln_nvfp4_token=_ADALN_NVFP4_TOKEN,
            crossblock_pending=pending,
            crossblock_previous=previous_block,
            crossblock_next=next_block,
            crossblock_token=_CROSSBLOCK_TOKEN,
            crossblock_defer=crossblock_defer,
            blocks_replace=blocks_replace,
        )
        if is_crossblock_pending(result):
            if not crossblock_defer:
                raise RuntimeError("unexpected H3 cross-block carrier")
            h = crossblock_pending_residual(
                result,
                block,
                next_block,
                mod_segments,
                mod_row_ids,
                _CROSSBLOCK_TOKEN,
            )
            pending = result
            previous_block = block
        else:
            h = result
            pending = None
            previous_block = None
    if pending is not None:
        raise RuntimeError("last H3 block returned a deferred carrier")
    if prefetch_queue is not None:
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)
    return h


'''

OLD_LOOP = '''        # blocks
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.blocks), device, transformer_options)

        def run_one(index, block, hidden):
            if ("double_block", index) in blocks_replace:
                def block_wrap(args):
                    return {"img": block(args["img"], args["t_emb"], args["mod_segments"], args["rope_freqs"],
                                         transformer_options=args["transformer_options"],
                                         mod_row_ids=args.get("mod_row_ids"))}
                return blocks_replace[("double_block", index)](
                    {"img": hidden, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": rope_freqs,
                     "transformer_options": transformer_options, "mod_row_ids": mod_row_ids},
                    {"original_block": block_wrap})["img"]
            return block(hidden, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options,
                         mod_row_ids=mod_row_ids, adaln_nvfp4_token=_ADALN_NVFP4_TOKEN)

        h = run_block_stack(
            self.blocks, h,
            run_one=run_one,
            prefetch_queue=prefetch_queue,
            device=device,
            transformer_options=transformer_options,
            timestep=timestep,
            minimax_payload=payload,
            blocks_replace=blocks_replace,
        )
'''

NEW_LOOP = '''        # blocks. Cache (default off) and cross-block auto are not mixed.
        blocks_replace = _loop_owned_blocks_replace(transformer_options)
        cache = resolve_cache(transformer_options, len(self.blocks), blocks_replace)
        if cache is not None:
            prefetch_queue = comfy.model_prefetch.make_prefetch_queue(
                list(self.blocks), device, transformer_options
            )

            def run_one(index, block, hidden):
                if ("double_block", index) in blocks_replace:
                    def block_wrap(args):
                        return {"img": block(args["img"], args["t_emb"], args["mod_segments"], args["rope_freqs"],
                                             transformer_options=args["transformer_options"],
                                             mod_row_ids=args.get("mod_row_ids"))}
                    return blocks_replace[("double_block", index)](
                        {"img": hidden, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": rope_freqs,
                         "transformer_options": transformer_options, "mod_row_ids": mod_row_ids},
                        {"original_block": block_wrap})["img"]
                return block(hidden, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options,
                             mod_row_ids=mod_row_ids, adaln_nvfp4_token=_ADALN_NVFP4_TOKEN)

            h = run_block_stack(
                self.blocks, h,
                run_one=run_one,
                prefetch_queue=prefetch_queue,
                device=device,
                transformer_options=transformer_options,
                timestep=timestep,
                minimax_payload=payload,
                blocks_replace=blocks_replace,
            )
        else:
            h = _run_dit_blocks(
                self.blocks, h, t_emb, mod_segments, rope_freqs,
                transformer_options, mod_row_ids, device,
            )
'''

OLD_FORWARD_START = "    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}, mod_row_ids=None, adaln_nvfp4_token=None):\n"


def replace_forward(src: str) -> str:
    start = src.find(OLD_FORWARD_START)
    if start < 0:
        raise SystemExit("live DiTBlock.forward not found")
    end = src.find("\nclass FinalLayer", start)
    if end < 0:
        raise SystemExit("FinalLayer not found after DiTBlock.forward")
    return src[:start] + FORWARD + src[end:]


def main() -> None:
    src = LIVE.read_text()
    old_imp = "from comfy.ldm.minimax.first_block_cache import run_block_stack\n"
    if old_imp not in src:
        raise SystemExit("live first_block_cache import not found")
    src = src.replace(old_imp, CROSSBLOCK_IMPORT, 1)
    marker = "class DiTBlock(nn.Module):\n"
    if marker not in src:
        raise SystemExit("DiTBlock class not found")
    src = src.replace(marker, HELPERS + marker, 1)
    src = replace_forward(src)
    fl = "\nclass FinalLayer(nn.Module):\n"
    if fl not in src:
        raise SystemExit("FinalLayer marker missing after forward replace")
    src = src.replace(fl, RUN_BLOCKS + fl, 1)
    if OLD_LOOP not in src:
        raise SystemExit("live block loop not found")
    src = src.replace(OLD_LOOP, NEW_LOOP, 1)
    OVERLAY.write_text(src)
    LIVE.write_text(src)
    print(f"wrote {OVERLAY} and {LIVE} ({len(src.splitlines())} lines)")


if __name__ == "__main__":
    main()
