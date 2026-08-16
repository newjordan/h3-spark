"""MiniMax H3 audio-video DiT.

Single-stream packed-token transformer denoising video (24ch, patch 1x2x2) and
stereo audio (32ch, 40 Hz) latents jointly, conditioned on Qwen3-VL layer-50 hidden states.
The packed sequence is:
[text | cond rows | audio | video] for t2va/fl2va
[text | reference blocks | audio | video] for ref2va

Timestep domain: the model receives the *video* sigma from the sampler and
derives per-token timesteps t = 1 - sigma internally; the audio stream runs on
its own shifted schedule (sigma_shift video 12.0 / audio 3.0), mapped from the
video sigma in closed form. The sampler carries the audio latent scaled onto the
video schedule (ModelSamplingAV); forward() undoes that scale and converts the
velocity back, so _forward only ever sees the stream's own latent.
"""

import math

import torch
import torch.nn as nn

import comfy.ldm.common_dit
import comfy.model_management
import comfy.model_prefetch
import comfy.ops
import comfy.patcher_extension
import comfy.quant_ops
from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.minimax.kernels import (
    segmented_adaln_bf16_,
    segmented_gate_bf16_,
    segmented_rms_adaln_bf16,
)
from comfy.ldm.minimax.adaln_nvfp4 import (
    ROW_IDS_TOKEN as _ADALN_NVFP4_TOKEN,
    parent_eligible as adaln_parent_eligible,
    try_linear as try_adaln_nvfp4_linear,
)
from comfy.ldm.minimax.rms_adaln_nvfp4 import try_linear as try_rms_adaln_nvfp4_linear
from comfy.ldm.minimax.gate_rms_adaln_nvfp4_static import try_mlp_transition as try_gate_rms_adaln_nvfp4_mlp
from comfy.ldm.minimax.first_block_cache import resolve_cache, run_block_stack
from comfy.ldm.minimax.crossblock_gate_qkv import (
    CROSSBLOCK_TOKEN as _CROSSBLOCK_TOKEN,
    block_pair_eligible as crossblock_pair_eligible,
    defer_qkv_transition as defer_crossblock_qkv_transition,
    is_pending as is_crossblock_pending,
    materialize_pending as materialize_crossblock_pending,
    pending_residual as crossblock_pending_residual,
    try_qkv_transition as try_crossblock_qkv_transition,
)
from comfy.ldm.minimax.linear_nvfp4 import (
    can_linear_from_amax_bits,
    can_linear_from_amax_bits_for_output,
    can_static_linear_for_output,
    try_linear as try_nvfp4_linear,
    try_linear_from_amax_bits as try_nvfp4_linear_from_amax_bits,
)
from comfy.ldm.minimax.swiglu_nvfp4 import try_linear as try_swiglu_nvfp4_linear

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0
VISUAL_COND_TIMESTEP = 0.999
AUDIO_COND_TIMESTEP = 1.0


class H3AttentionAmaxMarker:
    __slots__ = ("bits", "owner")

    def __init__(self, bits, owner):
        self.bits = bits
        self.owner = owner


def _h3_linear(linear, x, transformer_options):
    mode = transformer_options.get(
        "minimax_h3_linear_nvfp4_fusion",
        transformer_options.get("minimax_h3_swiglu_nvfp4_fusion", "stock"),
    )
    fused = try_nvfp4_linear(linear, x, mode)
    return linear(x) if fused is None else fused


def time_shift_sigma(sigma, from_shift, to_shift):
    # invert sigma = s*b/(1+(s-1)*b) to the base grid, re-apply the other shift
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def patchify_video(latent, patch_size=(1, 2, 2)):
    # [B, C, T, H, W] -> [B*t*h*w, C*pt*ph*pw]
    # permute == einsum("nctrhpwq->nthwcrpq") (gated)
    b, c, t_full, h_full, w_full = latent.shape
    pt, ph, pw = patch_size
    t, h, w = t_full // pt, h_full // ph, w_full // pw
    x = latent.reshape(b, c, t, pt, h, ph, w, pw)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return x.reshape(b * t * h * w, c * pt * ph * pw)


def unpatchify_video(rows, t, h, w, c=24, patch_size=(1, 2, 2)):
    # permute == einsum("nthwcrpq->nctrhpwq") (gated)
    pt, ph, pw = patch_size
    x = rows.reshape(-1, t, h, w, c, pt, ph, pw)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return x.reshape(-1, c, t * pt, h * ph, w * pw)


def cond_unique_blocks(latents, build_one):
    # build_one(z) is a function of the latent only. Returns
    # (unique_cat, inverse_ids, block_lens). inverse_ids[i] picks the unique
    # block for original item i. Linear is row-wise, so proj(unique) then
    # repeat blocks == proj(cat of copies).
    cache = []
    ids = []
    for z in latents:
        hit = None
        for i, (prev, prev_r) in enumerate(cache):
            if (
                prev.shape == z.shape
                and prev.dtype == z.dtype
                and prev.device == z.device
                and torch.equal(prev, z)
            ):
                hit = i
                break
        if hit is None:
            hit = len(cache)
            cache.append((z, build_one(z)))
        ids.append(hit)
    if not cache:
        return None, [], []
    unique = [r for _, r in cache]
    return torch.cat(unique, dim=0), ids, [r.shape[0] for r in unique]


def expand_unique_blocks(unique_cat, ids, lens):
    if unique_cat is None or not ids:
        return None
    blocks = unique_cat.split(lens, dim=0)
    return torch.cat([blocks[i] for i in ids], dim=0)


def cond_rows_from_unique(latents, build_one):
    # build_one(z) is a function of the latent tensor only (patchify/pack +
    # restarted RNG). Identical conds share one row; cat still places N copies.
    unique, ids, lens = cond_unique_blocks(latents, build_one)
    return expand_unique_blocks(unique, ids, lens)


def project_unique_cond_live(proj, live_rows, unique_cat, ids, lens, update, dtype):
    # One Linear on cat(unique, live), then repeat unique blocks into cond
    # slots. == proj(mix(expand(unique), live)) when unique is strictly
    # smaller than the expanded cond (FixedRef six-ref).
    if unique_cat is None:
        return proj(live_rows).to(dtype)
    n_u = unique_cat.shape[0]
    emb = proj(torch.cat([unique_cat, live_rows], dim=0)).to(dtype)
    cond_emb = expand_unique_blocks(emb[:n_u], ids, lens)
    out = emb.new_empty(update.shape[0], emb.shape[1])
    out[~update] = cond_emb
    out[update] = emb[n_u:]
    return out


def _embed_cond_or_mix(proj, live_rows, cond_rows, update, dtype, unique_pack):
    if cond_rows is None:
        return proj(live_rows).to(dtype)
    if unique_pack is not None:
        unique_cat, ids, lens = unique_pack
        if unique_cat is not None and unique_cat.shape[0] < cond_rows.shape[0]:
            return project_unique_cond_live(
                proj, live_rows, unique_cat, ids, lens, update, dtype
            )
    mixed = torch.empty(
        update.shape[0], live_rows.shape[1],
        dtype=torch.float32, device=live_rows.device,
    )
    mixed[~update] = cond_rows
    mixed[update] = live_rows
    return proj(mixed).to(dtype)


def _layout_text_tag_runs(layout, text_tags):
    # tag-run table is a function of the presentation tokens, not of t.
    if text_tags is None:
        return None
    cached = getattr(layout, "_tag_runs", None)
    if cached is not None and cached[0] is text_tags:
        return cached[1]
    span = next(((a, b) for a, b, k in layout.segments if k == "text"), None)
    if span is None:
        layout._tag_runs = (text_tags, None)
        return None
    a, b = span
    tags = text_tags.view(-1).tolist()
    runs = []
    run_start = 0
    n = b - a
    for i in range(1, n + 1):
        if i == n or tags[i] != tags[run_start]:
            runs.append((a + run_start, a + i, int(tags[run_start])))
            run_start = i
    layout._tag_runs = (text_tags, runs)
    return runs


def _layout_mod_row_ids(layout, segments, device):
    # row map is a function of coalesced (start, stop, row). The integers
    # stay put while unique_t sort order is unchanged.
    key = tuple(segments)
    cached = getattr(layout, "_mod_row_ids", None)
    if cached is not None and cached[0] == key and cached[1].device == device:
        return cached[1]
    ids = torch.empty(layout.seq_len, dtype=torch.int32)
    for a, b, row in segments:
        ids[a:b] = row
    ids = ids.to(device)
    layout._mod_row_ids = (key, ids)
    return ids


def pack_audio(latent):
    # [B, C=32, ch=2, T] -> [ch*T, 32] channel-major (ch0 t0..T-1, ch1 t0..T-1)
    b, c, ch, t = latent.shape
    return latent[0].permute(1, 2, 0).reshape(ch * t, c)


def unpack_audio(rows, ch=2):
    t = rows.shape[0] // ch
    return rows.reshape(ch, t, rows.shape[-1]).permute(2, 0, 1).unsqueeze(0)


def _axis_from_sqrt_area(dim, patch, sqrt_area):
    # linspace((1 - ratio) / 2, (1 + ratio) / 2, dim // patch, endpoint=False) * 32
    ratio = dim / sqrt_area
    n = dim // patch
    return (torch.arange(n, dtype=torch.float64) * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0


def _frame_grid(h, w):
    # area-normalized (h, w) coordinates of one latent frame's 2x2-patch rows
    area = math.sqrt(h * w)
    hh, ww = torch.meshgrid(_axis_from_sqrt_area(h, 2, area), _axis_from_sqrt_area(w, 2, area), indexing="ij")
    return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1), _axis_from_sqrt_area(w, 2, area)


def _video_t_spans(n):
    return [FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] for k in range(n)]


def _video_t_grid(n, origin):
    # origin + exclusive cumsum
    spans = torch.tensor(_video_t_spans(n), dtype=torch.float64)
    return float(origin) + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _audio_grid(cursor, t, w_low, w_high):
    # channel-major stereo rows: t advances per latent frame, w pinned to the grid extremes per stereo channel, h stays 0
    g = torch.zeros(t * 2, 3, dtype=torch.float64)
    g[:, 0] = (cursor + torch.arange(t, dtype=torch.float64)).repeat(2)
    g[:t, 2] = w_low
    g[t:, 2] = w_high
    return g


def _video_grid(vt, frame, cursor):
    g = torch.empty(vt, frame.shape[0], 3, dtype=torch.float64)
    g[:, :, 0] = _video_t_grid(vt, cursor)[:, None]
    g[:, :, 1:] = frame[None]
    return g.reshape(-1, 3)


class TimeEmbedder(nn.Module):
    def __init__(self, freq_dim, hidden, out, dtype=None, device=None, operations=None):
        super().__init__()
        self.freq_dim = freq_dim
        self.proj_in = operations.Linear(freq_dim, hidden, bias=True, dtype=dtype, device=device)
        self.proj_out = operations.Linear(hidden, out, bias=True, dtype=dtype, device=device)

    def forward(self, t):
        # t: [M] in [0, 1]; fp32 throughout, cos before sin
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t.to(torch.float32)[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.proj_out(nn.functional.silu(self.proj_in(emb)))


def rope_rotation_table(angles, dtype):
    """[S, rot_dim] pair angles -> [1, S, 1, rot_dim/2, 2, 2] rotation matrices."""
    half = angles.shape[-1] // 2
    ang = angles[:, :half]  # duplicated halves: [:, :half] == [:, half:]
    c, s = torch.cos(ang), torch.sin(ang)
    table = torch.stack([c, -s, s, c], dim=-1).reshape(1, angles.shape[0], 1, half, 2, 2)
    return table.to(dtype)


def rope_freqs_from_inv(position_ids, inv_freq, device):
    # [S, 3] float64 -> [S, 96] fp32. Function of (ids, inv_freq) only.
    pos = position_ids.to(torch.float32).to(device)
    inv = comfy.model_management.cast_to(inv_freq, device=device)
    per_axis = pos.unsqueeze(-1) * inv.view(1, 1, -1)      # [S, 3, 16]
    t_f, h_f, w_f = per_axis.unbind(dim=1)
    half = torch.cat((t_f, h_f, w_f), dim=-1)              # [S, 48]
    return torch.cat((half, half), dim=-1)                 # [S, 96]


def layout_device_tensor(layout, name, tensor, device):
    # PackedLayout masks/ids are a function of the layout signature.
    key = f"_dev_{name}"
    hit = getattr(layout, key, None)
    if hit is None or hit.device != device:
        hit = tensor.to(device)
        setattr(layout, key, hit)
    return hit


def layout_rope_table(layout, inv_freq, device, dtype):
    # rotation table is a function of (position_ids, inv_freq, dtype).
    key = (str(device), str(dtype), int(layout.seq_len),
           str(inv_freq.dtype), tuple(inv_freq.shape))
    cached = getattr(layout, "_rope_table", None)
    if cached is not None and cached[0] == key:
        table = cached[1]
        if table.device == device and table.dtype == dtype:
            return table
    table = rope_rotation_table(
        rope_freqs_from_inv(layout.position_ids, inv_freq, device), dtype
    )
    layout._rope_table = (key, table)
    return table


def project_mixed_rows(proj, live_rows, cond_rows, update, dtype, cache_owner=None, cache_attr=None):
    # Linear is row-wise: mix(proj(cond), proj(live)) == proj(mix(cond, live)).
    live = proj(live_rows).to(dtype)
    if cond_rows is None:
        return live
    cond = None
    if cache_owner is not None and cache_attr is not None:
        cond = getattr(cache_owner, cache_attr, None)
        if (
            cond is None
            or cond.device != live.device
            or cond.dtype != live.dtype
            or cond.shape[0] != cond_rows.shape[0]
            or cond.shape[-1] != live.shape[-1]
        ):
            cond = None
    if cond is None:
        cond = proj(cond_rows).to(dtype)
        if cache_owner is not None and cache_attr is not None:
            setattr(cache_owner, cache_attr, cond)
    out = live.new_empty(update.shape[0], live.shape[1])
    out[~update] = cond
    out[update] = live
    return out


class Attention(nn.Module):
    def __init__(self, hidden, heads, head_dim, eps, dtype=None, device=None, operations=None):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.qkv_proj = operations.Linear(hidden, inner * 3, bias=False, dtype=dtype, device=device)
        self.q_norm = operations.RMSNorm(head_dim, eps=eps, dtype=dtype, device=device)
        self.k_norm = operations.RMSNorm(head_dim, eps=eps, dtype=dtype, device=device)
        self.out_proj = operations.Linear(inner, hidden, bias=False, dtype=dtype, device=device)

    def _forward_projected(self, qkv, rope_freqs=None, transformer_options={}):
        s = qkv.shape[0]
        q, k, v = qkv.split(self.heads * self.head_dim, dim=-1)
        v = v.view(s, self.heads, self.head_dim)
        if rope_freqs is not None:
            # fused per-head RMSNorm + partial split-half rope, in place on the qkv buffer
            q = q.view(1, s, self.heads, self.head_dim)
            k = k.view(1, s, self.heads, self.head_dim)
            qw = comfy.model_management.cast_to(self.q_norm.weight, device=qkv.device)
            kw = comfy.model_management.cast_to(self.k_norm.weight, device=qkv.device)
            rot = rope_freqs.shape[-3] * 2
            q_fusion = transformer_options.get("minimax_h3_q_rms_rope_int8", None)
            override = transformer_options.get("optimized_attention_override", None)
            mode = transformer_options.get(
                "minimax_h3_linear_nvfp4_fusion",
                transformer_options.get("minimax_h3_swiglu_nvfp4_fusion", "stock"),
            )
            if (
                callable(getattr(q_fusion, "can_call", None))
                and callable(q_fusion)
                and getattr(q_fusion, "owner", None) is override
                and not comfy.model_management.in_training
                and not torch.is_grad_enabled()
                and not any(tensor.requires_grad for tensor in (q, k, v, rope_freqs, qw, kw))
                and not transformer_options.get("patches", {})
                and not transformer_options.get("patches_replace", {})
                and can_static_linear_for_output(
                    self.out_proj, s, qkv.device, qkv.dtype, mode
                )
                and q_fusion.can_call(
                    q, k, v.unsqueeze(0), rope_freqs, qw, kw,
                    self.q_norm.eps, self.heads, mask=None, scale=None,
                )
            ):
                out, marker = q_fusion(
                    q, k, v.unsqueeze(0), rope_freqs, qw, kw,
                    self.q_norm.eps, self.heads, mask=None, scale=None,
                )
                if not (
                    isinstance(marker, H3AttentionAmaxMarker)
                    and marker.owner is q_fusion
                ):
                    raise RuntimeError("H3 Q fusion returned an invalid provenance marker")
                if not (
                    isinstance(out, torch.Tensor)
                    and out.shape == (1, s, self.out_proj.in_features)
                    and out.dtype == torch.bfloat16
                    and out.device == qkv.device
                    and out.is_contiguous()
                    and not out.requires_grad
                ):
                    raise RuntimeError("H3 Q fusion returned an invalid static projection input")
                static_options = dict(transformer_options)
                static_options["minimax_h3_linear_nvfp4_fusion"] = "static"
                return _h3_linear(self.out_proj, out.squeeze(0), static_options)
            if (
                callable(getattr(q_fusion, "can_call", None))
                and callable(q_fusion)
                and getattr(q_fusion, "owner", None) is override
                and not comfy.model_management.in_training
                and not torch.is_grad_enabled()
                and not any(tensor.requires_grad for tensor in (q, k, v, rope_freqs, qw, kw))
                and not transformer_options.get("patches", {})
                and not transformer_options.get("patches_replace", {})
                and can_linear_from_amax_bits_for_output(
                    self.out_proj, s, qkv.device, qkv.dtype, mode
                )
                and q_fusion.can_call(
                    q, k, v.unsqueeze(0), rope_freqs, qw, kw,
                    self.q_norm.eps, self.heads, mask=None, scale=None,
                )
            ):
                out, marker = q_fusion(
                    q, k, v.unsqueeze(0), rope_freqs, qw, kw,
                    self.q_norm.eps, self.heads, mask=None, scale=None,
                )
                if not (
                    isinstance(marker, H3AttentionAmaxMarker)
                    and marker.owner is q_fusion
                ):
                    raise RuntimeError("H3 Q fusion returned an invalid provenance marker")
                fused = try_nvfp4_linear_from_amax_bits(
                    self.out_proj, out.squeeze(0), marker.bits, mode
                )
                if fused is None:
                    raise RuntimeError("H3 Q fusion lost its projection capability after launch")
                return fused
            if comfy.model_management.in_training:
                q, k = comfy.quant_ops.ck.rms_rope_split_half(
                    q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
            else:
                comfy.quant_ops.ck.rms_rope_split_half_(
                    q, k, rope_freqs, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot)
            q = q[0]
            k = k[0]
        else:
            q = self.q_norm(q.view(s, self.heads, self.head_dim))
            k = self.k_norm(k.view(s, self.heads, self.head_dim))
        # Keep the natural token-major layout. Attention backends can still
        # transpose internally when required, while NHD-native kernels avoid a
        # full output transpose/copy before the projection.
        q = q.reshape(1, s, self.heads * self.head_dim)
        k = k.reshape(1, s, self.heads * self.head_dim)
        v = v.reshape(1, s, self.heads * self.head_dim)
        producer = transformer_options.get("minimax_h3_attention_output_amax", None)
        override = transformer_options.get("optimized_attention_override", None)
        mode = transformer_options.get(
            "minimax_h3_linear_nvfp4_fusion",
            transformer_options.get("minimax_h3_swiglu_nvfp4_fusion", "stock"),
        )
        produced = None
        if (
            callable(getattr(producer, "try_call", None))
            and getattr(producer, "owner", None) is override
            and not comfy.model_management.in_training
            and not torch.is_grad_enabled()
            and not any(tensor.requires_grad for tensor in (q, k, v))
            and not transformer_options.get("patches", {})
            and not transformer_options.get("patches_replace", {})
            and can_linear_from_amax_bits(self.out_proj, q.squeeze(0), mode)
        ):
            produced = producer.try_call(
                q, k, v, self.heads, mask=None, skip_reshape=False
            )
        if produced is not None:
            out, marker = produced
            if not (
                isinstance(marker, H3AttentionAmaxMarker)
                and marker.owner is producer
            ):
                marker = None
        else:
            out = optimized_attention(q, k, v, self.heads, mask=None, skip_reshape=False, transformer_options=transformer_options)
            marker = None
        if marker is not None:
            fused = try_nvfp4_linear_from_amax_bits(
                self.out_proj, out.squeeze(0), marker.bits, mode
            )
            if fused is not None:
                return fused
        return _h3_linear(self.out_proj, out.squeeze(0), transformer_options)

    def forward(self, x, rope_freqs=None, transformer_options={}):
        return self._forward_projected(
            _h3_linear(self.qkv_proj, x, transformer_options),
            rope_freqs=rope_freqs,
            transformer_options=transformer_options,
        )


class MLP(nn.Module):
    def __init__(self, hidden, ffn, dtype=None, device=None, operations=None):
        super().__init__()
        self.fc1 = operations.Linear(hidden, ffn * 2, bias=False, dtype=dtype, device=device)
        self.fc2 = operations.Linear(ffn, hidden, bias=False, dtype=dtype, device=device)

    def _forward_raw(self, raw, transformer_options={}):
        fusion = transformer_options.get("minimax_h3_swiglu_nvfp4_fusion", "stock")
        fused = try_swiglu_nvfp4_linear(self.fc2, raw, fusion)
        if fused is not None:
            return fused
        return comfy.ops.linear_input_act(self.fc2, raw, "swiglu")

    def forward(self, x, transformer_options={}):
        return self._forward_raw(
            _h3_linear(self.fc1, x, transformer_options),
            transformer_options=transformer_options,
        )


class AdalnProj(nn.Module):
    def __init__(self, t_dim, hidden, expand, modalities, apply_silu=True,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.expand = expand
        self.modalities = modalities
        self.hidden = hidden
        self.apply_silu = apply_silu
        self.linear = operations.Linear(t_dim, expand * hidden * modalities, bias=True, dtype=dtype, device=device)

    def forward(self, t_emb):
        # [M, t_dim] -> expand tensors of [M*modalities, hidden]
        x = self.linear(nn.functional.silu(t_emb) if self.apply_silu else t_emb)
        x = x.view(x.shape[0] * self.modalities, self.expand * self.hidden)
        return x.chunk(self.expand, dim=-1)


def _mod_scale_shift(h, shift, scale, segments, row_ids=None):
    # segments: [(start, stop, mod_row)] covering h contiguously.
    if segmented_adaln_bf16_(h, scale, shift, row_ids):
        return h
    for a, b, row in segments:
        h[a:b].mul_(1.0 + scale[row].to(h.dtype)).add_(shift[row].to(h.dtype))
    return h


def _mod_gate(x, gate, other, segments, row_ids=None):
    # other is the fresh attn/mlp output: accumulate the gated residual into the stream in place, one fused kernel per segment
    if segmented_gate_bf16_(x, other, gate, row_ids):
        return x
    for a, b, row in segments:
        x[a:b].addcmul_(other[a:b], gate[row].to(x.dtype))
    return x


def _norm_mod(norm, x, shift, scale, segments, row_ids, aggressive=False):
    weight = getattr(norm, "weight", None)
    if aggressive and weight is not None and weight.device == x.device:
        h = segmented_rms_adaln_bf16(
            x, weight, scale, shift, row_ids, norm.eps
        )
        if h is not None:
            return h
    return _mod_scale_shift(norm(x), shift, scale, segments, row_ids)


def _h3_adaln_linear(linear, normed, shift, scale, row_ids, transformer_options):
    mode = transformer_options.get(
        "minimax_h3_adaln_nvfp4_fusion",
        transformer_options.get(
            "minimax_h3_linear_nvfp4_fusion",
            transformer_options.get("minimax_h3_swiglu_nvfp4_fusion", "stock"),
        ),
    )
    return try_adaln_nvfp4_linear(
        linear, normed, scale, shift, row_ids, _ADALN_NVFP4_TOKEN, mode, "exact"
    )


def _h3_rms_adaln_mode(transformer_options):
    return transformer_options.get(
        "minimax_h3_rms_adaln_nvfp4_fusion",
        transformer_options.get(
            "minimax_h3_adaln_nvfp4_fusion",
            transformer_options.get(
                "minimax_h3_linear_nvfp4_fusion",
                transformer_options.get("minimax_h3_swiglu_nvfp4_fusion", "stock"),
            ),
        ),
    )


def _h3_rms_adaln_linear(linear, norm, x, shift, scale, row_ids, transformer_options):
    return try_rms_adaln_nvfp4_linear(
        linear, norm, x, scale, shift, row_ids, _ADALN_NVFP4_TOKEN,
        _h3_rms_adaln_mode(transformer_options), "exact"
    )


def _coalesce_mod_segments(segments):
    """Merge contiguous spans that select the same modulation row."""
    merged = []
    for a, b, row in segments:
        if merged and merged[-1][1] == a and merged[-1][2] == row:
            merged[-1] = (merged[-1][0], b, row)
        else:
            merged.append((a, b, row))
    return merged


class RefinerBlock(nn.Module):
    def __init__(self, hidden, heads, head_dim, ffn, eps, qk_eps, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm1 = operations.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.norm2 = operations.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.attn = Attention(hidden, heads, head_dim, qk_eps, dtype=dtype, device=device, operations=operations)
        self.mlp = MLP(hidden, ffn, dtype=dtype, device=device, operations=operations)

    def forward(self, x, transformer_options={}):
        # attn/mlp outputs are fresh: accumulate residuals in place
        x = self.attn(self.norm1(x), transformer_options=transformer_options).add_(x)
        return self.mlp(self.norm2(x), transformer_options=transformer_options).add_(x)


class TokenRefiner(nn.Module):
    def __init__(self, num_layers, hidden, heads, head_dim, ffn, eps, qk_eps, final_eps,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            RefinerBlock(hidden, heads, head_dim, ffn, eps, qk_eps, dtype=dtype, device=device, operations=operations)
            for _ in range(num_layers)])
        self.final_norm = operations.RMSNorm(hidden, eps=final_eps, dtype=dtype, device=device)

    def forward(self, x, transformer_options={}):
        for block in self.blocks:
            x = block(x, transformer_options=transformer_options)
        return self.final_norm(x)



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


class DiTBlock(nn.Module):
    def __init__(self, hidden, heads, head_dim, ffn, t_dim, eps, qk_eps,
                 apply_silu=True, adaln_dtype=None, dtype=None, device=None, operations=None):
        super().__init__()
        self.norm1 = operations.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.norm2 = operations.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.attn = Attention(hidden, heads, head_dim, qk_eps, dtype=dtype, device=device, operations=operations)
        self.mlp = MLP(hidden, ffn, dtype=dtype, device=device, operations=operations)
        self.adaln_proj = AdalnProj(t_dim, hidden, 6, 3, apply_silu=apply_silu,
                                    dtype=adaln_dtype if adaln_dtype is not None else dtype,
                                    device=device, operations=operations)

    def forward(
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



class FinalLayer(nn.Module):
    def __init__(self, hidden, t_dim, video_dim, audio_dim, eps, apply_silu=True, adaln_dtype=None,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.norm = operations.RMSNorm(hidden, eps=eps, dtype=dtype, device=device)
        self.adaln_proj = AdalnProj(t_dim, hidden, 2, 1, apply_silu=apply_silu,
                                    dtype=adaln_dtype if adaln_dtype is not None else dtype,
                                    device=device, operations=operations)
        # output heads are the checkpoint's fp32 island; norm/adaln are stored at model dtype
        self.video_out = operations.Linear(hidden, video_dim, bias=True, dtype=torch.float32, device=device)
        self.audio_out = operations.Linear(hidden, audio_dim, bias=True, dtype=torch.float32, device=device)

    def forward(self, x, t_emb, video_seg, audio_seg):
        # video_seg / audio_seg: (start, stop, timestep_row) of the target streams
        shift, scale = self.adaln_proj(t_emb)
        va, vb, vrow = video_seg
        aa, ab, arow = audio_seg
        hv = (self.norm(x[va:vb]) * (1.0 + scale[vrow]) + shift[vrow]).to(torch.float32)
        ha = (self.norm(x[aa:ab]) * (1.0 + scale[arow]) + shift[arow]).to(torch.float32)
        return self.video_out(hv), self.audio_out(ha)


class PackedLayout:
    """Static packed-sequence structure for one shape/conditioning signature."""

    def __init__(self, text_len, latent_t, latent_h, latent_w, audio_t, keyframes=None, refs=None, frame_count=None):
        frame, w_grid = _frame_grid(latent_h, latent_w)
        frame_rows = frame.shape[0]

        segments = [("text", text_len)]  # (kind, n_rows)
        g = torch.zeros(text_len, 3, dtype=torch.float64)
        g[:, 0] = torch.arange(text_len, dtype=torch.float64)
        pos = [g]  # per segment: [n, 3] float64 (t, h, w)

        img_pos, img_update = [], []
        audio_pos, audio_update = [], []
        cursor = text_len
        row = text_len

        if keyframes:
            # fl2va: keyframe cond rows right after text, sharing the target spatial grid
            for kf in keyframes:
                pixel_index = kf["resolved_frame_index"]
                if pixel_index == 0:
                    cond_t = float(text_len)
                elif frame_count is not None and pixel_index == frame_count - 1:
                    cond_t = float(text_len) + sum(_video_t_spans(latent_t)) - FRAME_RESCALE
                else:
                    raise ValueError("only first/last keyframe anchors are supported")
                g = torch.empty(frame_rows, 3, dtype=torch.float64)
                g[:, 0] = cond_t
                g[:, 1:] = frame
                segments.append(("cond", frame_rows))
                pos.append(g)
                img_pos.append(torch.arange(row, row + frame_rows))
                img_update.append(torch.zeros(frame_rows, dtype=torch.bool))
                row += frame_rows

        target_audio_w = (float(w_grid[0]), float(w_grid[-1]))
        if refs:
            cursor = float(text_len)
            for blk in refs:
                kind = blk["kind"]
                if kind == "image":
                    r_frame, _ = _frame_grid(blk["latent_h"], blk["latent_w"])
                    n = r_frame.shape[0]
                    g = torch.empty(n, 3, dtype=torch.float64)
                    g[:, 0] = cursor
                    g[:, 1:] = r_frame
                    segments.append(("ref_img", n))
                    pos.append(g)
                    img_pos.append(torch.arange(row, row + n))
                    img_update.append(torch.zeros(n, dtype=torch.bool))
                    row += n
                    cursor += 1.0
                elif kind == "audio":
                    rt = blk["ref_audio_t"]
                    if rt > 0:
                        segments.append(("ref_audio", rt * 2))
                        pos.append(_audio_grid(cursor, rt, *target_audio_w))
                        audio_pos.append(torch.arange(row, row + rt * 2))
                        audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                        row += rt * 2
                    cursor += float(rt)
                elif kind in ("video", "video_audio"):
                    # the block's audio rows pack immediately before its video
                    # rows, both sharing the cursor origin
                    rt = blk["ref_audio_t"]
                    vt = blk["latent_t"]
                    r_frame, r_w_grid = _frame_grid(blk["latent_h"], blk["latent_w"])
                    if rt > 0:
                        segments.append(("ref_audio", rt * 2))
                        pos.append(_audio_grid(cursor, rt, float(r_w_grid[0]), float(r_w_grid[-1])))
                        audio_pos.append(torch.arange(row, row + rt * 2))
                        audio_update.append(torch.zeros(rt * 2, dtype=torch.bool))
                        row += rt * 2
                    n = vt * r_frame.shape[0]
                    segments.append(("ref_img", n))
                    pos.append(_video_grid(vt, r_frame, cursor))
                    img_pos.append(torch.arange(row, row + n))
                    img_update.append(torch.zeros(n, dtype=torch.bool))
                    row += n
                    cursor += max(float(rt), sum(_video_t_spans(vt)))

        # target audio then target video, always the last two segments
        segments.append(("audio", audio_t * 2))
        pos.append(_audio_grid(cursor, audio_t, *target_audio_w))
        audio_pos.append(torch.arange(row, row + audio_t * 2))
        audio_update.append(torch.ones(audio_t * 2, dtype=torch.bool))
        row += audio_t * 2

        n_video = latent_t * frame_rows
        segments.append(("video", n_video))
        pos.append(_video_grid(latent_t, frame, cursor))
        img_pos.append(torch.arange(row, row + n_video))
        img_update.append(torch.ones(n_video, dtype=torch.bool))
        row += n_video

        self.seq_len = row
        self.position_ids = torch.cat(pos)  # [S, 3] float64
        self.img_pos = torch.cat(img_pos)
        self.img_update = torch.cat(img_update)
        self.audio_pos = torch.cat(audio_pos)
        self.audio_update = torch.cat(audio_update)
        self.signature = (text_len, latent_t, latent_h, latent_w, audio_t)
        # contiguous segment table (start, stop, kind)
        # kinds: text / cond / ref_img / ref_audio / audio / video
        # the packed sequence is uniform per segment in (modality tag, timestep class),
        # except the text span (tag runs resolved at forward time from the presentation tags)
        seg_abs = []
        off = 0
        for kind, n in segments:
            seg_abs.append((off, off + n, kind))
            off += n
        self.segments = seg_abs


class MiniMaxH3Model(nn.Module):
    def __init__(self, hidden_size=5376, num_layers=50, token_refiner_num_layers=2,
                 num_attention_heads=56, attention_head_dim=128, ffn_hidden_size=14336,
                 latents_dim=24, audio_latents_dim=32, patch_size=(1, 2, 2), text_dim=5120,
                 timestep_input_dim=256, time_embed_hidden_size=5376, time_embed_dim=2688,
                 rope_inv_freq_len=16, norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5,
                 sigma_shift_video=12.0, sigma_shift_audio=3.0,
                 adaln_curve_grid=None,
                 image_model=None, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.patch_size = tuple(patch_size)
        self.latents_dim = latents_dim
        self.audio_latents_dim = audio_latents_dim
        self.sigma_shift_video = sigma_shift_video
        self.sigma_shift_audio = sigma_shift_audio
        self.use_adaln_curves = adaln_curve_grid is not None
        # curve-form checkpoints replace the time embedder and full-width adaln weights with a small shared basis of the time-embedding curve
        curve = {"apply_silu": not self.use_adaln_curves,
                 "adaln_dtype": torch.float32 if self.use_adaln_curves else dtype}
        video_patch_dim = latents_dim * self.patch_size[0] * self.patch_size[1] * self.patch_size[2]

        self.video_patch_proj = operations.Linear(video_patch_dim, hidden_size, bias=True, dtype=torch.float32, device=device)
        self.audio_patch_proj = operations.Linear(audio_latents_dim, hidden_size, bias=True, dtype=torch.float32, device=device)
        self.condition_proj = operations.Linear(text_dim, hidden_size, bias=True, dtype=dtype, device=device)
        if self.use_adaln_curves:
            self.register_buffer("adaln_t_table", torch.empty(adaln_curve_grid, time_embed_dim, dtype=torch.float32))
        else:
            self.time_embedder = TimeEmbedder(timestep_input_dim, time_embed_hidden_size, time_embed_dim,
                                              dtype=torch.float32, device=device, operations=operations)
        self.rope = nn.Module()
        self.rope.register_buffer("inv_freq", torch.empty(rope_inv_freq_len, dtype=torch.float32))
        self.token_refiner = TokenRefiner(token_refiner_num_layers, hidden_size, num_attention_heads,
                                          attention_head_dim, ffn_hidden_size, norm_eps, qk_norm_eps,
                                          final_norm_eps, dtype=dtype, device=device, operations=operations)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_attention_heads, attention_head_dim, ffn_hidden_size,
                     time_embed_dim, norm_eps, qk_norm_eps, **curve, dtype=dtype, device=device, operations=operations)
            for _ in range(num_layers)])
        self.final_layer = FinalLayer(hidden_size, time_embed_dim, video_patch_dim, audio_latents_dim,
                                      final_norm_eps, **curve, dtype=dtype, device=device, operations=operations)

    def preprocess_text_embeds(self, text_states):
        """[B, L, text_dim] Qwen states -> [B, L, hidden] refined text embeds."""
        if text_states.shape[-1] == self.hidden_size:
            return text_states
        return self.token_refiner(self.condition_proj(text_states[0])).unsqueeze(0)

    def rope_freqs(self, position_ids, device):
        return rope_freqs_from_inv(position_ids, self.rope.inv_freq, device)

    def _cond_video_rows(self, payload, device):
        """Concatenated visual condition rows (normalized latents -> patchified), with condition noise augmentation."""
        items = payload.get("cond_video_latents", [])
        if not items:
            return None
        aug = payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP)
        seed = int(payload.get("seed", 0))
        patch = self.patch_size

        layout = payload.get("layout")
        hit = getattr(layout, "_cond_video_rows", None) if layout is not None else None
        if hit is not None:
            if hit.device != device:
                hit = hit.to(device)
                layout._cond_video_rows = hit
                pack = getattr(layout, "_cond_video_unique", None)
                if pack is not None and pack[0] is not None and pack[0].device != device:
                    layout._cond_video_unique = (pack[0].to(device), pack[1], pack[2])
            return hit

        def build_one(z):
            # every condition intentionally restarts the same RNG stream
            r = patchify_video(z.to(torch.float32), patch)
            if aug < 1.0:
                gen = torch.Generator("cpu").manual_seed(seed)
                noise = torch.randn(r.shape, generator=gen, dtype=torch.float32)
                r = aug * r + (1.0 - aug) * noise.to(r.device)
            return r.to(device)

        unique, ids, lens = cond_unique_blocks(items, build_one)
        rows = expand_unique_blocks(unique, ids, lens)
        if layout is not None:
            layout._cond_video_rows = rows
            layout._cond_video_unique = (unique, ids, lens)
        return rows

    def _cond_audio_rows(self, payload, device):
        items = payload.get("cond_audio_latents", [])
        if not items:
            return None
        aug = payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP)
        seed = int(payload.get("seed", 0)) + 1

        layout = payload.get("layout")
        hit = getattr(layout, "_cond_audio_rows", None) if layout is not None else None
        if hit is not None:
            if hit.device != device:
                hit = hit.to(device)
                layout._cond_audio_rows = hit
                pack = getattr(layout, "_cond_audio_unique", None)
                if pack is not None and pack[0] is not None and pack[0].device != device:
                    layout._cond_audio_unique = (pack[0].to(device), pack[1], pack[2])
            return hit

        def build_one(z):
            r = pack_audio(z.to(torch.float32))
            if aug < 1.0:
                gen = torch.Generator("cpu").manual_seed(seed)
                noise = torch.randn(r.shape, generator=gen, dtype=torch.float32)
                r = aug * r + (1.0 - aug) * noise.to(r.device)
            return r.to(device)

        unique, ids, lens = cond_unique_blocks(items, build_one)
        rows = expand_unique_blocks(unique, ids, lens)
        if layout is not None:
            layout._cond_audio_rows = rows
            layout._cond_audio_unique = (unique, ids, lens)
        return rows

    def forward(self, x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
        # the sampler carries the audio as (sigma_v / sigma_a) * x_audio; undo it outside
        # the wrappers so they and the network see the stream's own latent and velocity
        scale = float((minimax_payload or {}).get("audio_scale", 1.0))
        audio_src = x[1]
        if scale != 1.0:
            shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", self.sigma_shift_video))
            shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", self.sigma_shift_audio))
            sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
            sigma_a = time_shift_sigma(sigma_v, shift_v, shift_a)
            carry = (sigma_a / sigma_v).to(audio_src.dtype)
            x = [x[0], audio_src * carry]

        out = comfy.patcher_extension.WrapperExecutor.new_class_executor(
            self._forward,
            self,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, timestep, context, transformer_options, minimax_payload=minimax_payload, **kwargs)

        if scale != 1.0:
            # d/d(sigma_v) of the carried variable
            out[1] = ((1.0 - scale) * (audio_src * carry)
                      + (1.0 + (scale - 1.0) * sigma_a).to(out[1].dtype) * out[1])
        return out

    def _forward(self, x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
        video_x, audio_x = x[0], x[1]
        orig_t, orig_h, orig_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
        video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, self.patch_size)
        if video_x.shape[0] != 1:
            raise ValueError("MiniMax H3 supports batch size 1")
        payload = minimax_payload or {}
        device = video_x.device
        dtype = context.dtype  # compute dtype

        latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
        audio_t = audio_x.shape[-1]
        text_len = context.shape[1]
        # extra_conds prebuilds the layout once per sampling run
        layout = payload.get("layout")
        if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t):
            layout = PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t,
                                  keyframes=payload.get("keyframes"),
                                  refs=payload.get("refs"),
                                  frame_count=payload.get("frame_count"))

        # model_base passes model_sampling.timestep(sigma) = sigma * 1000
        shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", self.sigma_shift_video))
        shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", self.sigma_shift_audio))
        sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
        t_v = float(1.0 - sigma_v)
        t_a = float(1.0 - time_shift_sigma(sigma_v, shift_v, shift_a))

        # distinct timesteps are known analytically: text/pad follow video, cond rows pin near 1
        vis_aug = float(payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP))
        aud_aug = float(payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP))
        has_vis_cond = any(k in ("cond", "ref_img") for _, _, k in layout.segments)
        has_aud_cond = any(k == "ref_audio" for _, _, k in layout.segments)
        seg_t = {"text": t_v, "video": t_v, "audio": t_a,
                 "cond": max(t_v, vis_aug), "ref_img": max(t_v, vis_aug),
                 "ref_audio": max(t_a, aud_aug)}
        unique_t = sorted({t_v, t_a} | ({seg_t["cond"]} if has_vis_cond else set())
                          | ({seg_t["ref_audio"]} if has_aud_cond else set()))
        t_row = {t: i for i, t in enumerate(unique_t)}
        seg_tag = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0, "ref_audio": 2}

        text_tags = payload.get("text_token_tags")
        tag_runs = _layout_text_tag_runs(layout, text_tags)
        mod_segments = []
        for a, b, kind in layout.segments:
            row_base = t_row[seg_t[kind]] * 3
            if kind == "text" and tag_runs is not None:
                # tag runs are a function of the presentation only; row_base
                # still tracks this step's unique_t order.
                for ra, rb, tag in tag_runs:
                    mod_segments.append((ra, rb, row_base + tag))
            else:
                mod_segments.append((a, b, row_base + seg_tag[kind]))
        # Adjacent reference blocks commonly share the same timestep and
        # modality. Coalescing preserves elementwise math while avoiding six
        # pointwise launches per redundant span in every transformer block.
        mod_segments = _coalesce_mod_segments(mod_segments)
        modulation_fusion = transformer_options.get("minimax_h3_modulation_fusion", "exact")
        mod_row_ids = None
        if modulation_fusion != "stock":
            mod_row_ids = _layout_mod_row_ids(layout, mod_segments, device)

        # embed
        img_update = layout_device_tensor(layout, "img_update", layout.img_update, device)
        audio_update = layout_device_tensor(layout, "audio_update", layout.audio_update, device)
        video_rows = patchify_video(video_x.to(torch.float32), self.patch_size)
        audio_rows = pack_audio(audio_x.to(torch.float32))
        cond_video_rows = self._cond_video_rows(payload, device)
        cond_audio_rows = self._cond_audio_rows(payload, device)

        all_video_rows = video_rows
        if cond_video_rows is not None:
            all_video_rows = torch.empty(img_update.shape[0], video_rows.shape[1], dtype=torch.float32, device=device)
            all_video_rows[~img_update] = cond_video_rows
            all_video_rows[img_update] = video_rows
        all_audio_rows = audio_rows
        if cond_audio_rows is not None:
            all_audio_rows = torch.empty(audio_update.shape[0], audio_rows.shape[1], dtype=torch.float32, device=device)
            all_audio_rows[~audio_update] = cond_audio_rows
            all_audio_rows[audio_update] = audio_rows

        video_embed = self.video_patch_proj(all_video_rows).to(dtype)
        audio_embed = self.audio_patch_proj(all_audio_rows).to(dtype)
        text_states = context[0]
        if text_states.shape[-1] != self.hidden_size:
            text_states = self.token_refiner(self.condition_proj(text_states),
                                             transformer_options=transformer_options)

        # segments are contiguous: assemble by slices, embed rows follow segment order
        h = torch.empty(layout.seq_len, self.hidden_size, dtype=dtype, device=device)
        voff = aoff = 0
        for a, b, kind in layout.segments:
            n = b - a
            if kind == "text":
                h[a:b] = text_states
            elif kind in ("cond", "ref_img", "video"):
                h[a:b] = video_embed[voff:voff + n]
                voff += n
            else:  # ref_audio / audio
                h[a:b] = audio_embed[aoff:aoff + n]
                aoff += n

        t_vals = torch.tensor(unique_t, dtype=torch.float32, device=device)
        if self.use_adaln_curves:
            # adaln projections consume interpolated coordinates of the time-embedding curve
            table = comfy.model_management.cast_to(self.adaln_t_table, device=device)
            pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)     # t in [0,1] -> fractional grid index, out-of-range t clamps to the curve ends
            i0 = pos.floor().long().clamp(max=table.shape[0] - 2)   # lower grid row, max-clamp keeps t=1.0 on the last interval instead of reading past the table
            t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))  # blend the two rows by the fractional part
        else:
            t_emb = self.time_embedder(t_vals).to(dtype)

        # rotation table is a function of the packed layout + inv_freq, not of x/t
        rope_freqs = layout_rope_table(layout, self.rope.inv_freq, device, dtype)

        # blocks. Cache (default off) and cross-block auto are not mixed.
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

        # target streams are single contiguous segments (audio then video, last two)
        video_seg = next((a, b, t_row[seg_t["video"]]) for a, b, k in layout.segments if k == "video")
        audio_seg = next((a, b, t_row[seg_t["audio"]]) for a, b, k in layout.segments if k == "audio")
        v, a = self.final_layer(h, t_emb, video_seg, audio_seg)

        video_out = unpatchify_video(v, latent_t, lat_h // 2, lat_w // 2, self.latents_dim, self.patch_size)
        video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
        audio_out = unpack_audio(a)

        return [-video_out.to(video_x.dtype), -audio_out.to(audio_x.dtype)]
