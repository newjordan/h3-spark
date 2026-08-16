"""MiniMax H3 text/vision conditioning: Qwen3-VL-32B (truncated to 50 layers).

The H3 presentation is NOT chat-templated: token ids are raw prompt/label text
(no special tokens) with explicit vision blocks spliced in:

  t2va:   <prompt>
  fl2va:  "<Picture 1>: " <vision block> ["<Picture 2>: " <vision block>] <prompt>
  ref2va: per condition in request order (1-based ordinals per type):
            image -> "<Picture i>: " <vision block>
            audio -> "<Audio j>: "              (audio never enters Qwen)
            video -> "<Video k>: " then per 2-frame temporal block
                     "<T.T seconds>" <vision block(2 frames)>
          then <prompt>

The conditioning is the unnormalized hidden state after LM layer 50 (the
converted checkpoint is truncated there, so this is simply the last-layer
output with no final norm). Vision-pad positions carry adaLN token tag 0
(video modality) in the DiT; text positions carry tag 1 — the tags are
returned alongside the embeddings as "minimax_token_tags".
"""

import math
import os

import torch
import torch.nn.functional as F
import comfy.sd1_clip
from .qwen3vl import Qwen3VL, Qwen3VLSDTokenizer
from .qwen_vl import process_qwen2vl_images

_VISION_BATCH_ENV = "COMFY_MINIMAX_H3_VISION_BATCH"
_VISION_BATCH_OFF = frozenset({"0", "off", "stock", "false", "no"})


def _vision_batch_enabled() -> bool:
    raw = os.environ.get(_VISION_BATCH_ENV, "auto")
    return str(raw).strip().lower() not in _VISION_BATCH_OFF


# Qwen 50-layer LM is invoked as fp32 from SDClipModel. Isolated S=2800 MLP
# is 245 ms/layer fp32 vs 28 ms fp16 (~12 s vs 1.4 s / 50). Not an identity.
# Default off. Opt-in: COMFY_MINIMAX_H3_QWEN_LM_DTYPE=bf16|fp16
_LM_DTYPE_ENV = "COMFY_MINIMAX_H3_QWEN_LM_DTYPE"


def qwen_lm_compute_dtype():
    raw = os.environ.get(_LM_DTYPE_ENV, "off")
    key = str(raw).strip().lower()
    if key in ("bf16", "bfloat16"):
        return torch.bfloat16
    if key in ("fp16", "float16", "half"):
        return torch.float16
    return None


def maybe_cast_lm_embeds(embeds, dtype):
    if dtype is None or embeds is None or embeds.dtype == dtype:
        return embeds
    return embeds.to(dtype)


# Persist W.t().contiguous() then x @ Wt (NN) is torch.equal to F.linear
# for gate/up/q/o at the measured row counts below (FixedRef 6-ref is 2800).
# Isolated extra-S (3a): gate 1400/1600/2048 and q 512/1600/2048 and
# o 512 are also byte-equal and faster. Other S for those KN are ULP
# (0 px > 0.01) — stay stock. down/k/v stay on F.linear.
# Rollback: COMFY_MINIMAX_H3_QWEN_NN_LINEAR=off
_NN_LINEAR_ENV = "COMFY_MINIMAX_H3_QWEN_NN_LINEAR"
_NN_LINEAR_OFF = frozenset({"0", "off", "stock", "false", "no"})
_NN_EXACT_KN = frozenset({
    (5120, 25600),
    (5120, 8192),
    (8192, 5120),
})
_NN_EXACT_ROWS = {
    (5120, 25600): frozenset({1400, 1600, 2048, 2800}),
    (5120, 8192): frozenset({512, 1600, 2048, 2800}),
    (8192, 5120): frozenset({512, 2800}),
}


def _nn_linear_enabled() -> bool:
    raw = os.environ.get(_NN_LINEAR_ENV, "auto")
    return str(raw).strip().lower() not in _NN_LINEAR_OFF


def qwen_nn_linear(x, weight, bias, owner=None):
    """F.linear via persist-NN. Caller must already shape-gate."""
    k, n = int(weight.shape[1]), int(weight.shape[0])
    wt = None
    if owner is not None:
        wt = getattr(owner, "_h3_Wt", None)
        if (
            wt is None
            or wt.shape[0] != k
            or wt.shape[1] != n
            or wt.dtype != weight.dtype
            or wt.device != weight.device
        ):
            wt = weight.transpose(0, 1).contiguous()
            owner._h3_Wt = wt
    if wt is None:
        wt = weight.transpose(0, 1).contiguous()
    y = x.matmul(wt)
    if bias is not None:
        y = y + bias
    return y


def _wrap_linear_nn(linear):
    if getattr(linear, "_h3_nn_wrapped", False):
        return
    orig = getattr(linear, "_forward", None)
    if orig is None:
        return

    def _forward(input, weight, bias):
        if not _nn_linear_enabled():
            return orig(input, weight, bias)
        if not isinstance(input, torch.Tensor) or not isinstance(weight, torch.Tensor):
            return orig(input, weight, bias)
        if input.shape[-1] != weight.shape[1]:
            return orig(input, weight, bias)
        rows = input.numel() // input.shape[-1]
        kn = (int(weight.shape[1]), int(weight.shape[0]))
        allowed = _NN_EXACT_ROWS.get(kn)
        if allowed is None or rows not in allowed:
            return orig(input, weight, bias)
        return qwen_nn_linear(input, weight, bias, owner=linear)

    linear._forward = _forward
    linear._h3_nn_wrapped = True


def install_qwen_nn_linear(root):
    if root is None or not _nn_linear_enabled():
        return
    for mod in root.modules():
        if hasattr(mod, "in_features") and hasattr(mod, "out_features"):
            kn = (int(mod.in_features), int(mod.out_features))
            if kn in _NN_EXACT_KN:
                _wrap_linear_nn(mod)

VISION_START = 151652
VISION_END = 151653
QWEN_IMAGE_MEAN = [0.5, 0.5, 0.5]
QWEN_IMAGE_STD = [0.5, 0.5, 0.5]


def process_video_block(frames, patch_size=16, temporal_patch_size=2, merge_size=2,
                        min_pixels=3136, max_pixels=12845056):
    """[2, H, W, C] frame pair -> (flatten_patches, grid_thw) with grid_t=1.

    Same resize/normalize policy as process_qwen2vl_images, but the two frames
    fill the temporal patch instead of repeating a single frame.
    """
    t, height, width, _ = frames.shape
    imgs = frames.permute(0, 3, 1, 2)
    factor = patch_size * merge_size
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    if height != h_bar or width != w_bar:
        imgs = F.interpolate(imgs, size=(h_bar, w_bar), mode="bilinear", align_corners=False)
    mean = imgs.new_tensor(QWEN_IMAGE_MEAN).view(1, 3, 1, 1)
    std = imgs.new_tensor(QWEN_IMAGE_STD).view(1, 3, 1, 1)
    imgs = (imgs - mean) / std

    grid_h = h_bar // patch_size
    grid_w = w_bar // patch_size
    patches = imgs.reshape(1, temporal_patch_size, 3, grid_h // merge_size, merge_size,
                           patch_size, grid_w // merge_size, merge_size, patch_size)
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten = patches.reshape(grid_h * grid_w, 3 * temporal_patch_size * patch_size * patch_size)
    grid_thw = torch.stack([torch.tensor([1, grid_h, grid_w], device=frames.device, dtype=torch.long)])
    return flatten, grid_thw


def vision_cache_lookup(cache, data, video_block):
    # visual(pixels) does not take <Picture i>. Identical pixels share one tower.
    if not isinstance(data, torch.Tensor):
        return None
    video_block = bool(video_block)
    for prev, prev_v, prev_out in cache:
        if (
            prev_v == video_block
            and prev.shape == data.shape
            and prev.dtype == data.dtype
            and prev.device == data.device
            and torch.equal(prev, data)
        ):
            return prev_out
    return None


def vision_cache_store(cache, data, video_block, out):
    if isinstance(data, torch.Tensor):
        cache.append((data.detach().clone(), bool(video_block), out))


def _install_qwen35_grid_caches():
    # rot_pos_emb / bilinear pos-embed interpolate are functions of the grid
    # (and frozen embeddings), not of pixels. Qwen2 caches existed; Qwen3-VL
    # uses Qwen35VisionModel and was recomputing them every tower call.
    from comfy.text_encoders.qwen35 import Qwen35VisionModel, Qwen35VisionRotaryEmbedding

    if getattr(Qwen35VisionModel, "_h3_grid_cache_patched", False):
        return

    orig_rot = Qwen35VisionModel.rot_pos_emb
    orig_pos = Qwen35VisionModel.fast_pos_embed_interpolate
    orig_freq = Qwen35VisionRotaryEmbedding.forward

    def rot_pos_emb(self, grid_thw):
        key = tuple(int(v) for v in grid_thw.reshape(-1).tolist()) + (str(grid_thw.device),)
        cache = getattr(self, "_h3_rot_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        out = orig_rot(self, grid_thw)
        self._h3_rot_cache = (key, out)
        return out

    def fast_pos_embed_interpolate(self, grid_thw):
        key = tuple(int(v) for v in grid_thw.reshape(-1).tolist()) + (
            str(self.pos_embed.weight.device), str(self.pos_embed.weight.dtype)
        )
        cache = getattr(self, "_h3_pos_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        out = orig_pos(self, grid_thw)
        self._h3_pos_cache = (key, out)
        return out

    def rotary_forward(self, seqlen):
        key = (int(seqlen), str(self.inv_freq.device))
        cache = getattr(self, "_h3_freq_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        out = orig_freq(self, seqlen)
        self._h3_freq_cache = (key, out)
        return out

    Qwen35VisionModel.rot_pos_emb = rot_pos_emb
    Qwen35VisionModel.fast_pos_embed_interpolate = fast_pos_embed_interpolate
    Qwen35VisionRotaryEmbedding.forward = rotary_forward
    Qwen35VisionModel._h3_grid_cache_patched = True


_install_qwen35_grid_caches()


def vision_equal_length_pack(cu_seqlens):
    """(n, s) when cu_seqlens is n>=2 equal-length image spans, else None.

    Packed same-grid images are independent sequences. One SDPA on batch n
    is the same function as the serial per-span loop (measured n_diff 0).
    """
    if not isinstance(cu_seqlens, torch.Tensor) or cu_seqlens.numel() < 3:
        return None
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    if lengths.numel() < 2:
        return None
    first = lengths[0]
    if int(first.item()) <= 0:
        return None
    if not bool(torch.all(lengths == first)):
        return None
    return int(lengths.numel()), int(first.item())


def vision_attn_packed(q, k, v, n, s, heads, optimized_attention):
    # q/k/v are [n*s, H, D] token-major, same layout the serial loop splits.
    q = q.view(n, s, heads, -1).transpose(1, 2)
    k = k.view(n, s, heads, -1).transpose(1, 2)
    v = v.view(n, s, heads, -1).transpose(1, 2)
    out = optimized_attention(q, k, v, heads, skip_reshape=True)
    return out.reshape(n * s, -1)


def _install_qwen35_vision_attn_pack():
    # Same-grid vision batch already cats tokens for GEMM. Attention still
    # looped per image via cu_seqlens. Equal S -> one batched SDPA.
    from comfy.text_encoders.llama import apply_rope
    from comfy.text_encoders.qwen35 import Qwen35VisionAttention

    if getattr(Qwen35VisionAttention, "_h3_attn_pack_patched", False):
        return

    orig = Qwen35VisionAttention.forward

    def forward(self, x, cu_seqlens, position_embeddings, optimized_attention=None):
        packed = vision_equal_length_pack(cu_seqlens)
        if packed is None or optimized_attention is None:
            return orig(self, x, cu_seqlens, position_embeddings, optimized_attention)
        n, s = packed
        seq_length = x.shape[0]
        if n * s != seq_length:
            return orig(self, x, cu_seqlens, position_embeddings, optimized_attention)
        query_states, key_states, value_states = (
            self.qkv(x).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        query_states, key_states = apply_rope(query_states, key_states, position_embeddings)
        attn_output = vision_attn_packed(
            query_states, key_states, value_states, n, s, self.num_heads, optimized_attention,
        )
        return self.proj(attn_output)

    Qwen35VisionAttention.forward = forward
    Qwen35VisionAttention._h3_attn_pack_patched = True


_install_qwen35_vision_attn_pack()


def _prepare_vision_flat(embed):
    data = embed.get("data")
    video = bool(embed.get("minimax_video_block", False))
    if video:
        flatten, grid = process_video_block(data)
    else:
        flatten, grid = process_qwen2vl_images(
            data, patch_size=16,
            image_mean=QWEN_IMAGE_MEAN, image_std=QWEN_IMAGE_STD,
        )
    return flatten, grid, video


def prefetch_vision_batch(transformer, token_batches, device):
    """One visual() per unique flatten shape / grid. Same cu_seqlens split
    the tower already uses for multi-image. Hits the per-pixel cache so
    preprocess_embed does not rerun the tower.
    """
    if (
        not _vision_batch_enabled()
        or transformer is None
        or getattr(transformer, "visual", None) is None
    ):
        return
    cache = getattr(transformer, "_h3_vision_cache", None)
    if cache is None:
        transformer._h3_vision_cache = cache = []
    pending = []
    for batch in token_batches:
        for item in batch:
            tok = item[0]
            if not (isinstance(tok, dict) and tok.get("type") == "image"):
                continue
            data = tok.get("data")
            video = bool(tok.get("minimax_video_block", False))
            if vision_cache_lookup(cache, data, video) is None:
                pending.append(tok)
    if len(pending) < 2:
        return
    prepared = []
    groups = {}
    for emb in pending:
        flatten, grid, video = _prepare_vision_flat(emb)
        flatten = flatten.to(device, dtype=torch.float32)
        grid = grid.to(device)
        n_tok = int(grid[0, 1].item() * grid[0, 2].item() // 4)
        key = (
            video,
            tuple(int(x) for x in flatten.shape),
            tuple(int(x) for x in grid.reshape(-1).tolist()),
        )
        groups.setdefault(key, []).append(len(prepared))
        prepared.append((emb, flatten, grid, n_tok, video))
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        flats = torch.cat([prepared[i][1] for i in idxs], dim=0)
        grids = torch.cat([prepared[i][2] for i in idxs], dim=0)
        merged, deepstack = transformer.visual(flats, grids)
        off = 0
        for i in idxs:
            n = prepared[i][3]
            piece = merged[off:off + n]
            ds = None
            if deepstack is not None:
                ds = [layer[off:off + n] for layer in deepstack]
            out = (piece, {"grid": prepared[i][2], "deepstack": ds})
            vision_cache_store(cache, prepared[i][0].get("data"), prepared[i][4], out)
            off += n


def token_tags_from_embeds_info(seq_len, embeds_info):
    # whole vision block VIDEO(0), including the flanking <|vision_start|>/<|vision_end|> tokens
    # embeds_info spans cover only the expanded embeddings, so widen by one on each side.
    tags = torch.ones(seq_len, dtype=torch.long)
    for e in embeds_info:
        if e.get("type") == "image":
            tags[max(0, e["index"] - 1):e["index"] + e["size"] + 1] = 0
    return tags


class MiniMaxQwen3VL(Qwen3VL):
    model_type = "qwen3vl_32b"

    def preprocess_embed(self, embed, device):
        cache = getattr(self, "_h3_vision_cache", None)
        if cache is None:
            self._h3_vision_cache = cache = []
        data = embed.get("data")
        video = bool(embed.get("minimax_video_block", False))
        if embed.get("type") == "image":
            hit = vision_cache_lookup(cache, data, video)
            if hit is not None:
                return hit
        if embed.get("type") == "image" and video:
            flatten, grid = process_video_block(data)
            merged, deepstack = self.visual(flatten.to(device, dtype=torch.float32), grid)
            out = merged, {"grid": grid, "deepstack": deepstack}
        else:
            out = super().preprocess_embed(embed, device)
        if embed.get("type") == "image" and out[0] is not None:
            vision_cache_store(cache, data, video, out)
        return out

    def forward(self, input_ids, attention_mask=None, embeds=None, num_tokens=None,
                intermediate_output=None, final_layer_norm_intermediate=True,
                dtype=None, embeds_info=[], **kwargs):
        seq = embeds.shape[1] if embeds is not None else input_ids.shape[1]
        self.last_token_tags = token_tags_from_embeds_info(seq, embeds_info)
        embeds = maybe_cast_lm_embeds(embeds, dtype)
        return super().forward(input_ids, attention_mask=attention_mask, embeds=embeds,
                               num_tokens=num_tokens, intermediate_output=intermediate_output,
                               final_layer_norm_intermediate=final_layer_norm_intermediate,
                               dtype=dtype, embeds_info=embeds_info, **kwargs)


class MiniMaxH3ClipModel(comfy.sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, model_options={}):
        super().__init__(device=device, layer="last", layer_idx=None, textmodel_json_config={},
                         dtype=dtype, special_tokens={"pad": 151643}, layer_norm_hidden_state=False,
                         model_class=MiniMaxQwen3VL, enable_attention_masks=False,
                         return_attention_masks=False, model_options=model_options)

    def forward(self, tokens):
        lm_dtype = qwen_lm_compute_dtype()
        if lm_dtype is None:
            return super().forward(tokens)
        if self.execution_device is None:
            device = self.transformer.get_input_embeddings().weight.device
        else:
            device = self.execution_device
        embeds, attention_mask, num_tokens, embeds_info = self.process_tokens(tokens, device)
        attention_mask_model = attention_mask if self.enable_attention_masks else None
        if isinstance(self.layer, list):
            intermediate_output = self.layer
        elif self.layer == "all":
            intermediate_output = "all"
        else:
            intermediate_output = self.layer_idx
        outputs = self.transformer(
            None, attention_mask_model, embeds=embeds, num_tokens=num_tokens,
            intermediate_output=intermediate_output,
            final_layer_norm_intermediate=self.layer_norm_hidden_state,
            dtype=lm_dtype, embeds_info=embeds_info,
        )
        z = outputs[0].float() if self.layer == "last" else outputs[1].float()
        if self.zero_out_masked:
            z *= attention_mask.unsqueeze(-1).float()
        pooled_output = None
        if len(outputs) >= 3:
            if not self.return_projected_pooled and len(outputs) >= 4 and outputs[3] is not None:
                pooled_output = outputs[3].float()
            elif outputs[2] is not None:
                pooled_output = outputs[2].float()
        extra = {}
        if self.return_attention_masks:
            extra["attention_mask"] = attention_mask
        if extra:
            return z, pooled_output, extra
        return z, pooled_output

    def encode_token_weights(self, token_weight_pairs):
        tr = self.transformer
        if tr is not None:
            tr._h3_vision_cache = []
            install_qwen_nn_linear(tr)
            try:
                device = tr.get_input_embeddings().weight.device
            except Exception:
                device = next(tr.parameters()).device
            prefetch_vision_batch(tr, token_weight_pairs, device)
        try:
            out = super().encode_token_weights(token_weight_pairs)
        finally:
            if tr is not None:
                tr._h3_vision_cache = []
        tags = getattr(self.transformer, "last_token_tags", None)
        if tags is not None:
            extra = out[2] if len(out) > 2 and isinstance(out[2], dict) else {}
            extra["minimax_token_tags"] = tags
            out = (out[0], out[1], extra)
        return out


class MiniMaxH3TEModel(comfy.sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, name="qwen3vl_32b",
                         clip_model=MiniMaxH3ClipModel, model_options=model_options)


class MiniMaxH3Tokenizer(comfy.sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer = lambda *a, **kw: Qwen3VLSDTokenizer(*a, **kw, embedding_size=5120, embedding_key="qwen3vl_32b")
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, name="qwen3vl_32b", tokenizer=tokenizer)

    def _text_ids(self, text):
        tok = self.qwen3vl_32b.tokenizer
        return tok(text, add_special_tokens=False)["input_ids"]

    @staticmethod
    def _vision_entry(data, video_block=False):
        emb = {"type": "image", "data": data, "original_type": "image"}
        if video_block:
            emb["minimax_video_block"] = True
        return emb

    def tokenize_with_weights(self, text, return_word_ids=False, images=[],
                              minimax_ref_items=None, **kwargs):
        entries = []

        def add_text(s):
            entries.extend((tid, 1.0) for tid in self._text_ids(s))

        def add_vision(data, video_block=False):
            entries.append((VISION_START, 1.0))
            entries.append((self._vision_entry(data, video_block), 1.0))
            entries.append((VISION_END, 1.0))

        if minimax_ref_items:
            counters = {"image": 0, "audio": 0, "video": 0}
            for item in minimax_ref_items:
                kind = item["type"]
                counters[kind] += 1
                if kind == "image":
                    add_text("<Picture %d>: " % counters["image"])
                    add_vision(item["data"])
                elif kind == "audio":
                    add_text("<Audio %d>: " % counters["audio"])
                elif kind == "video":
                    frames = item["data"]  # [T, H, W, C], sampled at 2 fps
                    timestamps = item.get("timestamps")
                    if timestamps is None:
                        timestamps = [i / 2.0 for i in range(frames.shape[0])]
                    if frames.shape[0] % 2 == 1:  # repeat-pad to temporal patch of 2
                        frames = torch.cat([frames, frames[-1:]], dim=0)
                        timestamps = list(timestamps) + [timestamps[-1]]
                    add_text("<Video %d>: " % counters["video"])
                    for i in range(0, frames.shape[0], 2):
                        block_ts = (timestamps[i] + timestamps[i + 1]) / 2.0
                        add_text("<%.1f seconds>" % block_ts)
                        add_vision(frames[i:i + 2], video_block=True)
        else:
            for i, img in enumerate(images):
                add_text("<Picture %d>: " % (i + 1))
                add_vision(img)

        add_text(text)
        if len(entries) == 0:
            entries.append((151643, 1.0))
        if return_word_ids:
            entries = [t + (0,) for t in entries]
        return {"qwen3vl_32b": [entries]}

    def untokenize(self, token_weight_pair):
        return self.qwen3vl_32b.untokenize(token_weight_pair)


def te(dtype_llama=None, llama_quantization_metadata=None, **kwargs):
    class MiniMaxH3TEModel_(MiniMaxH3TEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if dtype_llama is not None:
                dtype = dtype_llama
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return MiniMaxH3TEModel_
