# MiniMax H3 video VAE: 3D causal CNN encoder + ViT3D decoder.

import contextlib
import logging
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.ops
import comfy.quant_ops
import comfy.rmsnorm
from comfy.ldm.modules.attention import optimized_attention

# Same 256 px tiling, but same-shaped tiles share one _decode_pixels or
# _encode_moments call. Isolated decode 864x480 / 81 frames: 17.87 s
# serial -> 10.42 s batched, max_abs 0.0079, 0 pixels > 0.01.
# Rollback: COMFY_MINIMAX_H3_VAE_BATCH_TILES=off
_BATCH_TILES_ENV = "COMFY_MINIMAX_H3_VAE_BATCH_TILES"
_BATCH_TILES_MAX_ENV = "COMFY_MINIMAX_H3_VAE_BATCH_TILES_MAX"
_BATCH_TILES_DISABLED = frozenset({"0", "off", "stock", "false", "no"})
# Kitchen RMSNorm + split-half RoPE on VAE Q/K (affine-free, partial rot).
# Eager/CPU is byte-exact vs stock. CUDA is ULP (max_abs 0.03125 bf16) —
# faster (~1.8x isolated) but not an identity. Default off until a pixel gate.
# Opt-in: COMFY_MINIMAX_H3_VAE_RMS_ROPE=auto
_RMS_ROPE_ENV = "COMFY_MINIMAX_H3_VAE_RMS_ROPE"
# Native silu(gate)*up on [gate|up] last-dim (eager two-rounding). Rollback off.
_SILU_MUL_ENV = "COMFY_MINIMAX_H3_VAE_SILU_MUL"
_SILU_MUL_LIB_ENV = "COMFY_MINIMAX_H3_VAE_SILU_MUL_LIBRARY"
# Downcast decode pixels to fp16 on the GPU before Comfy's
# .to(cpu, fp16, copy=True). Combined fp32→cpu fp16 is ~1.9 s for FixedRef
# 618 MiB on this GB10; GPU fp16 then HtoD is ~5 ms and torch.equal.
# Rollback: COMFY_MINIMAX_H3_VAE_HOST_FP16=off
_HOST_FP16_ENV = "COMFY_MINIMAX_H3_VAE_HOST_FP16"
# ImageNet denorm already yields [0, 1]. The *2-1 here exists only so
# Comfy VAE.process_output can undo it with +1/2. That pair is identity
# on the clamped range. Skip both (this flag + process_output=identity).
# Rollback: COMFY_MINIMAX_H3_VAE_UNIT_INTERVAL=off
_UNIT_INTERVAL_ENV = "COMFY_MINIMAX_H3_VAE_UNIT_INTERVAL"
# Persistent host output buffer. Comfy VAE.decode does
# .to(cpu, fp16, copy=True) which first-touches a NEW 308 MiB
# allocation every call (~2.3 s on this GB10). Reusing a
# process-lifetime CPU tensor makes later HtoD ~5 ms and
# torch.equal. Flip-flop of 2 buffers keeps the previous
# IMAGE view valid. Rollback: COMFY_MINIMAX_H3_VAE_HOST_BUF=off
_HOST_BUF_ENV = "COMFY_MINIMAX_H3_VAE_HOST_BUF"
# Emit [B,T,H,W,C] on the GPU before host stage. Comfy movedim(1,-1)
# of NCHW is a non-contiguous view; SaveVideo's per-frame *255.byte()
# then copies. Contiguous NHWC is the same pixels (isolated h264
# 2.99 → 2.19 s on FixedRef-shaped fp16). Rollback:
# COMFY_MINIMAX_H3_VAE_NHWC=off
_NHWC_ENV = "COMFY_MINIMAX_H3_VAE_NHWC"
# Persist-Wt addmm for decoder w1/qkv/to_out/proj_out. Rollback off.


def _batch_tiles_enabled() -> bool:
    raw = os.environ.get(_BATCH_TILES_ENV, "auto")
    return str(raw).strip().lower() not in _BATCH_TILES_DISABLED


def _rms_rope_enabled() -> bool:
    raw = os.environ.get(_RMS_ROPE_ENV, "off")
    return str(raw).strip().lower() not in _BATCH_TILES_DISABLED


def _silu_mul_enabled() -> bool:
    raw = os.environ.get(_SILU_MUL_ENV, "auto")
    return str(raw).strip().lower() not in _BATCH_TILES_DISABLED


def _host_fp16_enabled() -> bool:
    raw = os.environ.get(_HOST_FP16_ENV, "auto")
    return str(raw).strip().lower() not in _BATCH_TILES_DISABLED


def _unit_interval_enabled() -> bool:
    raw = os.environ.get(_UNIT_INTERVAL_ENV, "auto")
    return str(raw).strip().lower() not in _BATCH_TILES_DISABLED


def _host_buf_enabled() -> bool:
    raw = os.environ.get(_HOST_BUF_ENV, "auto")
    return str(raw).strip().lower() not in _BATCH_TILES_DISABLED


def _nhwc_enabled() -> bool:
    raw = os.environ.get(_NHWC_ENV, "auto")
    return str(raw).strip().lower() not in _BATCH_TILES_DISABLED


# Persist W.t() then addmm(bias, x, Wt) == F.linear at FixedRef tile rows
# for w1/qkv/to_out/proj_out (not w2). Isolated ~0.68 s / decode.
# Rollback: COMFY_MINIMAX_H3_VAE_ADDMM=off
_ADDMM_ENV = "COMFY_MINIMAX_H3_VAE_ADDMM"
_ADDMM_KN = frozenset({
    (2048, 16384),
    (2048, 6144),
    (2048, 2048),
    (2048, 3072),
})


def _addmm_enabled() -> bool:
    raw = os.environ.get(_ADDMM_ENV, "auto")
    return str(raw).strip().lower() not in _BATCH_TILES_DISABLED


def vae_addmm_linear(x, weight, bias, owner=None):
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
    y = torch.addmm(bias, x.reshape(-1, k), wt)
    return y.view(*x.shape[:-1], n)


def _wrap_linear_addmm(linear):
    if getattr(linear, "_h3_addmm_wrapped", False):
        return
    orig = linear.forward

    def forward(x, *args, **kwargs):
        if (
            not _addmm_enabled()
            or not isinstance(x, torch.Tensor)
            or not x.is_cuda
            or x.requires_grad
            or getattr(linear, "bias", None) is None
            or getattr(linear, "weight", None) is None
        ):
            return orig(x, *args, **kwargs)
        weight = linear.weight
        if x.shape[-1] != weight.shape[1]:
            return orig(x, *args, **kwargs)
        kn = (int(weight.shape[1]), int(weight.shape[0]))
        if kn not in _ADDMM_KN:
            return orig(x, *args, **kwargs)
        try:
            return vae_addmm_linear(x, weight, linear.bias, owner=linear)
        except Exception:
            return orig(x, *args, **kwargs)

    linear.forward = forward
    linear._h3_addmm_wrapped = True


def install_vae_addmm(root):
    if root is None or not _addmm_enabled():
        return
    for mod in root.modules():
        if hasattr(mod, "in_features") and hasattr(mod, "out_features"):
            kn = (int(mod.in_features), int(mod.out_features))
            if kn in _ADDMM_KN:
                _wrap_linear_addmm(mod)


def maybe_nhwc_bt_hwc(dec):
    # [B,C,T,H,W] -> [B,T,H,W,C] contiguous. Same pixels as movedim(1,-1).
    if (
        not _nhwc_enabled()
        or not isinstance(dec, torch.Tensor)
        or dec.ndim != 5
        or dec.requires_grad
    ):
        return dec, False
    return dec.permute(0, 2, 3, 4, 1).contiguous(), True


def stage_output_host(owner, dec, device, dtype):
    # Same bits as Tensor.to(device, dtype, copy=True). The dest
    # storage is reused so the OS does not first-touch 308 MiB again.
    if (
        not _host_buf_enabled()
        or not isinstance(dec, torch.Tensor)
        or dec.requires_grad
    ):
        return dec.to(device=device, dtype=dtype, copy=True)
    if dec.dtype != dtype:
        dec = dec.to(dtype=dtype)
    key = (tuple(dec.shape), str(dtype), str(device))
    slots = getattr(owner, "_host_out_slots", None)
    if slots is None or getattr(owner, "_host_out_key", None) != key:
        slots = []
        for _ in range(2):
            buf = torch.empty(dec.shape, dtype=dtype, device=device)
            if device.type == "cpu" or str(device) == "cpu":
                buf.fill_(0)
            slots.append(buf)
        owner._host_out_slots = slots
        owner._host_out_key = key
        owner._host_out_i = 0
    i = int(getattr(owner, "_host_out_i", 0)) % 2
    buf = slots[i]
    buf.copy_(dec)
    owner._host_out_i = i + 1
    return buf


def unit_interval_from_signed(pixels):
    # Comfy VAE.process_output: (x+1)/2 clamp. Inverse of *2-1 on [0, 1].
    return pixels.add(1.0).div_(2.0).clamp_(0.0, 1.0)


def _maybe_host_fp16(dec):
    # Same bits as Comfy VAE.decode's .to(dtype=fp16) then HtoD, but the
    # downcast stays on the producing device.
    if (
        not _host_fp16_enabled()
        or not isinstance(dec, torch.Tensor)
        or not dec.is_cuda
        or dec.dtype == torch.float16
        or dec.requires_grad
    ):
        return dec
    return dec.to(dtype=torch.float16)


_SILU_MUL_LIB = None
_SILU_MUL_LIB_TRIED = False


def _silu_mul_lib():
    global _SILU_MUL_LIB, _SILU_MUL_LIB_TRIED
    if _SILU_MUL_LIB_TRIED:
        return _SILU_MUL_LIB
    _SILU_MUL_LIB_TRIED = True
    path = os.environ.get(_SILU_MUL_LIB_ENV, "").strip()
    if not path:
        here = os.path.dirname(os.path.abspath(__file__))
        for cand in (
            os.path.join(here, "silu_mul_native_v1.so"),
        ):
            if os.path.isfile(cand):
                path = cand
                break
    if not path or not os.path.isfile(path):
        return None
    try:
        import ctypes
        lib = ctypes.CDLL(path)
        for name in ("h3_silu_mul_bf16", "h3_silu_mul_fp16", "h3_silu_mul_fp32"):
            fn = getattr(lib, name)
            fn.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_longlong, ctypes.c_longlong, ctypes.c_void_p,
            ]
            fn.restype = ctypes.c_int
        _SILU_MUL_LIB = lib
    except OSError:
        _SILU_MUL_LIB = None
    return _SILU_MUL_LIB


def _stock_silu_mul(raw):
    gate, up = raw.chunk(2, dim=-1)
    return F.silu(gate).mul_(up)


def _apply_silu_mul(raw):
    # [..., 2*inner] -> [..., inner] = silu(gate)*up. Fail closed to eager.
    if (
        not _silu_mul_enabled()
        or not isinstance(raw, torch.Tensor)
        or raw.ndim < 1
        or raw.shape[-1] % 2 != 0
        or not raw.is_cuda
        or raw.requires_grad
    ):
        return _stock_silu_mul(raw)
    lib = _silu_mul_lib()
    if lib is None:
        return _stock_silu_mul(raw)
    if raw.dtype == torch.bfloat16:
        fn = lib.h3_silu_mul_bf16
    elif raw.dtype == torch.float16:
        fn = lib.h3_silu_mul_fp16
    elif raw.dtype == torch.float32:
        fn = lib.h3_silu_mul_fp32
    else:
        return _stock_silu_mul(raw)
    contig = raw.contiguous()
    inner = contig.shape[-1] // 2
    rows = contig.numel() // contig.shape[-1]
    out = contig.new_empty(*contig.shape[:-1], inner)
    stream = torch.cuda.current_stream(contig.device).cuda_stream
    if fn(contig.data_ptr(), out.data_ptr(), rows, inner, stream) != 0:
        return _stock_silu_mul(raw)
    return out


def _batch_tiles_max() -> int:
    raw = os.environ.get(_BATCH_TILES_MAX_ENV, "64")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 64


def _is_cuda_oom(err: BaseException) -> bool:
    if isinstance(err, torch.cuda.OutOfMemoryError):
        return True
    msg = str(err).lower()
    return "out of memory" in msg or "cuda oom" in msg


@contextlib.contextmanager
def _disable_batch_tiles():
    prev = os.environ.get(_BATCH_TILES_ENV)
    os.environ[_BATCH_TILES_ENV] = "off"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(_BATCH_TILES_ENV, None)
        else:
            os.environ[_BATCH_TILES_ENV] = prev

ops = comfy.ops.disable_weight_init

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
]

LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.68317747116088865, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.96531379222869875, 1.05698859691619875,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049925, 0.7197399735450745, 0.69362932443618775,
    2.961095094680786, 2.7694199085235595, 3.0496184825897215, 2.1088054180145265,
    3.276226282119751, 3.1627357006073, 2.28168129920959475, 2.6127843856811525,
]


# 3D causal CNN encoder

class CausalConv3d(ops.Conv3d):
    # Reflect spatial padding, causal (zeros, front-only) temporal padding.
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.causal_padding = (padding,) * 3 if isinstance(padding, int) else tuple(padding)

    def forward(self, x):
        if sum(self.causal_padding) == 0:
            return super().forward(x)

        x = F.pad(x, (self.causal_padding[2], self.causal_padding[2], self.causal_padding[1], self.causal_padding[1], 0, 0),  mode="reflect")
        if x.shape[2] == 1:
            # single frame: the causal front padding is all zeros truncate the temporal taps instead of convolving zero frames
            return super().forward(x, autopad="causal_zero")
        x = F.pad(x, (0, 0, 0, 0, self.causal_padding[0] * 2, 0), mode="constant")
        return super().forward(x)


class TemporalIsolatedGroupNorm(ops.GroupNorm):
    # GroupNorm with statistics computed per frame (time merged into batch).
    # [B*T, C, 1, H, W] == [B*T, C, H, W]: stats are over the last spatial axes.
    def forward(self, x):
        if x.dim() == 5:
            b, c, t, h, w = x.shape
            y = super().forward(
                x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            )
            return y.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        return super().forward(x)


def group_norm_3d(num_channels):
    return TemporalIsolatedGroupNorm(num_groups=32, num_channels=num_channels, eps=1e-6, affine=True)


class Downsample3D(nn.Module):
    def __init__(self, in_channels, out_channels, time_stride=1, space_stride=2):
        super().__init__()
        self.space_stride = space_stride
        self.conv = CausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=(1, 0, 0),
            stride=(time_stride, space_stride, space_stride),
        )

    def forward(self, x):
        if self.space_stride == 2:
            x = F.pad(x, (0, 1, 0, 1, 0, 0), mode="reflect")
        return self.conv(x)


class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = group_norm_3d(in_channels)
        self.norm2 = group_norm_3d(out_channels)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = CausalConv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x), inplace=True))
        h = self.conv2(F.silu(self.norm2(h), inplace=True))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return h.add_(x)


class EncoderFCN3D(nn.Module):
    def __init__(self, ch, ch_mult, space_down, time_down, num_res_blocks, in_channels, z_channels, double_z=True):
        super().__init__()
        self.num_levels = len(ch_mult)
        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks] * self.num_levels
        self.num_res_blocks = num_res_blocks

        block_mid = [ch * ch_mult[i] for i in range(self.num_levels)]
        block_in = [block_mid[0]] + block_mid[:-1]
        block_out = block_mid

        self.conv_in = CausalConv3d(in_channels, block_in[0], kernel_size=3, padding=1)

        self.down = nn.ModuleList()
        for i_level in range(self.num_levels):
            down = nn.Module()
            down.block = nn.ModuleList()
            for i in range(self.num_res_blocks[i_level]):
                down.block.append(
                    ResnetBlock3D(
                        in_channels=block_in[i_level] if i == 0 else block_mid[i_level],
                        out_channels=block_mid[i_level],
                    )
                )
            if space_down[i_level] * time_down[i_level] > 1:
                down.downsample = Downsample3D(
                    block_mid[i_level],
                    block_out[i_level],
                    time_stride=time_down[i_level],
                    space_stride=space_down[i_level],
                )
            self.down.append(down)

        self.norm_out = group_norm_3d(block_out[-1])
        self.conv_out = CausalConv3d(
            block_out[-1],
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        h = self.conv_in(x)
        for i_level in range(self.num_levels):
            for i_block in range(self.num_res_blocks[i_level]):
                h = self.down[i_level].block[i_block](h)
            if hasattr(self.down[i_level], "downsample"):
                h = self.down[i_level].downsample(h)
        h = F.silu(self.norm_out(h))
        return self.conv_out(h)


# ViT3D decoder

def create_token_ids(patch_dims, device, dtype):
    coords_list = []
    for dim_size in patch_dims:
        coords = torch.arange(0.5, dim_size, dtype=dtype, device=device)
        coords = coords / dim_size
        coords = 2.0 * coords - 1.0
        coords_list.append(coords)
    coords = torch.stack(torch.meshgrid(*coords_list, indexing="ij"), dim=-1)
    return coords.flatten(0, len(patch_dims) - 1).unsqueeze(0)


class RotaryEmbeddingND(nn.Module):
    def __init__(self, dim, rotary_base=100.0, n_dim=3):
        super().__init__()
        self.n_dim = n_dim
        self.angle_scale = 2.0 * math.pi
        inv_freq = 1 / rotary_base ** torch.arange(0, 1, 2 * n_dim / dim, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, img_ids):
        # [B, S, n_dim] -> [B, S, 1, pairs, 2, 2] rotation table for the kitchen split-half rope
        angles = (
            self.angle_scale
            * img_ids[:, :, :, None].float()
            * self.inv_freq.to(img_ids.device)[None, None, None, :]
        )
        angles = angles.flatten(2, 3)
        c, s = torch.cos(angles), torch.sin(angles)
        table = torch.stack([c, -s, s, c], dim=-1).reshape(*angles.shape[:2], 1, angles.shape[-1], 2, 2)
        return table.to(img_ids.dtype)


class FeedForward(nn.Module):
    # Gated SiLU FFN.
    def __init__(self, dim, mult=4, bias=True, operations=ops):
        super().__init__()
        inner_dim = dim * mult
        self.w1 = operations.Linear(dim, inner_dim * 2, bias=bias)
        self.w2 = operations.Linear(inner_dim, dim, bias=bias)

    def forward(self, x):
        return self.w2(_apply_silu_mul(self.w1(x)))


def _stock_qk_rms_rope(query, key, rotary_pos_emb, eps, q_weight, k_weight):
    query = comfy.rmsnorm.rms_norm(query, q_weight, eps)
    key = comfy.rmsnorm.rms_norm(key, k_weight, eps)
    if rotary_pos_emb is None:
        return query, key
    rot = rotary_pos_emb.shape[-3] * 2
    q_rot = query[..., :rot]
    k_rot = key[..., :rot]
    # In-place kitchen rope on the prefix is byte-equal to the out-of-place
    # assign and drops the extra write. Grad paths keep the out-of-place form.
    if not query.requires_grad and not key.requires_grad:
        try:
            comfy.quant_ops.ck.apply_rope_split_half_(q_rot, k_rot, rotary_pos_emb)
            return query, key
        except Exception:
            pass
    query[..., :rot], key[..., :rot] = comfy.quant_ops.ck.apply_rope_split_half(
        q_rot, k_rot, rotary_pos_emb)
    return query, key


def _apply_qk_rms_rope(query, key, rotary_pos_emb, eps, q_weight, k_weight, unit_scale=None):
    # Affine-free Q/K only: kitchen requires a scale tensor, so ones(head_dim).
    can_fuse = (
        _rms_rope_enabled()
        and rotary_pos_emb is not None
        and q_weight is None
        and k_weight is None
        and not query.requires_grad
        and not key.requires_grad
    )
    if can_fuse:
        try:
            rot = int(rotary_pos_emb.shape[-3]) * 2
            if unit_scale is None or unit_scale.shape[-1] != query.shape[-1]:
                unit_scale = query.new_ones(query.shape[-1])
            return comfy.quant_ops.ck.rms_rope_split_half(
                query, key, rotary_pos_emb, unit_scale, unit_scale,
                epsilon=eps, rot_dim=rot,
            )
        except Exception:
            pass
    return _stock_qk_rms_rope(query, key, rotary_pos_emb, eps, q_weight, k_weight)


class Attention(nn.Module):
    def __init__(self, heads, dim_head, bias=True, eps=1e-5, operations=ops):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm_q = ops.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.norm_k = ops.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.to_qkv = operations.Linear(inner_dim, inner_dim * 3, bias=bias)
        self.to_out = operations.Linear(inner_dim, inner_dim, bias=bias)

    def _unit_qk_scale(self, ref):
        buf = getattr(self, "_unit_qk_scale_buf", None)
        if (
            buf is None
            or buf.device != ref.device
            or buf.dtype != ref.dtype
            or buf.numel() != self.dim_head
        ):
            buf = torch.ones(self.dim_head, device=ref.device, dtype=ref.dtype)
            self._unit_qk_scale_buf = buf
        return buf

    def forward(self, x, rotary_pos_emb=None):
        batch_size, seq_len, _ = x.shape

        qkv = self.to_qkv(x)
        qkv = qkv.view(batch_size, seq_len, -1, 3 * self.dim_head)
        query, key, value = torch.chunk(qkv, 3, dim=-1)

        query, key = _apply_qk_rms_rope(
            query, key, rotary_pos_emb, self.norm_q.eps,
            self.norm_q.weight, self.norm_k.weight,
            unit_scale=self._unit_qk_scale(query),
        )

        # Finite Q/K/V → finite SDPA. nan_to_num_ was an identity write
        # (0 NaNs on FixedRef T=37 / 124 f, 13.9e9 attn outputs).
        out = optimized_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
            self.heads, skip_reshape=True,
        )
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, heads, dim_head, bias=True, eps=1e-5, operations=ops):
        super().__init__()
        dim = heads * dim_head
        self.norm1 = ops.RMSNorm(dim, elementwise_affine=True, eps=eps)
        self.attn = Attention(heads=heads, dim_head=dim_head, bias=bias, eps=eps, operations=operations)
        self.scale1 = nn.Parameter(torch.empty(dim))
        self.norm2 = ops.RMSNorm(dim, elementwise_affine=True, eps=eps)
        self.ff = FeedForward(dim=dim, bias=bias, operations=operations)
        self.scale2 = nn.Parameter(torch.empty(dim))

    def forward(self, x, rotary_pos_emb=None):
        x = x.addcmul_(self.attn(comfy.rmsnorm.rms_norm(x, self.norm1.weight, self.norm1.eps), rotary_pos_emb), comfy.ops.cast_to_input(self.scale1, x))
        return x.addcmul_(self.ff(comfy.rmsnorm.rms_norm(x, self.norm2.weight, self.norm2.eps)), comfy.ops.cast_to_input(self.scale2, x))


class ViT3DDecoder(nn.Module):
    def __init__(self, patch_size=16, patch_size_t=4, in_channels=24, out_channels=3, num_layers=36, heads=32, dim_head=64, rope_theta=100.0,
                 rope_dim_ratio=0.75, bias=True, eps=1e-5, num_register_tokens=4, operations=ops):
        super().__init__()
        dim = heads * dim_head
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels
        self.num_register_tokens = num_register_tokens

        self.pos_embed = RotaryEmbeddingND(int(dim_head * rope_dim_ratio), rope_theta, n_dim=3)
        self.x_embedder = ops.Linear(in_channels, dim)
        self.register_tokens = nn.Parameter(torch.empty(1, num_register_tokens, dim))
        # unused at inference; kept so the checkpoint loads without leftover keys
        self.register_buffer("mask_token", torch.empty(1, 1, dim))

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(heads=heads, dim_head=dim_head, bias=bias, eps=eps, operations=operations)
             for _ in range(num_layers)]
        )

        self.norm_out = ops.LayerNorm(dim, elementwise_affine=True, eps=eps)
        self.proj_out = ops.Linear(dim, out_channels * patch_size_t * patch_size * patch_size)

    def _cached_rotary(self, batch, latent_t, latent_h, latent_w, device, dtype):
        # ids/rope depend only on the latent grid, not on pixel values or batch.
        # B copies are expand() of the B=1 table.
        key = (latent_t, latent_h, latent_w, str(device), str(dtype))
        cache = getattr(self, "_rope_cache", None)
        if cache is None or cache[0] != key:
            num_suffix = 1 + self.num_register_tokens
            img_ids = create_token_ids((latent_t, latent_h, latent_w), device, dtype)
            suffix_ids = torch.zeros((1, num_suffix, 3), device=device, dtype=img_ids.dtype)
            rotary = self.pos_embed(torch.cat([img_ids, suffix_ids], dim=1))
            self._rope_cache = (key, rotary)
        else:
            rotary = cache[1]
        if batch != rotary.shape[0]:
            rotary = rotary.expand(batch, *rotary.shape[1:])
        return rotary

    def _cached_suffix(self, batch, ref):
        key = (str(ref.device), str(ref.dtype), ref.shape[-1])
        cache = getattr(self, "_suffix_cache", None)
        if cache is None or cache[0] != key:
            suf = torch.cat(
                [
                    comfy.ops.cast_to_input(self.register_tokens, ref),
                    ref.new_zeros(1, 1, ref.shape[-1]),
                ],
                dim=1,
            )
            self._suffix_cache = (key, suf)
        else:
            suf = cache[1]
        if batch != suf.shape[0]:
            suf = suf.expand(batch, -1, -1)
        return suf

    def forward(self, x):
        B, C, latent_T, latent_H, latent_W = x.shape

        h = self.x_embedder(x.flatten(2).transpose(1, 2))  # [B, T*H*W, C]

        num_patches = h.shape[1]
        h = torch.cat([h, self._cached_suffix(B, h)], dim=1)
        rotary_pos_emb = self._cached_rotary(
            B, latent_T, latent_H, latent_W, x.device, x.dtype
        )

        for block in self.transformer_blocks:
            h = block(h, rotary_pos_emb)

        output = self.proj_out(self.norm_out(h))

        output = output[:, :num_patches, :]

        output = output.view(
            B, latent_T, latent_H, latent_W,
            self.out_channels, self.patch_size_t, self.patch_size, self.patch_size,
        )
        output = output.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        output = output.reshape(
            B, self.out_channels,
            latent_T * self.patch_size_t,
            latent_H * self.patch_size,
            latent_W * self.patch_size,
        )
        return output


# Full VAE

class MiniMaxH3VideoVAE(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_ch=3,
        ch=128,
        embed_dim=24,
        z_channels=24,
        ch_mult=(1, 2, 2, 4, 4, 8),
        num_res_blocks=2,
        space_down=(2, 2, 2, 2, 1, 1),
        time_down=(1, 2, 2, 1, 1, 1),
        clip_length=17,
        token_drop=3,
        tile_size=256,
        tile_overlap_min=64,
        tiling=True,
        operations=ops,
    ):
        super().__init__()
        self.vae_ratio = int(math.prod(space_down))
        self.vae_ratio_t = int(math.prod(time_down))

        # temporal chunking parameters
        self.clip_length = clip_length
        self.token_drop = token_drop
        self.frame_pre_padding = (-clip_length) % self.vae_ratio_t
        self.tokens_chunk_size = math.ceil(clip_length / self.vae_ratio_t)
        self.token_overlap = (-token_drop) % self.tokens_chunk_size
        self.frame_overlap = max(self.token_overlap * self.vae_ratio_t - self.frame_pre_padding, 0)

        # spatial tiling parameters
        self.tiling = tiling
        self.tile_size = tile_size
        self.tile_overlap_min = tile_overlap_min

        self.encoder = EncoderFCN3D(
            ch=ch,
            ch_mult=list(ch_mult),
            space_down=list(space_down),
            time_down=list(time_down),
            num_res_blocks=num_res_blocks,
            in_channels=in_channels,
            z_channels=z_channels,
            double_z=True,
        )
        self.quant_conv = ops.Conv3d(z_channels * 2, 2 * embed_dim, 1)
        self.post_quant_conv = ops.Conv3d(embed_dim, z_channels, 1)
        self.decoder = ViT3DDecoder(
            patch_size=self.vae_ratio,
            patch_size_t=self.vae_ratio_t,
            in_channels=z_channels,
            out_channels=out_ch,
            operations=operations,
        )
        install_vae_addmm(self.decoder)

        self.register_buffer("latents_mean", torch.tensor(LATENTS_MEAN))
        self.register_buffer("latents_std", torch.tensor(LATENTS_STD))
        self.register_buffer("pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False)
        self.h3_channels_last = False

    # single-shot forward

    def _encode_moments(self, x):
        return self.quant_conv(self.encoder(x))

    def _decode_pixels(self, z):
        return self.decoder(self.post_quant_conv(z))

    def _adaptive_encode(self, x):
        if self.tiling:
            return self.tiled_encode(x)
        return self._encode_moments(x)

    def _adaptive_decode(self, z):
        if self.tiling:
            return self.tiled_decode(z)
        return self._decode_pixels(z)

    # spatial tiling

    def split_tiles(self, input_len):
        tile_size = self.tile_size
        if tile_size >= input_len:
            return [0], [input_len], []

        N = math.ceil(input_len / tile_size)
        while True:
            overlaps = [self.tile_overlap_min] * (N - 1)
            remaining = tile_size * N - sum(overlaps) - input_len
            if remaining < 0:
                N += 1
            else:
                break

        remaining_units = remaining // self.vae_ratio
        for i in range(remaining_units):
            overlaps[i % (N - 1)] += self.vae_ratio

        tile_start_idx = [0]
        for i in range(N - 1):
            tile_start_idx.append(tile_start_idx[-1] + tile_size - overlaps[i])

        return tile_start_idx, [tile_size] * N, overlaps

    def _blend_weights(self, blend_extent, device, dtype):
        # FixedRef has several overlap extents (144 / 112 / 96). A single-slot
        # cache thrashes and re-launches arange+div on every other blend.
        key = (int(blend_extent), str(device), str(dtype))
        cache = getattr(self, "_blend_w_map", None)
        if cache is None:
            cache = {}
            self._blend_w_map = cache
        hit = cache.get(key)
        if hit is not None:
            return hit
        positions = torch.arange(blend_extent, device=device, dtype=dtype)
        weight_a = 1 - positions / blend_extent
        weight_b = positions / blend_extent
        cache[key] = (weight_a, weight_b)
        return weight_a, weight_b

    def blend(self, a, b, blend_extent, dim):
        blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)
        weight_a, weight_b = self._blend_weights(blend_extent, b.device, b.dtype)

        shape = [1] * a.ndim
        shape[dim] = blend_extent
        weight_a = weight_a.view(shape)
        weight_b = weight_b.view(shape)

        slice_a = [slice(None)] * a.ndim
        slice_a[dim] = slice(-blend_extent, None)
        slice_b = [slice(None)] * b.ndim
        slice_b[dim] = slice(0, blend_extent)

        blended = a[tuple(slice_a)] * weight_a + b[tuple(slice_b)] * weight_b

        if blend_extent < b.shape[dim]:
            slice_b_rest = [slice(None)] * b.ndim
            slice_b_rest[dim] = slice(blend_extent, None)
            return torch.cat([blended, b[tuple(slice_b_rest)]], dim=dim)
        return blended

    def _should_batch_tiles(self, z):
        # Batch dim is per-item independent (GN is temporal-isolated).
        return (
            bool(self.tiling)
            and _batch_tiles_enabled()
            and isinstance(z, torch.Tensor)
            and z.ndim == 5
            and z.shape[0] >= 1
            and not z.requires_grad
        )

    def _spatial_tile_grid(self, height, width):
        y_idx, y_len, y_overlap = self.split_tiles(height)
        x_idx, x_len, x_overlap = self.split_tiles(width)
        return y_idx, y_len, y_overlap, x_idx, x_len, x_overlap

    def _spatial_latent_slices(self, z, y_idx, y_len, x_idx, x_len):
        slices = []
        for i_pos, i_len in zip(y_idx, y_len):
            zi, zl = i_pos // self.vae_ratio, i_len // self.vae_ratio
            for j_pos, j_len in zip(x_idx, x_len):
                zj, zw = j_pos // self.vae_ratio, j_len // self.vae_ratio
                slices.append(z[..., zi:zi + zl, zj:zj + zw])
        return slices

    def _decode_pixel_slices(self, slices):
        return self._run_batched_slices(slices, self._decode_pixels)

    def _encode_pixel_slices(self, slices):
        return self._run_batched_slices(slices, self._encode_moments)

    def _run_batched_slices(self, slices, fn):
        if not slices:
            return []
        if len(slices) == 1 or not _batch_tiles_enabled():
            return [fn(sl) for sl in slices]
        outs = [None] * len(slices)
        groups = {}
        for i, sl in enumerate(slices):
            groups.setdefault(tuple(sl.shape), []).append(i)
        limit = _batch_tiles_max()
        for idxs in groups.values():
            for start in range(0, len(idxs), limit):
                chunk = idxs[start:start + limit]
                if len(chunk) == 1:
                    outs[chunk[0]] = fn(slices[chunk[0]])
                    continue
                batch = torch.cat([slices[i] for i in chunk], dim=0)
                done = fn(batch)
                for offset, i in enumerate(chunk):
                    outs[i] = done[offset:offset + 1]
        return outs

    def _spatial_pixel_slices(self, x, y_idx, y_len, x_idx, x_len):
        slices = []
        for i_pos, i_len in zip(y_idx, y_len):
            for j_pos, j_len in zip(x_idx, x_len):
                slices.append(x[..., i_pos:i_pos + i_len, j_pos:j_pos + j_len])
        return slices

    def _assemble_spatial_tiles(
        self, tiles, height, width, y_idx, y_len, y_overlap, x_idx, x_len, x_overlap
    ):
        canvas = None
        row_tails = []
        out_y = 0
        n_x = len(x_idx)
        tile = None
        for iy in range(len(y_len)):
            new_tails = []
            left_tail = None
            out_x = 0
            for jx in range(len(x_len)):
                tile = tiles[iy * n_x + jx]
                if iy < len(y_idx) - 1:
                    new_tails.append(tile[..., -y_overlap[iy]:, :].clone())
                next_left_tail = (
                    tile[..., :, -x_overlap[jx]:].clone()
                    if jx < len(x_idx) - 1 else None
                )
                if iy > 0:
                    tile = self.blend(row_tails[jx], tile, y_overlap[iy - 1], dim=-2)
                if jx > 0:
                    tile = self.blend(left_tail, tile, x_overlap[jx - 1], dim=-1)
                left_tail = next_left_tail
                if iy < len(y_idx) - 1:
                    tile = tile[..., :-y_overlap[iy], :]
                if jx < len(x_idx) - 1:
                    tile = tile[..., :, :-x_overlap[jx]]
                if canvas is None:
                    canvas = torch.empty(
                        *tile.shape[:-2], height, width,
                        dtype=tile.dtype, device=tile.device,
                    )
                canvas[..., out_y:out_y + tile.shape[-2],
                       out_x:out_x + tile.shape[-1]].copy_(tile)
                out_x += tile.shape[-1]
            row_tails = new_tails
            out_y += tile.shape[-2]
        return canvas

    def tiled_encode(self, x):
        height, width = x.shape[-2], x.shape[-1]
        y_idx, y_len, y_overlap = self.split_tiles(height)
        x_idx, x_len, x_overlap = self.split_tiles(width)
        slices = self._spatial_pixel_slices(x, y_idx, y_len, x_idx, x_len)
        if self._should_batch_tiles(x) and len(slices) > 1:
            try:
                encoded = self._encode_pixel_slices(slices)
            except RuntimeError as err:
                if not _is_cuda_oom(err):
                    raise
                logging.warning(
                    "H3 VAE batched tile encode OOM; falling back to serial tiles"
                )
                if x.is_cuda:
                    torch.cuda.empty_cache()
                with _disable_batch_tiles():
                    encoded = [self._encode_moments(sl) for sl in slices]
        else:
            encoded = [self._encode_moments(sl) for sl in slices]
        n_x = len(x_idx)
        rows = [
            encoded[i * n_x:(i + 1) * n_x]
            for i in range(len(y_idx))
        ]

        latent_y_overlap = [o // self.vae_ratio for o in y_overlap]
        latent_x_overlap = [o // self.vae_ratio for o in x_overlap]

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend(rows[i - 1][j], tile, latent_y_overlap[i - 1], dim=-2)
                if j > 0:
                    tile = self.blend(row[j - 1], tile, latent_x_overlap[j - 1], dim=-1)
                if i < len(rows) - 1:
                    tile = tile[..., :-latent_y_overlap[i], :]
                if j < len(row) - 1:
                    tile = tile[..., :, :-latent_x_overlap[j]]
                result_row.append(tile)
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def tiled_decode(self, z):
        height, width = z.shape[-2] * self.vae_ratio, z.shape[-1] * self.vae_ratio
        grid = self._spatial_tile_grid(height, width)
        y_idx, y_len, y_overlap, x_idx, x_len, x_overlap = grid
        slices = self._spatial_latent_slices(z, y_idx, y_len, x_idx, x_len)
        if self._should_batch_tiles(z) and len(slices) > 1:
            try:
                tiles = self._decode_pixel_slices(slices)
            except RuntimeError as err:
                if not _is_cuda_oom(err):
                    raise
                logging.warning(
                    "H3 VAE batched tile decode OOM; falling back to serial tiles"
                )
                if z.is_cuda:
                    torch.cuda.empty_cache()
                with _disable_batch_tiles():
                    tiles = [self._decode_pixels(sl) for sl in slices]
        else:
            tiles = [self._decode_pixels(sl) for sl in slices]
        return self._assemble_spatial_tiles(
            tiles, height, width, y_idx, y_len, y_overlap, x_idx, x_len, x_overlap
        )

    # temporal chunking

    def encode_temporal(self, x):
        if x.shape[2] % self.clip_length != 0:
            pad_size = (-x.shape[2]) % self.clip_length
            pad_frames = x[:, :, -1:].repeat(1, 1, pad_size, 1, 1)
            x = torch.cat([x, pad_frames], dim=2)

        num_chunks = x.shape[2] // self.clip_length
        b, c, _, h, w = x.shape
        z = None
        if (
            _batch_tiles_enabled()
            and num_chunks > 1
            and not x.requires_grad
        ):
            try:
                clips = (
                    x.reshape(b, c, num_chunks, self.clip_length, h, w)
                    .permute(2, 0, 1, 3, 4, 5)
                    .reshape(num_chunks * b, c, self.clip_length, h, w)
                )
                z_all = self._adaptive_encode(clips)
                cz, t_lat, hl, wl = z_all.shape[1:]
                z = (
                    z_all.reshape(num_chunks, b, cz, t_lat, hl, wl)
                    .permute(1, 2, 0, 3, 4, 5)
                    .reshape(b, cz, num_chunks * t_lat, hl, wl)
                )
            except RuntimeError as err:
                if not _is_cuda_oom(err):
                    raise
                logging.warning(
                    "H3 VAE batched temporal encode OOM; falling back to serial clips"
                )
                if x.is_cuda:
                    torch.cuda.empty_cache()
                z = None
        if z is None:
            z_list = []
            for i in range(num_chunks):
                clip_x = x[:, :, i * self.clip_length:(i + 1) * self.clip_length, :, :]
                z_list.append(self._adaptive_encode(clip_x))
            z = torch.cat(z_list, dim=2)
        if self.token_drop > 0:
            z = z[:, :, :-self.token_drop]
        return z

    def _decode_temporal_pad_frames(self, z_len, pad_tokens):
        if pad_tokens <= 0:
            return 0
        intra_tail = self.clip_length % self.vae_ratio_t
        if intra_tail == 0:
            return pad_tokens * self.vae_ratio_t

        z_len_before_pad = z_len - pad_tokens
        return sum(
            (intra_tail if (z_len_before_pad + k) % self.tokens_chunk_size == 0
             else self.vae_ratio_t)
            for k in range(pad_tokens)
        )

    def _decode_temporal_frame_plan(self, z_len, num_chunks, pad_tokens):
        chunk_dec = self.tokens_chunk_size * self.vae_ratio_t
        split_count = int(self.token_drop > 0) + 1
        total_frames = 0
        final_overlap_frames = 0

        for i in range(num_chunks):
            t_start_idx = i * self.tokens_chunk_size
            t_end_idx = t_start_idx + self.tokens_chunk_size + self.token_overlap
            clip_token_len = max(0, min(t_end_idx, z_len) - min(t_start_idx, z_len))
            clip_frame_len = clip_token_len * self.vae_ratio_t

            for j in range(split_count):
                f_start_idx = j * chunk_dec
                f_end_idx = min(f_start_idx + chunk_dec, clip_frame_len)
                chunk_frames = max(0, f_end_idx - f_start_idx - self.frame_pre_padding)
                if j == 0:
                    total_frames += chunk_frames
                else:
                    final_overlap_frames = chunk_frames

        total_frames += final_overlap_frames
        return total_frames - self._decode_temporal_pad_frames(z_len, pad_tokens)

    def _pad_temporal_latents(self, z):
        pseudo_total_tokens = z.shape[2] + self.token_drop

        pad_tokens = 0
        remainder = pseudo_total_tokens % self.tokens_chunk_size
        if remainder != 0:
            pad_tokens = self.tokens_chunk_size - remainder
            pseudo_total_tokens += pad_tokens

        num_chunks = pseudo_total_tokens // self.tokens_chunk_size - int(self.token_drop > 0)
        if num_chunks < 1:
            # too few tokens for one chunk (e.g. T_lat == 2): pad one extra chunk
            pad_tokens += self.tokens_chunk_size
            num_chunks += 1

        if pad_tokens > 0:
            pad_z = z[:, :, -1:, :, :].repeat(1, 1, pad_tokens, 1, 1)
            z = torch.cat([z, pad_z], dim=2)

        output_frames = self._decode_temporal_frame_plan(z.shape[2], num_chunks, pad_tokens)
        return z, num_chunks, pad_tokens, output_frames

    def _fold_temporal_clips(self, clip_decs, output_frames):
        chunk_dec = self.tokens_chunk_size * self.vae_ratio_t
        split_count = int(self.token_drop > 0) + 1
        dec = None
        dec_overlap = None
        write_pos = 0

        def write_part(part):
            nonlocal dec, write_pos
            part_frames = part.shape[2]
            if part_frames <= 0:
                return
            if dec is None:
                out_shape = list(part.shape)
                out_shape[2] = output_frames
                dec = torch.empty(out_shape, dtype=part.dtype, device=part.device)
            copy_frames = min(part_frames, max(0, dec.shape[2] - write_pos))
            if copy_frames > 0:
                dec[:, :, write_pos:write_pos + copy_frames, :, :].copy_(
                    part[:, :, :copy_frames, :, :]
                )
                write_pos += copy_frames

        for i, clip_dec in enumerate(clip_decs):
            for j in range(split_count):
                f_start_idx = j * chunk_dec
                f_end_idx = min(f_start_idx + chunk_dec, clip_dec.shape[2])
                clip_dec_chunk = clip_dec[:, :, f_start_idx:f_end_idx, :, :]
                clip_dec_chunk = clip_dec_chunk[:, :, self.frame_pre_padding:, :, :]

                if j == 0:
                    if dec_overlap is not None:
                        clip_dec_chunk = self.blend(
                            dec_overlap, clip_dec_chunk, self.frame_overlap, dim=-3
                        )
                        dec_overlap = None
                    write_part(clip_dec_chunk)
                else:
                    dec_overlap = clip_dec_chunk.contiguous()

            if i == len(clip_decs) - 1 and dec_overlap is not None:
                write_part(dec_overlap)
                dec_overlap = None

        return dec

    def _decode_temporal_batched(self, z, num_chunks, output_frames):
        height = z.shape[-2] * self.vae_ratio
        width = z.shape[-1] * self.vae_ratio
        grid = self._spatial_tile_grid(height, width)
        y_idx, y_len, y_overlap, x_idx, x_len, x_overlap = grid
        n_tiles = len(y_idx) * len(x_idx)

        jobs = []
        for i in range(num_chunks):
            t_start_idx = i * self.tokens_chunk_size
            t_end_idx = t_start_idx + self.tokens_chunk_size + self.token_overlap
            clip_z = z[:, :, t_start_idx:t_end_idx, :, :]
            jobs.extend(self._spatial_latent_slices(clip_z, y_idx, y_len, x_idx, x_len))

        tiles_out = self._decode_pixel_slices(jobs)
        clip_decs = []
        for i in range(num_chunks):
            start = i * n_tiles
            clip_decs.append(self._assemble_spatial_tiles(
                tiles_out[start:start + n_tiles],
                height, width, y_idx, y_len, y_overlap, x_idx, x_len, x_overlap,
            ))
        return self._fold_temporal_clips(clip_decs, output_frames)

    def _decode_temporal_serial(self, z, num_chunks, output_frames):
        chunk_dec = self.tokens_chunk_size * self.vae_ratio_t
        split_count = int(self.token_drop > 0) + 1
        dec = None
        dec_overlap = None
        write_pos = 0

        def write_part(part):
            nonlocal dec, write_pos
            part_frames = part.shape[2]
            if part_frames <= 0:
                return
            if dec is None:
                out_shape = list(part.shape)
                out_shape[2] = output_frames
                dec = torch.empty(out_shape, dtype=part.dtype, device=part.device)
            copy_frames = min(part_frames, max(0, dec.shape[2] - write_pos))
            if copy_frames > 0:
                dec[:, :, write_pos:write_pos + copy_frames, :, :].copy_(
                    part[:, :, :copy_frames, :, :]
                )
                write_pos += copy_frames

        for i in range(num_chunks):
            t_start_idx = i * self.tokens_chunk_size
            t_end_idx = t_start_idx + self.tokens_chunk_size + self.token_overlap
            clip_z = z[:, :, t_start_idx:t_end_idx, :, :]

            clip_dec = self._adaptive_decode(clip_z)

            for j in range(split_count):
                f_start_idx = j * chunk_dec
                f_end_idx = min(f_start_idx + chunk_dec, clip_dec.shape[2])
                clip_dec_chunk = clip_dec[:, :, f_start_idx:f_end_idx, :, :]
                clip_dec_chunk = clip_dec_chunk[:, :, self.frame_pre_padding:, :, :]

                if j == 0:
                    if dec_overlap is not None:
                        clip_dec_chunk = self.blend(
                            dec_overlap, clip_dec_chunk, self.frame_overlap, dim=-3
                        )
                        dec_overlap = None
                    write_part(clip_dec_chunk)
                else:
                    dec_overlap = clip_dec_chunk.contiguous()

            if i == num_chunks - 1 and dec_overlap is not None:
                write_part(dec_overlap)
                dec_overlap = None

            del clip_dec, clip_z

        return dec

    def decode_temporal(self, z):
        z, num_chunks, _pad_tokens, output_frames = self._pad_temporal_latents(z)

        if self._should_batch_tiles(z):
            try:
                return self._decode_temporal_batched(z, num_chunks, output_frames)
            except RuntimeError as err:
                if not _is_cuda_oom(err):
                    raise
                logging.warning(
                    "H3 VAE batched temporal decode OOM; falling back to serial tiles"
                )
                if z.is_cuda:
                    torch.cuda.empty_cache()
                with _disable_batch_tiles():
                    return self._decode_temporal_serial(z, num_chunks, output_frames)

        return self._decode_temporal_serial(z, num_chunks, output_frames)


    def encode(self, x):
        # x: [B, 3, T, H, W] in [-1, 1] -> normalized latents [B, 24, T_lat, H/16, W/16]
        if x.ndim == 4:
            x = x.unsqueeze(2)

        x = x.add(1.0).mul_(0.5).sub_(self.pixel_mean.to(x)).div_(self.pixel_std.to(x))

        if x.shape[2] == 1:
            moments = self._adaptive_encode(x)
            moments = moments[:, :, -1:, :, :]
        else:
            moments = self.encode_temporal(x)

        mean = torch.chunk(moments.float(), 2, dim=1)[0]
        latents_mean, latents_std = self._latents_affine(mean)
        return (mean - latents_mean) / latents_std

    def encode_tiled(self, x, **kwargs):
        # tiling is always on internally with the reference's semantic tile sizes, ignore tiling fallbacks
        return self.encode(x)

    def decode_tiled(self, z, **kwargs):
        return self.decode(z)

    def _latents_affine(self, ref):
        # mean/std broadcast views depend only on (device, dtype), not on pixels.
        key = (str(ref.device), str(ref.dtype))
        cache = getattr(self, "_lat_aff", None)
        if cache is not None and cache[0] == key:
            return cache[1], cache[2]
        mean = self.latents_mean.view(1, -1, 1, 1, 1).to(ref)
        std = self.latents_std.view(1, -1, 1, 1, 1).to(ref)
        self._lat_aff = (key, mean, std)
        return mean, std

    def decode(self, z):
        # z: [B, 24, T_lat, H_lat, W_lat] normalized latents -> pixels [B, 3, T, H, W]
        # Default (unit-interval): ImageNet [0, 1] after clamp. Stock *2-1 is
        # inverted by Comfy process_output; skip both when the env is on.
        latents_mean, latents_std = self._latents_affine(z)
        z = z * latents_std + latents_mean

        if z.shape[2] == 1:
            dec = self._adaptive_decode(z)
            dec = dec[:, :, -1:, :, :]
        else:
            dec = self.decode_temporal(z)

        dec = dec.float()
        dec.mul_(self.pixel_std.to(dec)).add_(self.pixel_mean.to(dec)).clamp_(0.0, 1.0)
        if not _unit_interval_enabled():
            dec.mul_(2.0).sub_(1.0)
        dec = _maybe_host_fp16(dec)
        dec, last = maybe_nhwc_bt_hwc(dec)
        self.h3_channels_last = last
        return dec

    def stage_output_host(self, dec, device, dtype):
        return stage_output_host(self, dec, device, dtype)
