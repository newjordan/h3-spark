"""H3-scoped SageAttention 2++ model patch for DGX Spark calibration.

This deliberately overrides attention only through the selected MODEL. It does
not change ComfyUI's global attention backend, so the text encoder and VAEs keep
their stock kernels.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F

from comfy.ldm.minimax.model import H3AttentionAmaxMarker
from comfy.ldm.minimax import q_rms_rope_int8
from comfy.ldm.minimax.first_block_cache import (
    CrossStepCache,
    make_sample_wrapper,
    preset_config,
    resolve_mode,
)
import comfy.patcher_extension


# --- in-executor attention timing tap (added 2026-08-13). Default
# off: after 900 calls the live tap did cuda.synchronize() on every
# DiT attention and flushed jsonl every 50. Same sage kernel either
# way. Opt-in: H3_ATTN_TIMING=on
import json as _json
import time as _time
import os as _attn_os

_ATTN_TIMING_FILE = "$HOME/h3/results/in_exec_attn_20260813.jsonl"
_ATTN_CALLS = {}
_ATTN_SYNC_FLOOR = int(_attn_os.environ.get("_ATTN_SYNC_FLOOR", "900"))
_ATTN_TIMING_OFF = frozenset({"", "0", "off", "false", "no"})


def _attn_timing_enabled():
    raw = _attn_os.environ.get("H3_ATTN_TIMING", "off")
    return str(raw).strip().lower() not in _ATTN_TIMING_OFF


def _attn_sync_floor():
    try:
        with open("$HOME/h3/results/_attn_sync_floor.txt") as _f:
            return int(_f.read().strip())
    except Exception:
        return _ATTN_SYNC_FLOOR
_ATTN_GPU_EV0 = []
_ATTN_GPU_EV1 = []
_ATTN_GPU_META = []


def _attn_tick(mode, q_shape, layout, wall_ms, note="", gpu_events=None,
               contig=None):
    n = _ATTN_CALLS.get(mode, 0)
    _ATTN_CALLS[mode] = n + 1
    rec = {"mode": mode, "shape": list(q_shape), "layout": layout,
           "wall_ms": round(wall_ms, 3), "note": note, "n": n,
           "first_call": n == 0, "contig": contig}
    if gpu_events is not None:
        _ATTN_GPU_EV0.append(gpu_events[0])
        _ATTN_GPU_EV1.append(gpu_events[1])
        _ATTN_GPU_META.append(rec)
        if len(_ATTN_GPU_EV0) >= 50:
            torch.cuda.synchronize()
            for _ev0, _ev1, _meta in zip(_ATTN_GPU_EV0, _ATTN_GPU_EV1,
                                         _ATTN_GPU_META):
                _meta["gpu_ms"] = round(_ev0.elapsed_time(_ev1), 3)
                with open(_ATTN_TIMING_FILE, "a", encoding="utf-8") as _f:
                    _f.write(_json.dumps(_meta) + "\n")
            del _ATTN_GPU_EV0[:]
            del _ATTN_GPU_EV1[:]
            del _ATTN_GPU_META[:]
    else:
        with open(_ATTN_TIMING_FILE, "a", encoding="utf-8") as _f:
            _f.write(_json.dumps(rec) + "\n")


class H3SageAttentionOverride:
    def __init__(self, mode: str, calibrate_first_call: bool):
        self.mode = mode
        self.calibrate_first_call = calibrate_first_call
        self.calibrated = False
        self.fallback_logged = False

    def _eligible(self, q, k, v, heads, mask, skip_reshape):
        if (
            not all(isinstance(tensor, torch.Tensor) for tensor in (q, k, v))
            or q.shape != k.shape
            or q.shape != v.shape
            or heads <= 0
        ):
            return False
        if skip_reshape:
            shape_ok = (
                q.ndim == 4
                and q.shape[1] == heads
                and q.shape[-1] in (64, 128)
                and q.shape[-2] > 1024
            )
        else:
            shape_ok = (
                q.ndim == 3
                and q.shape[-1] % heads == 0
                and q.shape[-1] // heads in (64, 128)
                and q.shape[-2] > 1024
            )
        return (
            shape_ok
            and mask is None
            and q.device.type == "cuda"
            and q.dtype in (torch.float16, torch.bfloat16)
            and q.dtype == k.dtype == v.dtype
            and q.device == k.device == v.device
        )

    def __call__(
        self,
        baseline,
        q,
        k,
        v,
        heads,
        mask=None,
        attn_precision=None,
        skip_reshape=False,
        skip_output_reshape=False,
        **kwargs,
    ):
        if not self._eligible(q, k, v, heads, mask, skip_reshape):
            if not self.fallback_logged:
                logging.info(
                    "H3 SageAttention bypassed a non-H3 attention shape: %s", tuple(q.shape)
                )
                self.fallback_logged = True
            return baseline(
                q, k, v, heads, mask=mask, attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape, **kwargs,
            )

        from sageattention import sageattn_qk_int8_pv_fp8_cuda

        if self.mode == "quality":
            pv_accum_dtype = "fp32+fp32"
            smooth_k = True
        else:
            pv_accum_dtype = "fp32+fp16"
            smooth_k = False

        if skip_reshape:
            q_s, k_s, v_s = q, k, v
            tensor_layout = "HND"
            batch, _, seq, dim = q_s.shape
        else:
            batch, seq, inner = q.shape
            dim = inner // heads
            q_s = q.unsqueeze(3).reshape(batch, seq, heads, dim)
            k_s = k.unsqueeze(3).reshape(batch, seq, heads, dim)
            v_s = v.unsqueeze(3).reshape(batch, seq, heads, dim)
            tensor_layout = "NHD"

        sm_scale = kwargs.get("scale", None)
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(dim)

        if self.mode == "sage3_kfold":
            try:
                import importlib.util as _ilu
                import os as _os
                import sys as _sys
                _mname = "h3_sage3_kfold_backend"
                if _mname not in _sys.modules:
                    _spec = _ilu.spec_from_file_location(
                        _mname,
                        _os.path.join(
                            _os.path.dirname(_os.path.abspath(__file__)),
                            "h3_sage3_kfold.py",
                        ),
                    )
                    _mod = _ilu.module_from_spec(_spec)
                    _sys.modules[_mname] = _mod
                    _spec.loader.exec_module(_mod)
                sage3_kfold_attention = getattr(
                    _sys.modules[_mname], "sage3_kfold_attention")
                _tb0 = _time.perf_counter()
                if tensor_layout == "HND":
                    nq = q_s.transpose(1, 2).contiguous()
                    nk = k_s.transpose(1, 2).contiguous()
                    nv = v_s.transpose(1, 2).contiguous()
                else:
                    nq, nk, nv = q_s, k_s, v_s
                _ta0 = _time.perf_counter()
                if _attn_timing_enabled() and _ATTN_CALLS.get(self.mode, 0) >= _attn_sync_floor():
                    torch.cuda.synchronize()
                _ge0 = torch.cuda.Event(enable_timing=True)
                _ge0.record()
                out = sage3_kfold_attention(nq, nk, nv, sm_scale)
                _ge1 = torch.cuda.Event(enable_timing=True)
                _ge1.record()
                _attn_tick(self.mode, q.shape, tensor_layout,
                           (_time.perf_counter() - _ta0) * 1e3,
                           note="attn", gpu_events=(_ge0, _ge1),
                           contig=(q_s.is_contiguous(),
                                   k_s.is_contiguous(),
                                   v_s.is_contiguous()))
                _attn_tick(self.mode, q.shape, tensor_layout,
                           (_time.perf_counter() - _tb0) * 1e3,
                           note="branch",
                           contig=(q_s.is_contiguous(),
                                   k_s.is_contiguous(),
                                   v_s.is_contiguous()))
                if tensor_layout == "HND":
                    out = out.transpose(1, 2)
                if skip_output_reshape:
                    return out
                return out.reshape(batch, seq, heads * dim)
            except Exception as exc:  # pragma: no cover - defensive
                if not self.fallback_logged:
                    logging.warning(
                        "H3 sage3_kfold attention failed (%s); falling "
                        "back to the production kernel", exc)
                    self.fallback_logged = True

        _tp0 = _time.perf_counter()
        _timing = _attn_timing_enabled()
        if _timing and _ATTN_CALLS.get(self.mode, 0) >= _attn_sync_floor():
            torch.cuda.synchronize()
        if _timing:
            _ge0 = torch.cuda.Event(enable_timing=True)
            _ge0.record()
        candidate = sageattn_qk_int8_pv_fp8_cuda(
            q_s, k_s, v_s,
            tensor_layout=tensor_layout,
            is_causal=False,
            qk_quant_gran="per_warp",
            sm_scale=sm_scale,
            pv_accum_dtype=pv_accum_dtype,
            smooth_k=smooth_k,
        )
        if _timing:
            _ge1 = torch.cuda.Event(enable_timing=True)
            _ge1.record()
            _attn_tick(self.mode, q.shape, tensor_layout,
                       (_time.perf_counter() - _tp0) * 1e3,
                       note="prod", gpu_events=(_ge0, _ge1),
                       contig=(q_s.is_contiguous(),
                               k_s.is_contiguous(),
                               v_s.is_contiguous()))

        if self.calibrate_first_call and not self.calibrated:
            q_ref, k_ref, v_ref = q_s, k_s, v_s
            candidate_ref = candidate
            if tensor_layout == "NHD":
                q_ref, k_ref, v_ref = (t.transpose(1, 2) for t in (q_ref, k_ref, v_ref))
                candidate_ref = candidate_ref.transpose(1, 2)
            reference = F.scaled_dot_product_attention(
                q_ref, k_ref, v_ref, dropout_p=0.0, is_causal=False, scale=sm_scale
            )
            # Sample all feature dimensions across heads and sequence. This is
            # representative but avoids materializing two full FP32 outputs.
            ref = reference[:, ::7, ::16, :].float()
            got = candidate_ref[:, ::7, ::16, :].float()
            delta = got - ref
            relative_l2 = (
                torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(ref)
            ).item()
            cosine = F.cosine_similarity(ref.flatten(), got.flatten(), dim=0).item()
            logging.info(
                "H3 SageAttention real-activation calibration mode=%s shape=%s "
                "mae=%.8f max_abs=%.8f relative_l2=%.6f cosine=%.8f",
                self.mode,
                tuple(q_s.shape),
                delta.abs().mean().item(),
                delta.abs().max().item(),
                relative_l2,
                cosine,
            )
            self.calibrated = True

        if tensor_layout == "HND":
            if skip_output_reshape:
                return candidate
            return candidate.transpose(1, 2).reshape(batch, seq, heads * dim)
        if skip_output_reshape:
            return candidate.transpose(1, 2)
        return candidate.reshape(batch, seq, heads * dim)


class H3SageAttentionOutputAmax:
    def __init__(self, owner):
        self.owner = owner

    def try_call(self, q, k, v, heads, mask=None, skip_reshape=False):
        if (
            self.owner.mode != "quality"
            or (self.owner.calibrate_first_call and not self.owner.calibrated)
            or not self.owner._eligible(q, k, v, heads, mask, skip_reshape)
            or skip_reshape
            or q.dtype != torch.bfloat16
            or q.shape[0] != 1
            or heads != 56
            or q.shape[-1] != 7168
        ):
            return None

        try:
            from sageattention import (
                sageattn_output_amax_available,
                sageattn_qk_int8_pv_fp8_cuda_with_output_amax,
            )
            if not sageattn_output_amax_available():
                return None
        except (ImportError, OSError, RuntimeError):
            return None

        batch, seq, inner = q.shape
        dim = inner // heads
        q_s = q.reshape(batch, seq, heads, dim)
        k_s = k.reshape(batch, seq, heads, dim)
        v_s = v.reshape(batch, seq, heads, dim)
        candidate, marker = sageattn_qk_int8_pv_fp8_cuda_with_output_amax(
            q_s, k_s, v_s,
            tensor_layout="NHD",
            is_causal=False,
            qk_quant_gran="per_warp",
            sm_scale=1.0 / math.sqrt(dim),
            pv_accum_dtype="fp32+fp32",
            smooth_k=True,
        )
        return (
            candidate.reshape(batch, seq, inner),
            H3AttentionAmaxMarker(marker, self),
        )


class H3SageAttentionQFusion:
    __slots__ = ("owner",)

    def __init__(self, owner):
        self.owner = owner

    @staticmethod
    def _sage_capability():
        try:
            from sageattention import (
                sageattn_h3_prequantized_q_available,
                sageattn_h3_prequantized_q_pv_fp8_cuda,
            )
            if not sageattn_h3_prequantized_q_available(return_output_amax=True):
                return None
        except (ImportError, OSError, RuntimeError):
            return None
        return sageattn_h3_prequantized_q_pv_fp8_cuda

    def can_call(
        self, q, k, v, rope_freqs, q_weight, k_weight, epsilon, heads,
        mask=None, scale=None,
    ):
        return bool(
            self.owner.mode == "quality"
            and (not self.owner.calibrate_first_call or self.owner.calibrated)
            and mask is None
            and scale is None
            and heads == 56
            and isinstance(v, torch.Tensor)
            and v.shape == q.shape
            and v.dtype == torch.bfloat16
            and v.device == q.device
            and not v.requires_grad
            and not torch.is_grad_enabled()
            and q_rms_rope_int8.eligible(
                q, k, rope_freqs, q_weight, k_weight, epsilon
            )
            and self._sage_capability() is not None
        )

    def __call__(
        self, q, k, v, rope_freqs, q_weight, k_weight, epsilon, heads,
        mask=None, scale=None,
    ):
        if not self.can_call(
            q, k, v, rope_freqs, q_weight, k_weight, epsilon, heads,
            mask=mask, scale=scale,
        ):
            raise RuntimeError("H3 Q fusion was invoked without a valid preflight")
        attention = self._sage_capability()
        prequantized = q_rms_rope_int8.try_qk_rms_rope_q_int8(
            q, k, rope_freqs, q_weight, k_weight, epsilon, 16
        )
        if prequantized is None:
            raise RuntimeError("H3 Q fusion lost native capability before launch")
        q_int8, q_scale, transformed_k = prequantized
        out, marker = attention(
            q_int8, q_scale, transformed_k, v, return_output_amax=True
        )
        if not (
            isinstance(out, torch.Tensor)
            and out.shape == v.shape
            and out.dtype == torch.bfloat16
            and out.device == v.device
            and out.is_contiguous()
        ):
            raise RuntimeError("H3 prequantized attention returned an invalid output")
        out = out.reshape(out.shape[0], out.shape[1], heads * out.shape[-1])
        return out, H3AttentionAmaxMarker(marker, self)


class H3SageAttentionModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (["quality", "fast", "stock", "sage3_kfold"], {"default": "quality"}),
                "calibrate_first_call": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "modulation_fusion": (
                    ["exact", "aggressive", "stock"], {"default": "exact"}
                ),
                "swiglu_nvfp4_fusion": (
                    ["auto", "stock", "static"], {"default": "auto"}
                ),
                "rms_adaln_nvfp4_fusion": (
                    ["auto", "stock"], {"default": "auto"}
                ),
                "q_rms_rope_int8_fusion": (
                    ["auto", "stock"], {"default": "auto"}
                ),
                "cross_step_cache": (
                    ["off", "auto", "safe", "fast", "aggressive"],
                    {"default": "off"},
                ),
                "crossblock_gate_qkv_fusion": (
                    ["stock", "auto"], {"default": "stock"}
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model"

    def patch(
        self,
        model,
        mode,
        calibrate_first_call,
        modulation_fusion="exact",
        swiglu_nvfp4_fusion="auto",
        rms_adaln_nvfp4_fusion="auto",
        q_rms_rope_int8_fusion="auto",
        cross_step_cache="off",
        crossblock_gate_qkv_fusion="stock",
    ):
        if mode not in ("quality", "fast", "stock", "sage3_kfold"):
            raise ValueError(f"unsupported H3 attention mode: {mode}")
        if modulation_fusion not in ("exact", "aggressive", "stock"):
            raise ValueError(
                f"unsupported H3 modulation fusion: {modulation_fusion}"
            )
        if swiglu_nvfp4_fusion not in ("stock", "static", "auto"):
            raise ValueError(
                f"unsupported H3 SwiGLU NVFP4 fusion: {swiglu_nvfp4_fusion}"
            )
        if rms_adaln_nvfp4_fusion not in ("stock", "auto"):
            raise ValueError(
                f"unsupported H3 RMS+AdaLN NVFP4 fusion: {rms_adaln_nvfp4_fusion}"
            )
        if q_rms_rope_int8_fusion not in ("auto", "stock"):
            raise ValueError(
                f"unsupported H3 Q RMS-RoPE fusion: {q_rms_rope_int8_fusion}"
            )
        if crossblock_gate_qkv_fusion not in ("stock", "auto"):
            raise ValueError(
                f"unsupported H3 cross-block gate fusion: {crossblock_gate_qkv_fusion}"
            )
        cache_mode = resolve_mode(cross_step_cache)
        patched = model.clone()
        transformer_options = patched.model_options["transformer_options"]
        if mode != "stock":
            override = H3SageAttentionOverride(
                mode, calibrate_first_call
            )
            transformer_options["optimized_attention_override"] = override
            if mode == "quality":
                transformer_options["minimax_h3_attention_output_amax"] = H3SageAttentionOutputAmax(override)
                if q_rms_rope_int8_fusion == "auto":
                    transformer_options["minimax_h3_q_rms_rope_int8"] = H3SageAttentionQFusion(override)
                else:
                    transformer_options.pop("minimax_h3_q_rms_rope_int8", None)
            else:
                transformer_options.pop("minimax_h3_attention_output_amax", None)
                transformer_options.pop("minimax_h3_q_rms_rope_int8", None)
        else:
            producer = transformer_options.get("minimax_h3_attention_output_amax", None)
            q_fusion = transformer_options.get("minimax_h3_q_rms_rope_int8", None)
            override = transformer_options.get("optimized_attention_override", None)
            if isinstance(producer, H3SageAttentionOutputAmax):
                transformer_options.pop("minimax_h3_attention_output_amax", None)
            if isinstance(q_fusion, H3SageAttentionQFusion):
                transformer_options.pop("minimax_h3_q_rms_rope_int8", None)
            if isinstance(override, H3SageAttentionOverride):
                transformer_options.pop("optimized_attention_override", None)
        transformer_options["minimax_h3_modulation_fusion"] = modulation_fusion
        transformer_options["minimax_h3_swiglu_nvfp4_fusion"] = swiglu_nvfp4_fusion
        transformer_options["minimax_h3_rms_adaln_nvfp4_fusion"] = rms_adaln_nvfp4_fusion
        transformer_options["minimax_h3_crossblock_gate_qkv_fusion"] = (
            crossblock_gate_qkv_fusion
        )
        config = preset_config(cache_mode)
        if config is None:
            transformer_options.pop("minimax_h3_cross_step_cache", None)
        else:
            for key, name in (
                ("easycache", "EasyCache or LazyCache"),
                ("cache_dit_turbo", "CacheDiT"),
                ("minimax_h3_block_cache_t8", "MiniMax H3 Block Cache (T8)"),
            ):
                if key in transformer_options:
                    raise ValueError(
                        f"H3 cross-step cache cannot be combined with {name}"
                    )
            start_sigma = end_sigma = None
            getter = getattr(patched, "get_model_object", None)
            if callable(getter):
                try:
                    sampling = getter("model_sampling")
                    start_sigma = float(sampling.percent_to_sigma(config.start_percent))
                    end_sigma = float(sampling.percent_to_sigma(config.end_percent))
                except Exception:
                    start_sigma = end_sigma = None
            diffusion = None
            if callable(getter):
                try:
                    diffusion = getter("diffusion_model")
                except Exception:
                    diffusion = None
            block_count = len(getattr(diffusion, "blocks", ()) or ())
            if block_count < 2:
                block_count = 50
            cache = CrossStepCache(config, start_sigma, end_sigma, block_count)
            transformer_options["minimax_h3_cross_step_cache"] = cache
            add_wrapper = getattr(patched, "add_wrapper_with_key", None)
            if callable(add_wrapper):
                add_wrapper(
                    comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                    f"minimax_h3_cross_step_cache_{id(cache)}",
                    make_sample_wrapper(cache),
                )
        return (patched,)


NODE_CLASS_MAPPINGS = {"H3SageAttentionModel": H3SageAttentionModel}
NODE_DISPLAY_NAME_MAPPINGS = {"H3SageAttentionModel": "H3 Spark Optimizations"}
