"""H3-native first-block cross-step cache.

This is the remaining Sol Engine lever on DGX Spark: always execute block 0,
then reuse the residual of blocks 1..N-1 when that first residual has barely
moved. The path is fail-closed. EasyCache, DiT replacements, training, or a
missing schedule keep the full 50-block stack.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math

import torch

import comfy.model_prefetch


CONFLICT_KEYS = {
    "easycache": "EasyCache or LazyCache",
    "cache_dit_turbo": "CacheDiT",
    "minimax_h3_block_cache_t8": "MiniMax H3 Block Cache (T8)",
}

VALID_MODES = ("off", "stock", "auto", "safe", "fast", "aggressive")


@dataclass(frozen=True)
class CacheConfig:
    mode: str
    threshold: float
    start_percent: float = 0.10
    end_percent: float = 0.95
    max_consecutive_hits: int = 2
    temporal_guard: bool = True


PRESETS = {
    "safe": CacheConfig("safe", 0.08),
    "fast": CacheConfig("fast", 0.10),
    "aggressive": CacheConfig("aggressive", 0.12),
}


def resolve_mode(mode: str) -> str:
    if mode not in VALID_MODES:
        raise ValueError(f"unsupported H3 cross-step cache mode: {mode}")
    if mode == "auto":
        return "safe"
    return mode


def preset_config(mode: str) -> CacheConfig | None:
    resolved = resolve_mode(mode)
    if resolved in ("off", "stock"):
        return None
    return PRESETS[resolved]


def decide_reuse(
    diff: float | None,
    threshold: float,
    consecutive_hits: int,
    max_consecutive_hits: int,
    in_window: bool,
    temporal_diff: float | None = None,
) -> bool:
    if not in_window or diff is None or not math.isfinite(diff):
        return False
    decision = diff
    if temporal_diff is not None:
        if not math.isfinite(temporal_diff):
            return False
        decision = max(decision, temporal_diff)
    return decision <= threshold and consecutive_hits < max_consecutive_hits


def relative_mean_abs(current: torch.Tensor, previous: torch.Tensor) -> float:
    numerator = (current.float() - previous.float()).abs().mean()
    denominator = previous.float().abs().mean().clamp(min=1e-8)
    return float((numerator / denominator).item())


def video_temporal_diff(
    current: torch.Tensor,
    previous: torch.Tensor,
    video_slice: tuple[int, int] | None,
    latent_frames: int | None,
) -> float | None:
    if video_slice is None or not latent_frames:
        return None
    start, stop = video_slice
    current_video = current[start:stop]
    previous_video = previous[start:stop]
    if (
        current_video.shape != previous_video.shape
        or current_video.shape[0] % latent_frames
    ):
        return None
    rows_per_frame = current_video.shape[0] // latent_frames
    current_video = current_video.reshape(latent_frames, rows_per_frame, -1)
    previous_video = previous_video.reshape(latent_frames, rows_per_frame, -1)
    numerator = (current_video.float() - previous_video.float()).abs().mean(dim=(1, 2))
    denominator = previous_video.float().abs().mean(dim=(1, 2)).clamp(min=1e-8)
    return float((numerator / denominator).max().item())


class CacheContext:
    def __init__(self):
        self.previous_first_residual: torch.Tensor | None = None
        self.remaining_blocks_residual: torch.Tensor | None = None
        self.first_block_output: torch.Tensor | None = None
        self.pending_first_residual: torch.Tensor | None = None
        self.use_cache = False
        self.consecutive_hits = 0
        self.previous_sigma: float | None = None
        self.input_signature: tuple | None = None
        self.last_diff: float | None = None
        self.video_slice: tuple[int, int] | None = None
        self.latent_frames: int | None = None

    def clear_tensors(self):
        self.previous_first_residual = None
        self.remaining_blocks_residual = None
        self.first_block_output = None
        self.pending_first_residual = None
        self.use_cache = False
        self.consecutive_hits = 0
        self.previous_sigma = None
        self.input_signature = None
        self.last_diff = None
        self.video_slice = None
        self.latent_frames = None


class CrossStepCache:
    def __init__(
        self,
        config: CacheConfig,
        start_sigma: float | None,
        end_sigma: float | None,
        block_count: int,
    ):
        self.config = config
        self.start_sigma = start_sigma
        self.end_sigma = end_sigma
        self.block_count = block_count
        self.contexts: dict[tuple, CacheContext] = {}
        self.current: CacheContext | None = None
        self.full_steps = 0
        self.cached_steps = 0
        self.diff_values: list[float] = []
        self.temporal_diff_values: list[float] = []
        self.disabled_reason: str | None = None
        self._disable_logged = False

    def reset(self):
        for context in self.contexts.values():
            context.clear_tensors()
        self.contexts.clear()
        self.current = None
        self.full_steps = 0
        self.cached_steps = 0
        self.diff_values.clear()
        self.temporal_diff_values.clear()
        self.disabled_reason = None
        self._disable_logged = False

    def disable(self, reason: str):
        self.disabled_reason = reason
        if not self._disable_logged:
            logging.info("H3 cross-step cache disabled: %s", reason)
            self._disable_logged = True

    @staticmethod
    def _input_signature(hidden):
        return (tuple(hidden.shape), hidden.dtype, str(hidden.device))

    @staticmethod
    def _video_layout(minimax_payload):
        if not minimax_payload:
            return None, None
        layout = minimax_payload.get("layout")
        if layout is None or not getattr(layout, "segments", None):
            return None, None
        video_slice = next(
            ((start, stop) for start, stop, kind in layout.segments if kind == "video"),
            None,
        )
        signature = getattr(layout, "signature", None)
        latent_frames = signature[1] if signature and len(signature) > 1 else None
        return video_slice, latent_frames

    def _sigma(self, timestep):
        if torch.is_tensor(timestep):
            return float(timestep.flatten()[0].item()) / 1000.0
        return float(timestep) / 1000.0

    def _in_window(self, sigma: float) -> bool:
        if self.start_sigma is not None and self.end_sigma is not None:
            return self.end_sigma <= sigma <= self.start_sigma
        progress = max(0.0, min(1.0, 1.0 - sigma))
        return self.config.start_percent <= progress <= self.config.end_percent

    def begin_forward(self, hidden, timestep, transformer_options, minimax_payload=None):
        if self.disabled_reason is not None:
            self.current = None
            return
        if hidden.requires_grad:
            self.disable("hidden requires grad")
            self.current = None
            return
        sigma = self._sigma(timestep)
        uuids = transformer_options.get("uuids")
        key = tuple(str(value) for value in uuids) if uuids else ("default",)
        context = self.contexts.setdefault(key, CacheContext())
        signature = self._input_signature(hidden)
        if context.input_signature != signature or (
            context.previous_sigma is not None and sigma > context.previous_sigma + 1e-7
        ):
            context.clear_tensors()
        context.input_signature = signature
        context.previous_sigma = sigma
        context.video_slice, context.latent_frames = self._video_layout(minimax_payload)
        context.first_block_output = None
        context.pending_first_residual = None
        context.use_cache = False
        self.current = context

    def end_forward(self):
        self.current = None

    def should_skip(self, index: int) -> bool:
        return (
            self.current is not None
            and self.disabled_reason is None
            and index > 0
            and self.current.use_cache
        )

    @torch.compiler.disable()
    def after_first_block(self, first_input, first_output):
        context = self.current
        if context is None or self.disabled_reason is not None:
            return
        first_residual = (first_output - first_input).detach()
        previous = context.previous_first_residual
        tail = context.remaining_blocks_residual
        can_compare = (
            previous is not None
            and tail is not None
            and previous.shape == first_residual.shape
            and tail.shape == first_output.shape
        )
        diff = None
        temporal_diff = None
        if can_compare:
            diff = relative_mean_abs(first_residual, previous)
            self.diff_values.append(diff)
            if self.config.temporal_guard:
                temporal_diff = video_temporal_diff(
                    first_residual, previous, context.video_slice, context.latent_frames
                )
                if temporal_diff is not None:
                    self.temporal_diff_values.append(temporal_diff)
        use_cache = decide_reuse(
            diff,
            self.config.threshold,
            context.consecutive_hits,
            self.config.max_consecutive_hits,
            self._in_window(context.previous_sigma),
            temporal_diff,
        )
        context.last_diff = diff
        context.use_cache = use_cache
        if use_cache:
            context.consecutive_hits += 1
            context.first_block_output = None
            context.pending_first_residual = None
            return
        context.consecutive_hits = 0
        context.first_block_output = first_output.detach().clone()
        context.pending_first_residual = first_residual

    def after_full_stack(self, output):
        context = self.current
        if (
            context is None
            or self.disabled_reason is not None
            or context.first_block_output is None
            or context.pending_first_residual is None
        ):
            return
        context.remaining_blocks_residual = (output - context.first_block_output).detach()
        context.previous_first_residual = context.pending_first_residual
        context.first_block_output = None
        context.pending_first_residual = None
        self.full_steps += 1

    def apply_cached_tail(self, first_output):
        context = self.current
        if context is None or context.remaining_blocks_residual is None:
            raise RuntimeError("H3 cross-step cache has no residual to reuse")
        self.cached_steps += 1
        return first_output + context.remaining_blocks_residual

    def summary(self) -> str:
        steps = self.full_steps + self.cached_steps
        if self.disabled_reason is not None:
            return f"disabled ({self.disabled_reason})"
        if steps == 0:
            return "no model steps"
        executed_blocks = self.full_steps * self.block_count + self.cached_steps
        block_speedup = steps * self.block_count / max(executed_blocks, 1)
        details = ""
        finite_diffs = sorted(value for value in self.diff_values if math.isfinite(value))
        if finite_diffs:
            middle = len(finite_diffs) // 2
            median = (
                finite_diffs[middle]
                if len(finite_diffs) % 2
                else (finite_diffs[middle - 1] + finite_diffs[middle]) / 2
            )
            details = (
                f"; residual diff min/median/max "
                f"{finite_diffs[0]:.5f}/{median:.5f}/{finite_diffs[-1]:.5f}"
            )
        guard_details = ""
        finite_temporal = sorted(
            value for value in self.temporal_diff_values if math.isfinite(value)
        )
        if finite_temporal:
            guard_details = f"; temporal guard max {finite_temporal[-1]:.5f}"
        return (
            f"mode={self.config.mode} cached {self.cached_steps}/{steps} steps; "
            f"estimated block-stack speedup {block_speedup:.2f}x{details}{guard_details}"
        )


def resolve_cache(transformer_options, block_count, blocks_replace) -> CrossStepCache | None:
    cache = transformer_options.get("minimax_h3_cross_step_cache")
    if not isinstance(cache, CrossStepCache):
        return None
    if cache.disabled_reason is not None:
        return None
    if block_count < 2:
        cache.disable(f"need at least two blocks, got {block_count}")
        return None
    for key, name in CONFLICT_KEYS.items():
        if key in transformer_options:
            cache.disable(f"conflicts with {name}")
            return None
    if any(("double_block", index) in blocks_replace for index in range(block_count)):
        cache.disable("conflicts with a DiT block replacement")
        return None
    return cache


def run_block_stack(
    blocks,
    hidden,
    *,
    run_one,
    prefetch_queue,
    device,
    transformer_options,
    timestep,
    minimax_payload,
    blocks_replace,
):
    cache = resolve_cache(transformer_options, len(blocks), blocks_replace)
    if cache is not None:
        cache.begin_forward(hidden, timestep, transformer_options, minimax_payload)
    last = len(blocks) - 1
    try:
        for index, block in enumerate(blocks):
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block)
            if cache is not None and cache.should_skip(index):
                if index == last:
                    hidden = cache.apply_cached_tail(hidden)
                continue
            first_input = hidden.detach().clone() if cache is not None and index == 0 else None
            hidden = run_one(index, block, hidden)
            if cache is not None and index == 0:
                cache.after_first_block(first_input, hidden)
            elif cache is not None and index == last:
                cache.after_full_stack(hidden)
        if prefetch_queue is not None:
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)
        return hidden
    finally:
        if cache is not None:
            cache.end_forward()


def make_sample_wrapper(cache: CrossStepCache):
    def wrapper(executor, *args, **kwargs):
        cache.reset()
        logging.info(
            "H3 cross-step cache enabled: mode=%s threshold=%.3f window=%.2f-%.2f max_hits=%d temporal_guard=%s",
            cache.config.mode,
            cache.config.threshold,
            cache.config.start_percent,
            cache.config.end_percent,
            cache.config.max_consecutive_hits,
            cache.config.temporal_guard,
        )
        try:
            return executor(*args, **kwargs)
        finally:
            logging.info("H3 cross-step cache: %s", cache.summary())
            cache.reset()

    return wrapper
