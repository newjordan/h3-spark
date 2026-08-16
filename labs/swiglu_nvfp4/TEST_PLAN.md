# SwiGLU -> NVFP4 integration test plan

## Fail-closed contract

Use a real H3 FC2 module plus a lightweight predicate matrix. The fused branch
must be eligible only when all of these are true: inference mode; raw contiguous
CUDA BF16 `[S, 28672]` without gradients; NVFP4 non-transposed weight;
`in_features == 14336`; scalar FP32 `input_scale` already on the input
device; no pre-quant scale, full-precision override, force-cast, weight patch,
or bias patch; and all four bridge certificates are valid, including object
identity between `input_scale` and the producer-validated scale certificate.

Individually negate every condition. In particular, reject a missing/false
marker, truthy values other than the singleton `True`, non-`frozenset`
certificates, unknown `(S, 14336)` shapes, unknown dtype strings, wrong raw
rank/width/device/dtype/layout, and a scale on CPU or another GPU. Confirm each
case takes the unchanged eager path without raising. Known calibrated shapes
and dtypes must take the fused path; an adjacent row count must not.

## Casting and lifetime

Instrument `cast_bias_weight` and `uncast_bias_weight`. Verify the fused path
stages the weight once and always uncasts it, including exceptions. Make cast
return a full-precision, wrong-layout, or transposed weight and confirm the
adapter returns `None`, uncasts exactly once, and lets the caller run its normal
fallback. Exercise loaded, offloaded, and DynamicVRAM/vbar-streamed weights on
the real model; check output device, allocator stability, and absence of stale
offload-stream references over repeated calls.

## Kernel goldens

For rows `1, 15, 16, 17, 127, 128, 129` and production row counts, compare the
supplied-scale native and Triton kernels with BF16
`F.silu(gate).mul_(up)` followed by comfy-kitchen 0.2.27 CUDA
`quantize_nvfp4(..., pad_16x=True, hi_first=True)`.

- Require identical output and scale shapes, zeroed padded scale tiles, and
  byte-identical E4M3 block scales.
- Require byte-identical packed FP4. For the static supplied-scale path only, a
  mismatch may be classified as harmless signed zero only after decoding both
  nibbles and proving the magnitude bits are zero on both sides and the sign bit
  is the sole difference. Report this count; do not silently relax equality.
- Dequantize both packed results with comfy-kitchen and require exact BF16
  equality. Then run the real FC2 NVFP4 linear and compare its BF16 output with
  the unfused quantized reference exactly (or document any cuBLAS determinism
  limit separately).
- For the dynamic two-pass kernel, additionally require exact FP32 scale equality
  with `(BF16 amax / 2688.0).to(float32)`, exact block-scale bytes, and exact
  packed data. Include all-zero, large finite, signed-zero, Inf, and NaN probes
  as diagnostic edge cases; production eligibility remains finite-positive
  static scales certified by the bridge.

## Execution behavior

Capture supplied-scale fusion plus NVFP4 linear in a CUDA graph, replay it with
new data at the same certified shape, and compare with eager execution. Confirm
no `.item()`, host synchronization, allocation-shape change, or graph-unsafe
lazy compilation occurs after warmup. Profile eager, Triton, native static, and
native dynamic paths with CUDA events at every observed H3 row count, separating
first-build/autotune cost from steady state.

## End-to-end acceptance

Run the calibrated concept for 20 denoising steps with the bridge enabled.
Compare against the same seed/settings on the unfused static-scale path and the
original dynamic-scale path: final latent error, decoded-video metrics and
visual review, per-layer overflow counts, maximum `amax / (scale * 2688)`, and
NaN/Inf checks. Accept only with zero unexpected fallback, zero overflow under
the artifact margin, stable DynamicVRAM memory behavior, and no material quality
regression.
