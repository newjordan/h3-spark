# ComfyUI overlay

This is a source overlay for ComfyUI revision
`6db4fa2fcd38a92d4e0364c917cd8503620e2a74`, not a standalone package.

The finalized source includes:

- H3-scoped SageAttention quality routing and output-amax/prequantized-Q APIs;
- exact segmented modulation and gated residual kernels;
- guarded direct, SwiGLU, AdaLN, dynamic RMS/AdaLN, and gated-transition NVFP4
  adapters;
- promoted Q RMSNorm+rot96 RoPE -> Sage INT8 production;
- static Q out-projection coexistence;
- integrated one-pass static RMS/AdaLN and static gated FC1 adapters; and
- the five-attribute `StaticScalePlan` certificate, whose fifth attribute
  binds the exact plan-owned pre/post hook IDs;
- VAE tile-batch / SwiGLU native / identical-ref and step-invariant caches;
- first-block cache (default off) and staged cross-block gate→QKV (default
  `stock` / HOLD). Do not enable both.

Static source is integrated but dormant without an explicitly validated v3
artifact. Missing, partial, stale, foreign-hooked, or identity-mismatched
bridges fail closed before native mutation.

Use `scripts/install_comfyui_overlay.sh` from the repository root. It verifies
the upstream revision and refuses a dirty target. Repository tests are not
copied into the ComfyUI tree. Build native libraries with
`scripts/build_native.sh` against the exact target Python/PyTorch environment;
do not commit generated `.so` files. The attention node offers `quality`,
`fast`, and `stock`. `sage3_kfold` is not shipped and is rejected instead of
being silently replaced by another kernel.
