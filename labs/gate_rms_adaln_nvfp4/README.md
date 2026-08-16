# H3 attention-gate → RMS/AdaLN → NVFP4 lab

This bounded producer fusion targets the 50 in-block attention-to-MLP
transitions in MiniMax H3.  It replaces:

1. segmented BF16 residual gate (`x += attention * gate`),
2. weighted RMSNorm,
3. segmented AdaLN, and
4. dynamic NVFP4 input quantization for FC1,

with two native passes.  The first pass mutates the residual exactly, preserves
PyTorch's certified RMS reduction tree, and finds the exact post-AdaLN BF16
maximum.  The second pass emits the existing cuBLAS NVFP4 packed data and
swizzled block scales.  FC1 itself remains the stock comfy-kitchen GEMM.

## Archived v1 measurement (not promotion evidence)

On DGX Spark / GB10, Torch `2.13.0+cu130` git `cf30153`, production FixedRef shape
`S=20,423`, `D=5,376`, FP32 strided modulation tables:

- gated residual: byte exact
- RMS `rstd`: byte exact
- packed FP4 data: byte exact
- E4M3 block scales: byte exact
- dynamic global scale: byte exact
- staged path: `5.044336 ms`
- fused path: `4.247136 ms`
- saving: `0.797200 ms/path`

There are 50 eligible attention-to-MLP paths per model evaluation, so the
microbenchmark projects about `0.797 s` across a 20-step render.  This is a
projection, not an end-to-end serving claim.

The old `gate_129.json` and `gate_20423.json` preserve this useful performance
signal, but are explicitly rejected as promotion evidence: they predate full
alias rejection, the `x` version-counter contract, the v2 ABI/source identity,
and nondefault-stream qualification.  The projected saving therefore remains
a target, not a qualified v2 result.

## Qualified v2 gate

`gate_v2_20423.json` is the promotion evidence for the hardened source.  On
the same production shape with FP32 modulation and a nondefault CUDA stream:

- gated residual, RMS `rstd`, packed FP4, block scales, and global scale:
  byte exact
- `x` version-counter delta: exactly `1`
- staged path: `5.186528 ms`
- fused path: `4.477152 ms`
- saving: `0.709376 ms/path`

That projects about `0.709 s` over 50 paths × 20 model evaluations.  It is
still a microbenchmark projection, not an end-to-end serving measurement.

## Safety contract

The Comfy adapter is deliberately narrow:

- inference only, exact modulation only, dynamic NVFP4 only
- internally generated row-map identity token
- exact H3 FC1 pair `5,376 → 28,672`
- FP32 modulation tables (the production curve path)
- exact known RMSNorm implementation and epsilon
- no owner, Linear, RMSNorm, or global module hooks
- no weight/bias/low-VRAM patches, pre-quant scale, force-cast, or training
- residual and attention output must be distinct contiguous BF16 tensors
- writable residual storage must be disjoint from attention, all modulation
  inputs, row IDs, and the cast norm weight; native checks mirror the facade
- the successful mutating launch increments `x`'s version counter exactly once
- certified Torch/CUDA build, exact GB10 capability `(12, 1)`, immutable v2
  AOT ABI/build ID, and an `sm_120` cubin; never request-time JIT
- all launches use PyTorch's current CUDA stream
- norm cast is fully consumed and uncast before FC1 is cast, protecting shared
  DynamicVRAM cast buffers
- every module/input guard is repeated after `run_every_op()` and before the
  last safe fallback boundary

`try_mlp_transition()` returns `None` only before the residual can be mutated.
After native launch, any error is surfaced and stock fallback is forbidden,
because fallback would apply the attention gate twice.

The v2 lab gate is qualified, and the source overlay uses the sealed
`try_mlp_transition()` transaction without applying the attention residual
twice. No model or render smoke was used for this lab gate.

## Reproduce the short gate

```bash
COMFYUI_ROOT=/path/to/comfyui \
  /path/to/comfyui/.venv/bin/python gate_cuda.py \
  --rows 20423 --warmup 1 --reps 2 \
  --output gate_v2_20423.json
```

The production-shape command is seconds-long and does not load the H3 model.

CPU adapter checks:

```bash
PYTHONPATH=/path/to/comfyui \
  /path/to/comfyui/.venv/bin/python -m unittest -v test_comfy_adapter.py
```
