# Performance report

Measurements were collected on one NVIDIA GB10 (SM 12.1) with CUDA 13.0,
PyTorch 2.13.0+cu130, Triton 3.7.1, comfy-kitchen 0.2.27, and SageAttention
2.2.0. The production workload is 864x480, 124 frames, 24 fps, 20 denoising
steps, and 20,423 sequence rows.

## Profile-guided lever order

The original one-evaluation profile at 20,393 rows totaled 12.294 s and
16,624 CUDA launches. Work proceeded in descending measured cost: attention,
NVFP4 activation preparation, dynamic `abs+amax`, SwiGLU, segmented
modulation, RMS/AdaLN preparation, and residual-gate transitions.

SageAttention quality routing, exact modulation fusion, and token-major NHD
layout reduced the profiled evaluation to 8.968 s, a 27.1% reduction. On the
20,423-row serving workload, the warm production ladder moved from 9.21
s/step with stock dynamic preparation to 8.47 s/step with native SwiGLU,
8.24 s/step with direct QKV/out/FC1 preparation, and 8.01 s/step with the
staged exact AdaLN producer. That last 20-step run measured 186.66 s executor
time versus 191.6 s for the direct-auto baseline.

## Latest exact operator gates

These are isolated CUDA-event or one-shot producer measurements. Projections
multiply the observed per-path delta by the exact eligible call count; they
are not end-to-end render deltas.

| Lever at 20,423 rows | Reference -> candidate | Projected opportunity | State |
|---|---:|---:|---|
| Dynamic RMS+AdaLN -> NVFP4 v2 | 4.231872 -> 2.339328 ms | 3.785 s / render | promoted |
| Attention gate -> dynamic RMS+AdaLN -> FC1 | 5.186528 -> 4.477152 ms | 0.709 s / render | promoted |
| Q RMSNorm+rot96 RoPE -> Sage Q INT8 | 7.417600 -> 6.193312 ms | 1.224 s / render | promoted |
| One-pass supplied-scale RMS+AdaLN | 2.339328 -> 1.604224 ms | 1.470 s / render | accepted in final static run |
| Gate -> supplied-scale RMS+AdaLN -> FC1 | 4.354976 -> 3.840608 ms | 0.514 s / render | accepted in final static run |
| Cross-block final gate -> next QKV | qualified producers reused | 0.512 s / render (projection) | warm FixedRef A/B +1.7 s; default **HOLD** |

The Q release gate was intentionally one-shot: no warmup and no timing loop.
It completed in 2.8 seconds and passed exact Q INT8, Q scale, transformed K,
BF16 attention output, output-amax marker, dynamic/supplied consumer triplet,
and tensor version-delta checks on default and non-default streams. The sealed
receipt is `labs/q_rms_rope_int8/release_qualification_s20423.json`.

The static gate ran once with zero warmup and one repetition. It completed in
1.231 seconds, advanced the residual version once, and produced byte-exact
residual, rstd, packed E2M1, and E4M3 scales. The compact result is
`labs/gate_rms_adaln_nvfp4_static/gate_static_20423.json`.

The cross-block estimate combines 47 static transitions at 0.514368 ms saved
and two dynamic transitions at 0.709376 ms saved: 25.594 ms/evaluation, or
0.511881 s across 20 evaluations. That figure remains a **kernel projection**.
A paired full-FixedRef executor A/B on the same warm dynamic stack did **not**
show a 0.51 s end-to-end win.

| Live cross-block check (2026-08-12) | Result |
|---|---|
| 2-step sampler-only smoke | success; 50.1 s wall / 46.8 s executor |
| 20-step FixedRef auto (after smoke) | success; executor 182.8 s |
| 20-step FixedRef stock (warm twin) | success; executor 179.1 s |
| 20-step FixedRef auto (warm rematch) | success; executor 180.8 s |
| Warm pair delta (auto − stock) | **+1.7 s** (executor, includes VAE/MP4) |
| Stack | sage2-quality, exact, SwiGLU/RMS/Q auto, cache off, **dynamic** NVFP4 |
| Static FixedRef attempt | refused: architecture fingerprint mismatch vs live `model.py` |
| Serving default | still `stock` / HOLD |

Receipts: `labs/crossblock_gate_qkv/live_chef_20260812.json`,
`labs/crossblock_gate_qkv/live_ab_20260812.json`. The +1.7 s executor
delta is not a kernel regression proof and is not compared to the accepted
static FixedRef 213.0 s. Do not promote the serving default.

## Static activation boundary

Static source and binaries are integrated. The final path-neutral receipt is
`labs/nvfp4_static/final_rebind_receipt.json`: decision GO, source
`d7b196cc...`, architecture `b5808954...`, artifact `b3b4aee7...`, 196 layers,
3,920 calls, and max observed ratio 0.917149. The required dynamic exclusions
are:

```text
blocks.10.attn.qkv_proj
blocks.11.attn.qkv_proj
blocks.22.mlp.fc2
blocks.46.mlp.fc2
```

## Final exact acceptance render

The final 864x480, 124-frame, six-reference fixed-reference workload run used 20 denoising
steps and source `d7b196cc...`. It completed successfully:

| Metric | Result |
|---|---:|
| Sampler | 147.0 s total / 7.35 s per step |
| Executor | 213.0 s |
| Wall | 213.8 s |
| Static routing | 196 layers applied; 4 required dynamic fallbacks |

This is a measured acceptance result, not an isolated projection. Executor
time includes non-sampler work and is not compared to warm ladder rows. The
receipt is `docs/final_acceptance_receipt.json`; the MP4 itself is excluded.

## Accepted-stack profile (2026-08-12)

One warm DiT evaluation on the live accepted stack (sage2-quality, exact
modulation, SwiGLU/RMS/Q auto, cross-block **stock**, dynamic NVFP4).
Sampler-only 3-step, profiler wait=2 / active=1. Measured sequence **20,393**
rows (not the 20,423-row isolated-gate shape). Compact receipt:
`benchmarks/profiles/accepted_stack_20260812.json` (campaign worktree;
not included in this repository). Chrome trace stays outside Git.

| Rank | Kernel / boundary | Self CUDA / eval | Notes |
|---|---|---:|---|
| 1 | Sage INT8/FP8 attn + output-amax | 3.316 s | forbidden to reopen Sage3 |
| 2 | CUTLASS NVFP4 GEMM (4 linears) | 2.396 s | kitchen `scaled_mm`; FC1 993 ms |
| 3 | Command Buffer Full | 1.625 s | launch/sync; torch.compile forbidden |
| 4 | Q RMS-RoPE INT8 | 315 ms | already promoted |
| 5 | SwiGLU amax + NVFP4 pack | 283 + 261 ms | **largest remaining owned kernel** |
| 6 | TransposePadPermute | 150 ms | compact-K/V forbidden |
| 7 | RMS/AdaLN + gate producers | 147 + 143 + 132 ms | already promoted / cross-block HOLD |

Largest remaining *attackable* measured boundary was the dynamic SwiGLU
two-pass (amax then pack). A one-SwiGLU store+pack producer was implemented
in `labs/swiglu_nvfp4/native_cuda/` (`swiglu_nvfp4_dynamic_oneshot`). At
20,423 x 28,672 it is **byte-exact** vs the two-pass producer and vs eager
kitchen (packed FP4, E4M3, global scale) but **slower**: 11.969 ms vs
10.660 ms. Writing/rereading the 558 MiB BF16 activation costs more than
recomputing SiLU. Receipt:
`labs/swiglu_nvfp4/native_cuda/gate_oneshot_20423.json`. Serving is
unchanged. Next owned identity is not this workspace path. Do not reopen
Sage3, torch.compile, cuDNN SDPA, K-smooth, compact-K, or CTA Q=256.

The five-attribute bridge includes the exact plan-owned pre/post hook IDs. Any
extra local, backward, or global hook; stale scale object; partial bridge;
replacement; patch; dtype/shape/device mismatch; or identity mismatch rejects
the static producer before mutation. Failures after mutation propagate and do
not replay stock work.

## Correctness policy

- Native NVFP4 gates require zero packed-byte and block-scale mismatches and
  exact scalar-scale bits.
- Q promotion requires exact attention and consumer parity on two streams,
  not only producer-local equality.
- Static scales require source/checkpoint/workload binding and held-out
  validation; a kernel byte gate alone does not validate a calibration.
- Cold runs, calibration runs, changed seeds used to invalidate caches, and
  changed reference sizes are labeled and never mixed into clean deltas.
- Full renders are reserved for final acceptance after seconds-scale gates.
