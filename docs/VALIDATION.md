# Validation ledger

All entries below are bounded CPU/source gates unless explicitly labeled as a
seconds-scale CUDA gate. No validation step in this repository requires a
model load or render.

## Promoted Q deployment

| Gate | Result |
|---|---:|
| Focused Q/linear/RMS/gate/SwiGLU integration | 51 passed |
| Q native source contracts | 6 passed |
| Supplied-amax consumer structural contracts | 11 passed |
| Release artifact/source manifest checks | pass |
| One-shot 20,423-row release parity | pass in 2.8 s |

Release parity covered default and non-default streams and required exact Q,
Q scale, K, BF16 attention output, output-amax marker, consumer triplet, and
version deltas `[1, 1, 1]`. Artifact hashes and every boolean are recorded in
`labs/q_deployment/QUALIFICATION_RECEIPT.json`.

## Static coexistence and hook certificate

| Gate | Result |
|---|---:|
| Finalized Q/static candidate aggregate | 62 passed |
| Real `StaticScalePlan` cross-consumer lifecycle | 2 passed |
| Static-scale plan | 20 pytest cases + 15 unittest subtests |
| RMS/AdaLN static adapter | 40 passed |
| Gate/RMS/AdaLN static adapter | 29 passed |
| Aggregate frozen audit | 153 pytest cases + 15 unittest subtests |

The lifecycle suite proves normal activation, suppression in the presence of
pre-existing hooks, rejection of a foreign hook added after activation, exact
attribute restoration, and coexistence across Q/direct/SwiGLU/RMS/gate
consumers. The final artifact rebind and exact acceptance render are GO.

The path-neutral rebind receipt pins model source `d7b196cc...`, architecture
`b5808954...`, and artifact `b3b4aee7...`. The generic verifier passed 196
layers and 3,920 calls with max ratio 0.917149 and the four required dynamic
exclusions.

## Final serving acceptance

The exact 20-step, 20,423-row fixed-reference workload workflow completed successfully at
147.0 s sampler (7.35 s/step), 213.0 s executor, and 213.8 s wall. Runtime
logs reported 196 static layers applied and the four required dynamic
fallbacks. The path-neutral receipt is `docs/final_acceptance_receipt.json`;
the generated media is excluded.

## Static native gate

The single approved production-shape gate ran on a non-default stream with no
warmup and one repetition. It passed byte equality for the gated residual,
qualified rstd, packed FP4, and E4M3 scales; the residual version delta was
one. The candidate saved 0.514368 ms/path and the whole command completed in
1.231 seconds. Generated binaries are excluded; their hashes and resource
report remain in compact JSON.

## Cross-block candidate

CPU/source: 29 adapter/model-integration cases, including the loop-owned
`blocks_replace` identity tests. The candidate is rebased onto the finalized
Q/static integration model.

Seconds-scale CUDA gate `gate_cuda.py --rows 20423` passed in 1.567 s on
GB10 / SM 12.1. Default and non-default streams were byte-exact for the
gated residual, packed FP4, E4M3 scales, dynamic global scale, and rstd;
both producer and adapter paths advanced the residual version once;
materialize matched stock `_mod_gate`. Receipt:
`labs/crossblock_gate_qkv/gate_crossblock_20423.json`.

Same-branch source notes: `labs/crossblock_gate_qkv/AUDIT.md`.
Independent audit **GO**: `labs/crossblock_gate_qkv/INDEPENDENT_AUDIT.md`
(mutation/hooks/fallback/streams/identity; live adapter matches lab;
live `model.py` only adds first-block cache and disables it when
cross-block is `auto`). Live serving smoke and 20-step FixedRef renders succeeded with
`crossblock_gate_qkv_fusion=auto` and dynamic NVFP4. Warm paired FixedRef
executors: stock 179.1 s, auto 180.8 s (auto − stock = +1.7 s; includes
VAE/MP4). 0.511881 s remains a kernel projection. The accepted static
artifact refused an architecture fingerprint mismatch against live
`model.py`. The path is still **HOLD** for the serving default (`stock`).
Receipt: `labs/crossblock_gate_qkv/live_ab_20260812.json`.

## 2026-08-13 identity overlay

VAE tile-batch, VAE SwiGLU native, identical-ref encode/vision/Lanczos,
step-invariant DiT caches, and audio Snake/filter caches landed in the
source overlay. Unit tests live in `integration/comfyui/tests/test_h3_*.py`.
First-block skip algebra is not exact (`labs/first_block_cache/`). Cross-block
serving default remains `stock`. Kitchen GEMM clone hunt is closed; do not
retry those tiles.

## Repository curation gate

Before each handoff commit:

- compile every tracked Python file;
- run focused CPU/source suites with CUDA hidden;
- run `git diff --check`;
- reject tracked binaries, media, build products, raw traces, and caches;
- scan tracked content for absolute host paths, secrets, and credentials; and
- verify a clean worktree after commit.
