# Cross-block final gate -> next QKV

Status: **STAGED / HOLD. Not live and not promoted.**

The candidate defers block `i`'s final MLP residual gate into block `i+1`'s
RMS/AdaLN/QKV preparation. It reuses the already-qualified dynamic and static
gate producers at the same 5,376-wide boundary and changes no CUDA math.

The carrier is one-use and seals the adjacent blocks, residual, MLP output,
gate, modulation segments, row map, replacement mapping, route, producer,
scale object, and loop token. First/last-block behavior, prefetch ordering,
pre-claim materialization, post-claim no-replay, hooks, replacements, and
mutation errors are covered by the isolated tests.

Projected opportunity is 0.511881 s over 20 evaluations: 47 static
transitions at 0.514368 ms saved plus two dynamic transitions at 0.709376 ms
saved. This is a kernel projection, not an end-to-end claim.

Ownership identity is now fixed: the DiT loop owns one `blocks_replace`
mapping and passes that same object into every block. The earlier candidate
re-derived `transformer_options.get("patches_replace", {}).get("dit", {})`
inside each block, which allocated a fresh empty dict when no replacement was
configured and failed `is` checks.

The candidate is rebased onto the finalized Q/static integration model
(static gate import, `can_static_linear_for_output`, and the static Q path
are preserved). Default remains `stock`. Promotion still requires an
independent source/CPU audit, the 20,423-row exactness gate, and the single
final acceptance workflow before live serving default changes.

The 20,423-row CUDA gate is `gate_cuda.py`. On this GB10 it completed in
1.567 s with byte-exact residual / packed FP4 / scales / rstd on both
streams. The compact receipt is `gate_crossblock_20423.json`.

The repository contains the adapter, isolated candidate postimage, tests, the
seconds-scale gate, and a same-branch source audit. It contains no binary and
makes no live promotion claim.
