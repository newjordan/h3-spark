# Cross-block source audit

Scope: `comfy_adapter.py` plus the rebased `_run_dit_blocks` / `DiTBlock.forward`
on the final Q/static model. This is a same-branch source audit, not a
second-author review. It does not replace an independent reviewer.

## Mutation

- `defer_qkv_transition` does not write `x`. It only allocates a sealed
  carrier.
- `try_qkv_transition` returns `None` only before `pending._state = _CLAIMED`.
  After that point every failure raises. There is no stock residual replay.
- The native producer mutates `x` in place exactly once (version delta 1 in
  the 20,423-row gate).
- `materialize_pending` claims the carrier as `_MATERIALIZED` and calls the
  supplied `_mod_gate`. A claimed or materialized carrier cannot be reused;
  `_require_pending_identity` rejects `_state is not _PENDING`.

## Hooks

- `block_pair_eligible` requires empty `HOOK_ATTRS` on both blocks and empty
  process-global forward/backward hook tables.
- `parent_eligible` rechecks the QKV owner. `_dynamic_eligible` / static
  certificate recheck the linear.
- A local hook on block `i` disables transitions `(i-1, i)` and `(i, i+1)`.
  Covered by `test_local_block_hook_breaks_both_adjacent_transitions`.

## Fallback

- Missing capability, partial static bridge, foreign hooks, or a non-empty
  `blocks_replace` prevent deferral. The old block path runs.
- There is no static-to-dynamic fallback. Partial/stale/foreign bridge state
  selects neither producer.
- A pre-claim `try_qkv_transition` miss materializes through `_mod_gate`, then
  runs the unchanged attention path.
- Any `double_block` replacement disables every transition and still runs the
  original wrapper contract.

## Streams

- The adapter uses the current PyTorch CUDA stream. It does not pin the
  default stream.
- Norm/linear `cast_bias_weight(..., offloadable=True)` is paired with
  `uncast_bias_weight` in `finally`.
- The 20,423-row gate runs the default stream and a non-default stream.

## Identity

- The loop owns one `blocks_replace` mapping via `_loop_owned_blocks_replace`.
  It never re-derives `dict.get("dit", {})` inside a block.
- The carrier stores that object and later checks `pending._blocks_replace is
  blocks_replace`. A fresh empty dict fails closed.
- Residual, MLP, gate, segments, row map, adjacent blocks, token, native
  module, producer, route, and static scale are `is`-bound.

## Still HOLD

This audit plus the seconds-scale CUDA gate are necessary, not sufficient.
An independent reviewer still has to sign mutation/hooks/fallback/streams
before a production render.
