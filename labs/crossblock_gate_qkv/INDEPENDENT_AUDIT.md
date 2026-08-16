# Independent cross-block audit

Reviewer identity: 5-minute loop reviewer on dgx-spark-host (2026-08-12).
This is not the same-branch `AUDIT.md` and does not copy its conclusions.
A nested reviewer spawn did not persist; this report is from a fresh live+lab
read of the sources, tests, and 20,423-row receipt.

Date: 2026-08-12
Trees compared:

| Path | sha256 |
|---|---|
| live `comfyui-h3-current/.../crossblock_gate_qkv.py` | `2a448601143f9a17972efb43fc37cc32bf7f5ae870d7d3c11d21d51c7db03051` |
| lab `labs/crossblock_gate_qkv/comfy_adapter.py` | same |
| lab candidate `.../crossblock_gate_qkv.py` | same |
| integration `.../crossblock_gate_qkv.py` | same |
| live `.../model.py` | `d9853dfe8bcbbb6c8dcd2e3e4c7b20c6aaadf48e25d6211044dbb4cb1c434a46` |
| lab candidate `.../model.py` | `954274dca34d302302679d5dc8f657a6775bf0529c493a69ebeb609126c9edf0` |
| integration `.../model.py` | same as lab candidate |

Supporting: 29 CPU tests passed in 1.48 s
(`test_comfy_adapter.py` + `test_model_integration.py`).
`labs/crossblock_gate_qkv/gate_crossblock_20423.json` records both streams
byte-exact at 20,423 x 5376. `:18188` was down this tick; no render.

## Diff drift

Adapter is byte-identical across live, lab adapter, candidate, and
integration.

`DiTBlock.forward` is the same in live and lab. The only live `model.py`
delta is first-block cache coexistence:

- import `resolve_cache`
- `_run_dit_blocks(..., timestep=None, minimax_payload=None)`
- if cache is present and `crossblock_mode == "auto"`: `cache.disable(...)`
  then `cache = None`
- skip / after_first_block / after_full_stack / `end_forward` in `finally`

Lab integration tests load **candidate** `model.py`, so they do not execute
that live cache wrapper. The disable is conservative (cache off, fusion
allowed). Serving node default remains cache `off`, cross-block `stock`.

## Mutation

- `defer_qkv_transition` (`comfy_adapter.py:291-334`) returns a sealed
  `H3CrossBlockPending` or `None`. It does not write residual `x`.
- `try_qkv_transition` (`:584-809`) returns `None` only before
  `pending._state = _CLAIMED` (`:723`). After claim the native producer
  mutates `x` in place; later failures raise
  (`contract changed`, `invalid NVFP4 outputs`,
  `QKV owner/hook/weight changed`). There is no stock residual replay.
- `materialize_pending` (`:359-380`) sets `_MATERIALIZED` then calls the
  supplied `_mod_gate`. `is_pending` requires `_state is _PENDING`, so a
  claimed or materialized carrier cannot be reused.
- `DiTBlock.forward` (`live model.py:541-567`): `try_*` miss materializes
  through `_mod_gate` before the unchanged attention path. A post-claim
  exception skips materialize (`test_postmutation_failure_propagates_without_old_gate`).
- Gate receipt: producer and adapter residual `_version` delta is 1 on both
  streams.

## Hooks

- Loop pair check: empty `HOOK_ATTRS` on both DiTBlocks plus
  `_GLOBAL_HOOK_ATTRS` (forward and backward) via `_global_hooks_absent`
  (`comfy_adapter.py:117-122, 207-208`).
- Consumer recheck: `parent_eligible(attn)`, `_norm_eligible(norm1)`,
  `_linear_eligible(qkv_proj)`, and route-specific
  `_dynamic_eligible` / `_certified_static_scale` (`:396-416`). Those catch
  hooks on the modules whose forwards are bypassed.
- A local hook on block `i` makes pairs `(i-1,i)` and `(i,i+1)` ineligible
  (`test_local_block_hook_breaks_both_adjacent_transitions`).
- Live mix: `_run_dit_blocks` (`:664-667`) disables first-block cache when
  cross-block is `auto`. The node can still *set* both
  (`h3_sage_attention.py:383-418`); the loop then drops cache. Fail-closed.

## Fallback

- Default `_h3_crossblock_mode` is `stock` (`model.py:435-437`). Node
  optional default is `stock` (`h3_sage_attention.py:309-329`).
- Missing capability, partial/stale/foreign static bridge, foreign hooks,
  grad, training, or a non-empty `blocks_replace` prevent deferral
  (`_producer_capability` returns all-`None` on any non-absent incomplete
  bridge; no static-to-dynamic fallback).
- Pre-claim `try_*` miss → materialize once → stock attention.
- Post-claim failure → raise, no materialize.
- Any `double_block` replacement makes `blocks_replace` non-empty, so every
  transition is disabled; the wrapper still runs the original block
  (`test_any_block_replacement_disables_crossblock_and_preserves_wrapper`).
  If a pending exists when a replacement is hit, it is materialized first
  (`model.py:686-698`).

## Streams

- Adapter uses the current PyTorch CUDA stream. It does not pin stream 0.
- `cast_bias_weight(..., offloadable=True)` for norm and QKV linear is
  paired with `uncast_bias_weight` in `finally` (`:679-750`, `:774-809`).
  Success order is cast-norm → quantize → uncast-norm → cast-linear →
  linear → uncast-linear
  (`test_success_claims_once_and_projects_qkv`).
- `gate_cuda.py` activates the requested stream, asserts
  `current_stream == requested`, and synchronizes that stream. Receipt
  `gate_crossblock_20423.json`: default stream `0` and non-default
  `198432388320064`, both `pass: true`, 0 mismatches.

## Identity

- `_loop_owned_blocks_replace` (`model.py:440-454`) captures or allocates
  one mapping per DiT forward and passes that object into every block.
  `dict.get("dit", {})` is not re-derived inside a block.
- Carrier `is`-binds previous/current blocks, residual, MLP, gate,
  segments, row map, `blocks_replace`, loop token, native, producer, route,
  and static scale. Fresh `{}` fails
  `pending._blocks_replace is blocks_replace`
  (`test_fresh_empty_blocks_replace_is_not_the_sealed_mapping`,
  `test_loop_passes_one_owned_mapping_when_no_replacement_is_configured`).

## Test coverage vs claims

Covered: seal, one-use claim/materialize, pre-claim None, post-claim raise,
static-only vs partial-bridge, hook on block breaks both neighbors,
replacement disables all transitions, loop-owned mapping, stock default,
cast/uncast pairing, 20,423-row default + non-default stream exactness.

Gaps (residual, not a four-axis hole):

- Lab loop tests import candidate `model.py`, not live cache-wrapped
  `_run_dit_blocks`.
- No unit test that live `auto` + cache object calls `cache.disable`.
- Node does not refuse the combination; loop silently drops cache.
- 0.511881 s remains a kernel projection; no E2E delta this tick.

## Verdict

VERDICT: GO

Residual risks: rebase lab/integration `model.py` onto the live cache
wrapper or add a live-loop mix test before treating overlay re-apply as
identity-preserving; keep serving default `stock`; measure E2E only after
a named smoke/FixedRef run on an idle `:18188`.
