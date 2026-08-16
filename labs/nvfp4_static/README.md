# H3 static NVFP4 scale plan

Status: source integrated; final source-bound rebind and acceptance render GO.

The v3 artifact machinery captures exact per-linear activation maxima, binds
them to model/source/checkpoint/workload identity, validates held-out traffic,
and applies static scales through a scoped `StaticScalePlan`.

The plan now publishes five bridge attributes. The fifth,
`_h3_nvfp4_plan_hook_ids`, contains the exact plan-owned forward-pre and
forward-post hook IDs. Consumers require those exact hook dictionaries, empty
backward hook dictionaries, and no process-global hooks. A pre-existing hook
prevents publication; a foreign hook added after activation invalidates the
bridge. All attributes and hooks are restored on context exit.

`final_rebind_receipt.json` records the path-neutral GO decision for source
`d7b196cc...`: 196 layers, 3,920 calls, max ratio 0.917149, and the four
required exclusions. The final render applied 196 static layers with four
dynamic fallbacks and measured 7.35 sampler seconds/step. Scale tensors and
media remain outside this source repository.

Schema v1/v2 artifacts are incompatible. Recalibrate after any source,
checkpoint, shape, prompt, reference, sampler, schedule, resolution, length,
or dtype change. Activation requires `verify_promotion.py` to exit zero and
the four dynamic exclusions recorded in `docs/PERFORMANCE.md`.

```bash
"$H3_PYTHON" -m pytest -q test_static_scales.py
"$H3_PYTHON" verify_promotion.py ARTIFACT_PREFIX
```

Calibration safetensors and concept media are intentionally excluded from
this repository.
