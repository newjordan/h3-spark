# Provenance and exclusions

This snapshot was curated on 2026-08-12 from an H3 optimization worktree. It
preserves source and compact measurements while removing host-local and bulky
artifacts.

## Upstream revisions

| Component | Upstream | Revision | License note |
|---|---|---|---|
| ComfyUI | `https://github.com/Comfy-Org/ComfyUI.git` | `6db4fa2fcd38a92d4e0364c917cd8503620e2a74` | GPL-3.0; source overlay includes modified upstream files. |
| comfy-kitchen | `https://github.com/Comfy-Org/comfy-kitchen.git` | `5aab6c4bdb2bb73f4021277a58cc62d6185a7aed` | Apache-2.0; native NVFP4 packing/layout follows v0.2.27 CUDA code. |
| SageAttention | `https://github.com/thu-ml/SageAttention.git` | `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5` | Apache-2.0; the promoted Q deployment includes a narrow source patch and the earlier source overlay retains its license. |

Copies of the ComfyUI and comfy-kitchen upstream license texts are under
`third_party_licenses/`. They describe upstream portions; they do not declare a
new repository-wide license for original campaign work. No `LICENSE` file is
added because ownership and redistribution terms for that original work were
not provided.

## Included

- The modified H3 ComfyUI source overlay and tests.
- Custom H3 attention/profiler nodes and a path-neutral timing runner.
- Triton and CUDA kernel source, loaders, build files, and structural tests.
- Promoted Q deployment patches/manifests and compact qualification receipt.
- Five-attribute static hook-certificate source/tests and an explicitly held
  cross-block candidate.
- Small correctness/timing JSON and text profiler summaries.
- A serving-default configuration with no credentials.

## Intentionally excluded

- Model checkpoints, weights, quantization sidecars, safetensors, and hashes
  that reveal machine-local storage layout.
- Native `.so` files, cubins, objects, CMake caches, and JIT/build trees.
- Raw profiler traces, caches, generated contact sheets, images, audio, videos,
  public-site assets, prompts, and reference concepts.
- Calibration artifacts bound to a specific private concept or source path.
- Queue receipts, paused queue state, absolute host paths, shell history,
  credentials, tokens, and environment secrets.
- Full third-party source checkouts; exact upstream URLs/revisions are recorded
  above instead.

The source queue file `paused_h3_queue_20260810.json` was not read, modified,
or copied into this repository.
