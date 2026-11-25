---
license: other
library_name: comfyui
tags:
- comfyui
- video-generation
- minimax-h3
- nvfp4
- sageattention
- dgx-spark
- blackwell
- serving
- quantization
---

# MiniMax H3 serving source

Source overlay, native-kernel labs, and the serving environment for running
MiniMax H3 video generation through ComfyUI on one NVIDIA DGX Spark. This
Git repository does not contain model weights or datasets.

Canonical source: **https://github.com/newjordan/h3-spark**

The tree is the installable serving source. A separate weights package, when
one is published, is not required to read the code or to run the CPU tests
below. This pass did not download weights and did not run a GPU job, so the
README does not restate timings from earlier receipts.

## What is in the tree

| Path | Role |
|---|---|
| `integration/comfyui/` | Files copied onto a clean ComfyUI checkout |
| `labs/` | CUDA kernel sources, adapters, and lab gates |
| `config/h3_spark_serving.env` | Serving defaults |
| `scripts/install_comfyui_overlay.sh` | Revision-checked overlay install |
| `scripts/build_native.sh` | Configures every `labs/**/CMakeLists.txt` into `build/` |
| `scripts/h3_timed_render.py` | One-shot ComfyUI client that prints JSON |
| `docs/` | Provenance and recorded receipts from earlier runs |
| `assets/` | Demo video `knight_dragon_final.mp4` and chart `h3_spark_speed_chart.png` |
| `third_party_licenses/` | Upstream license texts, unchanged |
| `tests/` | CPU tests run by pytest and by `.github/workflows/ci.yml` |

Generated native binaries and CMake build trees are not part of the source.
`scripts/build_native.sh` writes them under each lab's `build/` directory,
which is gitignored.

## Serving behavior

`config/h3_spark_serving.env` is the serving default:

- SageAttention 2++ quality attention for the H3 model only (`H3_ATTENTION=sage2-quality`). Text-encoder and VAE attention stay on the ComfyUI backend.
- Exact segmented modulation (`H3_MODULATION_FUSION=exact`).
- Native NVFP4 packing for the SwiGLU and RMS/AdaLN paths when the matching library loads (`H3_SWIGLU_NVFP4_FUSION=auto`, `H3_RMS_ADALN_NVFP4_FUSION=auto`). A missing library fails closed to the stock op.
- Q RMSNorm + rot96 RoPE into Sage Q-INT8 when that library loads (`H3_Q_RMS_ROPE_INT8_FUSION=auto`).
- VAE same-tile batching, native SwiGLU `silu(gate)*up`, GPU fp16 host transfer, a persistent host decode buffer, unit-interval decode, and NHWC staging. Each one has a `COMFY_MINIMAX_H3_VAE_*` switch in the env file. VAE Q/K RMS + RoPE stays **off**.
- First-block cache stays **off** (`H3_CROSS_STEP_CACHE=off`).
- Cross-block final-gate to next QKV stays **stock**. The node accepts `auto`, and the lab candidate is not the serving default.
- Static NVFP4 activation scales stay **off**. `H3_NVFP4_STATIC_ARTIFACT` is empty until `labs/nvfp4_static/verify_promotion.py` exits 0 for an artifact you supply. Dynamic NVFP4 is the default that does not depend on that artifact.

The attention node accepts `quality`, `fast`, and `stock`. `sage3_kfold` is not in this tree; passing it raises, and it is not offered in the node.

Recorded GPU receipts under `docs/` and `labs/` are historical. Where a receipt and this env file disagree, the env file is what a new process exports. Read the receipt next to any timing claim. `docs/PERFORMANCE.md` and `docs/VALIDATION.md` are those notes.

## Requirements

CPU tests need Python 3.10 or newer and pytest. They do not import ComfyUI, PyTorch, or a GPU.

The serving stack recorded in `docs/REPRODUCIBILITY.md` and `docs/PROVENANCE.md` is:

- ComfyUI `6db4fa2fcd38a92d4e0364c917cd8503620e2a74` (GPL-3.0)
- comfy-kitchen `5aab6c4bdb2bb73f4021277a58cc62d6185a7aed` (Apache-2.0), noted there as the NVFP4 layout the native packers follow
- SageAttention `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5` (Apache-2.0)
- The Python, PyTorch, Triton, driver, and CUDA versions named in `docs/REPRODUCIBILITY.md`

Those pins were not re-validated on hardware in this pass. Native builds need the same interpreter ComfyUI uses, with PyTorch's CMake package visible to that interpreter.

## Setup

```bash
git clone https://github.com/newjordan/h3-spark
cd h3-spark

# Official upstream checkpoints. This does not fetch a pruned NVFP4 derivative.
hf download MiniMaxAI/MiniMax-H3 --local-dir models --include "Ref2VA/*" "FL2VA/*"
mkdir -p models/ComfyUI/models
cp -r models/Ref2VA models/ComfyUI/models/
cp -r models/FL2VA models/ComfyUI/models/

# Clean checkout of the pinned ComfyUI revision.
bash scripts/install_comfyui_overlay.sh /path/to/ComfyUI

export H3_PYTHON=/path/to/ComfyUI/.venv/bin/python
bash scripts/build_native.sh

set -a
source config/h3_spark_serving.env
set +a
```

`install_comfyui_overlay.sh` refuses a missing checkout, a non-git directory, the wrong `HEAD`, and a dirty worktree. It prints how many files it installed. It does not copy `integration/comfyui/tests/` or the overlay README. `H3_COMFYUI_REVISION` can override the pin for a deliberate experiment; the script warns when that override is not the qualified revision.

`build_native.sh --list` prints the CMake projects and does not configure them. `H3_TORCH_CUDA_ARCH_LIST` overrides the architecture each project defaults to. The script does not download weights.

## Run ComfyUI

From the ComfyUI checkout, with the env file sourced and `H3_PYTHON` on `PATH` as `python`:

```bash
python main.py --listen 127.0.0.1 --port 18188 \
  --models-directory /path/to/models/ComfyUI/models \
  --disable-pinned-memory --fp16-intermediates --highvram \
  --disable-dynamic-vram --fast fp16_accumulation cublas_ops autotune
```

That is the launch line this tree documents. It was not started here.

## Timed render

`scripts/h3_timed_render.py` loads a caller-supplied module that defines `workflow(...)`, submits one prompt to an idle ComfyUI, and prints JSON on stdout.

```bash
python3 scripts/h3_timed_render.py --preset serving --tag warm \
  --workflow-builder path/to/builder.py \
  --prompt "..." --refs ref_a.png,ref_b.png \
  --api http://127.0.0.1:18188
```

`--preset lab` (the default) is an A/B baseline: stock attention, exact modulation, stock SwiGLU and RMS fusions, Q fusion auto. `--preset serving` matches the env file, including quality attention and auto SwiGLU/RMS fusion. Cross-block fusion stays stock in both presets. An explicit flag overrides the preset.

Reference paths are split on commas and trimmed. The builder's prompt must contain node `127` (model), `124` and `126` (model consumers), and `136` (reference size). `--sampler-only` also expects node `125` and points a `PreviewAny` at it so the VAE and mux are not the output. A busy queue, a rejected submit, or an unreachable API prints a JSON error and returns non-zero instead of a traceback. The result object includes attention, both NVFP4 fusions, Q fusion, cross-block fusion, the preset, and sampler-only.

`--nvfp4-scales static` (and `calibrate` / `validate`) require `--nvfp4-prefix`. Calibrate and validate also require `--nvfp4-concept`.

## Tests

```bash
python -m pip install -e ".[test]"
python -m pytest
```

`make test` runs the same pytest suite. GitHub Actions workflow `.github/workflows/ci.yml` installs the test extra on Python 3.12, runs pytest, and byte-compiles `scripts/`, `integration/`, and `labs/`.

The ComfyUI overlay tests under `integration/comfyui/tests/` need a ComfyUI checkout and PyTorch. `scripts/run_comfy_cpu_pytest.py` forces ComfyUI's CPU arguments before pytest and exits with a short message if `comfy` cannot be imported. The CPU suite in `tests/` does not run those modules.

## Lab gates

Gates that import ComfyUI require `COMFYUI_ROOT`. They do not guess a checkout under the home directory.

- `H3_SWIGLU_NVFP4_LIBRARY` — serving SwiGLU library for the comparison gates.
- `H3_SWIGLU_NVFP4_CANDIDATE` — optional override for the inline-G candidate. The default path is `labs/sage_output_amax/nvfp4_consumer/build/h3_swiglu_nvfp4_native_v1.so`.
- `H3_SWIGLU_NVFP4_REFERENCE_LIBRARY` — previous serving library for that comparison.
- `H3_PROFILE_TXT` and `H3_PROFILE_JSON` — profile gates. Those profile files are not in the repository.
- `H3_SAGE_ROOT`, `H3_NVFP4_CONSUMER_ROOT`, and optionally `H3_Q_RMS_ROPE_INT8_LIBRARY` — `labs/q_rms_rope_int8/gate_release_parity.py`. The Q library defaults to `labs/q_rms_rope_int8/build/`.
- `H3_LIVE_MODEL` — `scripts/merge_identity_crossblock.py`. Required. The script writes both the overlay copy and that live file.

`labs/nvfp4_static/verify_promotion.py PREFIX` prints JSON and exits 0 only when the artifact passes. Exit 2 is a fail-closed refusal.

## License and attribution

- **Model weights**, if you obtain them separately, are under their upstream terms. Modified MiniMax H3 derivatives are licensed under the MiniMax H3 Community License Agreement. Read it, including the Applicable Territory definition and Sections III and V, before use or redistribution. This Git tree does not vendor that agreement or the weights.
- Qwen3-VL-32B text encoder: Apache-2.0.
- ComfyUI portions of the serving source: GPL-3.0. comfy-kitchen and SageAttention portions: Apache-2.0. See `third_party_licenses/`.
- Original serving-config work: see `docs/PROVENANCE.md`. No additional repository-wide license is added here.
