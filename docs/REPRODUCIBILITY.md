# Reproducibility

## Qualified stack

- NVIDIA DGX Spark, GB10, compute capability 12.1, 48 SMs
- NVIDIA driver 580.173.02; CUDA runtime 13.0
- Python 3.13.12; PyTorch 2.13.0+cu130
- Triton 3.7.1; comfy-kitchen 0.2.27; SageAttention 2.2.0
- ComfyUI `6db4fa2fcd38a92d4e0364c917cd8503620e2a74`
- SageAttention `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`

The qualified checkpoint exposes 200 NVFP4 linears across 50 H3 blocks.
Weights and calibration payloads are not included.

## Environment

```bash
export H3_SPARK_ROOT="$PWD"
export COMFYUI_ROOT="${COMFYUI_ROOT:?set COMFYUI_ROOT to a target checkout}"
export H3_PYTHON="${H3_PYTHON:-$COMFYUI_ROOT/.venv/bin/python}"
```

Install the overlay only into a clean checkout of the pinned ComfyUI revision:

```bash
./scripts/install_comfyui_overlay.sh "$COMFYUI_ROOT"
```

## CPU/source contracts

```bash
(cd labs/q_rms_rope_int8 && PYTHONPATH="$PWD" \
  CUDA_VISIBLE_DEVICES="" "$H3_PYTHON" -m pytest -q test_contract.py)

(cd labs/nvfp4_static && CUDA_VISIBLE_DEVICES="" \
  "$H3_PYTHON" -m pytest -q test_static_scales.py)

(cd labs/rms_adaln_nvfp4_static && \
  PYTHONPATH="$PWD:$COMFYUI_ROOT" CUDA_VISIBLE_DEVICES="" \
  "$H3_PYTHON" -m pytest -q test_contract.py test_comfy_adapter.py)

(cd labs/gate_rms_adaln_nvfp4_static && \
  PYTHONPATH="$PWD:$COMFYUI_ROOT" CUDA_VISIBLE_DEVICES="" \
  "$H3_PYTHON" -m pytest -q test_comfy_adapter.py)

PYTHONPATH="$H3_SPARK_ROOT/integration/comfyui:$COMFYUI_ROOT" \
  CUDA_VISIBLE_DEVICES="" "$H3_PYTHON" -m pytest -q \
  labs/q_static_coexistence/test_plan_hook_certificate.py \
  integration/comfyui/tests/test_minimax_h3_q_static_coexistence.py \
  integration/comfyui/tests/test_minimax_h3_linear_nvfp4.py \
  integration/comfyui/tests/test_minimax_h3_swiglu_nvfp4.py
```

The cross-block candidate has its own isolated source tree and remains HOLD:

```bash
(cd labs/crossblock_gate_qkv && \
  COMFYUI_ROOT="$COMFYUI_ROOT" PYTHONPATH="$COMFYUI_ROOT" \
  CUDA_VISIBLE_DEVICES="" "$H3_PYTHON" -m pytest -q \
  test_comfy_adapter.py test_model_integration.py)
```

## Native builds

Build against the target runtime; never copy a binary from a different Python
prefix. Generated products belong in ignored `build/` directories.

```bash
export H3_TORCH_CMAKE="$($H3_PYTHON -c \
  'import torch; print(torch.utils.cmake_prefix_path)')"

for lab in q_rms_rope_int8 rms_adaln_nvfp4_static \
           gate_rms_adaln_nvfp4_static; do
  cmake -S "labs/$lab" -B "labs/$lab/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE="$H3_PYTHON" \
    -DCMAKE_PREFIX_PATH="$H3_TORCH_CMAKE" \
    -DH3_TORCH_CUDA_ARCH_LIST=12.0
  cmake --build "labs/$lab/build" --parallel
done
```

`scripts/build_native.sh` configures every `labs/**/CMakeLists.txt` project
into that project's `build/` directory. `scripts/build_native.sh --list`
prints the projects and does not configure or compile. Set
`H3_TORCH_CUDA_ARCH_LIST` to override each project's default architecture.

`labs/q_deployment/rebuild_release.sh` and `validate_release.sh` freeze the
Sage/Q/consumer rebuild and identity checks. They require `H3_PYTHON` and
explicit source/binary arguments; they contain no machine defaults.

Lab gates do not fall back to a home-directory checkout. Set `COMFYUI_ROOT`
before a gate that imports ComfyUI. SwiGLU serving comparisons need
`H3_SWIGLU_NVFP4_LIBRARY`. The inline-G comparison reads the candidate from
`labs/sage_output_amax/nvfp4_consumer/build/` (or
`H3_SWIGLU_NVFP4_CANDIDATE`) and the previous library from
`H3_SWIGLU_NVFP4_REFERENCE_LIBRARY`. Profile gates need `H3_PROFILE_TXT` and
`H3_PROFILE_JSON`; those profile files are not in this repository.
`labs/q_rms_rope_int8/gate_release_parity.py` also needs `H3_SAGE_ROOT` and
`H3_NVFP4_CONSUMER_ROOT`, and looks for the Q library in
`labs/q_rms_rope_int8/build/` unless `H3_Q_RMS_ROPE_INT8_LIBRARY` is set.

## Seconds-scale CUDA gates

Run correctness before timing and only at the exact target shape. The release
Q parity script intentionally has no warmup or timing loop:

```bash
export H3_SAGE_ROOT="${H3_SAGE_ROOT:?set patched SageAttention root}"
export H3_NVFP4_CONSUMER_ROOT="${H3_NVFP4_CONSUMER_ROOT:?set consumer root}"
export H3_SWIGLU_NVFP4_LIBRARY="${H3_SWIGLU_NVFP4_LIBRARY:?set consumer binary}"
"$H3_PYTHON" labs/q_rms_rope_int8/gate_release_parity.py

(cd labs/gate_rms_adaln_nvfp4_static && \
  COMFYUI_ROOT="$COMFYUI_ROOT" "$H3_PYTHON" gate_cuda.py \
    --rows 20423 --warmup 0 --reps 1 --output gate_static_20423.json)

(cd labs/crossblock_gate_qkv && \
  COMFYUI_ROOT="$COMFYUI_ROOT" "$H3_PYTHON" gate_cuda.py \
    --rows 20423 --output gate_crossblock_20423.json)
```

Do not repeat a full render for each source change. Use CPU/source checks and
seconds gates until the source and artifact are frozen, then run one exact
acceptance render.

## Static activation and rollback

Static mode requires a v3 artifact bound to the exact model/source/checkpoint
and the four exclusions in `docs/PERFORMANCE.md`. Never reuse it after a
workload or source change. A conservative rollback is:

```bash
export H3_ATTENTION=stock
export H3_MODULATION_FUSION=stock
export H3_SWIGLU_NVFP4_FUSION=stock
export H3_RMS_ADALN_NVFP4_FUSION=stock
export H3_Q_RMS_ROPE_INT8_FUSION=stock
unset H3_NVFP4_STATIC_ARTIFACT
```

The portable timing runner refuses a busy queue and accepts caller-owned
workflow, prompt, reference paths, and static artifact inputs. Use its
`--sampler-only` mode for short serving-path checks and reserve a complete
render for final acceptance.
