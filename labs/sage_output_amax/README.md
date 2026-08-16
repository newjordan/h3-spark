# SageAttention output-amax to NVFP4 prototype

> **Status: staged research code; not installed in live ComfyUI.**

This prototype removes the standalone full-tensor amax pass before each H3
attention output projection. The selected SageAttention epilogue already has
the final BF16 output tile in shared memory. It reduces the magnitude bits of
those exact bytes and emits one device-side `int32` scalar. The paired NVFP4
entry point consumes that scalar and performs only the packing pass.

At the FixedRef workload shape, `[20,423, 7,168]`, the supplied-amax consumer is
byte-exact with the existing dynamic path and saves 1.316 ms per output
projection call. Across 50 blocks and 20 denoising steps, that is a measured
kernel opportunity of about **1.316 seconds per render**. Adding the reduction
to the dominant attention kernel caused no measurable regression in the
isolated median: 66.625 ms staged versus 66.721 ms stock.

## Exactness contract

- The output is contiguous BF16 and every value is finite.
- The producer reads each exact 128-bit shared-memory chunk immediately before
  the existing global BF16 store; it does not recompute attention arithmetic.
- Clearing each BF16 sign bit and taking integer max is exactly equivalent to
  max-absolute-value ordering for finite BF16 values.
- One warp maximum is combined with `atomicMax` into a zero-initialized,
  one-element CUDA `int32` tensor.
- The consumer preserves the existing global-scale operation order and NVFP4
  packing/layout. Its packed bytes, scale bytes, and FP32 global scale must all
  compare with `torch.equal` against `bf16_nvfp4_dynamic`.
- NaN handling is deliberately outside this contract. Live integration must
  fail closed to the existing dynamic route whenever the contract is not met.

The prototype enables tracking only for H3's quality SageAttention path:
SM89/SM12-compatible QK-int8, SV-FP8, FP32 accumulation, fused V scale,
instruction buffer, head dimension 128, NHD BF16 output. The new kernel
template parameter defaults to `false`; every other compiled call site passes
a null marker pointer and retains its prior behavior.

## Source layout

- `sageattention/csrc/qattn/` is an overlay for SageAttention revision
  `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`. The bundled Apache-2.0 license
  applies to those modified upstream files.
- `nvfp4_consumer/` contains the additive
  `bf16_nvfp4_from_amax_bits(input, max_bits)` entry point, its Python guard,
  CMake build, and CPU structural tests.
- `bench_output_amax.py` gates producer exactness and attention latency.
- `bench_consumer.py` gates all three NVFP4 outputs and times the removed pass.
- `results_20423.json` records the full-shape measurements used here.

## Reproduce

Apply the SageAttention overlay to a detached checkout of the pinned revision,
then build it in place:

```bash
export H3_PYTHON=/path/to/comfyui/.venv/bin/python
export H3_SAGE=/path/to/SageAttention
cp -R sageattention/csrc/qattn/. "$H3_SAGE/csrc/qattn/"
(cd "$H3_SAGE" && TORCH_CUDA_ARCH_LIST=12.1 "$H3_PYTHON" setup.py build_ext --inplace)
PYTHONPATH="$H3_SAGE" "$H3_PYTHON" bench_output_amax.py \
  --seq 20423 --repeats 9 --expect-amax
```

Build and gate the consumer separately:

```bash
export H3_TORCH_CMAKE="$($H3_PYTHON -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake -S nvfp4_consumer -B nvfp4_consumer/cmake-build-120 \
  -DPython_EXECUTABLE="$H3_PYTHON" \
  -DCMAKE_PREFIX_PATH="$H3_TORCH_CMAKE" \
  -DH3_TORCH_CUDA_ARCH_LIST=12.0
cmake --build nvfp4_consumer/cmake-build-120 --parallel
(cd nvfp4_consumer && "$H3_PYTHON" -m unittest -v test_layout.py)
PYTHONPATH=nvfp4_consumer "$H3_PYTHON" bench_consumer.py \
  --seq 20423 --width 7168 --repeats 11
```

## Live-integration gate

Do not expose the scalar by silently overloading SageAttention's public LSE
return or by storing tensor-keyed global state. Production wiring should add an
explicit, H3-scoped `(output, output_amax_bits)` contract between the selected
attention wrapper and the immediately following `out_proj`. It must guard the
backend, layout, dtype, head dimension, contiguity, inference/no-grad state,
patches/hooks, and the consumer ABI, then fall back atomically to the current
dynamic quantizer. A full 20-step A/B render is still required before promotion.

No file in this lab modifies or installs into live ComfyUI.
