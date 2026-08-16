# Native CUDA BF16 SwiGLU -> NVFP4 prototype

This isolated prototype fuses the MiniMax H3 MLP activation with activation
quantization. It accepts contiguous BF16 `raw[S, 28672]`, treats the first
14336 columns as the gate and the second 14336 as the up arm, computes
`silu(gate) * up`, and directly emits the representation expected by the
NVFP4 GEMM:

- packed uint8 FP4: `[roundup(S, 16), 7168]`;
- E4M3 block scales: `[roundup(S, 128), 896]`;
- 16 values per scale, high-nibble-first FP4 packing;
- cuBLAS scale swizzle `(row_block, col_block_group, 32, 4, 4)`.

The kernel mirrors comfy-kitchen v0.2.27's `quantize_nvfp4.cu` and
`float_utils.cuh`. In particular, the supplied FP32 scalar is a **global
decode scale**: each local scale is `E4M3((block_absmax / 6) / global_scale)`,
and FP4 encoding uses the rounded E4M3 value. The supplied-scale entry point
does no tensor-wide reduction.

A second, exact-dynamic entry point uses two CUDA passes. Pass 1 computes the
BF16-rounded SwiGLU maximum and preserves ComfyUI's operation order,
`(BF16 amax / 2688.0).to(float32)`; pass 2 recomputes SwiGLU and performs the
same fused quantization. This avoids writing and rereading the 558 MiB BF16
activation at 20,423 rows, at the cost of evaluating SwiGLU twice.

The same extension now also contains a direct BF16-to-NVFP4 path for the
remaining launch-heavy H3 activations. It is deliberately restricted to the
three exact H3 input widths:

- `5376`: QKV projection and FC1 input;
- `7168`: attention output projection input;
- `14336`: FC2 input when SwiGLU is not fused.

Its dynamic route uses two passes without materializing `abs(input)`: one
BF16-bit maximum reduction followed by quantization. The second pass derives
the stock BF16-rounded global scale inside the quantizer, avoiding a separate
scale-finalization launch. The supplied-scale route uses one pass. Both emit
the same high-nibble-first packed FP4 and cuBLAS-swizzled E4M3 scale layout as
comfy-kitchen.

By default the fusion reproduces the two BF16 storage roundings in ComfyUI's
eager `F.silu(gate).mul_(up)` path. An optional direct-FP32 mode is retained as
an accuracy experiment, but is not expected to be byte-identical to the eager
reference.

## Run

Use the same environment that provides PyTorch, CUDA 12.8+, and
comfy-kitchen 0.2.27:

```bash
H3_PYTHON=/path/to/ComfyUI/.venv/bin/python
H3_TORCH_CMAKE="$($H3_PYTHON -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake -S . -B cmake-build-120 \
  -DPython_EXECUTABLE="$H3_PYTHON" \
  -DCMAKE_PREFIX_PATH="$H3_TORCH_CMAKE" \
  -DH3_TORCH_CUDA_ARCH_LIST=12.0
cmake --build cmake-build-120 --parallel
```

The build resolves `libtorch_python` from the selected interpreter rather
than a hard-coded virtualenv. `H3_TORCH_CUDA_ARCH_LIST=12.0` produces the
SM120 family cubin used on GB10/SM121 Spark systems and prevents Torch's CMake
package from adding a host-probed architecture by accident.

Then run the SwiGLU comparison:

```bash
"$H3_PYTHON" benchmark.py \
  --correctness-rows 129 --rows 20423 --repeats 20 --dynamic
```

The loader prefers `cmake-build-120/h3_swiglu_nvfp4_native_v1.so`; if it is
missing, the first run JIT-compiles `swiglu_nvfp4_cuda.cu` and keeps those
artifacts in `native_cuda/build`. The script first checks packed bytes and scale bytes
against `F.silu(...).mul_(...)` followed by
`comfy_kitchen.quantize_nvfp4(..., pad_16x=True, hi_first=True)`, then measures
end-to-end CUDA-event latency for supplied-scale paths. With `--dynamic`, it
also compares the two-pass native dynamic path to eager SwiGLU + BF16 amax +
comfy-kitchen quantization. Results are written to
`native_cuda/results.json`.

The lightweight layout checks do not compile or use CUDA:

```bash
"$H3_PYTHON" -m unittest test_layout.py
```

For a small direct-path correctness gate (129 rows covers both row-padding
boundaries at all three H3 widths):

```bash
"$H3_PYTHON" check_direct.py --rows 129
```

Timing is optional and hard-capped at three samples by the harness:

```bash
"$H3_PYTHON" check_direct.py \
  --rows 129 --timing-width 5376 --timing-repeats 3
```

## Prototype status

The direct path remains intentionally isolated from ComfyUI while the current
H3 static-scale artifact is being calibrated. Before integration, check byte
equality on captured real QKV, attention-output, and FC1 inputs in addition to
the synthetic gate. The extension is
compiled without global `--use_fast_math` so eager-equivalent SiLU retains
PyTorch's precise `expf`; only the quantizer divisions explicitly use fast
division to mirror comfy-kitchen's build. If rare FP4 byte differences remain,
the likely sensitivity is PyTorch's exact SiLU implementation; scale-byte
equality is the most direct check of the blockwise contract.
