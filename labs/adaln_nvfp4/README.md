# H3 exact-staged AdaLN -> NVFP4 prototype

This isolated prototype targets the two 5,376-wide NVFP4 inputs in every H3
DiT block: `attn.qkv_proj` and `mlp.fc1`. It consumes the already materialized
BF16 result of the model's stock weighted RMSNorm, applies segmented AdaLN, and
emits the packed NVFP4 activation directly.

Keeping stock RMSNorm outside the fusion is deliberate. Its reduction order is
backend-specific; recomputing it in a custom kernel would make a bit-exact claim
unsafe. The staged boundary preserves the production RMS result exactly while
still removing the full AdaLN activation and one launch.

## Rounding contract

For every logical element, the CUDA kernel performs these storage boundaries:

```text
s  = BF16(scale[row_id, col])
m  = BF16(1.0f + FP32(s))
p  = BF16(FP32(normed[row, col]) * FP32(m))
sh = BF16(shift[row_id, col])
y  = BF16(FP32(p) + FP32(sh))
```

This matches H3 exact mode:

```python
h.mul_(1.0 + scale[row].to(torch.bfloat16))
h.add_(shift[row].to(torch.bfloat16))
```

Scale and shift tables may be BF16 or FP32 and need only have unit feature
stride. That supports the noncompact chunks returned by `AdalnProj`, whose row
stride is `6 * 5376`. `row_ids` remains the existing contiguous int32 map.

The dynamic path then preserves `TensorCoreNVFP4Layout`'s scalar ordering:

```text
global_decode_scale = BF16(BF16(amax(y)) / 2688.0) widened to FP32
```

Packing is 16 values per E4M3 block scale, high nibble first, with the cuBLAS
scale swizzle `(row_block, col_block_group, 32, 4, 4)`. A supplied-scale entry
point is included for calibrated/static experiments; dynamic is the production
target.

## Payoff at the fixed-reference workload target shape

For `S=20,423, D=5,376`, one BF16 activation is 219,588,096 bytes. After the
stock RMS output, the current dynamic path moves the large activation four
times:

```text
AdaLN read + AdaLN write + amax read + quantizer read = 878,352,384 bytes
```

The staged fusion reads the RMS output once in its fused amax pass and once in
its fused quantizer pass:

```text
fused amax read + fused quantizer read = 439,176,192 bytes
```

It therefore removes 439,176,192 bytes and one kernel launch at each QKV/FC1
site. Across 50 blocks, two sites, and 20 denoising steps, the upper-bound
traffic reduction is 878,352,384,000 bytes (878.35 GB) and 2,000 launches per
render. At Spark's nominal 273 GB/s this traffic alone has a 3.22-second
roofline; cache effects and recomputing the cheap AdaLN expression in two passes
will reduce the realized gain.

This does not fuse the attention output. Doing that exactly would require
modifying the selected attention backend's epilogue and coordinating a global
amax before the out projection. The local staged QKV/FC1 fusion is both broader
and substantially lower risk.

## Validation

CPU-only structural and rounding tests do not load CUDA:

```bash
cd "$H3_SPARK_ROOT/labs/adaln_nvfp4"
"$H3_PYTHON" -m unittest -v \
  test_integration_reference.py native_cuda.test_contract
```

Build without running a serving workload:

```bash
H3_PYTHON=/path/to/ComfyUI/.venv/bin/python
H3_TORCH_CMAKE="$($H3_PYTHON -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake -S native_cuda -B native_cuda/cmake-build-120 \
  -DPython_EXECUTABLE="$H3_PYTHON" \
  -DCMAKE_PREFIX_PATH="$H3_TORCH_CMAKE" \
  -DH3_TORCH_CUDA_ARCH_LIST=12.0
cmake --build native_cuda/cmake-build-120 --parallel
```

When the GPU is available, the small gate uses 129 rows to exercise both output
padding boundaries, BF16 and FP32 modulation tables, and noncompact table row
strides:

```bash
cd "$H3_SPARK_ROOT/labs/adaln_nvfp4/native_cuda"
"$H3_PYTHON" check_cuda.py --rows 129
```

Promotion requires zero packed-byte mismatches, zero scale-byte mismatches, and
an exactly equal dynamic global scale for both modulation dtypes. It should then
be repeated on captured real `norm1`/`norm2` outputs before any model wiring.

## Integration sketch

The narrow integration point is before `qkv_proj` and `fc1`, not inside shared
ComfyUI quantization:

1. Extend the existing H3 native facade with an additive
   `adaln_nvfp4_dynamic(normed, scale, shift, row_ids)` capability.
2. At `DiTBlock`, keep `norm1(x)` and `norm2(x)` stock. In exact mode, pass the
   RMS result plus the corresponding modulation tables and row map to a guarded
   H3 NVFP4 linear helper.
3. Construct the same `QuantizedTensor` layout already used by the direct
   BF16/NVFP4 and SwiGLU/NVFP4 paths, then call the original linear with that
   quantized input.
4. Fail closed for training, gradients, patches, low-VRAM functions, unsupported
   weights, wrong dtype/shape/stride, missing row ids, hooks, or a missing native
   entry point. Fall back to `_norm_mod(...)` followed by the normal linear.
5. Do not enable it for `RefinerBlock`; those blocks have no segmented AdaLN.

No live ComfyUI source or server state is modified by this lab.
`comfy_integration_reference.py` contains the corresponding non-installed
adapter, including training, gradient, patch, low-VRAM, hook, weight-layout,
static-certificate, shape, dtype, device, and stride guards.
