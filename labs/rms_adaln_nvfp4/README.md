# Exact RMSNorm + AdaLN to dynamic NVFP4 v2

This promoted MiniMax H3 kernel fuses the two remaining full-width activation
preparation stages on QKV and FC1 inputs. It reproduces the certified PyTorch
BF16 RMSNorm reduction tree, applies the exact staged BF16 AdaLN boundaries,
and writes the cuBLAS TensorCore NVFP4 layout without materializing a normalized
or modulated BF16 tensor.

The ComfyUI adapter is AOT-only and fail-closed. It requires the certified H3
width, shapes, epsilon, PyTorch build, SM12 device, trusted row-map identity,
unmodified RMSNorm and linear modules, inference/no-grad mode, and native ABI
and build ID. Ineligible calls take the already-integrated staged AdaLN path.
Native launch errors propagate; the adapter does not replay a possibly partial
CUDA operation.

## Qualified result

At the FixedRef production shape `[20,423, 5,376]`, the three-repeat CUDA-event
medians were:

| Path | Median |
|---|---:|
| RMSNorm plus staged dynamic AdaLN-to-NVFP4 | 4.231872 ms |
| Fused dynamic v2 | 2.339328 ms |
| Speedup | 1.809x |

The measured delta is 1.892544 ms for each of 100 eligible QKV/FC1 calls per
model evaluation: a projected 189.254 ms/evaluation, or about 3.785 seconds
across 20 evaluations. This is an operator-derived projection, not a measured
end-to-end render delta.

The production-shape gate was exact for row rstd, the materialized BF16 debug
boundary, packed FP4 bytes, E4M3 block-scale bytes, and dynamic global scale.
The 129-row finite hardening matrix covered FP32/BF16 modulation and
random/structured inputs. The eight-case NaN/+Inf matrix matched all
downstream bytes and global-scale bits. Evidence is limited to:

- `gate_results_v2_129_matrix.json`
- `gate_results_v2_20423.json`
- `gate_results_v2_exceptional.json`

The qualified binary is not committed. Its recorded SHA-256 is
`1038f12df9ebec597cf00f50511c32b088f14799fd2a0833dd8659b7ea1e4955`;
the required ABI is 2 and build ID is
`h3-rms-adaln-nvfp4-sm12-safe-row-v2`.

## Build and gate

```bash
export H3_PYTHON="$COMFYUI_ROOT/.venv/bin/python"
export H3_TORCH_CMAKE="$(
  "$H3_PYTHON" -c 'import torch; print(torch.utils.cmake_prefix_path)'
)"

cmake -S labs/rms_adaln_nvfp4 -B labs/rms_adaln_nvfp4/cmake-build-120 \
  -DPython_EXECUTABLE="$H3_PYTHON" -DCMAKE_PREFIX_PATH="$H3_TORCH_CMAKE" \
  -DH3_TORCH_CUDA_ARCH_LIST=12.0
cmake --build labs/rms_adaln_nvfp4/cmake-build-120 --parallel

"$H3_PYTHON" labs/rms_adaln_nvfp4/check_cuda.py \
  --rows 129 --modulation-dtype both --corpus both --warmup 2 --reps 3
"$H3_PYTHON" labs/rms_adaln_nvfp4/check_cuda.py \
  --rows 20423 --modulation-dtype fp32 --corpus random --warmup 2 --reps 3
```

Run the short correctness gates on the target GPU only after CPU/source review.
Do not substitute a diffusion render for these exact byte gates.
