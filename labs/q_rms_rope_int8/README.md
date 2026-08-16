# H3 Q/K RMS-RoPE -> Sage Q INT8

Status: **promoted**.

The v3 producer fuses H3 Q/K RMSNorm, split-half rot96 RoPE, and Sage's
per-warp Q INT8 quantization. It preserves K as the exact BF16 in-place output
expected by Sage smooth-K and emits Sage-compatible Q INT8 plus FP32 scales.
The route is additive, H3-only, inference-only, sequence-length guarded, and
fails closed on runtime, owner, hook, patch, shape, stride, dtype, device,
storage, ABI, or build-ID mismatches.

At 20,423 rows the qualified 16-warp producer reduced 7.417600 to 6.193312 ms
per block path, a 1.224288 ms delta or about 1.224 s across 50 blocks and 20
steps. The final one-shot v3 release gate also proved exact Q, Q scale, K,
attention output, output-amax marker, supplied-amax consumer output, and
version deltas on default and non-default streams in 2.8 seconds.

Important files:

- `h3_q_rms_rope_int8.py`, `q_rms_rope_int8_cuda.cu`: promoted facade/source.
- `results_production_s20423.json`: v2 producer timing/equality gate whose
  normalized SASS transfers to v3.
- `results_attention_production_s20423.json`: exact full-attention gate.
- `release_qualification_s20423.json`: final v3/Sage/consumer release receipt.
- `gate_release_parity.py`: deliberately one-shot release parity script.

Build with the target ComfyUI Python and run only after setting
`COMFYUI_ROOT`; generated binaries stay in the ignored `build/` directory.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$H3_PYTHON" \
  -DCMAKE_PREFIX_PATH="$($H3_PYTHON -c \
    'import torch; print(torch.utils.cmake_prefix_path)')" \
  -DH3_TORCH_CUDA_ARCH_LIST=12.0
cmake --build build --parallel
```
