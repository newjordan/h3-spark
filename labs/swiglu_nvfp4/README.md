# Fused BF16 SwiGLU to NVFP4 lab

Isolated Triton prototype for MiniMax H3's MLP down-projection input. It reads
the `[gate | up]` BF16 output of FC1 and directly emits packed NVFP4 bytes plus
the cuBLAS-swizzled E4M3 block-scale tensor using a supplied static FP32 scale.
It does not patch ComfyUI.

Run:

```bash
cd "$H3_SPARK_ROOT/labs/swiglu_nvfp4"
"$H3_PYTHON" benchmark.py
```

The baseline is the production operation order: eager BF16 SwiGLU followed by
the comfy-kitchen CUDA `quantize_nvfp4` implementation with the same supplied
scale. Results are written to `benchmark_results.json`.
