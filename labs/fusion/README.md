# H3 segmented fusion lab

This directory is an isolated prototype; it does not modify the live ComfyUI tree.

`h3_fused_ops.py` contains BF16 Triton candidates for H3's two repeated segmented
elementwise paths:

- weighted RMSNorm + AdaLN modulation in one launch;
- an exact segmented AdaLN kernel to pair with PyTorch RMSNorm in two launches;
- in-place gated residual accumulation in one launch.

`benchmark_h3_fusions.py` uses the exact 20,393 x 5,376 production shape and its
21 modulation runs. Run it with the live ComfyUI environment:

```bash
cd "$H3_SPARK_ROOT/labs/fusion"
"$H3_PYTHON" benchmark_h3_fusions.py
```

The benchmark compares the current PyTorch loops, the safely coalesced 16-run
loops, Comfy Kitchen's CUDA `rms_adaln` with the norm weight folded into its
scale, and the custom kernels. Results are written to `results.json`.

See `RESULTS.md` for the measured summary and integration recommendation.
