# H3 BF16 segmented fusion results on DGX Spark

Measured on NVIDIA GB10 (SM 12.1), PyTorch 2.13.0+cu130 and Triton 3.7.1 with
CUDA events. Shape is the production `[20,393, 5,376]` packed sequence. The
table has 21 runs: 13 Qwen text/vision-tag runs, six reference-image runs,
target audio, and target video. Adjacent identical reference runs can be safely
coalesced, leaving 16 runs.

Command:

```bash
cd "$H3_SPARK_ROOT/labs/fusion"
"$H3_PYTHON" \
  benchmark_h3_fusions.py --warmup 3 --repetitions 10 --output results.json
```

## Median CUDA-event latency

| Operation | Median | Speedup | Estimated launches |
|---|---:|---:|---:|
| Current RMSNorm + 21-run AdaLN loop | 5.447 ms | 1.00x | 106 |
| Current loop, six refs coalesced | 5.551 ms | 0.98x | 81 |
| Torch RMSNorm + exact fused segmented AdaLN | 3.664 ms | 1.49x | 2 |
| Fully fused RMSNorm + AdaLN, staged BF16 | 1.800 ms | 3.03x | 1 |
| Fully fused RMSNorm + AdaLN, FP32 math | 1.836 ms | 2.97x | 1 |
| Comfy Kitchen Triton, 21 calls + cat | 3.903 ms | 1.40x | 22 |
| Current gated residual, 21 runs | 2.838 ms | 1.00x | 42 |
| Current gate, coalesced | 2.799 ms | 1.01x | 32 |
| Fused segmented gated residual | 2.625 ms | 1.08x | 1 |

The exact two-launch RMS/AdaLN path and fused gate reduce the four repeated
elementwise operations in each DiT block from about 16.57 ms to 12.58 ms. Over
50 blocks that is about 200 ms per model evaluation, or 3.99 seconds across a
20-step sample, before accounting for CPU launch-pressure benefits.

The aggressive one-launch RMS/AdaLN path plus fused gate reduces those same
operations to about 8.85 ms per block: about 386 ms per model evaluation or
7.72 seconds across 20 steps. Launches fall from an estimated 296 to four per
block (14,600 fewer launches per model evaluation).

## Correctness

- Torch RMSNorm + fused AdaLN is bit-exact with the current BF16 path across all
  109,632,768 output elements.
- The fused gated residual is also bit-exact across the full tensor.
- The aggressive staged-BF16 RMS/AdaLN is 99.99978% bit-exact. Mean absolute
  error is `1.09e-8`, with maximum absolute error `0.03125`; the remaining tiny
  difference comes from RMS reduction order.
- The one-round FP32-math kernel differs more from current BF16 staging, but is
  closer to the direct FP32 formula on a 128-row calibration: mean absolute
  error `0.001060` versus `0.002096` for current BF16, and max error `0.01556`
  versus `0.05205`.

## Comfy Kitchen finding

`comfy_kitchen.rms_adaln` has no norm-weight argument, so the comparison folds
the weight into `effective_scale = weight.float() * (1 + scale) - 1` and calls
the operation once per segment followed by `cat`. The Triton backend is finite
but changes BF16 rounding. The CUDA backend is disqualified on this GB10 shape:
only 99.6168% of outputs are finite (NaNs/Infs appear consistently for
`D=5,376`). This is reproducible from one row through the full sequence.

## Recommendation

The safe first integration candidate is Torch RMSNorm followed by
`segmented_adaln_bf16_`, plus `segmented_gate_bf16_`: it is bit-exact and still
removes 290 estimated launches per block. The fully fused staged-BF16 kernel is
the performance candidate after a model-output calibration, since it halves
the tensor traffic and is over 3x faster than the present RMS/AdaLN path.

`row_ids` is a 20,393-element `int32` table (about 80 KiB), built once for the
packed layout and passed to each block. It avoids hard-coding transient AdaLN
row numbers, which can be reindexed when timestep classes merge near an endpoint.
