# Static gate -> RMS/AdaLN -> FC1 NVFP4

Status: source/native route integrated and CPU-green; final v3 rebind and exact
acceptance render GO.

This FC1-only producer combines the qualified BF16 attention-residual gate,
qualified RMS reduction, AdaLN staging, and supplied-scale NVFP4 packing. The
rounded gated row remains in registers, eliminating one 219,588,096-byte BF16
read at 20,423 rows.

The one approved no-warmup, one-repetition gate took 1.231 seconds wall time.
It was byte exact for residual, rstd, FP4 data, and E4M3 scales, advanced the
residual version once, and measured 4.354976 -> 3.840608 ms/path. That is
0.514368 s projected across 50 FC1 paths and 20 steps.

`comfy_adapter.py` is the later five-attribute hardened wrapper. The compact
gate receipt records the pre-certificate adapter hash because the hardening
changed routing checks, not CUDA math. `resource_report.v1.json` records 64
registers, 1,044 bytes shared memory, and zero stack/local/spills for the
qualified SM120 cubin. The binary itself is intentionally excluded.
