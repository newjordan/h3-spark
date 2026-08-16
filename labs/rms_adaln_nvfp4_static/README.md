# One-pass static RMS/AdaLN -> NVFP4

Status: source/native route integrated and CPU-green; final source-bound v3
rebind and exact acceptance render GO.

One 128-thread CTA owns a 5,376-wide row, reproduces the qualified RMS tree and
BF16 AdaLN boundary, then writes NVFP4 data and E4M3 block scales using the
validated scalar scale. At 20,423 rows the isolated exact gate measured
1.604224 ms versus 2.339328 ms for dynamic v2, about 1.470 s projected across
100 paths and 20 steps.

The adapter requires the complete five-attribute bridge, exact scale-object
identity, exact plan-owned hook certificate, source/runtime/native identity,
and all normal H3 owner/shape/dtype/device guards. Missing bridge attributes
delegate to dynamic v2; partial or stale bridges fail to stock before
mutation. Errors after mutation propagate.

`gate_129.json` and `gate_20423.json` are compact byte-equality evidence.
Generated libraries and calibration payloads are excluded.
