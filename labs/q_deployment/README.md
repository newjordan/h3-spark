# Promoted H3 Q deployment bundle

Status: **qualified and promoted**.

This source-only bundle freezes the Sage prequantized-Q/output-amax producer,
H3 Q/K RMSNorm+rot96-RoPE Q-INT8 producer ABI v3, supplied-amax NVFP4 consumer,
and narrow ComfyUI integration patch. Native binaries are not included.

Exact build IDs:

- `h3-q-rms-rope96-qint8-v3-torch2.13-sm120`
- `h3-nvfp4-supplied-amax-v1-torch2.13-sm120`
- `h3-sage-fused-current-stream-v1-torch2.13-sm120`
- `h3-sage-qattn-output-amax-v2-torch2.13-sm120`

`QUALIFICATION_RECEIPT.json` is the final 20,423-row one-shot gate. It pins the
four deployed artifact hashes and records exact default/non-default-stream
producer, attention, marker, consumer, and version equality. The gate exited
zero in 2.8 seconds.

Apply patches to disposable clean source trees in this order:

1. `sageattention_h3_prequant_output_amax.patch`
2. `q_native_v2_to_v3.patch`
3. `nvfp4_supplied_amax_build_id.patch`
4. `comfy_h3_q_integration.patch`

Set `H3_PYTHON` to the target ComfyUI Python before using the release scripts.
The validators enforce exact Torch/CUDA/source/build IDs, SM120 cubins,
release flags, RUNPATH, entrypoints, and normalized-SASS transfer.
