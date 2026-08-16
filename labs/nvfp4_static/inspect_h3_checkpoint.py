"""Header-only inspection of the MiniMax H3 NVFP4 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from safetensors import safe_open


DEFAULT_CHECKPOINT = os.environ.get("H3_CHECKPOINT")


def inspect_checkpoint(path: str | Path) -> dict:
    checkpoint = Path(path).resolve()
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        names = sorted(key[: -len(".comfy_quant")] for key in keys if key.endswith(".comfy_quant"))
        layers = []
        for name in names:
            weight = handle.get_slice(f"{name}.weight")
            weight_scale = handle.get_slice(f"{name}.weight_scale")
            weight_scale_2 = handle.get_slice(f"{name}.weight_scale_2")
            layers.append(
                {
                    "name": name,
                    "stored_weight_shape": list(weight.get_shape()),
                    "stored_weight_dtype": str(weight.get_dtype()),
                    "block_scale_shape": list(weight_scale.get_shape()),
                    "tensor_scale_shape": list(weight_scale_2.get_shape()),
                    "has_input_scale": f"{name}.input_scale" in keys,
                    "has_pre_quant_scale": f"{name}.pre_quant_scale" in keys,
                }
            )
    signature = json.dumps(layers, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "header_layer_sha256": hashlib.sha256(signature).hexdigest(),
        "nvfp4_layer_count": len(layers),
        "input_scale_count": sum(layer["has_input_scale"] for layer in layers),
        "pre_quant_scale_count": sum(layer["has_pre_quant_scale"] for layer in layers),
        "layers": layers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output")
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error("checkpoint is required (or set H3_CHECKPOINT)")
    result = inspect_checkpoint(args.checkpoint)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
