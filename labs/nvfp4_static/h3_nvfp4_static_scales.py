"""Calibration and scoped application of static NVFP4 activation scales.

This module deliberately does not patch ComfyUI.  The small adapter in
``nodes.py`` attaches the context managers here to a cloned ModelPatcher.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import datetime as dt
import fcntl
import hashlib
import inspect
import json
import math
import os
import statistics
import tempfile
import threading
import weakref
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file


SCHEMA_NAME = "comfy.h3.nvfp4_input_scales"
SCHEMA_VERSION = 3
ARTIFACT_TAG = "nvfp4-input-scales.v3"
VALIDATION_TAG = "nvfp4-input-scales.validation.v1"

# comfy-kitchen's E4M3 tensor scale (448) times E2M1 element max (6).
# This is the denominator used by comfy.quant_ops.TensorCoreNVFP4Layout.
NVFP4_SCALE_DENOMINATOR = 2688.0
H3_MODEL_CLASS = "comfy.ldm.minimax.model.MiniMaxH3Model"
FUSION_ENABLED_ATTR = "_h3_nvfp4_fusion_enabled"
FUSION_ALLOWED_SHAPES_ATTR = "_h3_nvfp4_allowed_shapes"
FUSION_ALLOWED_DTYPES_ATTR = "_h3_nvfp4_allowed_dtypes"
FUSION_VALIDATED_SCALE_ATTR = "_h3_nvfp4_validated_input_scale"
FUSION_PLAN_HOOK_IDS_ATTR = "_h3_nvfp4_plan_hook_ids"

_LAYER_IDENTITY_FIELDS = (
    "name",
    "module_class",
    "quant_format",
    "layout_type",
    "in_features",
    "out_features",
    "logical_weight_shape",
    "storage_weight_shape",
    "pre_quant_scale_shape",
)

_MODEL_LOCKS: weakref.WeakKeyDictionary[torch.nn.Module, threading.RLock] = weakref.WeakKeyDictionary()
_MODEL_LOCKS_GUARD = threading.Lock()
_FILE_HASH_CACHE: dict[tuple[str, int, int], str] = {}
_FILE_HASH_CACHE_GUARD = threading.Lock()


class ArtifactError(RuntimeError):
    pass


def _model_lock(root: torch.nn.Module) -> threading.RLock:
    with _MODEL_LOCKS_GUARD:
        lock = _MODEL_LOCKS.get(root)
        if lock is None:
            lock = threading.RLock()
            _MODEL_LOCKS[root] = lock
        return lock


def class_name(value: Any) -> str:
    cls = value.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def diffusion_root(model_or_patcher: Any) -> torch.nn.Module:
    """Return the diffusion module from a ModelPatcher, BaseModel, or module."""
    value = getattr(model_or_patcher, "model", model_or_patcher)
    return getattr(value, "diffusion_model", value)


def is_nvfp4_linear(module: torch.nn.Module) -> bool:
    return (
        getattr(module, "quant_format", None) == "nvfp4"
        and getattr(module, "layout_type", None) == "TensorCoreNVFP4Layout"
        and hasattr(module, "in_features")
        and hasattr(module, "out_features")
    )


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    return [int(v) for v in shape] if shape is not None else None


def _source_sha256(module: torch.nn.Module) -> str | None:
    source = inspect.getsourcefile(module.__class__)
    if source is None:
        return None
    path = Path(source)
    if not path.is_file():
        return None
    return file_sha256(path)


def file_sha256(path: str | os.PathLike[str]) -> str:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    key = (str(resolved), stat.st_size, stat.st_mtime_ns)
    with _FILE_HASH_CACHE_GUARD:
        cached = _FILE_HASH_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    with _FILE_HASH_CACHE_GUARD:
        _FILE_HASH_CACHE[key] = value
    return value


def infer_checkpoint_path(model_or_patcher: Any) -> Path | None:
    cached = getattr(model_or_patcher, "cached_patcher_init", None)
    if not cached or len(cached) < 2 or not cached[1]:
        return None
    candidate = cached[1][0]
    if not isinstance(candidate, (str, os.PathLike)):
        return None
    path = Path(candidate).expanduser()
    return path.resolve() if path.is_file() else None


def checkpoint_descriptor(path: str | os.PathLike[str]) -> dict[str, Any]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": file_sha256(resolved),
    }


def _metadata_artifact_sha256(metadata: dict[str, Any]) -> str:
    value = {key: item for key, item in metadata.items() if key not in ("artifact_sha256", "scale_file")}
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def layer_descriptor(name: str, module: torch.nn.Module) -> dict[str, Any]:
    weight = getattr(module, "weight", None)
    qdata = getattr(weight, "_qdata", None)
    orig_shape = getattr(module, "_orig_shape", None)
    if orig_shape is None:
        orig_shape = getattr(getattr(weight, "_params", None), "orig_shape", None)
    if orig_shape is None:
        orig_shape = getattr(weight, "shape", None)
    pre_quant_scale = getattr(module, "pre_quant_scale", None)
    return {
        "name": name,
        "module_class": class_name(module),
        "quant_format": getattr(module, "quant_format", None),
        "layout_type": getattr(module, "layout_type", None),
        "in_features": int(module.in_features),
        "out_features": int(module.out_features),
        "logical_weight_shape": [int(v) for v in orig_shape],
        "storage_weight_shape": _shape(qdata),
        "pre_quant_scale_shape": _shape(pre_quant_scale),
    }


def inspect_nvfp4_model(model_or_patcher: Any) -> tuple[torch.nn.Module, dict[str, torch.nn.Module], dict[str, Any]]:
    root = diffusion_root(model_or_patcher)
    modules = {name: module for name, module in root.named_modules() if is_nvfp4_linear(module)}
    descriptors = [layer_descriptor(name, modules[name]) for name in sorted(modules)]
    identity = {
        "family": class_name(root),
        "source_sha256": _source_sha256(root),
        "nvfp4_layer_count": len(descriptors),
        "layers": descriptors,
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    identity["architecture_sha256"] = hashlib.sha256(payload).hexdigest()
    return root, modules, identity


def require_minimax_h3(identity: dict[str, Any]) -> None:
    if identity["family"] != H3_MODEL_CLASS:
        raise ArtifactError(f"Expected {H3_MODEL_CLASS}, got {identity['family']}")
    if identity["nvfp4_layer_count"] != 200:
        raise ArtifactError(
            f"Expected 200 MiniMax H3 NVFP4 linears, got {identity['nvfp4_layer_count']}"
        )


@dataclasses.dataclass
class LayerObservation:
    maxima: list[torch.Tensor] = dataclasses.field(default_factory=list)
    dynamic_scales: list[torch.Tensor] = dataclasses.field(default_factory=list)
    input_shapes: set[tuple[int, ...]] = dataclasses.field(default_factory=set)
    activation_dtypes: set[str] = dataclasses.field(default_factory=set)

    @property
    def calls(self) -> int:
        return len(self.maxima)


def _scalar_lists(
    observations: dict[str, LayerObservation],
    field: str = "maxima",
) -> dict[str, list[float]]:
    flat: list[tuple[str, torch.Tensor]] = []
    for name, observation in observations.items():
        flat.extend((name, value) for value in getattr(observation, field))
    if not flat:
        raise ArtifactError(f"No NVFP4 activation {field} were captured")

    grouped: dict[tuple[str, torch.dtype], list[tuple[str, torch.Tensor]]] = defaultdict(list)
    for name, value in flat:
        grouped[(str(value.device), value.dtype)].append((name, value))
    values_by_name: dict[str, list[float]] = defaultdict(list)
    for entries in grouped.values():
        values = torch.stack([value for _, value in entries]).to(torch.float32).cpu().tolist()
        for (name, _), value in zip(entries, values):
            values_by_name[name].append(float(value))
    return dict(values_by_name)


def _has_function_patch(module: torch.nn.Module) -> bool:
    """Whether ComfyUI will bypass the direct quantized-linear path."""
    return (
        bool(getattr(module, "weight_function", ()))
        or bool(getattr(module, "bias_function", ()))
        or getattr(module, "weight_lowvram_function", None) is not None
        or getattr(module, "bias_lowvram_function", None) is not None
    )


def _has_preexisting_module_hooks(module: torch.nn.Module) -> bool:
    """Whether a fused helper would bypass observable module hook behavior."""
    return any(
        bool(getattr(module, attribute, ()))
        for attribute in (
            "_forward_pre_hooks",
            "_forward_hooks",
            "_backward_pre_hooks",
            "_backward_hooks",
        )
    )


def _static_scale_with_margin(exact_dynamic_scale: float, margin: float) -> float:
    """Return the FP32 scalar used by the static quantizer.

    Margin 1 preserves the exact observed FP32 scalar. Larger margins are
    rounded exactly once to FP32, matching artifact calibration behavior.
    """
    if margin == 1.0:
        return exact_dynamic_scale
    return float(torch.tensor(exact_dynamic_scale * margin, dtype=torch.float32).item())


def _validated_record_shapes(
    record: dict[str, Any],
    *,
    expected_width: int,
) -> frozenset[tuple[int, int]]:
    raw_shapes = record.get("input_shapes")
    if not isinstance(raw_shapes, list) or not raw_shapes:
        raise ArtifactError("Layer input_shapes must be a non-empty list")
    shapes: set[tuple[int, int]] = set()
    for shape in raw_shapes:
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or not all(
                isinstance(dimension, int)
                and not isinstance(dimension, bool)
                and dimension > 0
                for dimension in shape
            )
            or shape[1] != expected_width
        ):
            raise ArtifactError(f"Invalid calibrated input shape: {shape!r}")
        shapes.add((shape[0], shape[1]))
    envelope = record.get("input_envelope")
    if not isinstance(envelope, dict):
        raise ArtifactError("Layer input_envelope must be an object")
    expected_envelope = {
        "feature_width": expected_width,
        "min_rows": min(rows for rows, _ in shapes),
        "max_rows": max(rows for rows, _ in shapes),
    }
    if envelope != expected_envelope:
        raise ArtifactError(
            f"Layer input_envelope {envelope!r} does not match shapes {expected_envelope!r}"
        )
    return frozenset(shapes)


def _validated_record_dtypes(record: dict[str, Any]) -> frozenset[str]:
    raw_dtypes = record.get("activation_dtypes")
    if (
        not isinstance(raw_dtypes, list)
        or not raw_dtypes
        or not all(isinstance(dtype, str) and dtype for dtype in raw_dtypes)
        or raw_dtypes != sorted(set(raw_dtypes))
    ):
        raise ArtifactError("Layer activation_dtypes must be a sorted, unique, non-empty list")
    return frozenset(raw_dtypes)


@dataclasses.dataclass
class _AttributeSnapshot:
    kind: str
    value: Any = None


def _snapshot_plain_attribute(module: torch.nn.Module, name: str) -> _AttributeSnapshot:
    """Capture an instance attribute without consulting class descriptors."""
    if name in module.__dict__:
        return _AttributeSnapshot("attribute", module.__dict__[name])
    return _AttributeSnapshot("absent")


def _assign_plain_attribute(module: torch.nn.Module, name: str, value: Any) -> None:
    """Install non-buffer metadata that DynamicVRAM will not back up or move."""
    setattr(module, name, value)


def _restore_plain_attribute(
    module: torch.nn.Module,
    name: str,
    snapshot: _AttributeSnapshot,
) -> None:
    if snapshot.kind == "attribute":
        setattr(module, name, snapshot.value)
    else:
        module.__dict__.pop(name, None)


def _snapshot_input_scale(module: torch.nn.Module) -> _AttributeSnapshot:
    if "input_scale" in module._parameters:
        return _AttributeSnapshot("parameter", module._parameters["input_scale"])
    if "input_scale" in module._buffers:
        return _AttributeSnapshot("buffer", module._buffers["input_scale"])
    if hasattr(module, "input_scale"):
        return _AttributeSnapshot("attribute", getattr(module, "input_scale"))
    return _AttributeSnapshot("absent")


def _assign_input_scale(module: torch.nn.Module, snapshot: _AttributeSnapshot, scale: torch.Tensor) -> None:
    if snapshot.kind == "parameter":
        module._parameters["input_scale"] = torch.nn.Parameter(scale, requires_grad=False)
    elif snapshot.kind == "buffer":
        module._buffers["input_scale"] = scale
    elif snapshot.kind == "attribute":
        setattr(module, "input_scale", scale)
    else:
        setattr(module, "input_scale", scale)


def _restore_input_scale(module: torch.nn.Module, snapshot: _AttributeSnapshot) -> None:
    if snapshot.kind == "parameter":
        module._parameters["input_scale"] = snapshot.value
    elif snapshot.kind == "buffer":
        module._buffers["input_scale"] = snapshot.value
    elif snapshot.kind == "attribute":
        setattr(module, "input_scale", snapshot.value)
    else:
        module._parameters.pop("input_scale", None)
        module._buffers.pop("input_scale", None)
        module._non_persistent_buffers_set.discard("input_scale")
        if "input_scale" in module.__dict__:
            del module.__dict__["input_scale"]


class ActivationCalibrator:
    """Observe all dynamic NVFP4 linear inputs during one sampling run.

    A pre-hook computes the same abs+amax used by the dynamic ComfyUI path and
    gives that exact per-call scale to the module.  Thus calibration does not
    pay for a duplicate reduction and does not intentionally perturb that run.
    The scalar maxima remain on device until the run is complete.
    """

    def __init__(
        self,
        model_or_patcher: Any,
        *,
        margin: float = 1.20,
        model_id: str = "minimax_h3_ref2va_pruned_nvfp4.safetensors",
        concept_path: str | os.PathLike[str] | None = None,
        checkpoint_path: str | os.PathLike[str] | None = None,
        require_h3: bool = True,
    ) -> None:
        if not math.isfinite(margin) or margin < 1.0:
            raise ValueError("margin must be finite and >= 1.0")
        self.root = diffusion_root(model_or_patcher)
        self._lock = _model_lock(self.root)
        # ModelPatcher clones share the module tree. Inspect and filter only while
        # holding the same lock used by every calibration/application scope, so a
        # concurrent temporary input_scale cannot be mistaken for model state.
        with self._lock:
            inspected_root, all_modules, self.identity = inspect_nvfp4_model(model_or_patcher)
            if inspected_root is not self.root:
                raise ArtifactError("Selected diffusion model changed during calibration setup")
            if require_h3:
                require_minimax_h3(self.identity)
            self.skipped_patched_modules = {
                name for name, module in all_modules.items() if _has_function_patch(module)
            }
            # Existing static scales already skip the dynamic reduction. Preserve
            # them. Function-patched linears bypass ComfyUI's direct quantized path
            # and must never contribute or receive a static activation scale.
            self.modules = {
                name: module
                for name, module in all_modules.items()
                if getattr(module, "input_scale", None) is None
                and name not in self.skipped_patched_modules
            }
        if not self.modules:
            raise ArtifactError("No dynamically-scaled NVFP4 linears found")
        self.require_h3 = require_h3
        self.margin = float(margin)
        self.model_id = model_id
        self.concept_path = Path(concept_path).resolve() if concept_path else None
        inferred_checkpoint = Path(checkpoint_path).expanduser() if checkpoint_path else infer_checkpoint_path(model_or_patcher)
        self.checkpoint_path = inferred_checkpoint.resolve() if inferred_checkpoint else None
        if require_h3 and self.checkpoint_path is None:
            raise ArtifactError(
                "MiniMax H3 calibration requires a file-backed ModelPatcher from UNETLoader"
            )
        self.observations: dict[str, LayerObservation] = {}
        self._handles: list[Any] = []
        self._snapshots: dict[str, _AttributeSnapshot] = {}

    def _pre_hook(self, name: str, module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        if not args or not isinstance(args[0], torch.Tensor):
            return
        if _has_function_patch(module):
            # Patches can be installed lazily when DynamicVRAM loads the model,
            # after constructor-time inspection. Omit the whole layer if it is
            # patched on any observed call.
            self.skipped_patched_modules.add(name)
            return
        value = args[0].detach()
        pre_quant_scale = getattr(module, "pre_quant_scale", None)
        if pre_quant_scale is not None:
            value = value * pre_quant_scale.to(device=value.device, dtype=value.dtype)
        if value.ndim >= 3:
            value = value.reshape(-1, value.shape[-1])
        if value.ndim != 2:
            return

        dtype_name = str(value.dtype)
        if self.require_h3 and value.dtype != torch.bfloat16:
            raise ArtifactError(
                f"MiniMax H3 NVFP4 calibration requires torch.bfloat16 activations; "
                f"{name} received {dtype_name}"
            )

        # Keep operation order identical to TensorCoreNVFP4Layout.quantize:
        # amax in activation dtype, division, then conversion to float32.
        current_max = torch.amax(value.abs())
        current_scale = (current_max / NVFP4_SCALE_DENOMINATOR).to(torch.float32)
        observation = self.observations[name]
        observation.maxima.append(current_max.detach())
        observation.dynamic_scales.append(current_scale.detach())
        observation.input_shapes.add(tuple(int(v) for v in value.shape))
        observation.activation_dtypes.add(dtype_name)
        _assign_input_scale(module, self._snapshots[name], current_scale)

    def _post_hook(self, name: str, module: torch.nn.Module, _args: tuple[Any, ...], _output: Any) -> None:
        _restore_input_scale(module, self._snapshots[name])

    def reset(self) -> None:
        self.observations = {name: LayerObservation() for name in self.modules}

    @contextlib.contextmanager
    def capture(self) -> Iterator["ActivationCalibrator"]:
        with self._lock:
            if self._handles:
                raise RuntimeError("Calibration capture is already active")
            self.reset()
            self._snapshots = {name: _snapshot_input_scale(module) for name, module in self.modules.items()}
            try:
                for name, module in self.modules.items():
                    self._handles.append(
                        module.register_forward_pre_hook(
                            lambda mod, args, layer_name=name: self._pre_hook(layer_name, mod, args)
                        )
                    )
                    self._handles.append(
                        module.register_forward_hook(
                            lambda mod, args, output, layer_name=name: self._post_hook(layer_name, mod, args, output),
                            always_call=True,
                        )
                    )
                yield self
            finally:
                for handle in self._handles:
                    handle.remove()
                self._handles.clear()
                for name, module in self.modules.items():
                    _restore_input_scale(module, self._snapshots[name])
                self._snapshots.clear()

    def records(self) -> dict[str, dict[str, Any]]:
        maxima_by_name = _scalar_lists(self.observations, "maxima")
        scales_by_name = _scalar_lists(self.observations, "dynamic_scales")
        records: dict[str, dict[str, Any]] = {}
        for descriptor in self.identity["layers"]:
            name = descriptor["name"]
            if name not in self.modules or name in self.skipped_patched_modules:
                continue
            observation = self.observations[name]
            values = maxima_by_name.get(name, [])
            dynamic_scales = scales_by_name.get(name, [])
            if len(values) != len(dynamic_scales):
                raise ArtifactError(f"Incomplete dynamic scale observations for {name}")
            maximum = max(values, default=None)
            if maximum is None or not math.isfinite(maximum) or maximum <= 0.0:
                raise ArtifactError(f"Invalid or missing activation maximum for {name}: {maximum}")
            max_dynamic_scale = max(dynamic_scales, default=None)
            if (
                max_dynamic_scale is None
                or not math.isfinite(max_dynamic_scale)
                or max_dynamic_scale <= 0.0
            ):
                raise ArtifactError(
                    f"Invalid or missing exact dynamic input scale for {name}: {max_dynamic_scale}"
                )
            dtypes = sorted(observation.activation_dtypes)
            if not dtypes:
                raise ArtifactError(f"Missing activation dtype for {name}")
            if self.require_h3 and dtypes != [str(torch.bfloat16)]:
                raise ArtifactError(f"Unexpected H3 activation dtypes for {name}: {dtypes}")
            minimum = min(values)
            median = statistics.median(values)
            min_dynamic_scale = min(dynamic_scales)
            median_dynamic_scale = statistics.median(dynamic_scales)
            # Preserve the exact observed FP32 scalar for margin=1. For a larger
            # margin, round the product once to the FP32 scalar the quantizer uses.
            input_scale = _static_scale_with_margin(max_dynamic_scale, self.margin)
            records[name] = {
                **descriptor,
                "calls": observation.calls,
                "activation_dtypes": dtypes,
                "input_shapes": [list(shape) for shape in sorted(observation.input_shapes)],
                "input_envelope": {
                    "feature_width": descriptor["in_features"],
                    "min_rows": min(shape[0] for shape in observation.input_shapes),
                    "max_rows": max(shape[0] for shape in observation.input_shapes),
                },
                "max_abs": maximum,
                "call_max_abs_stats": {
                    "min": minimum,
                    "median": median,
                    "max": maximum,
                    "min_over_max": minimum / maximum,
                    "median_over_max": median / maximum,
                },
                "max_dynamic_input_scale": max_dynamic_scale,
                "call_dynamic_input_scale_stats": {
                    "min": min_dynamic_scale,
                    "median": median_dynamic_scale,
                    "max": max_dynamic_scale,
                    "min_over_max": min_dynamic_scale / max_dynamic_scale,
                    "median_over_max": median_dynamic_scale / max_dynamic_scale,
                },
                "input_scale": input_scale,
            }
        missing = sorted(set(self.modules) - self.skipped_patched_modules - set(records))
        if missing:
            raise ArtifactError(f"Calibration did not observe {len(missing)} layers; first missing: {missing[0]}")
        return records

    def artifact(self) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        records = self.records()
        concept: dict[str, Any] | None = None
        if self.concept_path is not None:
            if not self.concept_path.is_file():
                raise ArtifactError(f"Concept image does not exist: {self.concept_path}")
            concept = {
                "path": str(self.concept_path),
                "sha256": file_sha256(self.concept_path),
            }

        now = dt.datetime.now(dt.timezone.utc).isoformat()
        tensors: dict[str, torch.Tensor] = {}
        for name, record in records.items():
            record["scale_tensor"] = f"input_scale.{name}"
            record["max_tensor"] = f"max_abs.{name}"
            record["dynamic_scale_tensor"] = f"max_dynamic_input_scale.{name}"
            tensors[record["scale_tensor"]] = torch.tensor(record["input_scale"], dtype=torch.float32)
            tensors[record["max_tensor"]] = torch.tensor(record["max_abs"], dtype=torch.float32)
            tensors[record["dynamic_scale_tensor"]] = torch.tensor(
                record["max_dynamic_input_scale"], dtype=torch.float32
            )

        tensor_payload_sha256 = _tensor_payload_sha256(tensors)
        metadata = {
            "schema": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "created_utc": now,
            "model": {
                "id": self.model_id,
                "checkpoint": checkpoint_descriptor(self.checkpoint_path) if self.checkpoint_path else None,
                **{key: value for key, value in self.identity.items() if key != "layers"},
            },
            "calibration": {
                "margin": self.margin,
                "nvfp4_scale_denominator": NVFP4_SCALE_DENOMINATOR,
                "concept": concept,
                "captured_layer_count": len(records),
                "total_linear_calls": sum(record["calls"] for record in records.values()),
                "activation_dtypes": sorted(
                    {dtype for record in records.values() for dtype in record["activation_dtypes"]}
                ),
                "skipped_function_patched_layers": sorted(self.skipped_patched_modules),
            },
            "layers": records,
            "tensor_payload_sha256": tensor_payload_sha256,
        }
        metadata["artifact_sha256"] = _metadata_artifact_sha256(metadata)
        return metadata, tensors


def artifact_paths(prefix: str | os.PathLike[str]) -> tuple[Path, Path]:
    value = str(Path(prefix).expanduser())
    if value.endswith(".json"):
        value = value[:-5]
    elif value.endswith(".safetensors"):
        value = value[: -len(".safetensors")]
    if not value.endswith(ARTIFACT_TAG):
        value = f"{value}.{ARTIFACT_TAG}"
    base = Path(value).resolve()
    return Path(f"{base}.json"), Path(f"{base}.safetensors")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
        raise


def _atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        save_file(tensors, temp_name, metadata=metadata)
        os.replace(temp_name, path)
    except BaseException:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
        raise


@contextlib.contextmanager
def _artifact_file_lock(json_path: Path, *, exclusive: bool) -> Iterator[None]:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(f"{json_path}.lock")
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _tensor_payload_sha256(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _save_artifact_pair(
    metadata: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    prefix: str | os.PathLike[str],
) -> tuple[Path, Path]:
    """Finalize integrity metadata and atomically replace a locked artifact pair."""
    json_path, tensor_path = artifact_paths(prefix)
    finalized = copy.deepcopy(metadata)
    if finalized.get("schema") != SCHEMA_NAME or finalized.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactError("Can only save the current NVFP4 artifact schema")
    model = finalized.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("architecture_sha256"), str):
        raise ArtifactError("Artifact model architecture metadata is missing")
    if not tensors:
        raise ArtifactError("Cannot save an artifact with no tensors")
    finalized.pop("scale_file", None)
    finalized.pop("artifact_sha256", None)
    finalized["tensor_payload_sha256"] = _tensor_payload_sha256(tensors)
    finalized["artifact_sha256"] = _metadata_artifact_sha256(finalized)
    finalized["scale_file"] = tensor_path.name
    with _artifact_file_lock(json_path, exclusive=True):
        _atomic_safetensors(
            tensor_path,
            tensors,
            {
                "schema": SCHEMA_NAME,
                "schema_version": str(SCHEMA_VERSION),
                "architecture_sha256": model["architecture_sha256"],
                "artifact_sha256": finalized["artifact_sha256"],
                "tensor_payload_sha256": finalized["tensor_payload_sha256"],
            },
        )
        _atomic_json(json_path, finalized)
    return json_path, tensor_path


def save_calibration(calibrator: ActivationCalibrator, prefix: str | os.PathLike[str]) -> tuple[Path, Path]:
    metadata, tensors = calibrator.artifact()
    return _save_artifact_pair(metadata, tensors, prefix)


def load_artifact(prefix: str | os.PathLike[str]) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    json_path, tensor_path = artifact_paths(prefix)
    with _artifact_file_lock(json_path, exclusive=False):
        if not json_path.is_file() or not tensor_path.is_file():
            raise ArtifactError(f"Calibration pair is incomplete: {json_path} / {tensor_path}")
        with json_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if not isinstance(metadata, dict):
            raise ArtifactError("Calibration JSON root must be an object")
        if metadata.get("schema") != SCHEMA_NAME or metadata.get("schema_version") != SCHEMA_VERSION:
            raise ArtifactError("Unsupported NVFP4 scale artifact schema")
        if _metadata_artifact_sha256(metadata) != metadata.get("artifact_sha256"):
            raise ArtifactError("Calibration JSON checksum is invalid")
        artifact_model = metadata.get("model")
        calibration = metadata.get("calibration")
        layers = metadata.get("layers")
        if not isinstance(artifact_model, dict):
            raise ArtifactError("Calibration model metadata must be an object")
        if not isinstance(calibration, dict):
            raise ArtifactError("Calibration metadata must be an object")
        if not isinstance(layers, dict) or not layers:
            raise ArtifactError("Calibration layers must be a non-empty object")
        for name, record in layers.items():
            if not isinstance(name, str) or not isinstance(record, dict):
                raise ArtifactError("Every calibration layer must be a named object")
        expected_tensor_name = metadata.get("scale_file")
        if expected_tensor_name and expected_tensor_name != tensor_path.name:
            raise ArtifactError(
                f"JSON expects {expected_tensor_name}, but resolved tensor file is {tensor_path.name}"
            )

        tensors: dict[str, torch.Tensor] = {}
        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            tensor_metadata = handle.metadata() or {}
            if tensor_metadata.get("schema") != SCHEMA_NAME:
                raise ArtifactError("Safetensors schema does not match JSON")
            if tensor_metadata.get("schema_version") != str(SCHEMA_VERSION):
                raise ArtifactError("Safetensors schema version does not match JSON")
            if tensor_metadata.get("artifact_sha256") != metadata.get("artifact_sha256"):
                raise ArtifactError("JSON and safetensors artifact IDs do not match")
            if tensor_metadata.get("architecture_sha256") != artifact_model.get("architecture_sha256"):
                raise ArtifactError("Safetensors architecture fingerprint does not match JSON")
            expected_keys: set[str] = set()
            for record in layers.values():
                for field in ("scale_tensor", "max_tensor", "dynamic_scale_tensor"):
                    key = record.get(field)
                    if not isinstance(key, str) or key not in handle.keys():
                        raise ArtifactError(f"Missing artifact tensor: {key}")
                    expected_keys.add(key)
                    tensors[key] = handle.get_tensor(key)
            if set(handle.keys()) != expected_keys:
                raise ArtifactError("Safetensors contains unexpected keys")
    if _tensor_payload_sha256(tensors) != metadata.get("tensor_payload_sha256"):
        raise ArtifactError("Safetensors payload checksum does not match JSON")
    scale_tensors: dict[str, torch.Tensor] = {}
    for record in layers.values():
        for json_field, tensor_field in (
            ("input_scale", "scale_tensor"),
            ("max_abs", "max_tensor"),
            ("max_dynamic_input_scale", "dynamic_scale_tensor"),
        ):
            key = record[tensor_field]
            tensor = tensors[key]
            raw_value = record.get(json_field)
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ArtifactError(f"Invalid JSON scalar for {key}")
            try:
                expected = torch.tensor(raw_value, dtype=torch.float32)
            except (TypeError, ValueError, OverflowError) as error:
                raise ArtifactError(f"Invalid JSON scalar for {key}: {error}") from error
            if (
                tensor.shape != expected.shape
                or tensor.numel() != 1
                or not torch.equal(tensor, expected)
                or float(raw_value) != float(tensor.item())
                or not bool(torch.isfinite(tensor).item())
                or float(tensor.item()) <= 0.0
            ):
                raise ArtifactError(f"Artifact tensor and JSON value disagree or are invalid: {key}")
        scale_tensors[record["scale_tensor"]] = tensors[record["scale_tensor"]]
    return metadata, scale_tensors


def _positive_finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactError(f"{label} must be a positive finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ArtifactError(f"{label} must be a positive finite number") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ArtifactError(f"{label} must be a positive finite number")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ArtifactError(f"{label} must be a positive integer")
    return value


def normalize_excluded_layers(value: Any) -> frozenset[str]:
    """Parse a user-facing comma/newline list into exact module names."""
    if value is None:
        return frozenset()
    if isinstance(value, str):
        names = (
            name.strip()
            for chunk in value.splitlines()
            for name in chunk.split(",")
            if name.strip()
        )
    else:
        try:
            names = iter(value)
        except TypeError as error:
            raise ArtifactError("excluded_layers must be an iterable of names") from error
    result = frozenset(names)
    if not all(isinstance(name, str) and name for name in result):
        raise ArtifactError("excluded_layers must contain non-empty strings")
    return result


def _validated_call_stats(
    record: dict[str, Any],
    *,
    value_field: str,
    stats_field: str,
    layer_name: str,
) -> dict[str, float | None]:
    stats = record.get(stats_field)
    if not isinstance(stats, dict):
        raise ArtifactError(f"{layer_name} {stats_field} must be an object")
    minimum = _positive_finite_float(stats.get("min"), f"{layer_name} {stats_field}.min")
    maximum = _positive_finite_float(stats.get("max"), f"{layer_name} {stats_field}.max")
    record_maximum = _positive_finite_float(
        record.get(value_field), f"{layer_name} {value_field}"
    )
    if minimum > maximum or maximum != record_maximum:
        raise ArtifactError(f"{layer_name} {stats_field} min/max are inconsistent")

    median_raw = stats.get("median")
    median_ratio_raw = stats.get("median_over_max")
    if median_raw is None:
        if median_ratio_raw is not None:
            raise ArtifactError(f"{layer_name} {stats_field} has a ratio without a median")
        median = median_ratio = None
    else:
        median = _positive_finite_float(median_raw, f"{layer_name} {stats_field}.median")
        median_ratio = _positive_finite_float(
            median_ratio_raw, f"{layer_name} {stats_field}.median_over_max"
        )
        if not minimum <= median <= maximum or not math.isclose(
            median_ratio,
            median / maximum,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise ArtifactError(f"{layer_name} {stats_field} median is inconsistent")

    minimum_ratio = _positive_finite_float(
        stats.get("min_over_max"), f"{layer_name} {stats_field}.min_over_max"
    )
    if not math.isclose(
        minimum_ratio,
        minimum / maximum,
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        raise ArtifactError(f"{layer_name} {stats_field} minimum ratio is inconsistent")
    return {
        "min": minimum,
        "median": median,
        "max": maximum,
        "min_over_max": minimum_ratio,
        "median_over_max": median_ratio,
    }


def _validated_merge_source(
    prefix: str | os.PathLike[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    metadata, _ = load_artifact(prefix)
    model = metadata["model"]
    calibration = metadata["calibration"]
    layers = metadata["layers"]
    if "merge" in metadata:
        raise ArtifactError(
            "Merge inputs must be separately calibrated v3 artifacts, not an earlier merge"
        )

    for field in ("id", "family", "architecture_sha256"):
        if not isinstance(model.get(field), str) or not model[field]:
            raise ArtifactError(f"Calibration model {field} is invalid")
    source_sha256 = model.get("source_sha256")
    if source_sha256 is not None and (
        not isinstance(source_sha256, str) or len(source_sha256) != 64
    ):
        raise ArtifactError("Calibration model source_sha256 is invalid")
    if len(model["architecture_sha256"]) != 64:
        raise ArtifactError("Calibration model architecture_sha256 is invalid")
    _positive_int(model.get("nvfp4_layer_count"), "model nvfp4_layer_count")
    checkpoint = model.get("checkpoint")
    if checkpoint is not None:
        if not isinstance(checkpoint, dict):
            raise ArtifactError("Calibration checkpoint must be an object or null")
        if not isinstance(checkpoint.get("path"), str) or not checkpoint["path"]:
            raise ArtifactError("Calibration checkpoint path is invalid")
        _positive_int(checkpoint.get("bytes"), "checkpoint bytes")
        if (
            isinstance(checkpoint.get("mtime_ns"), bool)
            or not isinstance(checkpoint.get("mtime_ns"), int)
            or checkpoint["mtime_ns"] < 0
        ):
            raise ArtifactError("Calibration checkpoint mtime_ns is invalid")
        if not isinstance(checkpoint.get("sha256"), str) or len(checkpoint["sha256"]) != 64:
            raise ArtifactError("Calibration checkpoint sha256 is invalid")

    margin = _positive_finite_float(calibration.get("margin"), "calibration margin")
    if margin < 1.0:
        raise ArtifactError("Calibration margin must be >= 1.0")
    denominator = _positive_finite_float(
        calibration.get("nvfp4_scale_denominator"),
        "calibration NVFP4 denominator",
    )
    if denominator != NVFP4_SCALE_DENOMINATOR:
        raise ArtifactError("Calibration NVFP4 denominator does not match this implementation")
    captured_layer_count = _positive_int(
        calibration.get("captured_layer_count"),
        "calibration captured_layer_count",
    )
    if captured_layer_count != len(layers):
        raise ArtifactError("Calibration captured_layer_count does not match its layers")

    skipped = calibration.get("skipped_function_patched_layers")
    if (
        not isinstance(skipped, list)
        or not all(isinstance(name, str) for name in skipped)
        or skipped != sorted(set(skipped))
    ):
        raise ArtifactError("Calibration skipped-function layer list is invalid")

    normalized_layers: dict[str, dict[str, Any]] = {}
    all_dtypes: set[str] = set()
    total_calls = 0
    for name, record in layers.items():
        if record.get("name") != name:
            raise ArtifactError(f"Layer key/name mismatch: {name}")
        for field in ("module_class", "quant_format", "layout_type"):
            if not isinstance(record.get(field), str) or not record[field]:
                raise ArtifactError(f"{name} {field} is invalid")
        width = _positive_int(record.get("in_features"), f"{name} in_features")
        _positive_int(record.get("out_features"), f"{name} out_features")
        logical_shape = record.get("logical_weight_shape")
        if (
            not isinstance(logical_shape, list)
            or len(logical_shape) != 2
            or not all(
                isinstance(dimension, int)
                and not isinstance(dimension, bool)
                and dimension > 0
                for dimension in logical_shape
            )
        ):
            raise ArtifactError(f"{name} logical_weight_shape is invalid")
        for field in ("storage_weight_shape", "pre_quant_scale_shape"):
            shape = record.get(field)
            if shape is not None and (
                not isinstance(shape, list)
                or not all(
                    isinstance(dimension, int)
                    and not isinstance(dimension, bool)
                    and dimension > 0
                    for dimension in shape
                )
            ):
                raise ArtifactError(f"{name} {field} is invalid")
        calls = _positive_int(record.get("calls"), f"{name} calls")
        shapes = _validated_record_shapes(record, expected_width=width)
        dtypes = _validated_record_dtypes(record)
        max_abs = _positive_finite_float(record.get("max_abs"), f"{name} max_abs")
        max_dynamic_scale = _positive_finite_float(
            record.get("max_dynamic_input_scale"),
            f"{name} max_dynamic_input_scale",
        )
        input_scale = _positive_finite_float(record.get("input_scale"), f"{name} input_scale")
        expected_input_scale = _static_scale_with_margin(max_dynamic_scale, margin)
        if input_scale != expected_input_scale:
            raise ArtifactError(f"{name} input_scale does not match exact dynamic scale and margin")
        max_stats = _validated_call_stats(
            record,
            value_field="max_abs",
            stats_field="call_max_abs_stats",
            layer_name=name,
        )
        scale_stats = _validated_call_stats(
            record,
            value_field="max_dynamic_input_scale",
            stats_field="call_dynamic_input_scale_stats",
            layer_name=name,
        )
        if max_stats["median"] is None or scale_stats["median"] is None:
            raise ArtifactError(f"{name} raw calibration is missing per-call median statistics")
        normalized_layers[name] = {
            "record": record,
            "calls": calls,
            "shapes": shapes,
            "dtypes": dtypes,
            "max_abs": max_abs,
            "max_dynamic_input_scale": max_dynamic_scale,
            "max_stats": max_stats,
            "scale_stats": scale_stats,
        }
        total_calls += calls
        all_dtypes.update(dtypes)

    if _positive_int(
        calibration.get("total_linear_calls"),
        "calibration total_linear_calls",
    ) != total_calls:
        raise ArtifactError("Calibration total_linear_calls does not match layer calls")
    if calibration.get("activation_dtypes") != sorted(all_dtypes):
        raise ArtifactError("Calibration activation_dtypes does not match its layers")
    concept = calibration.get("concept")
    if concept is not None and not isinstance(concept, dict):
        raise ArtifactError("Calibration concept must be an object or null")
    if concept is not None and (
        not isinstance(concept.get("path"), str)
        or not concept["path"]
        or not isinstance(concept.get("sha256"), str)
        or len(concept["sha256"]) != 64
    ):
        raise ArtifactError("Calibration concept identity is invalid")
    if not isinstance(metadata.get("created_utc"), str) or not metadata["created_utc"]:
        raise ArtifactError("Calibration creation timestamp is invalid")
    artifact_id = metadata.get("artifact_sha256")
    if (
        not isinstance(artifact_id, str)
        or len(artifact_id) != 64
        or any(character not in "0123456789abcdef" for character in artifact_id)
    ):
        raise ArtifactError("Calibration artifact ID is invalid")
    source_json, _ = artifact_paths(prefix)
    source_summary = {
        "artifact_sha256": artifact_id,
        "created_utc": metadata.get("created_utc"),
        "json_path": str(source_json),
        "total_linear_calls": total_calls,
    }
    return metadata, normalized_layers, source_summary


def _merged_stats(
    sources: list[dict[str, float | None]],
    *,
    maximum: float,
) -> dict[str, float | None | str]:
    minimum = min(float(stats["min"]) for stats in sources)
    return {
        "min": minimum,
        "median": None,
        "max": maximum,
        "min_over_max": minimum / maximum,
        "median_over_max": None,
        "aggregation": "exact_min_max; pooled_median_unavailable",
    }


def merge_calibrations(
    source_prefixes: list[str | os.PathLike[str]] | tuple[str | os.PathLike[str], ...],
    output_prefix: str | os.PathLike[str],
    *,
    output_margin: float | None = None,
) -> tuple[Path, Path]:
    """Merge two or more independent v3 calibration pairs without using a GPU.

    Sources must share an identical model, checkpoint, architecture, concept,
    model id, margin, denominator, layer set/descriptors, and skipped-patch set.
    Exact observed shapes and dtypes are unioned. Per-layer calls are summed and
    the largest exact dynamic scale/max-abs are selected. ``output_margin`` may
    increase (but never decrease) the common source margin without rerunning
    calibration because every raw artifact retains the exact dynamic scale.
    """
    if isinstance(source_prefixes, (str, os.PathLike)):
        raise TypeError("source_prefixes must contain at least two artifact prefixes")
    prefixes = list(source_prefixes)
    if len(prefixes) < 2:
        raise ArtifactError("At least two calibration artifacts are required for a merge")

    source_pairs = [artifact_paths(prefix) for prefix in prefixes]
    if len(set(source_pairs)) != len(source_pairs):
        raise ArtifactError("Every merge source must be a distinct artifact pair")
    output_pair = artifact_paths(output_prefix)
    if output_pair in source_pairs:
        raise ArtifactError("Merge output must use a prefix distinct from every source")

    loaded = [_validated_merge_source(prefix) for prefix in prefixes]
    source_ids = [summary["artifact_sha256"] for _, _, summary in loaded]
    if len(set(source_ids)) != len(source_ids):
        raise ArtifactError("Duplicate source artifact content cannot be counted twice")

    baseline_metadata, baseline_layers, _ = loaded[0]
    baseline_model = baseline_metadata["model"]
    baseline_calibration = baseline_metadata["calibration"]
    baseline_names = set(baseline_layers)
    compatibility_fields = (
        "margin",
        "nvfp4_scale_denominator",
        "concept",
        "skipped_function_patched_layers",
    )
    for index, (metadata, layers, _) in enumerate(loaded[1:], start=2):
        if metadata["model"] != baseline_model:
            raise ArtifactError(
                f"Source {index} model/checkpoint/architecture/model-id contract differs"
            )
        for field in compatibility_fields:
            if metadata["calibration"].get(field) != baseline_calibration.get(field):
                raise ArtifactError(f"Source {index} calibration {field} contract differs")
        if set(layers) != baseline_names:
            raise ArtifactError(f"Source {index} calibrated layer set differs")
        for name in sorted(baseline_names):
            first_record = baseline_layers[name]["record"]
            record = layers[name]["record"]
            for field in _LAYER_IDENTITY_FIELDS:
                if record.get(field) != first_record.get(field):
                    raise ArtifactError(
                        f"Source {index} descriptor differs for {name}: {field}"
                    )

    source_margin = float(baseline_calibration["margin"])
    margin = (
        source_margin
        if output_margin is None
        else _positive_finite_float(output_margin, "output_margin")
    )
    if margin < source_margin:
        raise ArtifactError(
            f"output_margin {margin} cannot be smaller than source margin {source_margin}"
        )
    merged_layers: dict[str, dict[str, Any]] = {}
    tensors: dict[str, torch.Tensor] = {}
    all_dtypes: set[str] = set()
    for name in sorted(baseline_names):
        source_layers = [layers[name] for _, layers, _ in loaded]
        first_record = source_layers[0]["record"]
        calls = sum(int(layer["calls"]) for layer in source_layers)
        shapes = set().union(*(layer["shapes"] for layer in source_layers))
        dtypes = set().union(*(layer["dtypes"] for layer in source_layers))
        max_abs = max(float(layer["max_abs"]) for layer in source_layers)
        max_dynamic_scale = max(
            float(layer["max_dynamic_input_scale"]) for layer in source_layers
        )
        input_scale = _static_scale_with_margin(max_dynamic_scale, margin)
        shape_lists = [list(shape) for shape in sorted(shapes)]
        descriptor = {
            field: copy.deepcopy(first_record.get(field)) for field in _LAYER_IDENTITY_FIELDS
        }
        record: dict[str, Any] = {
            **descriptor,
            "calls": calls,
            "activation_dtypes": sorted(dtypes),
            "input_shapes": shape_lists,
            "input_envelope": {
                "feature_width": int(first_record["in_features"]),
                "min_rows": min(shape[0] for shape in shapes),
                "max_rows": max(shape[0] for shape in shapes),
            },
            "max_abs": max_abs,
            "call_max_abs_stats": _merged_stats(
                [layer["max_stats"] for layer in source_layers],
                maximum=max_abs,
            ),
            "max_dynamic_input_scale": max_dynamic_scale,
            "call_dynamic_input_scale_stats": _merged_stats(
                [layer["scale_stats"] for layer in source_layers],
                maximum=max_dynamic_scale,
            ),
            "input_scale": input_scale,
            "scale_tensor": f"input_scale.{name}",
            "max_tensor": f"max_abs.{name}",
            "dynamic_scale_tensor": f"max_dynamic_input_scale.{name}",
        }
        tensors[record["scale_tensor"]] = torch.tensor(input_scale, dtype=torch.float32)
        tensors[record["max_tensor"]] = torch.tensor(max_abs, dtype=torch.float32)
        tensors[record["dynamic_scale_tensor"]] = torch.tensor(
            max_dynamic_scale, dtype=torch.float32
        )
        merged_layers[name] = record
        all_dtypes.update(dtypes)

    source_summaries = sorted(
        (copy.deepcopy(summary) for _, _, summary in loaded),
        key=lambda summary: (summary["artifact_sha256"], summary["json_path"]),
    )
    metadata = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": copy.deepcopy(baseline_model),
        "calibration": {
            "margin": margin,
            "nvfp4_scale_denominator": NVFP4_SCALE_DENOMINATOR,
            "concept": copy.deepcopy(baseline_calibration.get("concept")),
            "captured_layer_count": len(merged_layers),
            "total_linear_calls": sum(record["calls"] for record in merged_layers.values()),
            "activation_dtypes": sorted(all_dtypes),
            "skipped_function_patched_layers": copy.deepcopy(
                baseline_calibration["skipped_function_patched_layers"]
            ),
        },
        "merge": {
            "version": 2,
            "method": "per_layer_max_exact_dynamic_scale",
            "source_margin": source_margin,
            "output_margin": margin,
            "source_count": len(source_summaries),
            "sources": source_summaries,
            "exactly_combined_fields": [
                "calls",
                "input_shapes",
                "activation_dtypes",
                "max_abs",
                "max_dynamic_input_scale",
                "minimum",
                "maximum",
            ],
            "unavailable_combined_fields": [
                "call_max_abs_stats.median",
                "call_max_abs_stats.median_over_max",
                "call_dynamic_input_scale_stats.median",
                "call_dynamic_input_scale_stats.median_over_max",
            ],
            "note": (
                "Pooled medians cannot be recovered from per-source counts and medians; "
                "source artifact IDs retain the original summaries."
            ),
        },
        "layers": merged_layers,
    }
    return _save_artifact_pair(metadata, tensors, output_prefix)


@dataclasses.dataclass
class StaticScaleReport:
    architecture_match: bool
    eligible_layers: int
    applied_layers: int
    already_static_layers: int
    fallback_layers: int
    excluded_layers: tuple[str, ...] = ()
    reason: str | None = None


class StaticScalePlan:
    """Validated static scales and a reversible activation context."""

    def __init__(
        self,
        model_or_patcher: Any,
        artifact_prefix: str | os.PathLike[str],
        *,
        strict: bool = False,
        expected_model_id: str | None = None,
        checkpoint_path: str | os.PathLike[str] | None = None,
        require_h3: bool = True,
        excluded_layers: Any = None,
    ) -> None:
        self.strict = strict
        self.expected_model_id = expected_model_id
        self.require_h3 = require_h3
        self.excluded_layers = normalize_excluded_layers(excluded_layers)
        self.load_device = getattr(model_or_patcher, "load_device", None)
        inferred_checkpoint = Path(checkpoint_path).expanduser() if checkpoint_path else infer_checkpoint_path(model_or_patcher)
        self.checkpoint_path = inferred_checkpoint.resolve() if inferred_checkpoint else None
        self.assignments: dict[str, float] = {}
        self.input_envelopes: dict[str, tuple[int, int]] = {}
        self.allowed_shapes: dict[str, frozenset[tuple[int, int]]] = {}
        self.activation_dtypes: dict[str, frozenset[str]] = {}
        self.artifact_sha256: str | None = None
        self._cache: dict[tuple[str, str], torch.Tensor] = {}
        self.root = diffusion_root(model_or_patcher)
        self._lock = _model_lock(self.root)
        with self._lock:
            inspected_root, self.modules, self.identity = inspect_nvfp4_model(model_or_patcher)
            if inspected_root is not self.root:
                raise ArtifactError("Selected diffusion model changed during static-scale setup")
            if require_h3:
                require_minimax_h3(self.identity)
            self.report = self._prepare(artifact_prefix)

    def _reject(self, reason: str) -> StaticScaleReport:
        if self.strict:
            raise ArtifactError(reason)
        return StaticScaleReport(
            architecture_match=False,
            eligible_layers=len(self.modules),
            applied_layers=0,
            already_static_layers=0,
            fallback_layers=len(self.modules),
            excluded_layers=tuple(sorted(self.excluded_layers)),
            reason=reason,
        )

    def _prepare(self, prefix: str | os.PathLike[str]) -> StaticScaleReport:
        try:
            return self._prepare_validated(prefix)
        except (
            ArtifactError,
            OSError,
            ValueError,
            SafetensorError,
            KeyError,
            TypeError,
            AttributeError,
            OverflowError,
        ) as error:
            return self._reject(str(error))

    def _prepare_validated(self, prefix: str | os.PathLike[str]) -> StaticScaleReport:
        metadata, tensors = load_artifact(prefix)
        self.artifact_sha256 = metadata["artifact_sha256"]
        unknown_exclusions = self.excluded_layers - (
            set(self.modules) & set(metadata["layers"])
        )
        if unknown_exclusions:
            raise ArtifactError(
                "Unknown excluded NVFP4 layer: " + sorted(unknown_exclusions)[0]
            )
        artifact_model = metadata["model"]
        if artifact_model.get("architecture_sha256") != self.identity["architecture_sha256"]:
            return self._reject("Artifact architecture fingerprint does not match the selected model")
        if self.expected_model_id and artifact_model.get("id") != self.expected_model_id:
            return self._reject(
                f"Artifact model id {artifact_model.get('id')!r} does not match {self.expected_model_id!r}"
            )
        artifact_checkpoint = artifact_model.get("checkpoint")
        if artifact_checkpoint is not None:
            if not isinstance(artifact_checkpoint, dict):
                raise ArtifactError("Artifact checkpoint metadata must be an object or null")
            if self.checkpoint_path is None:
                return self._reject("Artifact is checkpoint-bound, but the selected model path is unavailable")
            runtime_checkpoint = checkpoint_descriptor(self.checkpoint_path)
            for field in ("bytes", "sha256"):
                if runtime_checkpoint[field] != artifact_checkpoint.get(field):
                    return self._reject(f"Checkpoint {field} does not match the calibration artifact")

        runtime_descriptors = {
            descriptor["name"]: descriptor for descriptor in self.identity["layers"]
        }
        already_static = 0
        invalid: list[str] = []
        for name, module in self.modules.items():
            if name in self.excluded_layers:
                continue
            if getattr(module, "input_scale", None) is not None:
                already_static += 1
                continue
            if _has_function_patch(module):
                invalid.append(name)
                continue
            record = metadata["layers"].get(name)
            if record is None:
                invalid.append(name)
                continue
            expected_descriptor = runtime_descriptors[name]
            for field in (
                "quant_format",
                "layout_type",
                "in_features",
                "out_features",
                "logical_weight_shape",
                "pre_quant_scale_shape",
            ):
                if record.get(field) != expected_descriptor.get(field):
                    invalid.append(name)
                    break
            else:
                try:
                    dtypes = _validated_record_dtypes(record)
                except ArtifactError:
                    invalid.append(name)
                    continue
                if self.require_h3 and dtypes != {str(torch.bfloat16)}:
                    invalid.append(name)
                    continue
                tensor = tensors[record["scale_tensor"]]
                if tensor.numel() != 1:
                    invalid.append(name)
                    continue
                value = float(tensor.item())
                if not math.isfinite(value) or value <= 0.0:
                    invalid.append(name)
                    continue
                self.assignments[name] = value
                try:
                    allowed_shapes = _validated_record_shapes(
                        record,
                        expected_width=int(module.in_features),
                    )
                except ArtifactError:
                    invalid.append(name)
                    self.assignments.pop(name, None)
                    continue
                rows = [rows for rows, _ in allowed_shapes]
                self.input_envelopes[name] = (int(module.in_features), max(rows))
                self.allowed_shapes[name] = allowed_shapes
                self.activation_dtypes[name] = dtypes

        if self.strict and invalid:
            raise ArtifactError(
                f"Static scale coverage/metadata failed for {len(invalid)} layers; first: {invalid[0]}"
            )
        return StaticScaleReport(
            architecture_match=True,
            eligible_layers=len(self.modules),
            applied_layers=len(self.assignments),
            already_static_layers=already_static,
            fallback_layers=len(invalid) + len(self.excluded_layers),
            excluded_layers=tuple(sorted(self.excluded_layers)),
            reason=(
                f"{len(invalid) + len(self.excluded_layers)} layers retain dynamic scaling"
                if invalid or self.excluded_layers
                else None
            ),
        )

    def _scale_tensor(self, name: str, module: torch.nn.Module) -> torch.Tensor:
        weight = getattr(module, "weight", None)
        device = self.load_device or getattr(weight, "device", torch.device("cpu"))
        if getattr(device, "type", None) == "meta":
            device = torch.device("cpu")
        key = (name, str(device))
        value = self._cache.get(key)
        if value is None:
            value = torch.tensor(self.assignments[name], dtype=torch.float32, device=device)
            self._cache[key] = value
        return value

    @contextlib.contextmanager
    def activate(self, *, enable_fusion_bridge: bool = False) -> Iterator[StaticScaleReport]:
        """Set scales only for the enclosed sampling call and restore exactly.

        ``enable_fusion_bridge`` additionally scopes plain companion attributes
        consumed by guarded fused paths. They describe the exact activated
        Linear inputs and certify the plan-owned shape-guard hook IDs. Validation
        leaves the bridge disabled so normal Linear pre-hooks observe every call.
        """
        if not isinstance(enable_fusion_bridge, bool):
            raise TypeError("enable_fusion_bridge must be a bool")
        with self._lock:
            snapshots: dict[str, _AttributeSnapshot] = {}
            companion_snapshots: dict[
                str, dict[str, _AttributeSnapshot]
            ] = {}
            handles: list[Any] = []
            suspended: dict[str, _AttributeSnapshot] = {}

            def guard_pre(name: str, module: torch.nn.Module, args: tuple[Any, ...]) -> None:
                if not args or not isinstance(args[0], torch.Tensor):
                    return
                shape = args[0].shape
                effective_shape = (
                    (math.prod(shape[:-1]), int(shape[-1])) if len(shape) >= 3 else tuple(int(v) for v in shape)
                )
                shape_matches = effective_shape in self.allowed_shapes[name]
                dtype_matches = str(args[0].dtype) in self.activation_dtypes[name]
                if shape_matches and dtype_matches and not _has_function_patch(module):
                    return
                static_snapshot = _snapshot_input_scale(module)
                if static_snapshot.kind == "absent":
                    return
                suspended[name] = static_snapshot
                _restore_input_scale(module, snapshots[name])

            def guard_post(name: str, module: torch.nn.Module, _args: tuple[Any, ...], _output: Any) -> None:
                static_snapshot = suspended.pop(name, None)
                if static_snapshot is not None:
                    _assign_input_scale(module, static_snapshot, static_snapshot.value)

            try:
                for name in self.assignments:
                    module = self.modules[name]
                    # A patch can be installed after plan construction but before
                    # this scope starts. Leave that layer entirely dynamic and do
                    # not advertise fusion eligibility for it.
                    if _has_function_patch(module):
                        continue
                    snapshot = _snapshot_input_scale(module)
                    # A different patch may have installed a scale after validation.
                    if snapshot.kind != "absent" and snapshot.value is not None:
                        continue
                    snapshots[name] = snapshot
                    assigned_scale = self._scale_tensor(name, module)
                    _assign_input_scale(module, snapshot, assigned_scale)
                    if enable_fusion_bridge:
                        companion_snapshots[name] = {
                            FUSION_ENABLED_ATTR: _snapshot_plain_attribute(
                                module, FUSION_ENABLED_ATTR
                            ),
                            FUSION_ALLOWED_SHAPES_ATTR: _snapshot_plain_attribute(
                                module, FUSION_ALLOWED_SHAPES_ATTR
                            ),
                            FUSION_ALLOWED_DTYPES_ATTR: _snapshot_plain_attribute(
                                module, FUSION_ALLOWED_DTYPES_ATTR
                            ),
                            FUSION_VALIDATED_SCALE_ATTR: _snapshot_plain_attribute(
                                module, FUSION_VALIDATED_SCALE_ATTR
                            ),
                            FUSION_PLAN_HOOK_IDS_ATTR: _snapshot_plain_attribute(
                                module, FUSION_PLAN_HOOK_IDS_ATTR
                            ),
                        }
                        # A fused helper bypasses Module.__call__, so any hook
                        # already present here would otherwise disappear. Ensure
                        # even stale bridge metadata is absent for this scope;
                        # the original values are restored in finally.
                        if _has_preexisting_module_hooks(module):
                            for attribute_name in companion_snapshots[name]:
                                _restore_plain_attribute(
                                    module,
                                    attribute_name,
                                    _AttributeSnapshot("absent"),
                                )
                        else:
                            _assign_plain_attribute(module, FUSION_ENABLED_ATTR, True)
                            _assign_plain_attribute(
                                module,
                                FUSION_ALLOWED_SHAPES_ATTR,
                                self.allowed_shapes[name],
                            )
                            _assign_plain_attribute(
                                module,
                                FUSION_ALLOWED_DTYPES_ATTR,
                                self.activation_dtypes[name],
                            )
                            # The producer has already checked this scalar on
                            # the CPU. Object identity lets the fused consumer
                            # reject stale or independently mutated bridge
                            # metadata without synchronizing the GPU hot path.
                            _assign_plain_attribute(
                                module,
                                FUSION_VALIDATED_SCALE_ATTR,
                                assigned_scale,
                            )
                    pre_handle = module.register_forward_pre_hook(
                        lambda mod, args, layer_name=name: guard_pre(layer_name, mod, args)
                    )
                    handles.append(pre_handle)
                    post_handle = module.register_forward_hook(
                        lambda mod, args, output, layer_name=name: guard_post(layer_name, mod, args, output),
                        always_call=True,
                    )
                    handles.append(post_handle)
                    if (
                        enable_fusion_bridge
                        and name in companion_snapshots
                        and getattr(module, FUSION_ENABLED_ATTR, None) is True
                    ):
                        _assign_plain_attribute(
                            module,
                            FUSION_PLAN_HOOK_IDS_ATTR,
                            (pre_handle.id, post_handle.id),
                        )
                yield self.report
            finally:
                for handle in handles:
                    handle.remove()
                for name, static_snapshot in suspended.items():
                    _assign_input_scale(self.modules[name], static_snapshot, static_snapshot.value)
                for name, attribute_snapshots in reversed(
                    list(companion_snapshots.items())
                ):
                    module = self.modules[name]
                    for attribute_name, attribute_snapshot in attribute_snapshots.items():
                        _restore_plain_attribute(
                            module,
                            attribute_name,
                            attribute_snapshot,
                        )
                for name, snapshot in reversed(list(snapshots.items())):
                    _restore_input_scale(self.modules[name], snapshot)


@dataclasses.dataclass
class ValidationObservation(LayerObservation):
    static_applied: list[bool] = dataclasses.field(default_factory=list)


class StaticScaleValidator:
    """Held-out overflow audit; intentionally retains reduction cost while active."""

    def __init__(
        self,
        plan: StaticScalePlan,
        *,
        concept_path: str | os.PathLike[str] | None = None,
    ) -> None:
        if not plan.assignments:
            raise ArtifactError("Cannot validate a plan with no applicable static scales")
        self.plan = plan
        self.concept_path = Path(concept_path).resolve() if concept_path else None
        self.observations: dict[str, ValidationObservation] = {}
        self._handles: list[Any] = []

    def reset(self) -> None:
        self.observations = {name: ValidationObservation() for name in self.plan.assignments}

    def _pre_hook(self, name: str, module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        if not args or not isinstance(args[0], torch.Tensor):
            return
        value = args[0].detach()
        pre_quant_scale = getattr(module, "pre_quant_scale", None)
        if pre_quant_scale is not None:
            value = value * pre_quant_scale.to(device=value.device, dtype=value.dtype)
        if value.ndim >= 3:
            value = value.reshape(-1, value.shape[-1])
        if value.ndim != 2:
            return
        observation = self.observations[name]
        observation.maxima.append(torch.amax(value.abs()).detach())
        observation.input_shapes.add(tuple(int(v) for v in value.shape))
        observation.activation_dtypes.add(str(value.dtype))
        observation.static_applied.append(getattr(module, "input_scale", None) is not None)

    @contextlib.contextmanager
    def capture(self) -> Iterator["StaticScaleValidator"]:
        with self.plan._lock:
            if self._handles:
                raise RuntimeError("Static scale validation is already active")
            self.reset()
            try:
                for name in self.plan.assignments:
                    module = self.plan.modules[name]
                    self._handles.append(
                        module.register_forward_pre_hook(
                            lambda mod, args, layer_name=name: self._pre_hook(layer_name, mod, args)
                        )
                    )
                yield self
            finally:
                for handle in self._handles:
                    handle.remove()
                self._handles.clear()

    def report(self) -> dict[str, Any]:
        values_by_name = _scalar_lists(self.observations)
        layer_reports: dict[str, dict[str, Any]] = {}
        total_calls = total_static = total_overflows = 0
        global_max_ratio = 0.0
        missing_layers = invalid_layers = 0
        for name in sorted(self.plan.assignments):
            observation = self.observations[name]
            values = values_by_name.get(name, [])
            if len(values) != len(observation.static_applied):
                raise ArtifactError(f"Incomplete validation observations for {name}")
            threshold = self.plan.assignments[name] * NVFP4_SCALE_DENOMINATOR
            finite = all(math.isfinite(value) for value in values) and math.isfinite(threshold) and threshold > 0.0
            if not values:
                missing_layers += 1
            if not finite:
                invalid_layers += 1
            ratios = [value / threshold for value in values] if finite else []
            static_ratios = [
                ratio for ratio, applied in zip(ratios, observation.static_applied) if applied
            ]
            overflow_count = sum(ratio > 1.0 for ratio in static_ratios)
            total_calls += len(values)
            total_static += len(static_ratios)
            total_overflows += overflow_count
            if ratios:
                global_max_ratio = max(global_max_ratio, max(ratios))
            layer_reports[name] = {
                "calls": len(values),
                "static_calls": len(static_ratios),
                "dynamic_guard_fallback_calls": len(values) - len(static_ratios),
                "dynamic_shape_fallback_calls": len(values) - len(static_ratios),
                "activation_dtypes": sorted(observation.activation_dtypes),
                "input_shapes": [list(shape) for shape in sorted(observation.input_shapes)],
                "threshold_max_abs": threshold,
                "observed_max_abs": max(values) if values else None,
                "ratio_stats": {
                    "min": min(ratios) if ratios else None,
                    "median": statistics.median(ratios) if ratios else None,
                    "max": max(ratios) if ratios else None,
                },
                "overflow_count": overflow_count,
                "finite": finite,
            }
        concept = None
        if self.concept_path is not None:
            if not self.concept_path.is_file():
                raise ArtifactError(f"Validation concept image does not exist: {self.concept_path}")
            concept = {"path": str(self.concept_path), "sha256": file_sha256(self.concept_path)}
        return {
            "schema": f"{SCHEMA_NAME}.validation",
            "schema_version": 1,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "model": {
                "id": self.plan.expected_model_id,
                "architecture_sha256": self.plan.identity["architecture_sha256"],
                "checkpoint": checkpoint_descriptor(self.plan.checkpoint_path) if self.plan.checkpoint_path else None,
            },
            "concept": concept,
            "artifact": {
                "artifact_sha256": getattr(self.plan, "artifact_sha256", None),
                "excluded_layers": sorted(self.plan.excluded_layers),
            },
            "summary": {
                "layers": len(layer_reports),
                "excluded_layers": sorted(self.plan.excluded_layers),
                "calls": total_calls,
                "static_calls": total_static,
                "dynamic_guard_fallback_calls": total_calls - total_static,
                "dynamic_shape_fallback_calls": total_calls - total_static,
                "overflow_count": total_overflows,
                "max_ratio": global_max_ratio,
                "missing_layers": missing_layers,
                "invalid_layers": invalid_layers,
                "pass": (
                    total_overflows == 0
                    and total_static == total_calls
                    and missing_layers == 0
                    and invalid_layers == 0
                ),
            },
            "layers": layer_reports,
        }


def save_validation_report(
    validator: StaticScaleValidator,
    prefix: str | os.PathLike[str],
    report: dict[str, Any] | None = None,
) -> Path:
    json_path, _ = artifact_paths(prefix)
    stem = str(json_path)[: -len(".json")]
    path = Path(f"{stem}.{VALIDATION_TAG}.json")
    with _artifact_file_lock(json_path, exclusive=True):
        _atomic_json(path, report if report is not None else validator.report())
    return path


def validation_report_path(prefix: str | os.PathLike[str]) -> Path:
    json_path, _ = artifact_paths(prefix)
    stem = str(json_path)[: -len(".json")]
    return Path(f"{stem}.{VALIDATION_TAG}.json")


def verify_promotion(
    prefix: str | os.PathLike[str],
    *,
    excluded_layers: Any = None,
    report_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Fail closed unless a held-out report exactly certifies this artifact.

    The verifier rechecks the artifact's JSON/safetensors integrity chain, then
    binds every reported static layer to its exact shape, dtype, and threshold.
    This prevents a stale validation file from being promoted with a different
    scale artifact or hybrid exclusion policy.
    """
    metadata, _ = load_artifact(prefix)
    exclusions = normalize_excluded_layers(excluded_layers)
    unknown = exclusions - set(metadata["layers"])
    if unknown:
        raise ArtifactError("Unknown excluded NVFP4 layer: " + sorted(unknown)[0])
    path = Path(report_path) if report_path is not None else validation_report_path(prefix)
    try:
        with path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"Validation report cannot be loaded: {error}") from error
    if not isinstance(report, dict):
        raise ArtifactError("Validation report root must be an object")
    if (
        report.get("schema") != f"{SCHEMA_NAME}.validation"
        or report.get("schema_version") != 1
    ):
        raise ArtifactError("Validation report schema is unsupported")

    binding = report.get("artifact")
    if not isinstance(binding, dict):
        raise ArtifactError("Validation report is not artifact-bound")
    if binding.get("artifact_sha256") != metadata["artifact_sha256"]:
        raise ArtifactError("Validation report artifact SHA-256 does not match")
    expected_exclusions = sorted(exclusions)
    if binding.get("excluded_layers") != expected_exclusions:
        raise ArtifactError("Validation report exclusion policy does not match")

    report_model = report.get("model")
    artifact_model = metadata["model"]
    if not isinstance(report_model, dict):
        raise ArtifactError("Validation report model binding is missing")
    for field in ("id", "architecture_sha256"):
        if report_model.get(field) != artifact_model.get(field):
            raise ArtifactError(f"Validation report model {field} does not match")
    artifact_checkpoint = artifact_model.get("checkpoint")
    report_checkpoint = report_model.get("checkpoint")
    if artifact_checkpoint is not None:
        if not isinstance(report_checkpoint, dict):
            raise ArtifactError("Validation report checkpoint binding is missing")
        for field in ("bytes", "sha256"):
            if report_checkpoint.get(field) != artifact_checkpoint.get(field):
                raise ArtifactError(f"Validation report checkpoint {field} does not match")

    artifact_concept = metadata["calibration"].get("concept")
    report_concept = report.get("concept")
    if artifact_concept is not None:
        if not isinstance(report_concept, dict) or (
            report_concept.get("sha256") != artifact_concept.get("sha256")
        ):
            raise ArtifactError("Validation report concept does not match")

    expected_names = set(metadata["layers"]) - exclusions
    report_layers = report.get("layers")
    if not isinstance(report_layers, dict) or set(report_layers) != expected_names:
        raise ArtifactError("Validation report layer set does not match promotion policy")
    total_calls = total_static = total_overflows = 0
    maximum_ratio = 0.0
    for name in sorted(expected_names):
        artifact_record = metadata["layers"][name]
        record = report_layers[name]
        if not isinstance(record, dict):
            raise ArtifactError(f"Validation layer record is invalid: {name}")
        if record.get("input_shapes") != artifact_record.get("input_shapes"):
            raise ArtifactError(f"Validation input shapes do not match: {name}")
        if record.get("activation_dtypes") != artifact_record.get("activation_dtypes"):
            raise ArtifactError(f"Validation activation dtypes do not match: {name}")
        expected_threshold = (
            float(artifact_record["input_scale"]) * NVFP4_SCALE_DENOMINATOR
        )
        if record.get("threshold_max_abs") != expected_threshold:
            raise ArtifactError(f"Validation threshold does not match: {name}")
        calls = record.get("calls")
        static_calls = record.get("static_calls")
        ratio_stats = record.get("ratio_stats")
        if (
            not isinstance(calls, int)
            or isinstance(calls, bool)
            or calls <= 0
            or static_calls != calls
            or record.get("dynamic_guard_fallback_calls") != 0
            or record.get("dynamic_shape_fallback_calls") != 0
            or record.get("overflow_count") != 0
            or record.get("finite") is not True
            or not isinstance(ratio_stats, dict)
            or not isinstance(ratio_stats.get("max"), (int, float))
            or not math.isfinite(float(ratio_stats["max"]))
            or float(ratio_stats["max"]) > 1.0
        ):
            raise ArtifactError(f"Validation layer did not pass cleanly: {name}")
        total_calls += calls
        total_static += static_calls
        total_overflows += int(record["overflow_count"])
        maximum_ratio = max(maximum_ratio, float(ratio_stats["max"]))

    summary = report.get("summary")
    expected_summary = {
        "layers": len(expected_names),
        "excluded_layers": expected_exclusions,
        "calls": total_calls,
        "static_calls": total_static,
        "dynamic_guard_fallback_calls": 0,
        "dynamic_shape_fallback_calls": 0,
        "overflow_count": total_overflows,
        "missing_layers": 0,
        "invalid_layers": 0,
        "pass": True,
    }
    if not isinstance(summary, dict):
        raise ArtifactError("Validation summary is missing")
    for field, expected in expected_summary.items():
        if summary.get(field) != expected:
            raise ArtifactError(f"Validation summary {field} does not match")
    if summary.get("max_ratio") != maximum_ratio:
        raise ArtifactError("Validation summary max_ratio does not match layers")
    return {
        "pass": True,
        "artifact_sha256": metadata["artifact_sha256"],
        "validation_report": str(path.resolve()),
        "layers": len(expected_names),
        "excluded_layers": expected_exclusions,
        "calls": total_calls,
        "max_ratio": maximum_ratio,
    }
