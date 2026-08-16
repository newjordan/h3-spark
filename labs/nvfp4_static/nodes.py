"""ComfyUI custom-node adapter for the isolated H3 NVFP4 scale lab."""

from __future__ import annotations

import logging

import comfy.patcher_extension

from .h3_nvfp4_static_scales import (
    ActivationCalibrator,
    StaticScalePlan,
    StaticScaleValidator,
    artifact_paths,
    save_calibration,
    save_validation_report,
)


DEFAULT_PREFIX = "h3_nvfp4_artifacts/h3"
DEFAULT_CONCEPT = ""
DEFAULT_MODEL_ID = "minimax_h3_ref2va_pruned_nvfp4.safetensors"
CALIBRATE_KEY = "h3_nvfp4_calibrate_input_scales"
APPLY_KEY = "h3_nvfp4_apply_input_scales"
VALIDATE_KEY = "h3_nvfp4_validate_input_scales"


def _remove_lab_wrappers(model):
    for key in (CALIBRATE_KEY, APPLY_KEY, VALIDATE_KEY):
        model.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, key)


def _runtime_patcher(executor, fallback):
    patcher = getattr(executor.class_obj, "model_patcher", None)
    return patcher if patcher is not None else fallback


def _require_single_device(patcher):
    if patcher.get_additional_models_with_key("multigpu"):
        raise RuntimeError("H3 NVFP4 static-scale lab does not support multi-GPU clones")


def _artifact_change_token(prefix):
    values = []
    for path in artifact_paths(prefix):
        try:
            stat = path.stat()
            values.append((str(path), stat.st_size, stat.st_mtime_ns))
        except FileNotFoundError:
            values.append((str(path), None, None))
    return repr(values)


class H3CalibrateNVFP4InputScales:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "artifact_prefix": ("STRING", {"default": DEFAULT_PREFIX}),
                "margin": ("FLOAT", {"default": 1.20, "min": 1.0, "max": 4.0, "step": 0.01}),
                "concept_path": ("STRING", {"default": DEFAULT_CONCEPT}),
                "model_id": ("STRING", {"default": DEFAULT_MODEL_ID}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "artifact_json")
    FUNCTION = "attach"
    CATEGORY = "H3 Lab/NVFP4"

    def attach(self, model, artifact_prefix, margin, concept_path, model_id):
        patched = model.clone()
        _remove_lab_wrappers(patched)

        def calibration_wrapper(executor, *args, **kwargs):
            runtime_patcher = _runtime_patcher(executor, patched)
            _require_single_device(runtime_patcher)
            calibrator = ActivationCalibrator(
                runtime_patcher,
                margin=margin,
                model_id=model_id,
                concept_path=concept_path or None,
                require_h3=True,
            )
            with calibrator.capture():
                output = executor(*args, **kwargs)
                json_path, tensor_path = save_calibration(calibrator, artifact_prefix)
            logging.info(
                "H3 NVFP4 calibration wrote %d layer scales to %s and %s",
                len(calibrator.modules),
                json_path,
                tensor_path,
            )
            return output

        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            CALIBRATE_KEY,
            calibration_wrapper,
        )
        return patched, str(artifact_paths(artifact_prefix)[0])


class H3ApplyNVFP4InputScales:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "artifact_prefix": ("STRING", {"default": DEFAULT_PREFIX}),
                "on_mismatch": (["fallback_dynamic", "error"],),
                "expected_model_id": ("STRING", {"default": DEFAULT_MODEL_ID}),
            },
            "optional": {
                "excluded_layers": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "attach"
    CATEGORY = "H3 Lab/NVFP4"

    @classmethod
    def IS_CHANGED(
        cls, model, artifact_prefix, on_mismatch, expected_model_id,
        excluded_layers="",
    ):
        return repr((_artifact_change_token(artifact_prefix), excluded_layers))

    def attach(
        self, model, artifact_prefix, on_mismatch, expected_model_id,
        excluded_layers="",
    ):
        patched = model.clone()
        _remove_lab_wrappers(patched)

        def static_scale_wrapper(executor, *args, **kwargs):
            runtime_patcher = _runtime_patcher(executor, patched)
            _require_single_device(runtime_patcher)
            plan = StaticScalePlan(
                runtime_patcher,
                artifact_prefix,
                strict=on_mismatch == "error",
                expected_model_id=expected_model_id or None,
                require_h3=True,
                excluded_layers=excluded_layers,
            )
            if not plan.assignments:
                logging.warning("H3 NVFP4 static scales disabled: %s", plan.report.reason)
                return executor(*args, **kwargs)
            logging.info(
                "H3 NVFP4 static scales active: %d applied, %d dynamic fallbacks",
                plan.report.applied_layers,
                plan.report.fallback_layers,
            )
            with plan.activate(enable_fusion_bridge=True):
                return executor(*args, **kwargs)

        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            APPLY_KEY,
            static_scale_wrapper,
        )
        return (patched,)


class H3ValidateNVFP4InputScales:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "artifact_prefix": ("STRING", {"default": DEFAULT_PREFIX}),
                "validation_concept_path": ("STRING", {"default": DEFAULT_CONCEPT}),
                "expected_model_id": ("STRING", {"default": DEFAULT_MODEL_ID}),
                "on_mismatch": (["error", "fallback_dynamic"],),
            },
            "optional": {
                "excluded_layers": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "validation_json")
    FUNCTION = "attach"
    CATEGORY = "H3 Lab/NVFP4"

    @classmethod
    def IS_CHANGED(
        cls,
        model,
        artifact_prefix,
        validation_concept_path,
        expected_model_id,
        on_mismatch,
        excluded_layers="",
    ):
        return repr((_artifact_change_token(artifact_prefix), excluded_layers))

    def attach(
        self,
        model,
        artifact_prefix,
        validation_concept_path,
        expected_model_id,
        on_mismatch,
        excluded_layers="",
    ):
        patched = model.clone()
        _remove_lab_wrappers(patched)

        def validation_wrapper(executor, *args, **kwargs):
            runtime_patcher = _runtime_patcher(executor, patched)
            _require_single_device(runtime_patcher)
            plan = StaticScalePlan(
                runtime_patcher,
                artifact_prefix,
                strict=on_mismatch == "error",
                expected_model_id=expected_model_id or None,
                require_h3=True,
                excluded_layers=excluded_layers,
            )
            if not plan.assignments:
                logging.warning("H3 NVFP4 validation disabled: %s", plan.report.reason)
                return executor(*args, **kwargs)
            validator = StaticScaleValidator(
                plan,
                concept_path=validation_concept_path or None,
            )
            with plan.activate():
                with validator.capture():
                    output = executor(*args, **kwargs)
                    report = validator.report()
                    summary = report["summary"]
                    report_path = save_validation_report(validator, artifact_prefix, report=report)
            logging.info(
                "H3 NVFP4 held-out validation: pass=%s, overflow=%d, max_ratio=%.6f, report=%s",
                summary["pass"],
                summary["overflow_count"],
                summary["max_ratio"],
                report_path,
            )
            return output

        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            VALIDATE_KEY,
            validation_wrapper,
        )
        json_path, _ = artifact_paths(artifact_prefix)
        validation_path = f"{str(json_path)[:-len('.json')]}.nvfp4-input-scales.validation.v1.json"
        return patched, validation_path


NODE_CLASS_MAPPINGS = {
    "H3CalibrateNVFP4InputScales": H3CalibrateNVFP4InputScales,
    "H3ApplyNVFP4InputScales": H3ApplyNVFP4InputScales,
    "H3ValidateNVFP4InputScales": H3ValidateNVFP4InputScales,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3CalibrateNVFP4InputScales": "H3 Calibrate NVFP4 Input Scales (Lab)",
    "H3ApplyNVFP4InputScales": "H3 Apply NVFP4 Input Scales (Lab)",
    "H3ValidateNVFP4InputScales": "H3 Validate NVFP4 Input Scales (Lab)",
}
