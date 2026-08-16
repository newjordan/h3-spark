from pathlib import Path
import logging

import torch

import comfy.patcher_extension


class H3Profiler:
    def __init__(self, wait_calls, active_calls, out_prefix):
        self.wait_calls = wait_calls
        self.active_calls = active_calls
        self.out_prefix = Path(out_prefix)
        self.calls = 0
        self.profiler = None

    def __call__(self, executor, *args, **kwargs):
        self.calls += 1
        first = self.wait_calls + 1
        last = self.wait_calls + self.active_calls
        if self.calls == first:
            self.profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
            )
            self.profiler.start()
            logging.info("H3 profiler started on diffusion call %d", self.calls)

        try:
            return executor(*args, **kwargs)
        finally:
            if self.profiler is not None and first <= self.calls <= last:
                self.profiler.step()
            if self.profiler is not None and self.calls == last:
                self.profiler.stop()
                self.out_prefix.parent.mkdir(parents=True, exist_ok=True)
                table = self.profiler.key_averages(group_by_input_shape=True).table(
                    sort_by="cuda_time_total", row_limit=200, max_name_column_width=140)
                self.out_prefix.with_suffix(".txt").write_text(table + "\n", encoding="utf-8")
                self.profiler.export_chrome_trace(str(self.out_prefix.with_suffix(".json")))
                logging.info("H3 profiler wrote %s.[txt,json]", self.out_prefix)
                self.profiler = None


class H3ProfilerModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "wait_calls": ("INT", {"default": 2, "min": 0, "max": 100}),
            "active_calls": ("INT", {"default": 1, "min": 1, "max": 20}),
            "out_prefix": ("STRING", {"default": "/tmp/h3_prof"}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/debug"

    def patch(self, model, wait_calls, active_calls, out_prefix):
        patched = model.clone()
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            "h3_profiler",
            H3Profiler(wait_calls, active_calls, out_prefix),
        )
        return (patched,)


NODE_CLASS_MAPPINGS = {"H3ProfilerModel": H3ProfilerModel}
