from __future__ import annotations

import importlib.util
import os
import unittest


def _load():
    path = "$HOME/comfyui-h3-current/custom_nodes/h3_sage_attention.py"
    spec = importlib.util.spec_from_file_location("h3_sage_attention_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class AttnTimingOffTests(unittest.TestCase):
    def test_default_off(self):
        os.environ.pop("H3_ATTN_TIMING", None)
        # re-read env each call
        mod = _load()
        self.assertFalse(mod._attn_timing_enabled())

    def test_on_off(self):
        mod = _load()
        os.environ["H3_ATTN_TIMING"] = "on"
        self.assertTrue(mod._attn_timing_enabled())
        os.environ["H3_ATTN_TIMING"] = "off"
        self.assertFalse(mod._attn_timing_enabled())
        os.environ.pop("H3_ATTN_TIMING", None)
