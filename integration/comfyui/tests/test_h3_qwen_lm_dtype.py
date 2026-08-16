from __future__ import annotations

import os
import unittest

import torch

from comfy.text_encoders import minimax as mm


class QwenLmDtypeTests(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.get(mm._LM_DTYPE_ENV)

    def tearDown(self):
        if self._env is None:
            os.environ.pop(mm._LM_DTYPE_ENV, None)
        else:
            os.environ[mm._LM_DTYPE_ENV] = self._env

    def test_default_off_is_fp32_stock(self):
        os.environ.pop(mm._LM_DTYPE_ENV, None)
        self.assertIsNone(mm.qwen_lm_compute_dtype())
        os.environ[mm._LM_DTYPE_ENV] = "off"
        self.assertIsNone(mm.qwen_lm_compute_dtype())
        os.environ[mm._LM_DTYPE_ENV] = "stock"
        self.assertIsNone(mm.qwen_lm_compute_dtype())

    def test_bf16_and_fp16_opt_in(self):
        os.environ[mm._LM_DTYPE_ENV] = "bf16"
        self.assertEqual(mm.qwen_lm_compute_dtype(), torch.bfloat16)
        os.environ[mm._LM_DTYPE_ENV] = "fp16"
        self.assertEqual(mm.qwen_lm_compute_dtype(), torch.float16)

    def test_maybe_cast_lm_embeds(self):
        x = torch.randn(1, 8, 16)
        y = mm.maybe_cast_lm_embeds(x, torch.bfloat16)
        self.assertEqual(y.dtype, torch.bfloat16)
        self.assertEqual(y.shape, x.shape)
        self.assertIs(mm.maybe_cast_lm_embeds(x, None), x)
        self.assertIs(mm.maybe_cast_lm_embeds(y, torch.bfloat16), y)

    def test_default_does_not_cast_without_dtype(self):
        os.environ.pop(mm._LM_DTYPE_ENV, None)
        self.assertIsNone(mm.qwen_lm_compute_dtype())
        x = torch.randn(1, 4, 8)
        self.assertEqual(x.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
