from __future__ import annotations

import os
import unittest

import torch
import torch.nn.functional as F

from comfy.text_encoders import minimax as mm


class QwenNnLinearTests(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.get(mm._NN_LINEAR_ENV)
        os.environ[mm._NN_LINEAR_ENV] = "auto"

    def tearDown(self):
        if self._env is None:
            os.environ.pop(mm._NN_LINEAR_ENV, None)
        else:
            os.environ[mm._NN_LINEAR_ENV] = self._env

    def test_env_off_disables(self):
        os.environ[mm._NN_LINEAR_ENV] = "off"
        self.assertFalse(mm._nn_linear_enabled())
        os.environ[mm._NN_LINEAR_ENV] = "auto"
        self.assertTrue(mm._nn_linear_enabled())

    def test_wrap_skips_unknown_kn(self):
        lin = torch.nn.Linear(16, 8, bias=False)
        lin._forward = lambda x, w, b: F.linear(x, w, b)
        mm._wrap_linear_nn(lin)
        x = torch.randn(4, 16)
        y = lin._forward(x, lin.weight, None)
        self.assertTrue(torch.equal(y, F.linear(x, lin.weight, None)))

    def test_wrap_skips_wrong_rows(self):
        class Dummy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.in_features = 5120
                self.out_features = 8192
                self.weight = torch.nn.Parameter(torch.randn(8192, 5120))

            def _forward(self, x, w, b):
                return F.linear(x, w, b)

        d = Dummy()
        mm._wrap_linear_nn(d)
        x = torch.randn(7, 5120)
        y = d._forward(x, d.weight, None)
        self.assertTrue(torch.equal(y, F.linear(x, d.weight, None)))
        self.assertFalse(hasattr(d, "_h3_Wt"))

    def test_exact_row_whitelist(self):
        self.assertIn(2800, mm._NN_EXACT_ROWS[(5120, 25600)])
        self.assertIn(2048, mm._NN_EXACT_ROWS[(5120, 25600)])
        self.assertNotIn(800, mm._NN_EXACT_ROWS[(5120, 25600)])
        self.assertNotIn((25600, 5120), mm._NN_EXACT_ROWS)

    @unittest.skipUnless(torch.cuda.is_available(), "cuda")
    def test_nn_equals_linear_chef_rows(self):
        torch.manual_seed(14)
        device = "cuda"
        cases = (
            (2800, 5120, 25600),
            (2800, 5120, 8192),
            (2800, 8192, 5120),
            (2048, 5120, 25600),
            (1600, 5120, 8192),
            (512, 8192, 5120),
        )
        for m, k, n in cases:
            x = torch.randn(1, m, k, device=device, dtype=torch.float32)
            w = torch.randn(n, k, device=device, dtype=torch.float32)
            stock = F.linear(x, w, None)
            got = mm.qwen_nn_linear(x, w, None, owner=None)
            self.assertTrue(torch.equal(stock, got), msg=f"M={m} K={k} N={n}")


if __name__ == "__main__":
    unittest.main()
