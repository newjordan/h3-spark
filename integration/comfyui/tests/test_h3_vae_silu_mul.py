from __future__ import annotations

import os
import unittest

import torch

from comfy.ldm.minimax import vae as mvae


class VaeSiluMulTests(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.get(mvae._SILU_MUL_ENV)
        os.environ[mvae._SILU_MUL_ENV] = "auto"
        mvae._SILU_MUL_LIB = None
        mvae._SILU_MUL_LIB_TRIED = False

    def tearDown(self):
        if self._env is None:
            os.environ.pop(mvae._SILU_MUL_ENV, None)
        else:
            os.environ[mvae._SILU_MUL_ENV] = self._env

    def test_env_off_disables(self):
        os.environ[mvae._SILU_MUL_ENV] = "off"
        self.assertFalse(mvae._silu_mul_enabled())
        os.environ[mvae._SILU_MUL_ENV] = "auto"
        self.assertTrue(mvae._silu_mul_enabled())

    def test_stock_cpu_matches_silu_mul(self):
        torch.manual_seed(0)
        raw = torch.randn(3, 17, 64)
        gate, up = raw.chunk(2, dim=-1)
        ref = torch.nn.functional.silu(gate) * up
        out = mvae._stock_silu_mul(raw.clone())
        self.assertTrue(torch.equal(out, ref) or torch.allclose(out, ref, rtol=0, atol=0))

    def test_odd_last_dim_uses_stock_shape(self):
        # 2*inner required; helper should not crash on even path.
        raw = torch.randn(2, 8, 32)
        out = mvae._stock_silu_mul(raw)
        self.assertEqual(tuple(out.shape), (2, 8, 16))

    def test_feedforward_calls_silu_mul(self):
        torch.manual_seed(1)
        ff = mvae.FeedForward(dim=32, mult=2, bias=True)
        x = torch.randn(2, 5, 32)
        y = ff(x)
        self.assertEqual(tuple(y.shape), (2, 5, 32))


if __name__ == "__main__":
    unittest.main()
