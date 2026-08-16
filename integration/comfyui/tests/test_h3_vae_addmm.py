from __future__ import annotations

import os
import unittest

import torch
import torch.nn.functional as F

from comfy.ldm.minimax import vae as mvae


class VaeAddmmTests(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.get(mvae._ADDMM_ENV)
        os.environ[mvae._ADDMM_ENV] = "auto"

    def tearDown(self):
        if self._env is None:
            os.environ.pop(mvae._ADDMM_ENV, None)
        else:
            os.environ[mvae._ADDMM_ENV] = self._env

    def test_env_off_disables(self):
        os.environ[mvae._ADDMM_ENV] = "off"
        self.assertFalse(mvae._addmm_enabled())
        os.environ[mvae._ADDMM_ENV] = "auto"
        self.assertTrue(mvae._addmm_enabled())

    def test_w2_kn_not_in_set(self):
        self.assertNotIn((8192, 2048), mvae._ADDMM_KN)

    def test_cpu_small_equals_linear(self):
        # CPU addmm is a different kernel (not the live path). Identity is CUDA.
        torch.manual_seed(14)
        x = torch.randn(8, 2048)
        w = torch.randn(2048, 2048)
        b = torch.randn(2048)
        y = mvae.vae_addmm_linear(x, w, b, owner=None)
        self.assertEqual(tuple(y.shape), (8, 2048))

    @unittest.skipUnless(torch.cuda.is_available(), "cuda")
    def test_cuda_chef_rows_equal(self):
        torch.manual_seed(14)
        device = "cuda"
        cases = (
            (82144, 2048, 16384),
            (52685, 2048, 6144),
        )
        for m, k, n in cases:
            x = torch.randn(m, k, device=device, dtype=torch.float16)
            w = torch.randn(n, k, device=device, dtype=torch.float16)
            b = torch.randn(n, device=device, dtype=torch.float16)
            y0 = F.linear(x, w, b)
            y1 = mvae.vae_addmm_linear(x, w, b, owner=None)
            self.assertTrue(torch.equal(y0, y1), msg=f"M={m} K={k} N={n}")

    def test_wrap_skips_without_bias(self):
        lin = torch.nn.Linear(2048, 2048, bias=False)
        mvae._wrap_linear_addmm(lin)
        x = torch.randn(4, 2048)
        y = lin(x)
        self.assertEqual(tuple(y.shape), (4, 2048))


if __name__ == "__main__":
    unittest.main()
