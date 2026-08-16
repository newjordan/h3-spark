from __future__ import annotations

import os
import unittest

import torch

from comfy.ldm.minimax import vae as mvae


class VaeHostFp16Tests(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.get(mvae._HOST_FP16_ENV)
        os.environ[mvae._HOST_FP16_ENV] = "auto"

    def tearDown(self):
        if self._env is None:
            os.environ.pop(mvae._HOST_FP16_ENV, None)
        else:
            os.environ[mvae._HOST_FP16_ENV] = self._env

    def test_env_off_disables(self):
        os.environ[mvae._HOST_FP16_ENV] = "off"
        self.assertFalse(mvae._host_fp16_enabled())
        os.environ[mvae._HOST_FP16_ENV] = "auto"
        self.assertTrue(mvae._host_fp16_enabled())

    def test_cpu_tensor_unchanged(self):
        x = torch.randn(2, 3, 4, 5, 6)
        y = mvae._maybe_host_fp16(x)
        self.assertIs(y, x)
        self.assertEqual(y.dtype, torch.float32)

    def test_requires_grad_unchanged(self):
        if not torch.cuda.is_available():
            self.skipTest("cuda")
        x = torch.randn(2, 3, 4, 5, 6, device="cuda", requires_grad=True)
        y = mvae._maybe_host_fp16(x)
        self.assertIs(y, x)

    def test_cuda_fp32_matches_comfy_host_cast(self):
        if not torch.cuda.is_available():
            self.skipTest("cuda")
        torch.manual_seed(0)
        dec = torch.randn(1, 3, 8, 32, 48, device="cuda", dtype=torch.float32)
        stock = dec.to(device="cpu", dtype=torch.float16, copy=True)
        new = mvae._maybe_host_fp16(dec).to(device="cpu", dtype=torch.float16, copy=True)
        self.assertEqual(mvae._maybe_host_fp16(dec).dtype, torch.float16)
        self.assertTrue(torch.equal(stock, new))

    def test_env_off_keeps_fp32(self):
        if not torch.cuda.is_available():
            self.skipTest("cuda")
        os.environ[mvae._HOST_FP16_ENV] = "off"
        x = torch.randn(1, 3, 4, 8, 8, device="cuda", dtype=torch.float32)
        y = mvae._maybe_host_fp16(x)
        self.assertEqual(y.dtype, torch.float32)
        self.assertIs(y, x)
