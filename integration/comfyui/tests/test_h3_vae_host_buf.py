from __future__ import annotations

import os
import unittest

import torch

from comfy.ldm.minimax import vae as mvae


class VaeHostBufTests(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.get(mvae._HOST_BUF_ENV)
        os.environ[mvae._HOST_BUF_ENV] = "auto"

    def tearDown(self):
        if self._env is None:
            os.environ.pop(mvae._HOST_BUF_ENV, None)
        else:
            os.environ[mvae._HOST_BUF_ENV] = self._env

    def test_env_default_on(self):
        os.environ[mvae._HOST_BUF_ENV] = "auto"
        self.assertTrue(mvae._host_buf_enabled())
        os.environ[mvae._HOST_BUF_ENV] = "off"
        self.assertFalse(mvae._host_buf_enabled())

    def test_matches_to_copy_true(self):
        if not torch.cuda.is_available():
            self.skipTest("cuda")
        torch.manual_seed(0)
        dec = torch.randn(1, 3, 4, 16, 24, device="cuda", dtype=torch.float16)
        owner = type("O", (), {})()
        stock = dec.to(device="cpu", dtype=torch.float16, copy=True)
        staged = mvae.stage_output_host(owner, dec, torch.device("cpu"), torch.float16)
        self.assertTrue(torch.equal(stock, staged))
        self.assertEqual(staged.device.type, "cpu")

    def test_second_copy_matches_and_keeps_previous(self):
        if not torch.cuda.is_available():
            self.skipTest("cuda")
        torch.manual_seed(1)
        a = torch.randn(1, 3, 4, 8, 8, device="cuda", dtype=torch.float16)
        b = torch.randn(1, 3, 4, 8, 8, device="cuda", dtype=torch.float16)
        owner = type("O", (), {})()
        ha = mvae.stage_output_host(owner, a, torch.device("cpu"), torch.float16)
        a_cpu = ha.clone()
        hb = mvae.stage_output_host(owner, b, torch.device("cpu"), torch.float16)
        self.assertTrue(torch.equal(hb, b.cpu()))
        # flip-flop: first slot still holds a
        self.assertTrue(torch.equal(ha, a_cpu))

    def test_env_off_uses_fresh_to(self):
        if not torch.cuda.is_available():
            self.skipTest("cuda")
        os.environ[mvae._HOST_BUF_ENV] = "off"
        dec = torch.randn(1, 3, 2, 8, 8, device="cuda", dtype=torch.float16)
        owner = type("O", (), {})()
        out = mvae.stage_output_host(owner, dec, torch.device("cpu"), torch.float16)
        self.assertFalse(hasattr(owner, "_host_out_slots"))
        self.assertTrue(torch.equal(out, dec.to(device="cpu", dtype=torch.float16, copy=True)))

    def test_module_method_exists(self):
        v = mvae.MiniMaxH3VideoVAE()
        self.assertTrue(callable(v.stage_output_host))
