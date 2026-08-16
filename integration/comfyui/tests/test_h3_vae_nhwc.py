from __future__ import annotations

import os
import unittest

import torch

from comfy.ldm.minimax import vae as mvae


class VaeNhwcTests(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.get(mvae._NHWC_ENV)
        os.environ[mvae._NHWC_ENV] = "auto"

    def tearDown(self):
        if self._env is None:
            os.environ.pop(mvae._NHWC_ENV, None)
        else:
            os.environ[mvae._NHWC_ENV] = self._env

    def test_env_default_on(self):
        os.environ[mvae._NHWC_ENV] = "auto"
        self.assertTrue(mvae._nhwc_enabled())
        os.environ[mvae._NHWC_ENV] = "off"
        self.assertFalse(mvae._nhwc_enabled())

    def test_permute_equals_movedim(self):
        torch.manual_seed(0)
        x = torch.randn(1, 3, 4, 8, 10)
        y, last = mvae.maybe_nhwc_bt_hwc(x)
        self.assertTrue(last)
        self.assertTrue(y.is_contiguous())
        self.assertTrue(torch.equal(y, x.movedim(1, -1).contiguous()))

    def test_env_off_unchanged(self):
        os.environ[mvae._NHWC_ENV] = "off"
        x = torch.randn(1, 3, 2, 4, 6)
        y, last = mvae.maybe_nhwc_bt_hwc(x)
        self.assertFalse(last)
        self.assertIs(y, x)

    def test_requires_grad_unchanged(self):
        x = torch.randn(1, 3, 2, 4, 6, requires_grad=True)
        y, last = mvae.maybe_nhwc_bt_hwc(x)
        self.assertFalse(last)
        self.assertIs(y, x)

    def test_module_flag_default_false(self):
        v = mvae.MiniMaxH3VideoVAE()
        self.assertFalse(v.h3_channels_last)
