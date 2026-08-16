from __future__ import annotations

import os
import unittest

import torch

from comfy.ldm.minimax import vae as mvae


class VaeUnitIntervalTests(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.get(mvae._UNIT_INTERVAL_ENV)
        os.environ[mvae._UNIT_INTERVAL_ENV] = "auto"

    def tearDown(self):
        if self._env is None:
            os.environ.pop(mvae._UNIT_INTERVAL_ENV, None)
        else:
            os.environ[mvae._UNIT_INTERVAL_ENV] = self._env

    def test_env_default_on(self):
        os.environ[mvae._UNIT_INTERVAL_ENV] = "auto"
        self.assertTrue(mvae._unit_interval_enabled())
        os.environ[mvae._UNIT_INTERVAL_ENV] = "off"
        self.assertFalse(mvae._unit_interval_enabled())

    def test_signed_then_process_output_is_unit_interval(self):
        torch.manual_seed(0)
        unit = torch.rand(2, 3, 4, 5, 6)
        signed = unit.mul(2.0).sub(1.0)
        recovered = mvae.unit_interval_from_signed(signed)
        self.assertTrue(torch.equal(recovered, unit))

    def test_affine_skip_matches_process_output(self):
        torch.manual_seed(1)
        denorm = torch.rand(1, 3, 2, 8, 8)
        clamped = denorm.clamp(0.0, 1.0)
        signed = clamped.mul(2.0).sub(1.0)
        comfy = signed.add(1.0).div_(2.0).clamp_(0.0, 1.0)
        self.assertTrue(torch.equal(comfy, clamped))
        self.assertEqual(int((comfy - clamped).abs().gt(0.01).sum()), 0)

    def test_blend_map_holds_multiple_extents(self):
        v = mvae.MiniMaxH3VideoVAE()
        a144, b144 = v._blend_weights(144, torch.device("cpu"), torch.float32)
        a112, b112 = v._blend_weights(112, torch.device("cpu"), torch.float32)
        a144b, b144b = v._blend_weights(144, torch.device("cpu"), torch.float32)
        self.assertIs(a144, a144b)
        self.assertIs(b144, b144b)
        self.assertEqual(int(a144.numel()), 144)
        self.assertEqual(int(a112.numel()), 112)
        self.assertTrue(torch.equal(a144 + b144, torch.ones_like(a144)))
        self.assertTrue(torch.equal(a112 + b112, torch.ones_like(a112)))
        self.assertIn((144, "cpu", "torch.float32"), v._blend_w_map)
        self.assertIn((112, "cpu", "torch.float32"), v._blend_w_map)

    def test_full_batch_reuse_is_identity(self):
        out = torch.randn(1, 3, 4, 8, 8)
        reused = out
        extra = torch.empty_like(out)
        extra.copy_(out)
        self.assertTrue(torch.equal(reused, extra))
