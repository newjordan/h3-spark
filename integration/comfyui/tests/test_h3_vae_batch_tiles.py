from __future__ import annotations

import os
import unittest
from unittest import mock

import torch

from comfy.ldm.minimax import vae as mvae
import comfy.ops as cop


def _fake_decode_pixels(z):
    """Channel-mean upsample: batching is bit-exact vs per-slice calls."""
    marker = z.to(torch.float32).mean(dim=1, keepdim=True).repeat(1, 3, 1, 1, 1)
    return (
        marker.repeat_interleave(4, dim=2)
        .repeat_interleave(16, dim=3)
        .repeat_interleave(16, dim=4)
    )


def _fake_encode_moments(x):
    """Mean-pool 16x: batching is bit-exact vs per-slice calls."""
    marker = x.to(torch.float32).mean(dim=1, keepdim=True).repeat(1, 48, 1, 1, 1)
    return marker[..., ::16, ::16]


class VaeBatchTilesTests(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.get(mvae._BATCH_TILES_ENV)
        self._max = os.environ.get(mvae._BATCH_TILES_MAX_ENV)
        os.environ[mvae._BATCH_TILES_ENV] = "auto"
        os.environ[mvae._BATCH_TILES_MAX_ENV] = "64"
        self.vae = mvae.MiniMaxH3VideoVAE(
            operations=cop.disable_weight_init, tile_size=256
        )
        self.vae._decode_pixels = _fake_decode_pixels
        self.vae._encode_moments = _fake_encode_moments

    def tearDown(self):
        if self._env is None:
            os.environ.pop(mvae._BATCH_TILES_ENV, None)
        else:
            os.environ[mvae._BATCH_TILES_ENV] = self._env
        if self._max is None:
            os.environ.pop(mvae._BATCH_TILES_MAX_ENV, None)
        else:
            os.environ[mvae._BATCH_TILES_MAX_ENV] = self._max

    def test_env_off_disables_batching(self):
        os.environ[mvae._BATCH_TILES_ENV] = "off"
        self.assertFalse(mvae._batch_tiles_enabled())
        os.environ[mvae._BATCH_TILES_ENV] = "stock"
        self.assertFalse(mvae._batch_tiles_enabled())
        os.environ[mvae._BATCH_TILES_ENV] = "auto"
        self.assertTrue(mvae._batch_tiles_enabled())

    def test_decode_pixel_slices_groups_and_chunks(self):
        calls = []

        def tap(z):
            calls.append(tuple(z.shape))
            return _fake_decode_pixels(z)

        self.vae._decode_pixels = tap
        a = torch.randn(1, 24, 7, 16, 16)
        b = torch.randn(1, 24, 7, 16, 16)
        c = torch.randn(1, 24, 5, 16, 16)
        os.environ[mvae._BATCH_TILES_MAX_ENV] = "2"
        outs = self.vae._decode_pixel_slices([a, b, c, a.clone()])
        self.assertEqual(len(outs), 4)
        self.assertEqual(outs[0].shape[0], 1)
        # two 7-token tiles + clone of a fit in one max=2 group, leftover 7-token
        # clone grouped with the first two by shape; third shape is separate.
        shapes_b = [s[0] for s in calls]
        self.assertIn(2, shapes_b)
        self.assertTrue(all(o.shape[0] == 1 for o in outs))

    def test_temporal_batch_matches_serial_and_reduces_launches(self):
        torch.manual_seed(0)
        z = torch.randn(1, 24, 22, 30, 54)

        calls = {"n": 0, "max_b": 0}

        def tap(lat):
            calls["n"] += 1
            calls["max_b"] = max(calls["max_b"], lat.shape[0])
            return _fake_decode_pixels(lat)

        self.vae._decode_pixels = tap
        os.environ[mvae._BATCH_TILES_ENV] = "off"
        serial = self.vae.decode(z)
        serial_launches = calls["n"]

        calls["n"] = 0
        calls["max_b"] = 0
        os.environ[mvae._BATCH_TILES_ENV] = "auto"
        batched = self.vae.decode(z)

        self.assertEqual(tuple(serial.shape), tuple(batched.shape))
        self.assertTrue(torch.equal(serial, batched))
        self.assertGreater(serial_launches, 1)
        self.assertLess(calls["n"], serial_launches)
        self.assertGreater(calls["max_b"], 1)

    def test_oom_falls_back_to_serial(self):
        torch.manual_seed(1)
        z = torch.randn(1, 24, 8, 30, 54)
        self.vae._decode_pixels = _fake_decode_pixels
        os.environ[mvae._BATCH_TILES_ENV] = "off"
        expected = self.vae.decode(z.clone())

        def boom(lat):
            if lat.shape[0] > 1:
                raise RuntimeError("CUDA out of memory")
            return _fake_decode_pixels(lat)

        self.vae._decode_pixels = boom
        os.environ[mvae._BATCH_TILES_ENV] = "auto"
        with mock.patch.object(mvae.logging, "warning") as warn:
            actual = self.vae.decode(z)
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(warn.called)

    def test_requires_grad_stays_serial(self):
        z = torch.randn(1, 24, 8, 30, 54, requires_grad=True)
        self.assertFalse(self.vae._should_batch_tiles(z))

    def test_batch_dim_greater_than_one_is_allowed(self):
        z = torch.randn(3, 24, 8, 16, 16)
        self.assertTrue(self.vae._should_batch_tiles(z))

    def test_tiled_encode_batches_and_matches_serial(self):
        torch.manual_seed(3)
        x = torch.randn(1, 3, 1, 512, 800)
        calls = {"n": 0, "max_b": 0}

        def tap(tile):
            calls["n"] += 1
            calls["max_b"] = max(calls["max_b"], tile.shape[0])
            return _fake_encode_moments(tile)

        self.vae._encode_moments = tap
        os.environ[mvae._BATCH_TILES_ENV] = "off"
        serial = self.vae.tiled_encode(x)
        serial_n = calls["n"]
        calls["n"] = 0
        calls["max_b"] = 0
        os.environ[mvae._BATCH_TILES_ENV] = "auto"
        batched = self.vae.tiled_encode(x)
        self.assertEqual(tuple(serial.shape), tuple(batched.shape))
        self.assertTrue(torch.equal(serial, batched))
        self.assertGreater(serial_n, 1)
        self.assertLess(calls["n"], serial_n)
        self.assertGreater(calls["max_b"], 1)

    def test_encode_temporal_batches_clips(self):
        torch.manual_seed(4)
        self.vae.tiling = False
        x = torch.randn(1, 3, 34, 64, 64)
        calls = {"n": 0, "max_b": 0}

        def tap(clip):
            calls["n"] += 1
            calls["max_b"] = max(calls["max_b"], clip.shape[0])
            return _fake_encode_moments(clip)

        self.vae._encode_moments = tap
        os.environ[mvae._BATCH_TILES_ENV] = "off"
        serial = self.vae.encode_temporal(x.clone())
        serial_n = calls["n"]
        calls["n"] = 0
        calls["max_b"] = 0
        os.environ[mvae._BATCH_TILES_ENV] = "auto"
        batched = self.vae.encode_temporal(x.clone())
        self.assertEqual(tuple(serial.shape), tuple(batched.shape))
        self.assertTrue(torch.equal(serial, batched))
        self.assertEqual(serial_n, 2)
        self.assertEqual(calls["n"], 1)
        self.assertEqual(calls["max_b"], 2)


class TemporalGNTests(unittest.TestCase):
    def test_4d_reshape_matches_t1_5d(self):
        torch.manual_seed(0)
        gn = mvae.TemporalIsolatedGroupNorm(32, 64, eps=1e-6, affine=True)
        torch.nn.init.normal_(gn.weight, 1.0, 0.1)
        torch.nn.init.normal_(gn.bias, 0.0, 0.1)
        x = torch.randn(2, 64, 3, 8, 10)
        got = gn(x)
        b, c, t, h, w = x.shape
        ref = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, 1, h, w)
        ref = torch.nn.GroupNorm.forward(gn, ref)
        ref = ref.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        self.assertTrue(torch.equal(got, ref) or torch.allclose(got, ref, rtol=0, atol=0))


if __name__ == "__main__":
    unittest.main()
