from __future__ import annotations

import unittest

import torch

from comfy.ldm.minimax.model import patchify_video, unpatchify_video
from comfy.ldm.minimax.audio_vae import LowPassFilter1d, UpSample1d


class PatchifyPermuteTests(unittest.TestCase):
    def test_patchify_unpatchify_roundtrip(self):
        torch.manual_seed(0)
        x = torch.randn(1, 24, 4, 8, 12)
        rows = patchify_video(x, (1, 2, 2))
        y = unpatchify_video(rows, 4, 4, 6, c=24, patch_size=(1, 2, 2))
        self.assertTrue(torch.equal(x, y))

    def test_permute_matches_einsum(self):
        torch.manual_seed(1)
        x = torch.randn(2, 24, 2, 6, 8)
        b, c, tf, hf, wf = x.shape
        pt, ph, pw = 1, 2, 2
        t, h, w = tf // pt, hf // ph, wf // pw
        view = x.reshape(b, c, t, pt, h, ph, w, pw)
        ein = torch.einsum("nctrhpwq->nthwcrpq", view)
        perm = view.permute(0, 2, 4, 6, 1, 3, 5, 7)
        self.assertTrue(torch.equal(ein, perm))
        rows = ein.reshape(b * t * h * w, c * pt * ph * pw)
        back_v = rows.reshape(b, t, h, w, c, pt, ph, pw)
        self.assertTrue(torch.equal(
            torch.einsum("nthwcrpq->nctrhpwq", back_v),
            back_v.permute(0, 4, 1, 5, 2, 6, 3, 7),
        ))


class GroupedFilterCacheTests(unittest.TestCase):
    def test_lowpass_cache_hit(self):
        lp = LowPassFilter1d()
        x = torch.randn(1, 4, 32)
        w1 = lp._grouped_filter(4, x)
        w2 = lp._grouped_filter(4, x)
        self.assertIs(w1, w2)
        y1 = lp(x)
        y2 = lp(x)
        self.assertTrue(torch.equal(y1, y2))

    def test_upsample_cache_hit(self):
        up = UpSample1d(ratio=2)
        x = torch.randn(1, 3, 16)
        w1 = up._grouped_filter(3, x)
        w2 = up._grouped_filter(3, x)
        self.assertIs(w1, w2)
        self.assertTrue(torch.equal(up(x), up(x)))


if __name__ == "__main__":
    unittest.main()
