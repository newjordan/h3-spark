from __future__ import annotations

import unittest

import torch

from comfy.ldm.minimax.model import cond_rows_from_unique, pack_audio, patchify_video


class CondRowsUniqueTests(unittest.TestCase):
    def test_six_identical_build_once(self):
        calls = {"n": 0}
        z = torch.randn(1, 24, 2, 4, 6)

        def build(x):
            calls["n"] += 1
            return patchify_video(x, (1, 2, 2))

        items = [z.clone() for _ in range(6)]
        out = cond_rows_from_unique(items, build)
        self.assertEqual(calls["n"], 1)
        once = patchify_video(z, (1, 2, 2))
        self.assertTrue(torch.equal(out, torch.cat([once] * 6, dim=0)))

    def test_two_different_build_each(self):
        calls = {"n": 0}

        def build(x):
            calls["n"] += 1
            return pack_audio(x)

        a = torch.zeros(1, 32, 2, 4)
        b = torch.ones(1, 32, 2, 4)
        out = cond_rows_from_unique([a, b], build)
        self.assertEqual(calls["n"], 2)
        self.assertTrue(torch.equal(out, torch.cat([pack_audio(a), pack_audio(b)], dim=0)))

    def test_empty(self):
        self.assertIsNone(cond_rows_from_unique([], lambda z: z))


if __name__ == "__main__":
    unittest.main()
