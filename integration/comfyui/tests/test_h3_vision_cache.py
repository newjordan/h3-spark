from __future__ import annotations

import unittest

import torch

from comfy.text_encoders.minimax import vision_cache_lookup, vision_cache_store


class VisionCacheTests(unittest.TestCase):
    def test_identical_pixels_hit(self):
        cache = []
        img = torch.rand(1, 8, 8, 3)
        out = (torch.ones(1, 4, 8), {"grid": torch.tensor([1, 2, 2])})
        vision_cache_store(cache, img, False, out)
        hit = vision_cache_lookup(cache, img.clone(), False)
        self.assertIs(hit, out)

    def test_six_identical_one_store(self):
        cache = []
        img = torch.rand(2, 16, 16, 3)
        outs = []
        for i in range(6):
            hit = vision_cache_lookup(cache, img, False)
            if hit is None:
                hit = (torch.full((1, 3, 4), float(i + 1)), {"i": i})
                vision_cache_store(cache, img, False, hit)
            outs.append(hit)
        self.assertEqual(len(cache), 1)
        self.assertTrue(all(o is outs[0] for o in outs))

    def test_video_flag_is_part_of_key(self):
        cache = []
        img = torch.rand(2, 8, 8, 3)
        a = (torch.zeros(1, 2, 4), {})
        b = (torch.ones(1, 2, 4), {})
        vision_cache_store(cache, img, False, a)
        vision_cache_store(cache, img, True, b)
        self.assertIs(vision_cache_lookup(cache, img, False), a)
        self.assertIs(vision_cache_lookup(cache, img, True), b)

    def test_different_pixels_miss(self):
        cache = []
        vision_cache_store(cache, torch.zeros(1, 4, 4, 3), False, ("z",))
        self.assertIsNone(vision_cache_lookup(cache, torch.ones(1, 4, 4, 3), False))

    def test_snapshot_survives_later_mutate(self):
        cache = []
        img = torch.zeros(1, 4, 4, 3)
        vision_cache_store(cache, img, False, ("ok",))
        img.add_(1)
        later = torch.zeros(1, 4, 4, 3)
        self.assertEqual(vision_cache_lookup(cache, later, False), ("ok",))


if __name__ == "__main__":
    unittest.main()
