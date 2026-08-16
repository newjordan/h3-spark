from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from comfy.text_encoders.minimax import process_video_block
from comfy.text_encoders.qwen_vl import process_qwen2vl_images
from comfy_extras.nodes_minimax_h3 import _resize, _resize_unique


class ProcessQwenImagesTests(unittest.TestCase):
    def test_same_size_does_not_mutate_input(self):
        torch.manual_seed(0)
        img = torch.rand(1, 64, 96, 3)
        clone = img.clone()
        process_qwen2vl_images(
            img, patch_size=16, merge_size=2,
            image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5],
        )
        self.assertTrue(torch.equal(img, clone))

    def test_same_size_matches_forced_interpolate(self):
        torch.manual_seed(1)
        img = torch.rand(1, 64, 96, 3)
        a, ga = process_qwen2vl_images(
            img, patch_size=16, merge_size=2,
            image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5],
        )
        # Stock interpolate path on an already-aligned tensor.
        x = img[0].permute(2, 0, 1).unsqueeze(0)
        y = F.interpolate(x, size=(64, 96), mode="bilinear", align_corners=False)
        self.assertTrue(torch.equal(x, y))
        self.assertEqual(tuple(ga.shape), (1, 3))
        self.assertEqual(int(ga[0, 1]), 64 // 16)
        self.assertEqual(int(ga[0, 2]), 96 // 16)
        self.assertEqual(a.shape[0], (64 // 16) * (96 // 16))

    def test_vectorized_norm_matches_loop(self):
        torch.manual_seed(2)
        pix = torch.rand(3, 32, 32)
        mean_l = [0.5, 0.4, 0.3]
        std_l = [0.2, 0.25, 0.3]
        loop = pix.clone()
        for c in range(3):
            loop[c] = (pix[c] - mean_l[c]) / std_l[c]
        mean = pix.new_tensor(mean_l).view(3, 1, 1)
        std = pix.new_tensor(std_l).view(3, 1, 1)
        vec = (pix - mean) / std
        self.assertTrue(torch.equal(loop, vec))


class ProcessVideoBlockTests(unittest.TestCase):
    def test_same_size_does_not_mutate(self):
        torch.manual_seed(5)
        frames = torch.rand(2, 64, 96, 3)
        clone = frames.clone()
        process_video_block(frames)
        self.assertTrue(torch.equal(frames, clone))


class ResizeUniqueTests(unittest.TestCase):
    def test_identical_sources_resize_once(self):
        calls = {"n": 0}
        orig = _resize

        def tap(image, width, height, crop):
            calls["n"] += 1
            return orig(image, width, height, crop)

        import comfy_extras.nodes_minimax_h3 as n
        n._resize = tap
        try:
            cache = []
            img = torch.rand(1, 48, 64, 3)
            a = _resize_unique(img, 32, 32, "disabled", cache)
            b = _resize_unique(img.clone(), 32, 32, "disabled", cache)
            self.assertEqual(calls["n"], 1)
            self.assertIs(a, b)
        finally:
            n._resize = orig


if __name__ == "__main__":
    unittest.main()
