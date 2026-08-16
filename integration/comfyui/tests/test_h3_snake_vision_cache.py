from __future__ import annotations

import unittest

import torch

from comfy.ldm.minimax.audio_vae import Snake1d, SnakeBeta, snake
from comfy.ldm.minimax import vae as mvae
from comfy.text_encoders.qwen_vl import VisionRotaryEmbedding
import comfy.ops as cop


class SnakeCacheTests(unittest.TestCase):
    def test_snake1d_matches_stock_and_caches(self):
        torch.manual_seed(0)
        m = Snake1d(8)
        torch.nn.init.normal_(m.alpha, 0.5, 0.1)
        x = torch.randn(2, 8, 16)
        a = m(x.clone())
        a2 = m(x.clone())
        alpha = m.alpha.detach()
        ref = snake(x.clone(), alpha, (alpha + 1e-9).reciprocal())
        self.assertTrue(torch.equal(a, ref) or torch.allclose(a, ref, rtol=0, atol=0))
        self.assertTrue(torch.equal(a, a2))
        c1 = m._coeffs(x)
        c2 = m._coeffs(x)
        self.assertIs(c1[0], c2[0])
        self.assertIs(c1[1], c2[1])

    def test_snakebeta_matches_eager_exp(self):
        torch.manual_seed(1)
        m = SnakeBeta(6)
        torch.nn.init.normal_(m.alpha, 0.0, 0.2)
        torch.nn.init.normal_(m.beta, 0.0, 0.2)
        x = torch.randn(1, 6, 20)
        out = m(x.clone())
        alpha = torch.exp(m.alpha.detach()).view(1, -1, 1)
        beta = torch.exp(m.beta.detach()).view(1, -1, 1)
        ref = snake(x.clone(), alpha, (beta + 1e-9).reciprocal())
        self.assertTrue(torch.equal(out, ref) or torch.allclose(out, ref, rtol=0, atol=0))


class LatentsAffineCacheTests(unittest.TestCase):
    def test_affine_second_call_same_object(self):
        v = mvae.MiniMaxH3VideoVAE(operations=cop.disable_weight_init, tile_size=256)
        z = torch.randn(1, 24, 2, 4, 4)
        a1, b1 = v._latents_affine(z)
        a2, b2 = v._latents_affine(z)
        self.assertIs(a1, a2)
        self.assertIs(b1, b2)
        self.assertEqual(tuple(a1.shape), (1, 24, 1, 1, 1))


class VisionGridCacheTests(unittest.TestCase):
    def test_rotary_cache_hit(self):
        rot = VisionRotaryEmbedding(dim=8)
        f1 = rot(5, torch.device("cpu"))
        f2 = rot(5, torch.device("cpu"))
        self.assertIs(f1, f2)
        f3 = rot(6, torch.device("cpu"))
        self.assertEqual(f3.shape[0], 6)
        self.assertFalse(f3.shape[0] == f1.shape[0] and f3.data_ptr() == f1.data_ptr())


if __name__ == "__main__":
    unittest.main()
