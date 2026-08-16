from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from comfy_extras.nodes_minimax_h3 import _encode_unique_image


class _CountingVae:
    def __init__(self, mutate=False):
        self.calls = 0
        self.mutate = mutate

    def encode(self, x):
        self.calls += 1
        if self.mutate:
            x.add_(0.25)
        return x.mean().reshape(1, 1, 1, 1, 1).expand(1, 24, 1, 2, 2).clone()


class RefEncodeDedupTests(unittest.TestCase):
    def test_identical_refs_encode_once(self):
        vae = _CountingVae()
        img = torch.rand(1, 8, 8, 3)
        cache = []
        z0 = _encode_unique_image(vae, img, cache)
        z1 = _encode_unique_image(vae, img.clone(), cache)
        z2 = _encode_unique_image(vae, img.clone(), cache)
        self.assertEqual(vae.calls, 1)
        self.assertIs(z0, z1)
        self.assertIs(z1, z2)

    def test_different_refs_encode_each(self):
        vae = _CountingVae()
        cache = []
        a = torch.zeros(1, 4, 4, 3)
        b = torch.ones(1, 4, 4, 3)
        za = _encode_unique_image(vae, a, cache)
        zb = _encode_unique_image(vae, b, cache)
        self.assertEqual(vae.calls, 2)
        self.assertFalse(torch.equal(za, zb))

    def test_snapshot_survives_in_place_encode(self):
        vae = _CountingVae(mutate=True)
        cache = []
        img = torch.zeros(1, 4, 4, 3)
        _encode_unique_image(vae, img, cache)
        later = torch.zeros(1, 4, 4, 3)
        _encode_unique_image(vae, later, cache)
        self.assertEqual(vae.calls, 1)

    def test_shape_mismatch_does_not_hit(self):
        vae = _CountingVae()
        cache = []
        _encode_unique_image(vae, torch.zeros(1, 4, 4, 3), cache)
        _encode_unique_image(vae, torch.zeros(1, 8, 8, 3), cache)
        self.assertEqual(vae.calls, 2)


if __name__ == "__main__":
    unittest.main()
