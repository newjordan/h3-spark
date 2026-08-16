from __future__ import annotations

import unittest

import torch

from comfy.ldm.minimax import vae as mvae
import comfy.ops as cop


class RopeAndBlendCacheTests(unittest.TestCase):
    def test_rotary_cache_hit_is_exact(self):
        dec = mvae.ViT3DDecoder(
            num_layers=1, heads=2, dim_head=8, in_channels=4, out_channels=3,
            operations=cop.disable_weight_init,
        )
        r1 = dec._cached_rotary(2, 2, 4, 4, torch.device("cpu"), torch.float32)
        calls = {"n": 0}
        orig = dec.pos_embed.forward

        def tap(ids):
            calls["n"] += 1
            return orig(ids)

        dec.pos_embed.forward = tap
        r2 = dec._cached_rotary(2, 2, 4, 4, torch.device("cpu"), torch.float32)
        self.assertEqual(calls["n"], 0)
        self.assertTrue(torch.equal(r1, r2))

    def test_rotary_cache_misses_on_grid(self):
        dec = mvae.ViT3DDecoder(
            num_layers=1, heads=2, dim_head=8, in_channels=4, out_channels=3,
            operations=cop.disable_weight_init,
        )
        dec(torch.randn(1, 4, 2, 4, 4))
        calls = {"n": 0}
        orig = dec.pos_embed.forward

        def tap(ids):
            calls["n"] += 1
            return orig(ids)

        dec.pos_embed.forward = tap
        dec(torch.randn(2, 4, 2, 4, 4))
        self.assertEqual(calls["n"], 0)
        dec(torch.randn(1, 4, 3, 4, 4))
        self.assertEqual(calls["n"], 1)

    def test_rotary_expand_matches_full_batch(self):
        dec = mvae.ViT3DDecoder(
            num_layers=1, heads=2, dim_head=8, in_channels=4, out_channels=3,
            operations=cop.disable_weight_init,
        )
        expanded = dec._cached_rotary(3, 2, 4, 4, torch.device("cpu"), torch.float32)
        dec._rope_cache = None
        img_ids = mvae.create_token_ids((2, 4, 4), torch.device("cpu"), torch.float32).expand(3, -1, -1)
        suffix = torch.zeros((3, 1 + dec.num_register_tokens, 3), dtype=torch.float32)
        full = dec.pos_embed(torch.cat([img_ids, suffix], dim=1))
        self.assertTrue(torch.equal(expanded, full))

    def test_suffix_expand_matches_full_batch(self):
        dec = mvae.ViT3DDecoder(
            num_layers=1, heads=2, dim_head=8, in_channels=4, out_channels=3,
            operations=cop.disable_weight_init,
        )
        ref = torch.randn(3, 4, 16)
        expanded = dec._cached_suffix(3, ref)
        dec._suffix_cache = None
        full = torch.cat(
            [
                cop.cast_to_input(dec.register_tokens, ref).expand(3, -1, -1),
                ref.new_zeros(3, 1, ref.shape[-1]),
            ],
            dim=1,
        )
        self.assertTrue(torch.equal(expanded, full))

    def test_blend_cache_matches_fresh_weights(self):
        vae = mvae.MiniMaxH3VideoVAE(
            operations=cop.disable_weight_init, tile_size=256
        )
        a = torch.arange(8, dtype=torch.float32).view(1, 1, 1, 8)
        b = torch.arange(8, 16, dtype=torch.float32).view(1, 1, 1, 8)
        out1 = vae.blend(a, b, 4, dim=-1)
        out2 = vae.blend(a, b, 4, dim=-1)
        self.assertTrue(torch.equal(out1, out2))
        vae._blend_w_cache = None
        out3 = vae.blend(a, b, 4, dim=-1)
        self.assertTrue(torch.equal(out1, out3))


if __name__ == "__main__":
    unittest.main()
