from __future__ import annotations

import os
import unittest

import torch

from comfy.text_encoders import minimax as mm
from comfy.text_encoders.qwen_vl import process_qwen2vl_images


def _fake_visual(flat, grid):
    tokens = []
    off = 0
    for _t, h, w in grid.tolist():
        n = int(h * w)
        chunk = flat[off:off + n]
        off += n
        merged = chunk.reshape(n // 4, 4, -1).mean(dim=1)
        if merged.shape[-1] < 8:
            merged = torch.nn.functional.pad(merged, (0, 8 - merged.shape[-1]))
        else:
            merged = merged[..., :8]
        tokens.append(merged)
    out = torch.cat(tokens, 0)
    return out, [out.clone(), out.clone() + 1]


class _FakeTE:
    def __init__(self):
        self.visual = _fake_visual
        self._h3_vision_cache = []


class VisionBatchTests(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.get(mm._VISION_BATCH_ENV)
        os.environ[mm._VISION_BATCH_ENV] = "auto"

    def tearDown(self):
        if self._env is None:
            os.environ.pop(mm._VISION_BATCH_ENV, None)
        else:
            os.environ[mm._VISION_BATCH_ENV] = self._env

    def test_env_off_disables(self):
        os.environ[mm._VISION_BATCH_ENV] = "off"
        self.assertFalse(mm._vision_batch_enabled())
        os.environ[mm._VISION_BATCH_ENV] = "auto"
        self.assertTrue(mm._vision_batch_enabled())

    def test_same_grid_one_visual_call(self):
        torch.manual_seed(0)
        a = torch.rand(1, 64, 64, 3)
        b = torch.rand(1, 64, 64, 3)
        te = _FakeTE()
        calls = []

        def tap(flat, grid):
            calls.append((tuple(flat.shape), int(grid.shape[0])))
            return _fake_visual(flat, grid)

        te.visual = tap
        tokens = [[
            ({"type": "image", "data": a}, 1.0),
            ({"type": "image", "data": b}, 1.0),
        ]]
        mm.prefetch_vision_batch(te, tokens, a.device)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], 2)
        self.assertEqual(len(te._h3_vision_cache), 2)

    def test_batch_matches_serial_fake(self):
        torch.manual_seed(1)
        imgs = [torch.rand(1, 64, 64, 3) for _ in range(3)]
        serial = []
        for img in imgs:
            flat, grid = process_qwen2vl_images(
                img, patch_size=16,
                image_mean=mm.QWEN_IMAGE_MEAN, image_std=mm.QWEN_IMAGE_STD,
            )
            merged, ds = _fake_visual(flat, grid)
            serial.append((merged, ds))
        te = _FakeTE()
        te.visual = _fake_visual
        tokens = [[({"type": "image", "data": im}, 1.0) for im in imgs]]
        mm.prefetch_vision_batch(te, tokens, imgs[0].device)
        for img, (ref_m, ref_ds) in zip(imgs, serial):
            hit = mm.vision_cache_lookup(te._h3_vision_cache, img, False)
            self.assertIsNotNone(hit)
            got_m, extra = hit
            self.assertTrue(torch.equal(got_m, ref_m))
            self.assertTrue(torch.equal(extra["deepstack"][0], ref_ds[0]))
            self.assertTrue(torch.equal(extra["deepstack"][1], ref_ds[1]))

    def test_mixed_grids_group_separately(self):
        torch.manual_seed(2)
        a = torch.rand(1, 64, 64, 3)
        b = torch.rand(1, 96, 64, 3)
        c = torch.rand(1, 64, 64, 3)
        te = _FakeTE()
        calls = []

        def tap(flat, grid):
            calls.append(int(grid.shape[0]))
            return _fake_visual(flat, grid)

        te.visual = tap
        tokens = [[
            ({"type": "image", "data": a}, 1.0),
            ({"type": "image", "data": b}, 1.0),
            ({"type": "image", "data": c}, 1.0),
        ]]
        mm.prefetch_vision_batch(te, tokens, a.device)
        # 64x64 pair batched (B=2); 96x64 left to serial preprocess
        self.assertEqual(sorted(calls), [2])
        self.assertEqual(len(te._h3_vision_cache), 2)

    def test_off_skips_prefetch(self):
        os.environ[mm._VISION_BATCH_ENV] = "off"
        te = _FakeTE()
        img = torch.rand(1, 64, 64, 3)
        mm.prefetch_vision_batch(
            te, [[({"type": "image", "data": img}, 1.0)]], img.device
        )
        self.assertEqual(te._h3_vision_cache, [])

    def test_qwen35_grid_caches_hit(self):
        from comfy.text_encoders.qwen35 import Qwen35VisionModel
        import comfy.ops as cop

        cfg = dict(
            hidden_size=64, intermediate_size=128, depth=1, num_heads=4,
            patch_size=16, temporal_patch_size=2, in_channels=3,
            spatial_merge_size=2, num_position_embeddings=16,
            out_hidden_size=32,
        )
        v = Qwen35VisionModel(cfg, device="cpu", dtype=torch.float32, ops=cop.disable_weight_init)
        grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
        r1 = v.rot_pos_emb(grid)
        r2 = v.rot_pos_emb(grid)
        p1 = v.fast_pos_embed_interpolate(grid)
        p2 = v.fast_pos_embed_interpolate(grid)
        self.assertIs(r1, r2)
        self.assertIs(p1, p2)

    def test_equal_length_pack_detects_uniform_spans(self):
        cu = torch.tensor([0, 16, 32, 48], dtype=torch.int32)
        self.assertEqual(mm.vision_equal_length_pack(cu), (3, 16))
        self.assertIsNone(mm.vision_equal_length_pack(torch.tensor([0, 16], dtype=torch.int32)))
        self.assertIsNone(mm.vision_equal_length_pack(torch.tensor([0, 16, 40], dtype=torch.int32)))
        self.assertIsNone(mm.vision_equal_length_pack(None))

    def test_packed_sdpa_matches_serial_cpu(self):
        from comfy.ldm.modules.attention import attention_pytorch

        torch.manual_seed(3)
        n, s, h, d = 3, 8, 4, 16
        q = torch.randn(n * s, h, d)
        k = torch.randn(n * s, h, d)
        v = torch.randn(n * s, h, d)
        serial = []
        for i in range(n):
            a, b = i * s, (i + 1) * s
            serial.append(attention_pytorch(
                q[a:b].transpose(0, 1).unsqueeze(0),
                k[a:b].transpose(0, 1).unsqueeze(0),
                v[a:b].transpose(0, 1).unsqueeze(0),
                h, skip_reshape=True,
            ))
        ref = torch.cat(serial, dim=1).reshape(n * s, -1)
        got = mm.vision_attn_packed(q, k, v, n, s, h, attention_pytorch)
        self.assertTrue(torch.equal(ref, got))

    def test_attn_pack_install_is_idempotent(self):
        from comfy.text_encoders.qwen35 import Qwen35VisionAttention
        self.assertTrue(getattr(Qwen35VisionAttention, "_h3_attn_pack_patched", False))
        mm._install_qwen35_vision_attn_pack()
        self.assertTrue(Qwen35VisionAttention._h3_attn_pack_patched)


if __name__ == "__main__":
    unittest.main()
