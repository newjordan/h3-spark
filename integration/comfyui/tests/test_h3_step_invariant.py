from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from comfy.ldm.minimax.model import (
    PackedLayout,
    _embed_cond_or_mix,
    _layout_mod_row_ids,
    _layout_text_tag_runs,
    cond_rows_from_unique,
    cond_unique_blocks,
    expand_unique_blocks,
    layout_device_tensor,
    layout_rope_table,
    pack_audio,
    patchify_video,
    project_mixed_rows,
    project_unique_cond_live,
    rope_freqs_from_inv,
    rope_rotation_table,
)


class StepInvariantTests(unittest.TestCase):
    def test_layout_rope_second_call_is_same_object(self):
        layout = PackedLayout(4, 2, 4, 4, 3)
        inv = torch.exp(-torch.arange(16, dtype=torch.float32) / 16.0)
        t1 = layout_rope_table(layout, inv, torch.device("cpu"), torch.float32)
        t2 = layout_rope_table(layout, inv, torch.device("cpu"), torch.float32)
        self.assertIs(t1, t2)
        fresh = rope_rotation_table(
            rope_freqs_from_inv(layout.position_ids, inv, torch.device("cpu")),
            torch.float32,
        )
        self.assertTrue(torch.equal(t1, fresh))

    def test_layout_device_tensor_cached(self):
        layout = PackedLayout(2, 2, 4, 4, 2)
        a = layout_device_tensor(layout, "img_update", layout.img_update, torch.device("cpu"))
        b = layout_device_tensor(layout, "img_update", layout.img_update, torch.device("cpu"))
        self.assertIs(a, b)
        self.assertTrue(torch.equal(a, layout.img_update))

    def test_project_mixed_rows_equals_mix_then_proj(self):
        torch.manual_seed(0)
        proj = nn.Linear(8, 16, bias=True)
        live = torch.randn(5, 8)
        cond = torch.randn(3, 8)
        update = torch.tensor([False, False, True, True, True, False, True, True])
        mixed = torch.empty(update.shape[0], 8)
        mixed[~update] = cond
        mixed[update] = live
        ref = proj(mixed)
        out = project_mixed_rows(proj, live, cond, update, torch.float32)
        self.assertTrue(torch.equal(out, ref) or torch.allclose(out, ref, rtol=0, atol=0))

    def test_project_mixed_rows_caches_cond(self):
        torch.manual_seed(1)
        proj = nn.Linear(4, 6, bias=True)
        live = torch.randn(2, 4)
        cond = torch.randn(2, 4)
        update = torch.tensor([True, False, True, False])
        owner = type("O", (), {})()
        a = project_mixed_rows(proj, live, cond, update, torch.float32, owner, "_e")
        cached = owner._e
        b = project_mixed_rows(proj, live + 1, cond, update, torch.float32, owner, "_e")
        self.assertIs(owner._e, cached)
        self.assertTrue(torch.equal(a[~update], b[~update]))
        self.assertFalse(torch.equal(a[update], b[update]))

    def test_cond_rows_unique_then_layout_cache(self):
        layout = PackedLayout(2, 2, 4, 4, 2)
        z = torch.randn(1, 24, 2, 4, 4)
        calls = {"n": 0}

        def build(x):
            calls["n"] += 1
            return patchify_video(x, (1, 2, 2))

        rows = cond_rows_from_unique([z, z.clone()], build)
        self.assertEqual(calls["n"], 1)
        layout._cond_video_rows = rows
        self.assertIs(layout._cond_video_rows, rows)

    def test_pack_audio_roundtrip_shape(self):
        x = torch.randn(1, 32, 2, 6)
        self.assertEqual(tuple(pack_audio(x).shape), (12, 32))

    def test_six_identical_unique_is_one_block(self):
        z = torch.randn(1, 24, 2, 4, 4)
        calls = {"n": 0}

        def build(x):
            calls["n"] += 1
            return patchify_video(x, (1, 2, 2))

        unique, ids, lens = cond_unique_blocks([z.clone() for _ in range(6)], build)
        self.assertEqual(calls["n"], 1)
        self.assertEqual(ids, [0, 0, 0, 0, 0, 0])
        self.assertEqual(lens, [unique.shape[0]])
        expanded = expand_unique_blocks(unique, ids, lens)
        self.assertTrue(torch.equal(expanded, torch.cat([unique] * 6, dim=0)))

    def test_project_unique_equals_mix_then_proj(self):
        torch.manual_seed(2)
        proj = nn.Linear(8, 12, bias=True)
        live = torch.randn(4, 8)
        block = torch.randn(2, 8)
        unique = block
        ids = [0, 0, 0]
        lens = [2]
        cond = torch.cat([block] * 3, dim=0)
        update = torch.tensor([False] * 6 + [True] * 4)
        mixed = torch.empty(10, 8)
        mixed[~update] = cond
        mixed[update] = live
        ref = proj(mixed)
        out = project_unique_cond_live(proj, live, unique, ids, lens, update, torch.float32)
        self.assertTrue(torch.equal(out, ref) or torch.allclose(out, ref, rtol=0, atol=0))

    def test_embed_uses_unique_when_smaller(self):
        torch.manual_seed(3)
        proj = nn.Linear(4, 5, bias=True)
        live = torch.randn(3, 4)
        u = torch.randn(2, 4)
        cond = torch.cat([u, u], dim=0)
        update = torch.tensor([False, False, True, True, True, False, False])
        pack = (u, [0, 0], [2])
        out = _embed_cond_or_mix(proj, live, cond, update, torch.float32, pack)
        mixed = torch.empty(7, 4)
        mixed[~update] = cond
        mixed[update] = live
        self.assertTrue(torch.equal(out, proj(mixed)))

    def test_tag_runs_cached_and_stable(self):
        layout = PackedLayout(6, 2, 4, 4, 2)
        tags = torch.tensor([1, 1, 0, 0, 0, 1])
        r1 = _layout_text_tag_runs(layout, tags)
        r2 = _layout_text_tag_runs(layout, tags)
        self.assertIs(r1, r2)
        self.assertEqual(r1, [(0, 2, 1), (2, 5, 0), (5, 6, 1)])

    def test_mod_row_ids_cached_for_same_segments(self):
        layout = PackedLayout(4, 2, 4, 4, 2)
        segs = [(0, 4, 1), (4, layout.seq_len, 0)]
        a = _layout_mod_row_ids(layout, segs, torch.device("cpu"))
        b = _layout_mod_row_ids(layout, segs, torch.device("cpu"))
        self.assertIs(a, b)
        self.assertTrue((a[:4] == 1).all())
        c = _layout_mod_row_ids(layout, [(0, 4, 2), (4, layout.seq_len, 0)], torch.device("cpu"))
        self.assertIsNot(a, c)
        self.assertTrue((c[:4] == 2).all())


if __name__ == "__main__":
    unittest.main()
