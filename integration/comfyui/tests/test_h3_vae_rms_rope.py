from __future__ import annotations

import os
import unittest

import torch

from comfy.ldm.minimax import vae as mvae


def _split_half_table(batch, seq, pairs, device, dtype):
    # Kitchen split-half layout: [B, S, 1, pairs, 2, 2]
    angle = torch.linspace(-1.2, 1.1, batch * seq * pairs, device=device, dtype=torch.float32)
    angle = angle.view(batch, seq, pairs)
    c, s = torch.cos(angle), torch.sin(angle)
    table = torch.stack([c, -s, s, c], dim=-1).reshape(batch, seq, 1, pairs, 2, 2)
    return table.to(dtype)


def _stock(query, key, rope, eps):
    return mvae._stock_qk_rms_rope(query, key, rope, eps, None, None)


def _assign_rope_after_rms(query, key, rope, eps):
    """Old stock: RMS then out-of-place prefix assign. In-place must match this."""
    import comfy.quant_ops as qops

    query = torch.nn.functional.rms_norm(query, (query.shape[-1],), eps=eps)
    key = torch.nn.functional.rms_norm(key, (key.shape[-1],), eps=eps)
    rot = rope.shape[-3] * 2
    query[..., :rot], key[..., :rot] = qops.ck.apply_rope_split_half(
        query[..., :rot], key[..., :rot], rope
    )
    return query, key


class VaeRmsRopeTests(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.get(mvae._RMS_ROPE_ENV)
        os.environ[mvae._RMS_ROPE_ENV] = "auto"

    def tearDown(self):
        if self._env is None:
            os.environ.pop(mvae._RMS_ROPE_ENV, None)
        else:
            os.environ[mvae._RMS_ROPE_ENV] = self._env

    def test_env_off_disables_fuse(self):
        os.environ[mvae._RMS_ROPE_ENV] = "off"
        self.assertFalse(mvae._rms_rope_enabled())
        os.environ[mvae._RMS_ROPE_ENV] = "stock"
        self.assertFalse(mvae._rms_rope_enabled())
        os.environ.pop(mvae._RMS_ROPE_ENV, None)
        self.assertFalse(mvae._rms_rope_enabled())
        os.environ[mvae._RMS_ROPE_ENV] = "auto"
        self.assertTrue(mvae._rms_rope_enabled())

    def test_ones_scale_matches_weightless_rms(self):
        torch.manual_seed(0)
        x = torch.randn(2, 17, 32, 64, dtype=torch.float32)
        none = torch.nn.functional.rms_norm(x, (x.shape[-1],), eps=1e-5)
        ones = torch.nn.functional.rms_norm(
            x, (x.shape[-1],), weight=torch.ones(64), eps=1e-5
        )
        self.assertTrue(torch.equal(none, ones))

    def test_fused_matches_stock_partial_rot(self):
        torch.manual_seed(1)
        b, s, h, d, rot = 2, 17, 32, 64, 48
        query = torch.randn(b, s, h, d)
        key = torch.randn(b, s, h, d)
        rope = _split_half_table(b, s, rot // 2, query.device, query.dtype)
        stock_q, stock_k = _stock(query.clone(), key.clone(), rope, 1e-5)
        fused_q, fused_k = mvae._apply_qk_rms_rope(
            query.clone(), key.clone(), rope, 1e-5, None, None
        )
        self.assertTrue(torch.equal(stock_q, fused_q))
        self.assertTrue(torch.equal(stock_k, fused_k))
        # Tail beyond rot_dim is RMS only — not rotated.
        tail = stock_q[..., rot:]
        raw_rms = torch.nn.functional.rms_norm(query, (d,), eps=1e-5)[..., rot:]
        self.assertTrue(torch.equal(tail, raw_rms))

    def test_env_off_uses_stock(self):
        torch.manual_seed(2)
        b, s, h, d, rot = 1, 9, 32, 64, 48
        query = torch.randn(b, s, h, d)
        key = torch.randn(b, s, h, d)
        rope = _split_half_table(b, s, rot // 2, query.device, query.dtype)
        os.environ[mvae._RMS_ROPE_ENV] = "off"
        out_q, out_k = mvae._apply_qk_rms_rope(
            query.clone(), key.clone(), rope, 1e-5, None, None
        )
        stock_q, stock_k = _stock(query.clone(), key.clone(), rope, 1e-5)
        self.assertTrue(torch.equal(out_q, stock_q))
        self.assertTrue(torch.equal(out_k, stock_k))

    def test_affine_weight_stays_stock(self):
        torch.manual_seed(3)
        b, s, h, d, rot = 1, 5, 32, 64, 48
        query = torch.randn(b, s, h, d)
        key = torch.randn(b, s, h, d)
        weight = torch.randn(d)
        rope = _split_half_table(b, s, rot // 2, query.device, query.dtype)
        out_q, out_k = mvae._apply_qk_rms_rope(
            query.clone(), key.clone(), rope, 1e-5, weight, weight
        )
        stock_q, stock_k = mvae._stock_qk_rms_rope(
            query.clone(), key.clone(), rope, 1e-5, weight, weight
        )
        self.assertTrue(torch.equal(out_q, stock_q))
        self.assertTrue(torch.equal(out_k, stock_k))

    def test_inplace_rope_matches_assign_cpu(self):
        torch.manual_seed(4)
        query = torch.randn(2, 17, 32, 64)
        key = torch.randn(2, 17, 32, 64)
        rope = _split_half_table(2, 17, 24, query.device, query.dtype)
        got_q, got_k = mvae._stock_qk_rms_rope(
            query.clone(), key.clone(), rope, 1e-5, None, None
        )
        ref_q, ref_k = _assign_rope_after_rms(query.clone(), key.clone(), rope, 1e-5)
        self.assertTrue(torch.equal(got_q, ref_q))
        self.assertTrue(torch.equal(got_k, ref_k))

    @unittest.skipUnless(torch.cuda.is_available(), "cuda")
    def test_inplace_rope_matches_assign_cuda_fp16(self):
        torch.manual_seed(5)
        query = torch.randn(4, 129, 32, 64, device="cuda", dtype=torch.float16)
        key = torch.randn(4, 129, 32, 64, device="cuda", dtype=torch.float16)
        rope = _split_half_table(4, 129, 24, query.device, query.dtype)
        got_q, got_k = mvae._stock_qk_rms_rope(
            query.clone(), key.clone(), rope, 1e-5, None, None
        )
        ref_q, ref_k = _assign_rope_after_rms(query.clone(), key.clone(), rope, 1e-5)
        self.assertTrue(torch.equal(got_q, ref_q))
        self.assertTrue(torch.equal(got_k, ref_k))


if __name__ == "__main__":
    unittest.main()
