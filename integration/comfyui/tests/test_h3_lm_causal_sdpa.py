from __future__ import annotations

import unittest

import torch

from comfy.ops import scaled_dot_product_attention
from comfy.text_encoders.llama import precompute_freqs_cis


def _triu_mask(n, dtype, device):
    mask = torch.empty(n, n, dtype=dtype, device=device)
    mask.fill_(torch.finfo(dtype).min / 4)
    return mask.triu_(1)


class CausalSdpaTests(unittest.TestCase):
    def test_pytorch_is_causal_matches_triu_mask_cpu(self):
        # attention_pytorch at the pinned ComfyUI revision does not
        # forward is_causal (the LM path uses explicit triu masks), so
        # the bitwise-equivalence gate targets comfy.ops, the wrapper
        # the shipped overlay actually calls with is_causal=True
        # (audio VAE).
        torch.manual_seed(0)
        b, h, s, d = 1, 4, 16, 32
        q = torch.randn(b, h, s, d)
        k = torch.randn(b, h, s, d)
        v = torch.randn(b, h, s, d)
        mask = _triu_mask(s, q.dtype, q.device)
        via_mask = scaled_dot_product_attention(q, k, v, attn_mask=mask)
        via_flag = scaled_dot_product_attention(q, k, v, is_causal=True)
        self.assertTrue(torch.equal(via_mask, via_flag))

    @unittest.skipUnless(torch.cuda.is_available(), "cuda")
    def test_pytorch_is_causal_matches_triu_mask_cuda_fp32(self):
        torch.manual_seed(1)
        b, h, s, d = 1, 8, 64, 64
        q = torch.randn(b, h, s, d, device="cuda", dtype=torch.float32)
        k = torch.randn(b, h, s, d, device="cuda", dtype=torch.float32)
        v = torch.randn(b, h, s, d, device="cuda", dtype=torch.float32)
        mask = _triu_mask(s, q.dtype, q.device)
        via_mask = scaled_dot_product_attention(q, k, v, attn_mask=mask)
        via_flag = scaled_dot_product_attention(q, k, v, is_causal=True)
        self.assertTrue(torch.equal(via_mask, via_flag))

    @unittest.skipUnless(torch.cuda.is_available(), "cuda")
    def test_gqa_is_causal_matches_mask(self):
        torch.manual_seed(2)
        b, hq, hk, s, d = 1, 8, 2, 32, 64
        q = torch.randn(b, hq, s, d, device="cuda", dtype=torch.float32)
        k = torch.randn(b, hk, s, d, device="cuda", dtype=torch.float32)
        v = torch.randn(b, hk, s, d, device="cuda", dtype=torch.float32)
        mask = _triu_mask(s, q.dtype, q.device)
        via_mask = scaled_dot_product_attention(q, k, v, enable_gqa=True, attn_mask=mask)
        via_flag = scaled_dot_product_attention(q, k, v, enable_gqa=True, is_causal=True)
        self.assertTrue(torch.equal(via_mask, via_flag))

    def test_freqs_cis_deterministic(self):
        pos = torch.arange(0, 8).unsqueeze(0)
        a = precompute_freqs_cis(32, pos, 10000.0, device=pos.device)
        b = precompute_freqs_cis(32, pos, 10000.0, device=pos.device)
        self.assertTrue(torch.equal(a[0], b[0]))
        self.assertTrue(torch.equal(a[1], b[1]))


if __name__ == "__main__":
    unittest.main()
