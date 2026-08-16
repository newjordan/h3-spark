from __future__ import annotations

import unittest

import torch

from comfy.ldm.modules.attention import attention_pytorch
from comfy.text_encoders.llama import precompute_freqs_cis


def _triu_mask(n, dtype, device):
    mask = torch.empty(n, n, dtype=dtype, device=device)
    mask.fill_(torch.finfo(dtype).min / 4)
    return mask.triu_(1)


class CausalSdpaTests(unittest.TestCase):
    def test_pytorch_is_causal_matches_triu_mask_cpu(self):
        torch.manual_seed(0)
        b, h, s, d = 1, 4, 16, 32
        q = torch.randn(b, h, s, d)
        k = torch.randn(b, h, s, d)
        v = torch.randn(b, h, s, d)
        mask = _triu_mask(s, q.dtype, q.device)
        via_mask = attention_pytorch(q, k, v, h, mask=mask, skip_reshape=True)
        via_flag = attention_pytorch(q, k, v, h, mask=None, skip_reshape=True, is_causal=True)
        self.assertTrue(torch.equal(via_mask, via_flag))

    @unittest.skipUnless(torch.cuda.is_available(), "cuda")
    def test_pytorch_is_causal_matches_triu_mask_cuda_fp32(self):
        torch.manual_seed(1)
        b, h, s, d = 1, 8, 64, 64
        q = torch.randn(b, h, s, d, device="cuda", dtype=torch.float32)
        k = torch.randn(b, h, s, d, device="cuda", dtype=torch.float32)
        v = torch.randn(b, h, s, d, device="cuda", dtype=torch.float32)
        mask = _triu_mask(s, q.dtype, q.device)
        via_mask = attention_pytorch(q, k, v, h, mask=mask, skip_reshape=True)
        via_flag = attention_pytorch(q, k, v, h, mask=None, skip_reshape=True, is_causal=True)
        self.assertTrue(torch.equal(via_mask, via_flag))

    @unittest.skipUnless(torch.cuda.is_available(), "cuda")
    def test_gqa_is_causal_matches_mask(self):
        torch.manual_seed(2)
        b, hq, hk, s, d = 1, 8, 2, 32, 64
        q = torch.randn(b, hq, s, d, device="cuda", dtype=torch.float32)
        k = torch.randn(b, hk, s, d, device="cuda", dtype=torch.float32)
        v = torch.randn(b, hk, s, d, device="cuda", dtype=torch.float32)
        mask = _triu_mask(s, q.dtype, q.device)
        via_mask = attention_pytorch(
            q, k, v, hq, mask=mask, skip_reshape=True, enable_gqa=True
        )
        via_flag = attention_pytorch(
            q, k, v, hq, mask=None, skip_reshape=True, enable_gqa=True, is_causal=True
        )
        self.assertTrue(torch.equal(via_mask, via_flag))

    def test_freqs_cis_deterministic(self):
        pos = torch.arange(0, 8).unsqueeze(0)
        a = precompute_freqs_cis(32, pos, 10000.0, device=pos.device)
        b = precompute_freqs_cis(32, pos, 10000.0, device=pos.device)
        self.assertTrue(torch.equal(a[0], b[0]))
        self.assertTrue(torch.equal(a[1], b[1]))


if __name__ == "__main__":
    unittest.main()
