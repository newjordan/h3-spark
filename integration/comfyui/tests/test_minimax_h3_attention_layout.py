import torch

from comfy.ldm.modules.attention import attention_pytorch


def test_token_major_attention_layout_matches_head_major_path():
    generator = torch.Generator().manual_seed(260812)
    batch, seq, heads, dim = 1, 17, 4, 8
    q = torch.randn(batch, seq, heads, dim, generator=generator)
    k = torch.randn(batch, seq, heads, dim, generator=generator)
    v = torch.randn(batch, seq, heads, dim, generator=generator)

    head_major = attention_pytorch(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        heads, skip_reshape=True,
    )
    token_major = attention_pytorch(
        q.reshape(batch, seq, heads * dim),
        k.reshape(batch, seq, heads * dim),
        v.reshape(batch, seq, heads * dim),
        heads, skip_reshape=False,
    )
    assert torch.equal(token_major, head_major)
