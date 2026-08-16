from types import SimpleNamespace

import torch

import comfy.ldm.minimax.model as minimax_model
from comfy.ldm.minimax import adaln_nvfp4


class TorchOperations:
    Linear = torch.nn.Linear
    RMSNorm = torch.nn.RMSNorm


def make_block():
    return minimax_model.DiTBlock(
        hidden=4,
        heads=2,
        head_dim=2,
        ffn=8,
        t_dim=2,
        eps=1e-6,
        qk_eps=1e-6,
        operations=TorchOperations,
    )


def run_block(block, options=None, token=None):
    return block(
        torch.randn(3, 4),
        torch.randn(1, 2),
        [(0, 3, 0)],
        None,
        transformer_options=options or {},
        mod_row_ids=torch.zeros(3, dtype=torch.int32),
        adaln_nvfp4_token=token,
    )


def test_parent_hook_preserves_untouched_module_path(monkeypatch):
    block = make_block()
    calls = []
    handles = [
        block.attn.register_forward_hook(lambda *_args: calls.append("attn")),
        block.mlp.register_forward_hook(lambda *_args: calls.append("mlp")),
    ]
    monkeypatch.setattr(
        minimax_model,
        "_h3_adaln_linear",
        lambda *_args: (_ for _ in ()).throw(AssertionError("fusion attempted")),
    )
    monkeypatch.setattr(
        minimax_model,
        "optimized_attention",
        lambda _q, _k, v, _heads, **_kwargs: v,
    )
    try:
        output = run_block(
            block,
            {"minimax_h3_modulation_fusion": "exact"},
            minimax_model._ADALN_NVFP4_TOKEN,
        )
    finally:
        for handle in handles:
            handle.remove()

    assert output.shape == (3, 4)
    assert calls == ["attn", "mlp"]


def test_untrusted_row_map_never_attempts_producer_fusion(monkeypatch):
    block = make_block()
    monkeypatch.setattr(
        minimax_model,
        "_h3_adaln_linear",
        lambda *_args: (_ for _ in ()).throw(AssertionError("fusion attempted")),
    )
    monkeypatch.setattr(
        minimax_model,
        "optimized_attention",
        lambda _q, _k, v, _heads, **_kwargs: v,
    )
    assert run_block(
        block,
        {"minimax_h3_modulation_fusion": "exact"},
        token=None,
    ).shape == (3, 4)


def test_actual_route_failure_reenters_parent_modules(monkeypatch):
    block = make_block()
    attempted = []
    parent_calls = []
    original_attn = block.attn.forward
    original_mlp = block.mlp.forward

    def failed(linear, *_args):
        attempted.append(linear)
        return None

    def attention_forward(*args, **kwargs):
        parent_calls.append("attn")
        return original_attn(*args, **kwargs)

    def mlp_forward(*args, **kwargs):
        parent_calls.append("mlp")
        return original_mlp(*args, **kwargs)

    monkeypatch.setattr(minimax_model, "_h3_adaln_linear", failed)
    monkeypatch.setattr(block.attn, "forward", attention_forward)
    monkeypatch.setattr(block.mlp, "forward", mlp_forward)
    monkeypatch.setattr(
        minimax_model,
        "optimized_attention",
        lambda _q, _k, v, _heads, **_kwargs: v,
    )
    output = run_block(
        block,
        {"minimax_h3_modulation_fusion": "exact"},
        minimax_model._ADALN_NVFP4_TOKEN,
    )

    assert output.shape == (3, 4)
    assert attempted == [block.attn.qkv_proj, block.mlp.fc1]
    assert parent_calls == ["attn", "mlp"]


def test_actual_fusion_uses_projected_and_raw_paths_once(monkeypatch):
    block = make_block()
    attempted = []
    qkv = torch.randn(3, 12)
    raw = torch.randn(3, 16)

    def fused(linear, *_args):
        attempted.append(linear)
        if linear is block.attn.qkv_proj:
            return qkv
        assert linear is block.mlp.fc1
        return raw

    monkeypatch.setattr(minimax_model, "_h3_adaln_linear", fused)
    monkeypatch.setattr(
        minimax_model,
        "optimized_attention",
        lambda _q, _k, v, _heads, **_kwargs: v,
    )
    output = run_block(
        block,
        {"minimax_h3_modulation_fusion": "exact"},
        minimax_model._ADALN_NVFP4_TOKEN,
    )

    assert output.shape == (3, 4)
    assert attempted == [block.attn.qkv_proj, block.mlp.fc1]


def test_aggressive_and_stock_modulation_never_attempt_fusion(monkeypatch):
    block = make_block()
    monkeypatch.setattr(
        minimax_model,
        "_h3_adaln_linear",
        lambda *_args: (_ for _ in ()).throw(AssertionError("fusion attempted")),
    )
    monkeypatch.setattr(
        minimax_model,
        "optimized_attention",
        lambda _q, _k, v, _heads, **_kwargs: v,
    )
    for mode in ("stock", "aggressive"):
        assert run_block(
            block,
            {"minimax_h3_modulation_fusion": mode},
            minimax_model._ADALN_NVFP4_TOKEN,
        ).shape == (3, 4)


def test_adaln_mode_inherits_existing_fusion_and_allows_override(monkeypatch):
    seen = []
    monkeypatch.setattr(
        minimax_model,
        "try_adaln_nvfp4_linear",
        lambda *args: seen.append(args[-2:]) or None,
    )
    value = object()
    common = (object(), value, object(), object(), object())
    minimax_model._h3_adaln_linear(
        *common,
        {"minimax_h3_swiglu_nvfp4_fusion": "auto"},
    )
    minimax_model._h3_adaln_linear(
        *common,
        {
            "minimax_h3_swiglu_nvfp4_fusion": "auto",
            "minimax_h3_adaln_nvfp4_fusion": "stock",
        },
    )
    assert seen == [("auto", "exact"), ("stock", "exact")]


def test_native_route_is_dynamic_only_and_missing_native_fails_closed(monkeypatch):
    monkeypatch.setattr(adaln_nvfp4, "_inputs_eligible", lambda *_args: True)
    monkeypatch.setattr(adaln_nvfp4, "_linear_eligible", lambda *_args: True)
    monkeypatch.setattr(adaln_nvfp4, "_dynamic_eligible", lambda *_args: True)
    monkeypatch.setattr(adaln_nvfp4, "_load_native", lambda: None)
    monkeypatch.setattr(
        adaln_nvfp4,
        "cast_bias_weight",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cast attempted")
        ),
    )
    args = (
        SimpleNamespace(),
        object(),
        object(),
        object(),
        object(),
        adaln_nvfp4.ROW_IDS_TOKEN,
    )
    assert adaln_nvfp4.try_linear(*args, "stock", "exact") is None
    assert adaln_nvfp4.try_linear(*args, "static", "exact") is None
    assert adaln_nvfp4.try_linear(*args, "auto", "aggressive") is None
    assert adaln_nvfp4.try_linear(*args, "auto", "exact") is None


def test_input_guard_requires_matching_modulation_dtype(monkeypatch):
    monkeypatch.setattr(adaln_nvfp4, "_cuda_device_supported", lambda _device: True)
    normed = torch.empty(2, 5376, dtype=torch.bfloat16)
    scale = torch.empty(1, 5376, dtype=torch.bfloat16)
    shift = torch.empty(1, 5376, dtype=torch.float32)
    row_ids = torch.zeros(2, dtype=torch.int32)
    linear = SimpleNamespace(in_features=5376, out_features=21504)
    # CPU is independently ineligible, but the dtype mismatch remains covered
    # by proving even a CUDA/device-compatible shim cannot pass the full guard.
    assert not adaln_nvfp4._inputs_eligible(
        linear, normed, scale, shift, row_ids
    )


def test_parent_guard_rejects_local_and_global_hooks():
    module = torch.nn.Identity()
    assert adaln_nvfp4.parent_eligible(module)
    local = module.register_forward_hook(lambda *_args: None)
    try:
        assert not adaln_nvfp4.parent_eligible(module)
    finally:
        local.remove()

    global_handle = torch.nn.modules.module.register_module_forward_pre_hook(
        lambda *_args: None
    )
    try:
        assert not adaln_nvfp4.parent_eligible(module)
    finally:
        global_handle.remove()
