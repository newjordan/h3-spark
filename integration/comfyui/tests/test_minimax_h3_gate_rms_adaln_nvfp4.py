from __future__ import annotations

import pytest
import torch

import comfy.ldm.minimax.model as minimax_model


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


def prepare_block(monkeypatch):
    block = make_block()
    modulation = tuple(torch.zeros(1, 4) for _ in range(6))
    monkeypatch.setattr(block.adaln_proj, "forward", lambda _t_emb: modulation)
    attention = torch.randn(3, 4)
    monkeypatch.setattr(
        block.attn,
        "forward",
        lambda _x, rope_freqs=None, transformer_options={}: attention,
    )

    monkeypatch.setattr(minimax_model, "_h3_rms_adaln_linear", lambda *_args: None)
    monkeypatch.setattr(minimax_model, "_h3_adaln_linear", lambda *_args: None)
    return block, modulation, attention


def run_block(block, *, mode="auto", token=minimax_model._ADALN_NVFP4_TOKEN):
    return block(
        torch.randn(3, 4),
        torch.randn(1, 2),
        [(0, 3, 0)],
        None,
        transformer_options={
            "minimax_h3_modulation_fusion": "exact",
            "minimax_h3_rms_adaln_nvfp4_fusion": mode,
        },
        mod_row_ids=torch.zeros(3, dtype=torch.int32),
        adaln_nvfp4_token=token,
    )


def test_fused_transition_skips_attention_gate_and_old_mlp(monkeypatch):
    block, modulation, attention = prepare_block(monkeypatch)
    gate_msa = modulation[2]
    gate_mlp = modulation[5]
    fused_mlp = torch.randn(3, 4)
    transitions = []
    gates = []

    def transition(*args):
        transitions.append(args)
        return fused_mlp

    def gate(x, selected_gate, other, *_args):
        gates.append((selected_gate, other))
        return x

    monkeypatch.setattr(
        minimax_model, "try_gate_rms_adaln_nvfp4_mlp", transition
    )
    monkeypatch.setattr(
        minimax_model,
        "_h3_rms_adaln_linear",
        lambda linear, *_args: (
            (_ for _ in ()).throw(
                AssertionError("old MLP RMS producer entered after gate fusion")
            )
            if linear is block.mlp.fc1
            else None
        ),
    )
    monkeypatch.setattr(
        minimax_model,
        "_h3_adaln_linear",
        lambda linear, *_args: (
            (_ for _ in ()).throw(
                AssertionError("old MLP AdaLN producer entered after gate fusion")
            )
            if linear is block.mlp.fc1
            else None
        ),
    )
    monkeypatch.setattr(minimax_model, "_mod_gate", gate)
    monkeypatch.setattr(
        block.mlp,
        "forward",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("stock MLP entered after gate fusion")
        ),
    )

    output = run_block(block)

    assert output.shape == (3, 4)
    assert len(transitions) == 1
    assert transitions[0][0] is block.mlp
    assert transitions[0][3] is attention
    assert transitions[0][4] is gate_msa
    assert transitions[0][9:11] == ("auto", "exact")
    assert gates == [(gate_mlp, fused_mlp)]


def test_none_transition_runs_stock_gate_and_mlp_once(monkeypatch):
    block, modulation, attention = prepare_block(monkeypatch)
    gate_msa = modulation[2]
    gate_mlp = modulation[5]
    mlp_output = torch.randn(3, 4)
    gates = []
    mlp_calls = []
    monkeypatch.setattr(
        minimax_model, "try_gate_rms_adaln_nvfp4_mlp", lambda *_args: None
    )
    monkeypatch.setattr(
        minimax_model,
        "_mod_gate",
        lambda x, selected_gate, other, *_args: (
            gates.append((selected_gate, other)) or x
        ),
    )
    monkeypatch.setattr(
        block.mlp,
        "forward",
        lambda *_args, **_kwargs: mlp_calls.append(True) or mlp_output,
    )

    assert run_block(block).shape == (3, 4)
    assert mlp_calls == [True]
    assert gates == [(gate_msa, attention), (gate_mlp, mlp_output)]


def test_post_mutation_failure_propagates_without_stock_fallback(monkeypatch):
    block, _modulation, _attention = prepare_block(monkeypatch)
    monkeypatch.setattr(
        minimax_model,
        "try_gate_rms_adaln_nvfp4_mlp",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("after mutation")),
    )
    monkeypatch.setattr(
        minimax_model,
        "_mod_gate",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("stock gate entered after mutation")
        ),
    )
    monkeypatch.setattr(
        block.mlp,
        "forward",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("stock MLP entered after mutation")
        ),
    )

    with pytest.raises(RuntimeError, match="after mutation"):
        run_block(block)


@pytest.mark.parametrize(
    ("mode", "token"),
    (
        ("stock", minimax_model._ADALN_NVFP4_TOKEN),
        ("auto", None),
    ),
)
def test_stock_and_untrusted_replacement_paths_disable_transition(
    monkeypatch, mode, token
):
    block, _modulation, _attention = prepare_block(monkeypatch)
    monkeypatch.setattr(
        minimax_model,
        "try_gate_rms_adaln_nvfp4_mlp",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("gate transition attempted")
        ),
    )
    monkeypatch.setattr(minimax_model, "_h3_rms_adaln_linear", lambda *_args: None)
    monkeypatch.setattr(minimax_model, "_h3_adaln_linear", lambda *_args: None)

    assert run_block(block, mode=mode, token=token).shape == (3, 4)


def test_local_and_global_hooks_disable_transition_and_still_fire(monkeypatch):
    block, _modulation, _attention = prepare_block(monkeypatch)
    monkeypatch.setattr(
        minimax_model,
        "try_gate_rms_adaln_nvfp4_mlp",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("hooked parent bypassed")
        ),
    )
    monkeypatch.setattr(minimax_model, "_h3_rms_adaln_linear", lambda *_args: None)
    monkeypatch.setattr(minimax_model, "_h3_adaln_linear", lambda *_args: None)
    calls = []
    local = block.mlp.register_forward_hook(lambda *_args: calls.append("local"))
    global_hook = torch.nn.modules.module.register_module_forward_hook(
        lambda module, *_args: calls.append("global") if module is block.mlp else None
    )
    try:
        assert run_block(block).shape == (3, 4)
    finally:
        global_hook.remove()
        local.remove()

    assert calls.count("local") == 1
    assert calls.count("global") == 1
