"""CPU-only unit check for the packed path's model helpers (search_ft/model/arch.py, search_ft/model/attention.py).

Builds a 2-layer random-init Qwen3 model on CPU from a config (no download) with attn_implementation='eager' and
checks that
  * resolve_model_parts finds the body and the head, and every decoder layer is a transformers
    GradientCheckpointingLayer (what per-layer toggling relies on; verify_attention_setup asserts the same);
  * Trainer-style gradient_checkpointing_enable sets the flag on every layer;
  * set_layer_checkpointing(layers, use_ckpt=True, ckpt_every=2) leaves layers[0] True and layers[1] False, and a
    forward/backward runs with that mixed setting;
  * eval_attn_sdpa switches the model to sdpa for the duration of an eval and restores the training implementation,
    also when the eval body raises, and is a no-op on a model that already runs sdpa.
Run from the repository root:  python tests/test_ckpt_toggle.py   (no GPU, no flash-attn, no generator build needed)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.modeling_layers import GradientCheckpointingLayer

from search_ft.model.arch import set_layer_checkpointing, resolve_model_parts, unwrap_model_for_body
from search_ft.model.attention import eval_attn_sdpa, verify_attention_setup, select_attn_implementation


def tiny_qwen(attn_implementation="eager"):
    cfg = Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
        attn_implementation=attn_implementation,
    )
    return Qwen3ForCausalLM(cfg), cfg  # random init, CPU, fp32


def test_ckpt_toggle():
    torch.manual_seed(0)
    model, cfg = tiny_qwen("eager")
    assert model.config._attn_implementation == "eager", model.config._attn_implementation

    parts = resolve_model_parts(model)
    layers = parts.layers
    assert parts.arch == "qwen" and parts.inner is model.model and parts.lm_head is model.lm_head
    assert unwrap_model_for_body(model) is model
    assert len(layers) == 2
    assert all(isinstance(l, GradientCheckpointingLayer) for l in layers), [type(l) for l in layers]
    assert not any(l.gradient_checkpointing for l in layers), "flag must start False"

    # The post-load verification the loader runs (requested == loaded, GCL layers; no kernel import for eager).
    assert verify_attention_setup(model, "eager") == "eager"
    try:
        verify_attention_setup(model, "flash_attention_2")
        raise AssertionError("verify_attention_setup must reject a mismatching request")
    except AssertionError as e:
        assert "requested" in str(e), e

    # What Trainer does at train start: flag True on every layer, use_reentrant=False.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    assert all(l.gradient_checkpointing for l in layers), "gradient_checkpointing_enable must set every layer"

    # The per-micro-batch toggle used by hf_forward_body.
    set_layer_checkpointing(layers, use_ckpt=True, ckpt_every=2)
    assert layers[0].gradient_checkpointing is True, layers[0].gradient_checkpointing
    assert layers[1].gradient_checkpointing is False, layers[1].gradient_checkpointing

    model.train()
    ids = torch.randint(0, cfg.vocab_size, (1, 12))
    pos = torch.arange(12).unsqueeze(0)
    out = model.model(input_ids=ids, position_ids=pos, attention_mask=None, use_cache=False)
    assert out.past_key_values is None
    h = out.last_hidden_state[0]
    assert h.shape == (12, cfg.hidden_size), h.shape
    loss = model.lm_head(h).float().logsumexp(-1).mean()
    loss.backward()
    assert model.model.layers[0].mlp.down_proj.weight.grad is not None
    assert model.model.layers[1].mlp.down_proj.weight.grad is not None

    set_layer_checkpointing(layers, use_ckpt=False, ckpt_every=2)
    assert not any(l.gradient_checkpointing for l in layers)
    set_layer_checkpointing(layers, use_ckpt=True, ckpt_every=1)
    assert all(l.gradient_checkpointing for l in layers)
    print("test_ckpt_toggle: layers[0]=True, layers[1]=False at ckpt_every=2  OK")


def test_eval_attn_sdpa():
    torch.manual_seed(0)
    model, cfg = tiny_qwen("eager")
    verify_attention_setup(model, "eager")   # records the training implementation for the drift assertion

    # Eval pinning: eager -> sdpa inside, eager restored after (same mechanism as flash_attention_* -> sdpa on GPU).
    with eval_attn_sdpa(model):
        assert model.config._attn_implementation == "sdpa", model.config._attn_implementation
        assert model.model.layers[0].self_attn.config._attn_implementation == "sdpa"
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        model(input_ids=ids, labels=ids)   # a teacher-forced eval forward runs under sdpa
    assert model.config._attn_implementation == "eager", model.config._attn_implementation
    assert model.model.layers[0].self_attn.config._attn_implementation == "eager"

    # Restored even when the eval body raises.
    try:
        with eval_attn_sdpa(model):
            assert model.config._attn_implementation == "sdpa"
            raise RuntimeError("eval failed")
    except RuntimeError:
        pass
    assert model.config._attn_implementation == "eager"

    # Round trip through a DDP-like wrapper (attribute access goes through .module).
    class _Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.module = m
    wrapped = _Wrapper(model)
    with eval_attn_sdpa(wrapped):
        assert model.config._attn_implementation == "sdpa"
    assert model.config._attn_implementation == "eager"

    # No-op on a model that already runs sdpa.
    model_sdpa, _ = tiny_qwen("sdpa")
    with eval_attn_sdpa(model_sdpa):
        assert model_sdpa.config._attn_implementation == "sdpa"
    assert model_sdpa.config._attn_implementation == "sdpa"
    print("test_eval_attn_sdpa: eager -> sdpa -> eager round trip, exception-safe, DDP-wrapped, sdpa no-op  OK")


def test_select_attn_implementation():
    # No GPU here: auto resolves to FA2; fa2/fa3 force; anything else is rejected.
    assert select_attn_implementation("fa2") == "flash_attention_2"
    assert select_attn_implementation("fa3") == "flash_attention_3"
    if not torch.cuda.is_available():
        assert select_attn_implementation("auto") == "flash_attention_2"
    try:
        select_attn_implementation("sdpa")
        raise AssertionError("must reject unknown kernel names")
    except ValueError:
        pass
    print("test_select_attn_implementation: OK")


def main():
    test_ckpt_toggle()
    test_eval_attn_sdpa()
    test_select_attn_implementation()
    print("test_ckpt_toggle: ALL OK")


if __name__ == "__main__":
    main()
