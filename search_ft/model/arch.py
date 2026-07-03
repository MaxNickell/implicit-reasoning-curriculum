from dataclasses import dataclass

import torch.nn as nn


@dataclass
class ModelParts:
    arch: str               # "gpt_neox" or "qwen"
    embed: nn.Module        # Token embedding layer
    layers: nn.ModuleList   # Transformer layer stack
    norm: nn.Module         # Final layer norm
    rotary_emb: nn.Module   # Rotary position embedding
    lm_head: nn.Module      # Language model head (output projection)
    num_heads: int          # Number of attention heads
    num_kv_heads: int       # Number of KV heads (GQA; == num_heads if no GQA)
    head_dim: int           # Dimension per head
    rotary_ndims: int       # Dimensions that get rotary embedding (< head_dim for Pythia)
    parallel_residual: bool # True = GPT-NeoX parallel attn+mlp, False = sequential


_QWEN_TYPES = {"qwen3"}


def detect_architecture(model) -> ModelParts:
    # Unwrap DDP/FSDP wrapping
    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module

    config = unwrapped.config
    model_type = getattr(config, "model_type", "")

    if model_type == "gpt_neox":
        return _extract_gpt_neox(unwrapped, config)
    elif model_type in _QWEN_TYPES:
        return _extract_qwen(unwrapped, config)
    else:
        raise ValueError(
            f"Unsupported model_type: '{model_type}'. "
            f"Supported: 'gpt_neox', {sorted(_QWEN_TYPES)}"
        )


def _extract_gpt_neox(model, config) -> ModelParts:
    inner = model.gpt_neox

    head_dim = config.hidden_size // config.num_attention_heads
    rotary_pct = getattr(config, 'rotary_pct', 1.0)
    rotary_ndims = int(head_dim * rotary_pct) if rotary_pct < 1.0 else head_dim

    return ModelParts(
        arch='gpt_neox',
        embed=inner.embed_in,
        layers=inner.layers,
        norm=inner.final_layer_norm,
        rotary_emb=inner.rotary_emb,
        lm_head=model.embed_out,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_attention_heads,  # GPT-NeoX has no GQA
        head_dim=head_dim,
        rotary_ndims=rotary_ndims,
        parallel_residual=getattr(config, 'use_parallel_residual', True),
    )


def _extract_qwen(model, config) -> ModelParts:
    inner = model.model
    if not hasattr(inner, 'embed_tokens') and hasattr(inner, 'model'):
        inner = inner.model

    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

    return ModelParts(
        arch='qwen',
        embed=inner.embed_tokens,
        layers=inner.layers,
        norm=inner.norm,
        rotary_emb=inner.rotary_emb,
        lm_head=model.lm_head,
        num_heads=config.num_attention_heads,
        num_kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
        head_dim=head_dim,
        rotary_ndims=head_dim,
        parallel_residual=False,
    )
