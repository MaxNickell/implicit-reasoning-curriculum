from dataclasses import dataclass

import torch.nn as nn


@dataclass
class ModelParts:
    arch: str               # "gpt_neox" or "qwen"
    unwrapped: nn.Module    # The HF *ForCausalLM after stripping DDP wrappers
    inner: nn.Module        # Transformer body (Qwen3Model / GPTNeoXModel): embeddings, decoder layers, final norm
    lm_head: nn.Module      # Output head (lm_head for Qwen, embed_out for GPT-NeoX)

    @property
    def layers(self) -> nn.ModuleList:
        return self.inner.layers


def unwrap_model_for_body(model):
    """Strip DistributedDataParallel (.module) wrappers; returns the HF *ForCausalLM."""
    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    return unwrapped


def resolve_model_parts(model) -> ModelParts:
    """The transformer body and output head that the packed training forward calls directly
    (see trainer/forward.py). The body is run instead of the CausalLM so that the head, the loss
    and the gate metrics stay ours."""
    unwrapped = unwrap_model_for_body(model)

    if hasattr(unwrapped, "gpt_neox"):
        # GPT-NeoX (Pythia): body .gpt_neox, separate output head .embed_out
        return ModelParts(
            arch="gpt_neox",
            unwrapped=unwrapped,
            inner=unwrapped.gpt_neox,
            lm_head=unwrapped.embed_out,
        )
    elif hasattr(unwrapped, "model") and hasattr(unwrapped, "lm_head"):
        # Qwen / Llama-style: body .model, output head .lm_head
        inner = unwrapped.model
        if not hasattr(inner, "embed_tokens") and hasattr(inner, "model"):
            inner = inner.model
        return ModelParts(
            arch="qwen",
            unwrapped=unwrapped,
            inner=inner,
            lm_head=unwrapped.lm_head,
        )
    else:
        raise ValueError(
            f"Unsupported model layout: {type(unwrapped).__name__} has neither .gpt_neox/.embed_out "
            f"(GPT-NeoX) nor .model/.lm_head (Qwen/Llama-style)"
        )


def set_layer_checkpointing(layers, use_ckpt: bool, ckpt_every: int) -> None:
    """Per-micro-batch selective gradient checkpointing. transformers' GradientCheckpointingLayer
    reads its per-instance `gradient_checkpointing` flag at call time, and Trainer sets it True on
    every layer at train start; writing `use_ckpt and (li % ckpt_every == 0)` before the forward
    recomputes only every ckpt_every-th layer. Gradients are identical either way; this is a
    memory/compute dial only."""
    for li, layer in enumerate(layers):
        layer.gradient_checkpointing = bool(use_ckpt and (li % ckpt_every == 0))
