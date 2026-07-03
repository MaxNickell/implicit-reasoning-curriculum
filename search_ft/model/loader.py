import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from ..distributed.utils import rank_print, is_main_process


def _apply_liger_kernels(model_name: str) -> bool:
    is_qwen = "qwen" in model_name.lower()
    if not is_qwen:
        rank_print("[LIGER] Skipping — no compatible Liger kernels for this architecture")
        return False

    try:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3
        apply_liger_kernel_to_qwen3(
            rope=True, rms_norm=True, swiglu=True,
            cross_entropy=False, fused_linear_cross_entropy=False,
        )
        rank_print("[LIGER] Applied fused RoPE/RMSNorm/SwiGLU kernels (Qwen3)")
        return True
    except ImportError:
        rank_print("[LIGER] liger-kernel not installed, continuing without")
        return False


def load_model_and_tokenizer(
    model_name: str,
    dtype: str = "bfloat16",
    gradient_checkpointing: bool = True,
    reinit_weights: bool = False,
    use_liger: bool = True,
):
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Apply Liger fused kernels BEFORE model instantiation
    if use_liger:
        _apply_liger_kernels(model_name)

    # Model
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "attn_implementation": "flash_attention_2",
    }

    if reinit_weights:
        config = AutoConfig.from_pretrained(
            model_name, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
        )
        rank_print("[INIT] Randomly initialized model weights (reinit_weights=True)")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Force-tie lm_head.weight to embed_tokens.weight.
    # HF's tie_weights() is broken for Qwen3 + Trainer.resume_from_checkpoint:
    # it leaves lm_head at pretrained init after resume (loss explodes to ~30).
    # Direct aliasing ensures both params point to the same tensor.
    if hasattr(model, "lm_head") and hasattr(model, "get_input_embeddings"):
        embed = model.get_input_embeddings()
        if embed is not None and model.lm_head.weight is not embed.weight:
            model.lm_head.weight = embed.weight
            rank_print("[TIE] Force-tied lm_head.weight to embed_tokens.weight")

    # Gradient checkpointing setup
    if gradient_checkpointing:
        if hasattr(model, "config"):
            model.config.use_cache = False
            rank_print("[MEM] model.config.use_cache = False (training)")

    if is_main_process():
        attn_impl = getattr(model.config, "_attn_implementation", "unknown")
        rank_print(f"[ATTENTION] Using: {attn_impl}")
        n_params = sum(p.numel() for p in model.parameters())
        rank_print(f"[MODEL] {model_name} | {n_params/1e6:.1f}M params | dtype={torch_dtype}")

    return model, tokenizer
