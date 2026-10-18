"""Attention-kernel selection for training and SDPA pinning for in-run evaluation.

Training runs the HF transformer body through transformers' built-in packed (varlen) flash-attention
path, so the kernel is chosen once at model load: ``model.attn_kernel`` auto = flash_attention_3 on
Hopper (sm90), flash_attention_2 elsewhere; fa2/fa3 force one. Every in-run evaluation (teacher-forced
loss, greedy decoding) runs under SDPA, the transformers default, and the training implementation is
restored afterwards (``eval_attn_sdpa``).
"""
import contextlib

import torch

from ..config import ATTN_KERNELS
from ..distributed.utils import rank_print
from .arch import unwrap_model_for_body, resolve_model_parts


# config._attn_implementation the model was loaded with (recorded by verify_attention_setup);
# eval_attn_sdpa asserts that the restore lands back on it.
_TRAIN_ATTN_IMPL = None


def is_hopper() -> bool:
    """FA3 is built around Hopper (sm90a: H100/H200). Auto mode uses FA3 only there."""
    try:
        return bool(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] == 9)
    except Exception:
        return False


def select_attn_implementation(attn_kernel: str = "auto") -> str:
    """Map model.attn_kernel onto the transformers attn_implementation string used at model load.
    The paper's Qwen runs used FA2 and the Pythia runs FA3; attn_kernel: fa2 reproduces the Qwen
    kernel exactly. transformers accepts flash_attention_3 wherever the flash_attn_3 package imports,
    so the Hopper gate lives here."""
    if attn_kernel not in ATTN_KERNELS:
        raise ValueError(f"model.attn_kernel must be one of {ATTN_KERNELS}, got {attn_kernel!r}")
    if attn_kernel == "fa3" or (attn_kernel == "auto" and is_hopper()):
        return "flash_attention_3"
    return "flash_attention_2"


def _check_flash_attention_kwargs() -> None:
    """The packed forward hands cu_seq_lens_q/k and max_length_q/k to the body as keyword arguments;
    they reach the kernel through transformers' FlashAttentionKwargs only (written against 4.57.5)."""
    try:
        from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
    except ImportError as e:
        raise ImportError("transformers.modeling_flash_attention_utils.FlashAttentionKwargs is missing; "
                          "the packed forward was written against transformers==4.57.5") from e
    fields = (set(getattr(FlashAttentionKwargs, "__annotations__", {}))
              | set(getattr(FlashAttentionKwargs, "__dataclass_fields__", {})))
    needed = {"cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"}
    if not needed <= fields:
        raise ImportError(f"FlashAttentionKwargs does not accept {sorted(needed)} (fields: {sorted(fields)}); "
                          "the packed forward was written against transformers==4.57.5")


def verify_attention_setup(model, requested_impl: str) -> str:
    """After model load: check that the load honoured the requested attention implementation, that
    every decoder layer is a transformers GradientCheckpointingLayer (per-layer selective checkpointing
    relies on it), and that transformers imports the varlen kernel from the expected package (FA3 lives
    in the top-level ``flash_attn_interface`` module, FA2 in ``flash_attn``). Records the training
    implementation for eval_attn_sdpa and returns it."""
    global _TRAIN_ATTN_IMPL
    _check_flash_attention_kwargs()

    impl = getattr(model.config, "_attn_implementation", None)
    assert impl == requested_impl, \
        f"model loaded with attn_implementation={impl!r}, requested {requested_impl!r}"

    from transformers.modeling_layers import GradientCheckpointingLayer
    parts = resolve_model_parts(model)
    not_gcl = [type(l).__name__ for l in parts.layers if not isinstance(l, GradientCheckpointingLayer)]
    assert not not_gcl, ("per-layer selective checkpointing needs every decoder layer to be a "
                         f"transformers GradientCheckpointingLayer; found {not_gcl}")

    kernel_mod = "n/a"
    if impl.startswith("flash_attention"):
        import transformers.modeling_flash_attention_utils as mfa
        mfa.lazy_import_flash_attention(impl, force_import=True)
        vfn = mfa._flash_varlen_fn
        kernel_mod = getattr(vfn, "__module__", "") or ""
        want = "flash_attn_interface" if impl == "flash_attention_3" else "flash_attn"
        assert vfn is not None and kernel_mod.split(".")[0] == want, \
            f"transformers imported the varlen kernel from {kernel_mod!r}; expected package {want!r} for {impl}"

    _TRAIN_ATTN_IMPL = impl
    rank_print(f"[ATTN-KERNEL] verified: config._attn_implementation={impl}; varlen kernel from "
               f"{kernel_mod}; {len(parts.layers)} {parts.arch} decoder layers are GradientCheckpointingLayer")
    return impl


@contextlib.contextmanager
def eval_attn_sdpa(model):
    """Run an in-run eval (teacher-forced loss, generate) with SDPA, the transformers default, and
    restore the training attention implementation afterwards.

    Operates on the unwrapped PreTrainedModel (DistributedDataParallel does not forward attribute
    access). set_attn_implementation rewrites the config shared by every attention module, which
    reads it at call time, so the switch is atomic across layers. The restore is asserted: a model
    left on SDPA would turn the packed position_ids with attention_mask=None into a dense
    [1, 1, T, T] mask on the next training micro-batch. No-op when the model already runs SDPA."""
    pm = unwrap_model_for_body(model)
    train_impl = pm.config._attn_implementation
    if train_impl == "sdpa":
        yield
        return
    pm.set_attn_implementation("sdpa")
    assert pm.config._attn_implementation == "sdpa", pm.config._attn_implementation
    try:
        yield
    finally:
        pm.set_attn_implementation(train_impl)
        assert pm.config._attn_implementation == train_impl, \
            f"attention implementation not restored after eval: {pm.config._attn_implementation!r} != {train_impl!r}"
        if _TRAIN_ATTN_IMPL is not None:
            assert pm.config._attn_implementation == _TRAIN_ATTN_IMPL, \
                f"training attention implementation drifted: {pm.config._attn_implementation!r} != {_TRAIN_ATTN_IMPL!r}"
