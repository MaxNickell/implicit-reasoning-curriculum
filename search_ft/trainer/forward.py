"""Training forward on one packed micro-batch.

The micro-batch is one flat sequence [1, T] with per-sequence position_ids restarting at 0 and the
sequence boundaries as int32 cu_seq_lens (data/collate.py). hf_forward_body runs the transformer BODY
only (Qwen3Model / GPTNeoXModel after stripping the DDP wrapper) through transformers' built-in packed
flash-attention path, passing attention_mask=None, use_cache=False and explicit cu_seq_lens_q/k plus
python-int max_length_q/k. The output head, the cross-entropy and the gate metrics are computed by the
trainer (head_ce_and_preds, PackedSequenceTrainer.compute_loss).

Two properties follow from calling the body directly and are kept on purpose: DistributedDataParallel's
forward is bypassed, so gradients are NOT averaged across ranks unless training.ddp_grad_average is set,
and no autocast is entered (the model runs in its native dtype, bf16 in training).
"""
import torch
import torch.nn.functional as F

from ..model.arch import ModelParts, set_layer_checkpointing


def hf_forward_body(
    parts: ModelParts,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    max_length: int,
    use_ckpt: bool,
    ckpt_every: int,
    check: bool = False,
) -> torch.Tensor:
    """Run the HF transformer body on one packed micro-batch; returns h [T, H] after the final norm.

    With attention_mask=None transformers' flash-attention mask builder returns None and the varlen branch
    runs with our cu_seq_lens; RoPE cos/sin come from the model's own rotary embedding module, and Liger's
    module-level patches (RoPE, RMSNorm, SwiGLU) apply unchanged. Under SDPA/eager (CPU tests) the same
    inputs yield a block-diagonal causal mask derived from the restarting position_ids, so the numbers
    agree with a per-sequence forward.

    `check` enables one-time assertions on the collator contract (int32 cu_seq_lens on the model device,
    python-int max_length), on autocast being off and on no KV cache being allocated."""
    set_layer_checkpointing(parts.layers, use_ckpt, ckpt_every)
    device = input_ids.device
    # int32 on purpose: transformers annotates cu_seq_lens_* as LongTensor but flash_attn 2.8.x and
    # flash_attn_3 take int32, and transformers' own position-ids path builds int32. Do not "fix" to int64.
    if check:
        assert cu_seq_lens.dtype == torch.int32 and cu_seq_lens.device == device and isinstance(max_length, int), \
            (f"cu_seq_lens must be int32 on {device} with a python-int max_length, got "
             f"{cu_seq_lens.dtype} {cu_seq_lens.device} {type(max_length)}")
        assert not torch.is_autocast_enabled(), "autocast must stay off in the training forward"
    cu = cu_seq_lens.to(device=device, dtype=torch.int32)
    max_len = int(max_length)
    out = parts.inner(
        input_ids=input_ids,          # [1, T]
        position_ids=position_ids,    # [1, T], restarts at 0 per sequence
        attention_mask=None,
        use_cache=False,
        cu_seq_lens_q=cu, cu_seq_lens_k=cu,
        max_length_q=max_len, max_length_k=max_len,
    )
    if check:
        assert getattr(out, "past_key_values", None) is None, "body allocated a KV cache despite use_cache=False"
    return out.last_hidden_state[0]   # [T, H], after the final norm


def head_ce_and_preds(lm_head, shift_h, shift_labels, valid_mask, mode: str = "sparse", chunk_size: int = 4096):
    """Output head + per-row cross-entropy of one packed micro-batch.

    shift_h [Tm1, H], shift_labels [Tm1] (-100 = unlabelled), valid_mask = shift_labels != -100. Returns
    (ce [n_valid] in valid-row order, preds [Tm1] long, head_rows) where head_rows is how many rows went
    through lm_head. ce is F.cross_entropy(reduction='none') on the lm_head output with no dtype conversion
    here: in training (autocast off, bf16 model) the CE runs on bf16 logits in both modes.

    mode='full': lm_head and argmax over EVERY row in chunk_size slices (the full [Tm1, V] logits would not
    fit), CE at the valid rows of each slice; head_rows = Tm1.
    mode='sparse': lm_head only at rows = valid_mask.nonzero() (one call when n_valid <= chunk_size,
    otherwise chunk_size row pieces); preds is -1 at every unlabelled row and the argmax at labelled rows,
    which is all the gate metrics ever read; head_rows = n_valid. With ~1% of the packed rows labelled
    this skips about a quarter of all FLOPs per step at 156M head / 596M params.
    The two modes differ only by GEMM rounding (different row counts per cuBLAS call)."""
    Tm1 = shift_h.size(0)
    device = shift_h.device
    chunk_size = chunk_size or 4096
    ce_parts = []
    if mode == "full":
        preds = torch.empty(Tm1, dtype=torch.long, device=device)
        for cs in range(0, Tm1, chunk_size):
            ce_end = min(cs + chunk_size, Tm1)
            chunk_logits = lm_head(shift_h[cs:ce_end])  # [chunk, V]
            preds[cs:ce_end] = chunk_logits.detach().argmax(dim=-1)
            chunk_vm = valid_mask[cs:ce_end]
            if chunk_vm.any():
                ce_parts.append(F.cross_entropy(
                    chunk_logits[chunk_vm],
                    shift_labels[cs:ce_end][chunk_vm],
                    reduction='none'
                ))
            del chunk_logits
        head_rows = Tm1
    elif mode == "sparse":
        rows = valid_mask.nonzero(as_tuple=True)[0]  # long, ascending, on device
        preds = torch.full((Tm1,), -1, dtype=torch.long, device=device)
        for cs in range(0, rows.numel(), chunk_size):
            rows_c = rows[cs:cs + chunk_size]
            logits_v = lm_head(shift_h[rows_c])  # [n_valid_chunk, V]
            preds[rows_c] = logits_v.detach().argmax(dim=-1)
            ce_parts.append(F.cross_entropy(logits_v, shift_labels[rows_c], reduction='none'))
            del logits_v
        head_rows = int(rows.numel())
    else:
        raise ValueError(f"head mode must be 'sparse' or 'full', got {mode!r}")
    ce = torch.cat(ce_parts) if ce_parts else shift_h.new_zeros(0)
    return ce, preds, head_rows


def blend_first_token_ce(ce, lm_head, shift_h, valid_mask, first_indices, first_valid, targets_by_seq, w: float):
    """Soft first-token CE blend, in place on ce (the per-valid-row CE from head_ce_and_preds): for every
    sequence si with first_valid[si], the CE at its first answer row first_indices[si] becomes
    w * soft_ce + (1 - w) * ce, soft_ce = -mean log p over targets_by_seq[si] (skipped when that list is
    empty). The first rows' logits come from a second lm_head call on exactly those rows, so the term is
    identical under both head modes. Returns the number of rows that second call ran through lm_head
    (0 when w <= 0, which is the paper's setting), for the head_rows accounting."""
    if w <= 0 or not first_valid.any():
        return 0
    device = shift_h.device
    ce_indices = torch.cumsum(valid_mask.int(), dim=0) - 1  # [Tm1]
    valid_fi = first_indices[first_valid]
    valid_ce = ce_indices[valid_fi]
    fi_logits = lm_head(shift_h[valid_fi])  # [N_first, V]
    batch_logp = F.log_softmax(fi_logits, dim=-1)
    del fi_logits

    ce_list = valid_ce.tolist()
    valid_seq_indices = torch.nonzero(first_valid, as_tuple=True)[0].tolist()
    for j, si in enumerate(valid_seq_indices):
        vf = targets_by_seq[si]
        if not vf:
            continue
        ids = torch.tensor(vf, device=device, dtype=torch.long)
        soft_ce = -batch_logp[j, ids].mean()
        ci = ce_list[j]
        ce[ci] = w * soft_ce + (1.0 - w) * ce[ci]
    return int(valid_fi.numel())
