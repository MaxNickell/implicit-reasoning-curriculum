import os
import time
from collections import deque
from typing import Optional

import torch

from transformers import Trainer

from ..distributed.utils import get_world_size, is_main_process, rank_print, dist_is_initialized
from ..model.arch import ModelParts, resolve_model_parts
from .forward import hf_forward_body, head_ce_and_preds, blend_first_token_ce


def _dataloader_worker_init(worker_id: int) -> None:
    """Pin each dataloader worker to one CPU thread. Workers inherit OMP_NUM_THREADS from the trainer
    process, so num_workers of them per rank oversubscribe the node; sample generation gains nothing
    from OpenMP and only pays the contention."""
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PackedSequenceTrainer(Trainer):
    """Trainer for packed sequences: one flat [1, T] micro-batch with per-sequence position_ids and
    cu_seq_lens. compute_loss runs the HF body through transformers' packed flash-attention path
    (forward.hf_forward_body); the head, loss and gate metrics are computed here."""

    def __init__(
        self,
        *args,
        accuracy_window: int = 800,
        head: str = "sparse",
        ce_chunk_size: int = 4096,
        first_token_soft_weight: float = 0.0,
        ckpt_every_n_layers: int = 1,
        ckpt_relax_max_tokens: int = 0,
        ddp_grad_average: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        if head not in ("sparse", "full"):
            raise ValueError(f"head must be 'sparse' or 'full', got {head!r}")
        self.head = head
        self.ce_chunk_size = ce_chunk_size
        # Soft first-token CE blend weight; 0 (the paper's setting) disables the blend entirely.
        self.first_token_soft_weight = first_token_soft_weight
        self.ckpt_every_n_layers = max(1, int(ckpt_every_n_layers))
        self.ckpt_relax_max_tokens = int(ckpt_relax_max_tokens)

        self.recent_losses = deque(maxlen=100)
        # accuracy_window is the *global* sample count (total across ranks): predictions are all-gathered
        # across ranks every micro-batch so all ranks see the same deque state.
        self.first_token_correct = deque(maxlen=accuracy_window)
        self.full_word_correct = deque(maxlen=accuracy_window)

        self._last_batch_samples = 0
        self._last_batch_tokens = 0

        # Accumulate across micro-batches within one optimizer step (for grad_acc > 1); the curriculum
        # callback reads and resets these at the end of every optimizer step.
        self._step_samples = 0
        self._step_tokens = 0
        self._step_head_rows = 0   # rows the lm_head processed this optimizer step

        # Lightweight timing
        self._step_end_time: Optional[float] = None
        self._steps_timed = 0
        self._total_step_time = 0.0

        # Cached model internals (populated on first compute_loss call)
        self._cached_parts: Optional[ModelParts] = None
        self._first_batch_checked = False   # one-time collator-contract / no-autocast / no-cache check
        self._ckpt_gate_last = None
        self._ckpt_gate_flips = 0

        # Cross-rank gradient averaging. Off: compute_loss bypasses DistributedDataParallel.forward, so
        # each rank trains its own replica between checkpoint reloads (the paper's setting). On: all-reduce
        # (AVG) every trainable gradient after backward, before clipping, i.e. what DDP would do.
        self._ddp_grad_average = bool(ddp_grad_average)
        if self._ddp_grad_average:
            rank_print("[DDP] cross-rank gradient averaging: ON (training.ddp_grad_average=true; all-reduce AVG "
                       "of trainable grads after backward, before clipping)")
        else:
            rank_print("[DDP] cross-rank gradient averaging: OFF (each rank trains its own replica; the "
                       "checkpoint is rank 0's; set training.ddp_grad_average=true to all-reduce)")
        if self.head == "sparse":
            rank_print(f"[HEAD] sparse: lm_head/argmax/CE on labelled rows only, in {self.ce_chunk_size}-row "
                       f"pieces; per-step rows logged as head_rows next to tokens in loss_history")
        else:
            rank_print(f"[HEAD] full: chunked lm_head over every packed row, {self.ce_chunk_size} rows per chunk")

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        if self.tokenizer is not None and output_dir is not None:
            self.tokenizer.save_pretrained(output_dir)

    def get_train_dataloader(self):
        """Bypass Accelerate's dataloader wrapping; one dataset item is one packed micro-batch."""
        from torch.utils.data import DataLoader
        nw = self.args.dataloader_num_workers
        return DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=False,  # idx provides randomness via seeding
            collate_fn=lambda x: x[0],  # Unwrap single-item batch
            num_workers=nw,
            prefetch_factor=self.args.dataloader_prefetch_factor if nw > 0 else None,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=nw > 0,
            worker_init_fn=_dataloader_worker_init if nw > 0 else None,
        )

    def training_step(self, model, inputs, num_items_in_batch=None):
        t0 = time.perf_counter()
        loss = super().training_step(model, inputs, num_items_in_batch)
        # Trainer.training_step has run backward. sync_gradients is True on the last micro-batch of the
        # accumulation window; clipping and the optimizer step follow in the training loop, so this is
        # the point where DistributedDataParallel would have reduced.
        if self.accelerator.sync_gradients:
            if self._ddp_grad_average and dist_is_initialized():
                self._all_reduce_grads(model)
        t1 = time.perf_counter()
        self._total_step_time += t1 - t0
        self._steps_timed += 1
        self._step_end_time = t1
        return loss

    def _all_reduce_grads(self, model):
        """training.ddp_grad_average: average every trainable parameter's gradient across ranks
        (ReduceOp.AVG), i.e. what DistributedDataParallel would do if compute_loss went through its forward."""
        with torch.no_grad():
            for p in model.parameters():
                if not p.requires_grad:
                    continue
                if p.grad is None:
                    # A rank whose whole accumulation window produced no valid labels has no grads; it must
                    # still take part in every collective or the ranks' all-reduce sequences desynchronise.
                    # Zero is the correct contribution to the average.
                    p.grad = torch.zeros_like(p)
                torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)

    def get_step_time_ms(self) -> float:
        """Mean wall time of one micro-batch (forward + backward [+ all-reduce]) since the last reset."""
        if self._steps_timed == 0:
            return 0.0
        return (self._total_step_time / self._steps_timed) * 1000

    def reset_timing(self):
        self._step_end_time = None
        self._steps_timed = 0
        self._total_step_time = 0.0

    def _get_parts(self, model) -> ModelParts:
        if self._cached_parts is None:
            self._cached_parts = resolve_model_parts(model)
        return self._cached_parts

    def _resolve_ckpt_every(self, tokens: int) -> int:
        """Selective-checkpointing stride for this micro-batch: model.ckpt_every_n_layers, falling back to
        every layer when the micro-batch exceeds model.ckpt_relax_max_tokens tokens (0 = never). Activation
        memory tracks tokens per micro-batch, which grows with the curriculum stage, so the gate is keyed on
        the actual token count rather than on the stage. Gradients are identical either way."""
        ckpt_every = self.ckpt_every_n_layers
        relax_max = self.ckpt_relax_max_tokens
        gated = ckpt_every > 1 and relax_max > 0 and tokens > relax_max
        if gated:
            ckpt_every = 1
        # Packing makes tokens per micro-batch vary by tens of percent within a stage, so near the threshold
        # the setting flips often; log the first few transitions in full, then one line per 500 flips.
        if ckpt_every != self._ckpt_gate_last:
            self._ckpt_gate_last = ckpt_every
            self._ckpt_gate_flips += 1
            log_first, log_every = 5, 500
            if is_main_process() and (self._ckpt_gate_flips <= log_first or self._ckpt_gate_flips % log_every == 0):
                tail = f"  [flip {self._ckpt_gate_flips:,}]" if self._ckpt_gate_flips > log_first else ""
                print(f"[CKPT-GATE] recomputing every {ckpt_every} layer(s) at {tokens:,} tokens/micro-batch "
                      f"(every_n={self.ckpt_every_n_layers}, relax_max={relax_max:,}"
                      f"{', gate ACTIVE' if gated else ''}){tail}", flush=True)
        return ckpt_every

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extract metadata
        sequence_info = inputs.pop("sequence_info")
        num_sequences = inputs.pop("num_sequences")
        cu_seq_lens = inputs.pop("cu_seq_lens")   # int32 [S + 1]
        max_length = inputs.pop("max_length")     # python int

        self._last_batch_samples = num_sequences
        self._last_batch_tokens = int(cu_seq_lens[-1])

        self._step_samples += self._last_batch_samples
        self._step_tokens += self._last_batch_tokens

        input_ids = inputs["input_ids"]       # [1, total_tokens]
        labels_pad = inputs["labels"]         # [1, total_tokens]
        position_ids = inputs["position_ids"] # [1, total_tokens]

        device = input_ids.device

        use_grad_checkpoint = getattr(self.args, 'gradient_checkpointing', False)
        ckpt_every = self._resolve_ckpt_every(self._last_batch_tokens)
        use_ckpt = bool(use_grad_checkpoint) and model.training

        parts = self._get_parts(model)
        all_labels = labels_pad[0]        # [total_tokens]

        # Transformer body -> h [total_tokens, H] after the final norm. Bypasses the DDP wrapper and runs
        # without autocast (forward.py).
        h = hf_forward_body(parts, input_ids, position_ids, cu_seq_lens, max_length,
                            use_ckpt, ckpt_every, check=not self._first_batch_checked)
        self._first_batch_checked = True

        # Shift for autoregressive loss (per-sequence boundaries are masked by -100 labels)
        shift_h = h[:-1, :]           # [Tm1, H]
        shift_labels = all_labels[1:] # [Tm1]
        Tm1 = shift_h.size(0)

        valid_mask = (shift_labels != -100)
        n_valid = valid_mask.sum().item()

        if n_valid > 0:
            # Pre-compute first_idx and answer spans for all sequences
            S = len(sequence_info)
            first_indices = torch.zeros(S, dtype=torch.long, device=device)
            seq_starts = torch.zeros(S, dtype=torch.long, device=device)
            seq_ends = torch.zeros(S, dtype=torch.long, device=device)
            is_search = torch.zeros(S, dtype=torch.bool, device=device)
            for si, seq in enumerate(sequence_info):
                pl = seq["prompt_len"]
                is_search[si] = pl > 0  # instruction samples have prompt_len=0
                first_indices[si] = seq["start_idx"] + max(pl, 1) - 1
                seq_starts[si] = seq["start_idx"] + max(pl, 1) - 1
                seq_ends[si] = min(seq["end_idx"] - 1, Tm1)

            first_in_range = (first_indices >= 0) & (first_indices < Tm1) & is_search
            first_valid = first_in_range & valid_mask[first_indices.clamp(0, Tm1 - 1)]

            # Output head + per-valid-row CE (sparse: labelled rows only, preds = -1 elsewhere; every consumer
            # below reads labelled rows only). ce is ordered by ascending valid row either way.
            ce, preds, head_rows = head_ce_and_preds(parts.lm_head, shift_h, shift_labels, valid_mask,
                                                     mode=self.head, chunk_size=self.ce_chunk_size)

            # Soft first-token CE blend (no-op at weight 0, the paper's setting)
            head_rows += blend_first_token_ce(ce, parts.lm_head, shift_h, valid_mask, first_indices, first_valid,
                                              [seq["valid_first_targets"] for seq in sequence_info],
                                              self.first_token_soft_weight)
            self._step_head_rows += head_rows

            loss = ce.sum() / n_valid

            # Track accuracy (batched GPU->CPU transfers)
            with torch.no_grad():
                pred_matches = (preds == shift_labels)
                first_pred_list = preds[first_indices[first_valid].clamp(0, Tm1 - 1)].tolist()
                first_si_list = torch.nonzero(first_valid, as_tuple=True)[0].tolist()
                starts = seq_starts.tolist()
                ends = seq_ends.tolist()
                pred_matches_cpu = pred_matches.cpu()
                valid_mask_cpu = valid_mask.cpu()

                # First-token accuracy
                local_first = [first_pred_list[j] in sequence_info[si]["valid_first_targets"]
                               for j, si in enumerate(first_si_list)]

                # Full-word accuracy (search samples only)
                is_search_cpu = is_search.cpu()
                local_full = []
                for si in range(S):
                    if not is_search_cpu[si]:
                        continue
                    s, e = starts[si], ends[si]
                    if s >= e:
                        continue
                    span_valid = valid_mask_cpu[s:e]
                    if span_valid.any():
                        local_full.append(pred_matches_cpu[s:e][span_valid].all().item())

                # All-gather across ranks so the rolling buffers are GPU-count independent
                if dist_is_initialized():
                    ws = get_world_size()
                    g_first = [None] * ws
                    g_full = [None] * ws
                    torch.distributed.all_gather_object(g_first, local_first)
                    torch.distributed.all_gather_object(g_full, local_full)
                    self.first_token_correct.extend(b for sl in g_first for b in sl)
                    self.full_word_correct.extend(b for sl in g_full for b in sl)
                else:
                    self.first_token_correct.extend(local_first)
                    self.full_word_correct.extend(local_full)
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        self.recent_losses.append(loss.item())

        return loss

    def get_first_token_acc(self) -> float:
        if not self.first_token_correct:
            return 0.0
        return sum(self.first_token_correct) / len(self.first_token_correct)

    def get_full_word_acc(self) -> float:
        if not self.full_word_correct:
            return 0.0
        return sum(self.full_word_correct) / len(self.full_word_correct)
