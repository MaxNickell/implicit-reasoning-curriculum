import time
from collections import deque
from typing import Dict, Any, Optional, List

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from transformers import Trainer

from ..distributed.utils import get_rank, get_world_size, is_main_process, dist_is_initialized
from ..model.arch import ModelParts, detect_architecture
from .forward import forward_layer_varlen


class PackedSequenceTrainer(Trainer):

    CE_CHUNK_SIZE = 4096

    def __init__(self, *args, accuracy_window=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        self.recent_losses = deque(maxlen=100)
        self.first_token_correct = deque(maxlen=accuracy_window)
        self.full_word_correct = deque(maxlen=accuracy_window)

        self._last_batch_samples = 0
        self._last_batch_tokens = 0

        # Accumulate across micro-batches within one optimizer step (for grad_acc > 1)
        self._step_samples = 0
        self._step_tokens = 0

        # Lightweight timing
        self._step_end_time: Optional[float] = None
        self._steps_timed = 0
        self._total_step_time = 0.0

        # Cached model internals (populated on first compute_loss call)
        self._cached_parts: Optional[ModelParts] = None

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        if self.tokenizer is not None and output_dir is not None:
            self.tokenizer.save_pretrained(output_dir)

    def get_train_dataloader(self):
        from torch.utils.data import DataLoader
        nw = self.args.dataloader_num_workers
        return DataLoader(
            self.train_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: x[0],
            num_workers=nw,
            prefetch_factor=self.args.dataloader_prefetch_factor if nw > 0 else None,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=nw > 0,
        )

    def training_step(self, model, inputs, num_items_in_batch=None):
        t0 = time.perf_counter()
        loss = super().training_step(model, inputs, num_items_in_batch)
        t1 = time.perf_counter()
        self._total_step_time += t1 - t0
        self._steps_timed += 1
        self._step_end_time = t1
        return loss

    def get_step_time_ms(self) -> float:
        if self._steps_timed == 0:
            return 0.0
        return (self._total_step_time / self._steps_timed) * 1000

    def reset_timing(self):
        self._step_end_time = None
        self._steps_timed = 0
        self._total_step_time = 0.0

    def _get_parts(self, model) -> ModelParts:
        if self._cached_parts is None:
            self._cached_parts = detect_architecture(model)
        return self._cached_parts

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extract metadata
        sequence_info = inputs.pop("sequence_info")
        num_sequences = inputs.pop("num_sequences")
        cu_seqlens_list = inputs.pop("cu_seqlens_list")
        max_seqlen_list = inputs.pop("max_seqlen_list")
        inputs.pop("_efficiency", None)

        self._last_batch_samples = num_sequences
        self._last_batch_tokens = sum(cu[-1] for cu in cu_seqlens_list)

        self._step_samples += self._last_batch_samples
        self._step_tokens += self._last_batch_tokens

        input_ids = inputs["input_ids"]       # [1, total_tokens]
        labels_pad = inputs["labels"]         # [1, total_tokens]
        position_ids = inputs["position_ids"] # [1, total_tokens]

        device = input_ids.device
        use_grad_checkpoint = getattr(self.args, 'gradient_checkpointing', False)

        parts = self._get_parts(model)

        # Single row, no padding — directly use the tensors
        flat_ids = input_ids[0]           # [total_tokens]
        all_labels = labels_pad[0]        # [total_tokens]
        all_pos = position_ids[0:1]       # [1, total_tokens]
        merged_cu = torch.tensor(cu_seqlens_list[0], dtype=torch.int32, device=device)
        global_max_seqlen = max_seqlen_list[0]

        # Embed tokens
        h = parts.embed(flat_ids)  # [total_tokens, H]

        # Compute RoPE cos/sin once (same for all layers)
        cos, sin = parts.rotary_emb(h, all_pos)

        # Layer-by-layer forward with flash_attn_varlen_func
        for layer in parts.layers:
            if use_grad_checkpoint and model.training:
                h = checkpoint.checkpoint(
                    forward_layer_varlen,
                    layer, h, merged_cu, global_max_seqlen,
                    parts, cos, sin,
                    use_reentrant=False,
                )
            else:
                h = forward_layer_varlen(
                    layer, h, merged_cu, global_max_seqlen,
                    parts, cos, sin,
                )

        h = parts.norm(h)  # [total_tokens, H]

        # Shift for autoregressive loss
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
                is_search[si] = pl > 0
                first_indices[si] = seq["start_idx"] + max(pl, 1) - 1
                seq_starts[si] = seq["start_idx"] + max(pl, 1) - 1
                seq_ends[si] = min(seq["end_idx"] - 1, Tm1)

            first_in_range = (first_indices >= 0) & (first_indices < Tm1) & is_search
            first_valid = first_in_range & valid_mask[first_indices.clamp(0, Tm1 - 1)]

            # Chunked lm_head to avoid OOM
            chunk_size = self.CE_CHUNK_SIZE
            ce_parts = []
            preds = torch.empty(Tm1, dtype=torch.long, device=device)

            for cs in range(0, Tm1, chunk_size):
                ce_end = min(cs + chunk_size, Tm1)
                chunk_logits = parts.lm_head(shift_h[cs:ce_end])  # [chunk, V]
                preds[cs:ce_end] = chunk_logits.detach().argmax(dim=-1)
                chunk_vm = valid_mask[cs:ce_end]
                if chunk_vm.any():
                    ce_parts.append(F.cross_entropy(
                        chunk_logits[chunk_vm],
                        shift_labels[cs:ce_end][chunk_vm],
                        reduction='none'
                    ))
                del chunk_logits

            ce = torch.cat(ce_parts)
            loss = ce.sum() / n_valid

            # Track accuracy
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

                # All-gather across ranks for GPU-count-independent accuracy
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
