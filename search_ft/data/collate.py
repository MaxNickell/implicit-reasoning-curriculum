"""Packing of tokenized samples into one flat micro-batch for transformers' packed flash-attention path.

Every micro-batch is a single row [1, T] with no padding: the samples are concatenated, position_ids
restart at 0 for each sample, and the per-sequence boundaries are passed to the attention kernel as
cu_seq_lens (int32, cumulative lengths, [S + 1]) plus max_length (python int). The trainer hands these
to the HF transformer body as cu_seq_lens_q/k and max_length_q/k (see trainer/forward.py).
"""
from typing import Dict, List, Any

import torch


def pack_batch(samples: List[Dict]) -> Dict[str, Any]:
    all_input_ids = []
    all_labels = []
    all_position_ids = []
    cu_seqlens = [0]
    max_seqlen = 0
    all_sequence_info = []

    offset = 0
    for sample in samples:
        seq_len = sample["seq_len"]

        all_input_ids.extend(sample["input_ids"])
        all_labels.extend(sample["labels"])
        all_position_ids.extend(range(seq_len))

        offset += seq_len
        cu_seqlens.append(offset)

        max_seqlen = max(max_seqlen, seq_len)

        all_sequence_info.append({
            "start_idx": cu_seqlens[-2],
            "end_idx": cu_seqlens[-1],
            "prompt_len": sample["prompt_len"],
            "valid_first_targets": sample["valid_first_targets"],
        })

    return {
        "input_ids": torch.tensor([all_input_ids], dtype=torch.long),        # [1, T]
        "labels": torch.tensor([all_labels], dtype=torch.long),              # [1, T]
        "position_ids": torch.tensor([all_position_ids], dtype=torch.long),  # [1, T], restarts at 0 per sequence
        # int32 on purpose: flash_attn 2.8.x / flash_attn_3 take int32 cumulative lengths and transformers'
        # own position-ids path builds int32; max_length must be a python int.
        "cu_seq_lens": torch.tensor(cu_seqlens, dtype=torch.int32),          # [S + 1]
        "max_length": int(max_seqlen),
        "sequence_info": all_sequence_info,
        "num_sequences": len(samples),
    }
