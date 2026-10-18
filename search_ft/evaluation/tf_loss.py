import math
import random
from typing import List

import torch

from ..distributed.utils import (
    get_rank, get_world_size, barrier, dist_is_initialized, rank_print
)
from ..model.attention import eval_attn_sdpa


def _tokenize_leading_space(tokenizer, s: str) -> List[int]:
    return tokenizer(" " + s, add_special_tokens=False)["input_ids"]


def run_eval_tf_loss(
    model,
    tokenizer,
    inputs: List[str],
    labels: List[List[str]],
    seed: int = 0,
    use_chat_template: bool = False,
) -> float:
    """Teacher-forced eval loss, run with SDPA (eval_attn_sdpa) whatever the training attention
    implementation is; the training implementation is restored on exit."""
    with eval_attn_sdpa(model):
        return _run_eval_tf_loss_impl(model, tokenizer, inputs, labels,
                                      seed=seed, use_chat_template=use_chat_template)


@torch.no_grad()
def _run_eval_tf_loss_impl(
    model,
    tokenizer,
    inputs: List[str],
    labels: List[List[str]],
    seed: int = 0,
    use_chat_template: bool = False,
) -> float:
    barrier()
    rank = get_rank()
    world_size = get_world_size()
    device = next(model.parameters()).device

    # Shard data across ranks
    total_len = len(inputs)
    chunk_size = math.ceil(total_len / world_size)
    start_idx = rank * chunk_size
    end_idx = min(start_idx + chunk_size, total_len)

    my_inputs = inputs[start_idx:end_idx]
    my_labels = labels[start_idx:end_idx]

    rng = random.Random((seed or 0) + 777)

    local_loss_sum = 0.0
    local_count = 0
    unwrapped = model.module if hasattr(model, "module") else model

    for x, ys in zip(my_inputs, my_labels):
        ys = ys if isinstance(ys, list) else [ys]
        chosen = rng.choice(ys)

        if use_chat_template:
            msgs = [
                {"role": "user", "content": x},
                {"role": "assistant", "content": chosen},
            ]
            full_ids = tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=False,
                enable_thinking=False,
            )
            prefix_ids = tokenizer.apply_chat_template(
                msgs[:1], tokenize=True, add_generation_prompt=True,
                enable_thinking=False,
            )
            input_ids = torch.tensor([full_ids], device=device)
            label_ids = torch.tensor([[-100] * len(prefix_ids) + full_ids[len(prefix_ids):]], device=device)
        else:
            prompt_ids = tokenizer(x, add_special_tokens=True, truncation=False)["input_ids"]
            ans_ids = _tokenize_leading_space(tokenizer, chosen)
            end_ids = tokenizer(". ", add_special_tokens=False)["input_ids"]
            input_ids = torch.tensor([prompt_ids + ans_ids + end_ids], device=device)
            label_ids = torch.tensor([[-100] * len(prompt_ids) + ans_ids + end_ids], device=device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = unwrapped(input_ids=input_ids, labels=label_ids)

        local_loss_sum += out.loss.item()
        local_count += 1

    # Aggregate across ranks
    metrics = torch.tensor([local_loss_sum, float(local_count)], dtype=torch.float64, device=device)
    if dist_is_initialized():
        torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)

    total_loss = metrics[0].item()
    total_count = int(metrics[1].item())

    barrier()
    avg_loss = total_loss / total_count if total_count > 0 else 0.0
    rank_print(f"[EVAL-TF] Loss={avg_loss:.4f} (n={total_count})")
    return avg_loss
