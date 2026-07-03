import math
import re
from typing import Dict, List, Any

import torch
from transformers import GenerationConfig

from ..distributed.utils import (
    get_rank, get_world_size, is_main_process, barrier, dist_is_initialized, rank_print
)


def _tokenize_leading_space(tokenizer, s: str) -> List[int]:
    return tokenizer(" " + s, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def run_eval_greedy(
    model,
    tokenizer,
    inputs: List[str],
    labels: List[List[str]],
    print_examples: int = 0,
    use_chat_template: bool = False,
) -> Dict[str, Any]:
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
    actual_count = len(my_inputs)

    # Pad to uniform chunk_size (synced_gpus requires all ranks generate same count)
    pad_needed = chunk_size - actual_count
    if pad_needed > 0:
        dummy_in = my_inputs[-1] if my_inputs else ""
        dummy_la = my_labels[-1] if my_labels else []
        my_inputs.extend([dummy_in] * pad_needed)
        my_labels.extend([dummy_la] * pad_needed)

    rank_print(f"[EVAL-GREEDY] Starting generation on {total_len} samples "
               f"(Local: {actual_count}, Pad: {pad_needed})")

    local_correct_first = 0
    local_correct_full = 0
    local_total = 0
    local_printed = 0

    allow_print = is_main_process() and (print_examples > 0)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    greedy_config = GenerationConfig(
        max_new_tokens=24,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True
    )

    for i, (x, ys) in enumerate(zip(my_inputs, my_labels)):
        is_padding = (i >= actual_count)
        ys = ys if isinstance(ys, list) else [ys]

        if use_chat_template:
            msgs = [{"role": "user", "content": x}]
            input_ids = tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True,
                enable_thinking=False, return_tensors="pt",
            ).to(device)
            enc = {"input_ids": input_ids}
            prompt_len = input_ids.shape[1]
        else:
            enc = tokenizer(x, return_tensors="pt").to(device)
            prompt_len = enc["input_ids"].shape[1]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            unwrapped = model.module if hasattr(model, "module") else model
            gen_out = unwrapped.generate(
                **enc,
                max_new_tokens=24,
                generation_config=greedy_config,
                synced_gpus=(world_size > 1),
            )

        if is_padding:
            continue

        # Extract predicted word
        gen_ids = gen_out[0][prompt_len:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        pred_word = re.split(r"[.,\s]+", gen_text)[0] if gen_text else ""

        # First-token accuracy
        valid_first_ids = {
            _tokenize_leading_space(tokenizer, y)[0]
            for y in ys
            if _tokenize_leading_space(tokenizer, y)
        }
        first_ok = (len(gen_ids) > 0 and gen_ids[0].item() in valid_first_ids)
        local_correct_first += int(first_ok)

        # Full-word accuracy (case-insensitive)
        full_ok = any(pred_word.lower() == y.lower() for y in ys)
        local_correct_full += int(full_ok)
        local_total += 1

        # Print mistakes (rank 0 only)
        if allow_print and (not first_ok or not full_ok) and local_printed < print_examples:
            print("\n" + "=" * 60)
            print(f"[GREEDY MISTAKE] (Rank {rank})")
            print(f"Prompt (tail): ...{x[-100:]}")
            print(f"Gold Options: {ys}")
            print(f"Predicted Word: '{pred_word}'")
            print(f"Full Generation: '{gen_text}'")
            local_printed += 1

    # Aggregate across ranks
    metrics = torch.tensor(
        [local_correct_first, local_correct_full, local_total],
        dtype=torch.long, device=device
    )
    if dist_is_initialized():
        torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)

    global_first = metrics[0].item()
    global_full = metrics[1].item()
    global_total = metrics[2].item()

    barrier()
    rank_print(f"[EVAL-GREEDY] Completed. Global samples: {global_total}")

    return {
        "first_token_acc": (global_first / global_total) if global_total else 0.0,
        "full_word_acc": (global_full / global_total) if global_total else 0.0,
        "first_token_hits": global_first,
        "full_word_hits": global_full,
        "total": global_total,
    }
