import torch
from typing import Dict, List


def chat_format(tokenizer, user_content: str) -> str:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True,
        )


def loglik_completion(model, tokenizer, prefix: str, completion: str) -> float:
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    full_ids = tokenizer.encode(prefix + completion, add_special_tokens=False)

    # Find longest common prefix of token-id lists
    split = 0
    while split < len(prefix_ids) and split < len(full_ids) and prefix_ids[split] == full_ids[split]:
        split += 1
    if split == len(full_ids):
        return 0.0  # completion absorbed into last prefix token

    inp = torch.tensor([full_ids], device=model.device)
    with torch.no_grad():
        logits = model(inp).logits[0]  # [seq_len, vocab]
    log_probs = torch.log_softmax(logits, dim=-1)

    # log p(token[t] | tokens[:t]) = log_probs[t-1, token[t]]
    total = 0.0
    for t in range(split, len(full_ids)):
        total += log_probs[t - 1, full_ids[t]].item()
    return total


def score_choices(
    model,
    tokenizer,
    prompt: str,
    choices: List[str],
    use_chat: bool,
) -> Dict[str, any]:
    prefix = chat_format(tokenizer, prompt) if use_chat else prompt

    # Tokenize each choice with leading space
    choice_token_ids = {}
    for c in choices:
        ids = tokenizer.encode(" " + c, add_special_tokens=False)
        choice_token_ids[c] = ids

    all_single = all(len(ids) == 1 for ids in choice_token_ids.values())

    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    inp = torch.tensor([prefix_ids], device=model.device)

    with torch.no_grad():
        logits = model(inp).logits[0, -1, :]  # next-token logits
    log_probs = torch.log_softmax(logits, dim=-1)

    if all_single:
        # Fast path: read directly from next-token distribution
        scores = {c: log_probs[ids[0]].item() for c, ids in choice_token_ids.items()}
    else:
        # Slow path: teacher-forced log-likelihood for multi-token choices
        scores = {c: loglik_completion(model, tokenizer, prefix, " " + c) for c in choices}

    prediction = max(scores, key=scores.get)

    return {"logprobs": scores, "prediction": prediction}


def generate(
    model,
    tokenizer,
    prompt: str,
    use_chat: bool,
    max_new_tokens: int = 100,
) -> str:
    prefix = chat_format(tokenizer, prompt) if use_chat else prompt
    input_ids = tokenizer.encode(prefix, return_tensors="pt", add_special_tokens=False)
    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def fewshot_block_from_pool(pool, k, prompt_fn, gold_fn, seed=1234):
    import random as _random
    rng = _random.Random(seed)
    shuffled = list(pool)
    rng.shuffle(shuffled)
    picked = []
    seen_golds = []
    for ex in shuffled:
        g = gold_fn(ex)
        if g is None:
            continue
        if g in seen_golds and len(seen_golds) < k and len(picked) + (k - len(seen_golds)) > k:
            continue
        if g not in seen_golds:
            seen_golds.append(g)
        picked.append((ex, g))
        if len(picked) >= k:
            break
    blocks = [f"{prompt_fn(ex)} {g}" for ex, g in picked]
    used_ids = {id(ex) for ex, _ in picked}
    return "\n\n".join(blocks) + ("\n\n" if blocks else ""), used_ids


def subsample(items: list, cap: int, seed: int = 42) -> list:
    if cap <= 0:
        return []
    if len(items) <= cap:
        return list(items)
    import random
    return random.Random(seed).sample(list(items), cap)


def distribute_n(n: int, num_subsets: int, sizes: List[int] = None) -> List[int]:
    if num_subsets <= 0:
        return []
    if n is None or n <= 0:
        return [None] * num_subsets

    if sizes is None:
        base = n // num_subsets
        rem = n - base * num_subsets
        return [base + (1 if i < rem else 0) for i in range(num_subsets)]

    # Adaptive: if a subset has fewer than its share, redistribute leftover.
    caps = [min(sizes[i], n // num_subsets) for i in range(num_subsets)]
    leftover = n - sum(caps)
    # Round-robin add leftover to subsets with headroom
    while leftover > 0:
        progress = False
        for i in range(num_subsets):
            if caps[i] < sizes[i]:
                caps[i] += 1
                leftover -= 1
                progress = True
                if leftover == 0:
                    break
        if not progress:
            break
    return caps
