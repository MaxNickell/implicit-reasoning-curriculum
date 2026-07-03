import hashlib
import json
import os
import random
from typing import Dict, List, Any, Optional, Set, Tuple

from .nl_generator import NaturalLanguageGraphGenerator


def build_heldout_set(
    *,
    size: int,
    max_input_size: int,
    max_lookahead: int,
    seed: int = 12345,
    reserved_inputs: Optional[Set[str]] = None,
    max_attempts_per_example: int = 2000,
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    avoid = reserved_inputs or set()

    heldout: List[Dict[str, Any]] = []
    heldout_inputs: Set[str] = set()

    alpha = 1.0
    stage = max_lookahead

    i = 0
    while len(heldout) < size:
        base_seed = seed + i * 1_000_003

        for attempt in range(max_attempts_per_example):
            sample_seed = (base_seed + attempt * 104729) & 0x7FFFFFFF

            gen = NaturalLanguageGraphGenerator(max_input_size, seed=sample_seed)
            batch = gen.generate_batch(
                batch_size=1,
                reserved_inputs=avoid,
                alpha=alpha,
                max_lookahead=max_lookahead,
            )

            if not batch or not batch[0] or not batch[0].output_texts:
                continue

            ex = batch[0]

            if ex.input_text in avoid:
                continue
            if ex.input_text in heldout_inputs:
                continue

            row = {
                "input_text": ex.input_text,
                "output_texts": list(ex.output_texts),
                "alpha": alpha,
                "stage": stage,
                "seed": int(sample_seed),
            }
            heldout.append(row)
            heldout_inputs.add(ex.input_text)
            break
        else:
            raise RuntimeError(
                f"[build_heldout_set] Failed to fill heldout slot {len(heldout)+1}/{size} "
                f"after {max_attempts_per_example} attempts."
            )

        i += 1

    return heldout, heldout_inputs


def generate_eval_at_alpha(
    *,
    n_samples: int,
    max_input_size: int,
    max_lookahead: int,
    alpha: float,
    tokenizer,
    reserved_inputs: Optional[Set[str]] = None,
    seed: Optional[int] = None,
) -> Tuple[List[str], List[List[str]]]:
    reserved = reserved_inputs or set()
    g = NaturalLanguageGraphGenerator(max_input_size, seed=seed)
    eval_inputs: List[str] = []
    eval_labels: List[List[str]] = []

    max_len = getattr(tokenizer, "model_max_length", 2048)
    rng = random.Random((seed or 0) + 424242)

    attempts = 0
    while len(eval_inputs) < n_samples and attempts < n_samples * 10:
        attempts += 1
        batch = g.generate_batch(
            batch_size=1,
            reserved_inputs=reserved,
            alpha=alpha,
            max_lookahead=max_lookahead,
        )
        if not (batch and batch[0] and batch[0].output_texts):
            continue
        ex = batch[0]
        if ex.input_text in reserved:
            continue

        # Check tokenized length fits
        prompt_ids = tokenizer(ex.input_text, add_special_tokens=True, truncation=False)["input_ids"]
        chosen = rng.choice(ex.output_texts)
        ans_ids = tokenizer(" " + chosen, add_special_tokens=False)["input_ids"]
        end_ids = tokenizer(". ", add_special_tokens=False)["input_ids"]
        full_len = len(prompt_ids) + len(ans_ids) + len(end_ids)
        if full_len > max_len:
            continue

        eval_inputs.append(ex.input_text)
        eval_labels.append(list(ex.output_texts))

    return eval_inputs, eval_labels


def eval_data_fingerprint(inputs: List[str], labels: List[List[str]]) -> str:
    if not inputs:
        return "empty"
    content = f"{len(inputs)}|{inputs[0][:100]}|{inputs[-1][:100]}|{len(labels)}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def save_heldout(
    path: str,
    examples: List[Dict[str, Any]],
    *,
    max_lookahead: int,
    max_input_size: int,
    seed: int,
) -> None:
    data = {
        "meta": {
            "max_lookahead": max_lookahead,
            "max_input_size": max_input_size,
            "seed": seed,
            "samples": len(examples),
        },
        "examples": examples,
    }

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_heldout(path: str) -> Tuple[List[Dict[str, Any]], Set[str]]:
    with open(path) as f:
        data = json.load(f)

    examples = data["examples"]
    reserved = {ex["input_text"] for ex in examples}

    meta = data.get("meta", {})
    print(f"  [heldout] Loaded {len(examples)} examples from {path}")
    print(f"  [heldout] max_lookahead={meta.get('max_lookahead')}, "
          f"seed={meta.get('seed')}")

    return examples, reserved
