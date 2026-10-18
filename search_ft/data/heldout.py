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
    vocab_pool: str = "none",
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
                vocab_pool=vocab_pool,
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
    vocab_pool: str = "none",
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
            vocab_pool=vocab_pool,
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
    vocab_pool: str = "none",
) -> None:
    data = {
        "meta": {
            "max_lookahead": max_lookahead,
            "max_input_size": max_input_size,
            "seed": seed,
            "samples": len(examples),
            "vocab_pool": vocab_pool,
        },
        "examples": examples,
    }

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_heldout(
    path: str,
    *,
    expect_max_input_size: Optional[int] = None,
    expect_vocab_pool: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """Load a held-out file. The file is named by max_lookahead only, so a training run passes the context size
    and vocab_pool it was generated with and a mismatch is an error (a held-out set at the wrong context or with
    the wrong naming would silently score a different task)."""
    with open(path) as f:
        data = json.load(f)

    examples = data["examples"]
    reserved = {ex["input_text"] for ex in examples}

    meta = data.get("meta", {})
    print(f"  [heldout] Loaded {len(examples)} examples from {path}")
    print(f"  [heldout] max_lookahead={meta.get('max_lookahead')}, max_input_size={meta.get('max_input_size')}, "
          f"vocab_pool={meta.get('vocab_pool', 'none')}, seed={meta.get('seed')}")

    if expect_max_input_size is not None and meta.get("max_input_size") not in (None, expect_max_input_size):
        raise ValueError(f"{path} was generated at max_input_size={meta.get('max_input_size')}, the run uses "
                         f"{expect_max_input_size}; regenerate it with --max_input_size {expect_max_input_size}")
    if expect_vocab_pool is not None and meta.get("vocab_pool", "none") != expect_vocab_pool:
        raise ValueError(f"{path} was generated with vocab_pool={meta.get('vocab_pool', 'none')}, the run uses "
                         f"{expect_vocab_pool}; regenerate it with --vocab_pool {expect_vocab_pool}")

    return examples, reserved
