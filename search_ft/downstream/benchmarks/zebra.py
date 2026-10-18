"""ZebraLogic-MC (Lin et al. 2025), the multiple-choice mode of the ZebraLogic grid puzzles
(HF WildEval/ZebraLogic, config mc_mode, split test). 3-shot, log-likelihood over each
puzzle's own choice set, stratified by grid size (25 sizes from 2x2 to 6x6).

The three demos come from the smallest puzzles (2x2 and 2x3) with distinct answers, so
the prompt stays short; the puzzles actually picked are removed from the scored pool.
The content-free baseline is recomputed per item because the choice set is per item.
"""
import random
from typing import Dict

from ..registry import register, register_prepare
from ..scoring import eval_strata, subsample, distribute_n

DEFAULT_N = 1000
K_SHOT = 3


def load():
    from datasets import load_dataset
    return list(load_dataset("WildEval/ZebraLogic", "mc_mode", split="test"))


def extract_size(ex_id: str) -> str:
    try:
        return ex_id.split("-")[2].replace("x", "*")
    except Exception:
        return "?"


def fmt(ex) -> str:
    return f"{ex['puzzle']}\n\nQuestion: {ex['question']}\nAnswer:"


@register_prepare("zebra_mc")
def prepare(n=None) -> Dict:
    n = DEFAULT_N if n is None else n
    rows = load()
    print(f"  [zebra_mc] loaded {len(rows)} questions")
    by_size = {}
    for ex in rows:
        by_size.setdefault(extract_size(ex["id"]), []).append(ex)

    demo_pool = (by_size.get("2*2", []) or []) + (by_size.get("2*3", []) or [])
    rng = random.Random(1234)
    rng.shuffle(demo_pool)
    demos, seen_answers, picked = [], [], []
    for ex in demo_pool:
        a = str(ex["answer"])
        if a in seen_answers and len(demos) + 1 < K_SHOT:
            continue
        seen_answers.append(a)
        demos.append(f"{fmt(ex)} {a}")
        picked.append(ex)
        if len(demos) >= K_SHOT:
            break
    fs_block = ("\n\n".join(demos) + "\n\n") if demos else ""
    demo_ids = {id(ex) for ex in picked}   # the puzzles actually picked, not demo_pool[:3]

    sizes = sorted(by_size.keys(),
                   key=lambda s: tuple(int(x) for x in s.split("*")) if "*" in s else (99, 99))
    counts = [sum(1 for ex in by_size[s] if id(ex) not in demo_ids) for s in sizes]
    caps = distribute_n(n, len(sizes), counts)
    strata = [(s, subsample([ex for ex in by_size[s] if id(ex) not in demo_ids], c))
              for s, c in zip(sizes, caps)]
    print(f"  [zebra_mc] {K_SHOT}-shot demos {seen_answers}; scored "
          + " ".join(f"{s}={len(v)}" for s, v in strata))
    return dict(strata=strata, labels=lambda ex: [str(c) for c in ex["choices"]], fmt=fmt,
                gold_of=lambda ex: str(ex["answer"]), item_id=lambda ex: ex["id"],
                demos_block=fs_block, cf_prompt=fs_block + "N/A\n\nQuestion: N/A\nAnswer:")


@register("zebra_mc")
def eval_zebra_mc(model, tokenizer, use_chat: bool, n: int = None) -> Dict:
    return eval_strata(model, tokenizer, use_chat, "zebra_mc", **prepare(n))
