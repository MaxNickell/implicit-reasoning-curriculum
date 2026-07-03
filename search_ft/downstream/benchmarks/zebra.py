import random as _random
from typing import Dict

from ..registry import register
from ..scoring import score_choices, subsample, distribute_n

DEFAULT_N = 1000


def _load_zebra():
    from datasets import load_dataset
    try:
        return load_dataset("WildEval/ZebraLogic", "mc_mode", split="test")
    except Exception as e:
        print(f"  [zebra_mc] load failed: {e}")
        return None


def _extract_size(ex_id: str) -> str:
    try:
        return ex_id.split("-")[2].replace("x", "*")
    except Exception:
        return "?"


def _format_prompt(ex) -> str:
    return f"{ex['puzzle']}\n\nQuestion: {ex['question']}\nAnswer:"


def _build_few_shot_block(by_size, k=1) -> tuple:
    demo_pool = (by_size.get("2*2", []) or []) + (by_size.get("2*3", []) or [])
    rng = _random.Random(1234)
    rng.shuffle(demo_pool)

    demos = []
    for ex in demo_pool:
        a = str(ex["answer"])
        demos.append(f"{_format_prompt(ex)} {a}")
        if len(demos) >= k:
            break

    fs_block = "\n\n".join(demos) + "\n\n" if demos else ""
    demo_ids = {id(ex) for ex in demo_pool[:k]}
    return fs_block, demo_ids


@register("zebra_mc")
def eval_zebra_mc(model, tokenizer, use_chat: bool, n: int = None) -> Dict:
    n = n if n is not None else DEFAULT_N

    ds = _load_zebra()
    if ds is None:
        return {"error": "ZebraLogic load failed"}

    by_size = {}
    for ex in ds:
        by_size.setdefault(_extract_size(ex["id"]), []).append(ex)

    fs_block, demo_ids = _build_few_shot_block(by_size, k=1)

    sizes = sorted(
        by_size.keys(),
        key=lambda s: tuple(int(x) for x in s.split("*")) if "*" in s else (99, 99)
    )
    size_counts = [sum(1 for ex in by_size[s] if id(ex) not in demo_ids) for s in sizes]
    caps = distribute_n(n, len(sizes), size_counts)

    print(f"  [zebra_mc] Loaded {len(ds)} questions, 1-shot demo built")

    results = {}
    for size, cap in zip(sizes, caps):
        pool = [ex for ex in by_size[size] if id(ex) not in demo_ids]
        correct = 0
        total = 0

        for ex in subsample(pool, cap):
            prompt = fs_block + _format_prompt(ex)
            choices = [str(c) for c in ex["choices"]]
            gold = ex["answer"]

            out = score_choices(model, tokenizer, prompt, choices, use_chat)

            if out["prediction"] == gold:
                correct += 1
            total += 1

        acc = correct / total if total > 0 else 0.0
        results[size] = {"accuracy": acc, "correct": correct, "total": total}
        print(f"  [zebra_mc] size={size}: {correct}/{total} = {acc:.1%}")

    total_correct = sum(r["correct"] for r in results.values())
    total_count = sum(r["total"] for r in results.values())
    results["overall"] = {"accuracy": total_correct / total_count if total_count > 0 else 0.0,
                          "correct": total_correct, "total": total_count}
    print(f"  [zebra_mc] overall: {total_correct}/{total_count} = {results['overall']['accuracy']:.1%}")
    return results
