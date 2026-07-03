from typing import Dict

from ..registry import register
from ..scoring import score_choices, subsample, distribute_n, fewshot_block_from_pool

LABELS = [
    "aunt", "uncle", "father", "mother", "brother", "sister",
    "son", "daughter", "grandmother", "grandfather",
    "granddaughter", "grandson", "niece", "nephew",
    "husband", "wife", "father-in-law", "mother-in-law",
    "son-in-law", "daughter-in-law", "sister-in-law",
]
DEFAULT_N = 1000


def _load_clutrr():
    from datasets import load_dataset
    for repo in ("kendrivp/CLUTRR_v1_extracted", "kliang5/CLUTRR_huggingface_dataset"):
        try:
            return load_dataset(repo, split="test")
        except Exception:
            continue
    return None


def _extract_k(ex) -> int:
    t = ex.get("task_name") or ex.get("task", "")
    try:
        return int(t.split(".")[-1])
    except Exception:
        return 0


def _format_prompt(ex) -> str:
    story = ex.get("clean_story") or ex.get("story", "")
    q = ex.get("query")
    if isinstance(q, (list, tuple)):
        a, b = q[0], q[1]
    elif isinstance(q, str) and "(" in q:
        import ast
        try:
            t = ast.literal_eval(q)
            a, b = t[0], t[1]
        except Exception:
            a, b = "?", "?"
    else:
        a, b = "?", "?"
    return (f"{story}\n"
            f"Question: How is {b} related to {a}?\n"
            f"Answer:")


def _extract_gold(ex) -> str:
    gold = str(ex.get("target_text") or ex.get("target") or ex.get("answer") or "").strip().lower()
    return gold if gold in LABELS else None


@register("clutrr")
def eval_clutrr(model, tokenizer, use_chat: bool, n: int = None) -> Dict:
    n = n if n is not None else DEFAULT_N

    ds = _load_clutrr()
    if ds is None:
        return {"error": "CLUTRR load failed"}

    by_k = {}
    for ex in ds:
        by_k.setdefault(_extract_k(ex), []).append(ex)

    ks = [k for k in sorted(by_k.keys()) if k > 0]

    all_pool = [ex for k in ks for ex in by_k[k]]
    fs_block, demo_ids = fewshot_block_from_pool(
        all_pool, k=5, prompt_fn=_format_prompt, gold_fn=_extract_gold)

    sizes = [sum(1 for ex in by_k[k] if id(ex) not in demo_ids) for k in ks]
    caps = distribute_n(n, len(ks), sizes)

    print(f"  [clutrr] Loaded {len(ds)} examples across k={ks}, 5-shot demos built")

    results = {}
    for k, cap in zip(ks, caps):
        pool = [ex for ex in by_k[k] if id(ex) not in demo_ids]
        correct = 0
        total = 0

        for ex in subsample(pool, cap):
            gold = _extract_gold(ex)
            if gold is None:
                continue
            prompt = fs_block + _format_prompt(ex)
            out = score_choices(model, tokenizer, prompt, LABELS, use_chat)

            if out["prediction"] == gold:
                correct += 1
            total += 1

        acc = correct / total if total > 0 else 0.0
        results[f"k={k}"] = {"accuracy": acc, "correct": correct, "total": total}
        print(f"  [clutrr] k={k}: {correct}/{total} = {acc:.1%}")

    total_correct = sum(r["correct"] for r in results.values())
    total_count = sum(r["total"] for r in results.values())
    results["overall"] = {"accuracy": total_correct / total_count if total_count > 0 else 0.0,
                          "correct": total_correct, "total": total_count}
    print(f"  [clutrr] overall: {total_correct}/{total_count} = {results['overall']['accuracy']:.1%}")
    return results
