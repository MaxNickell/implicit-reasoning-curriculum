from typing import Dict

from ..registry import register
from ..scoring import score_choices, subsample, distribute_n, fewshot_block_from_pool

LABELS = ["True", "False"]
DEFAULT_N = 1000


def _load_by_depth():
    from datasets import load_dataset

    by_depth = {}
    try:
        ds = load_dataset("tasksource/ruletaker", split="test")
    except Exception as e:
        try:
            ds = load_dataset("hitachi-nlp/ruletaker", split="test")
        except Exception as e2:
            print(f"  [ruletaker] load failed: {e}; {e2}")
            return None

    cfg_field = "config" if "config" in ds.column_names else None
    for ex in ds:
        d = ex.get(cfg_field, "depth-?") if cfg_field else "depth-?"
        by_depth.setdefault(str(d), []).append(ex)
    return by_depth


def _format_prompt(ex) -> str:
    return (f"Rules and facts:\n{ex.get('context', '')}\n\n"
            f"Statement: {ex.get('question', '')}\n"
            f"Based only on the rules and facts above, is the statement true or false?\n"
            f"Answer:")


def _extract_gold(ex) -> str:
    lbl = str(ex.get("label", "")).strip().lower()
    if lbl in ("true", "entailment", "1"):
        return "True"
    if lbl in ("false", "not entailment", "not_entailment", "0"):
        return "False"
    return None


@register("ruletaker")
def eval_ruletaker(model, tokenizer, use_chat: bool, n: int = None) -> Dict:
    n = n if n is not None else DEFAULT_N

    by_depth = _load_by_depth()
    if by_depth is None:
        return {"error": "RuleTaker load failed"}

    cfgs = sorted(by_depth.keys())

    # 5-shot demos from the pool
    all_pool = [ex for cfg in cfgs for ex in by_depth[cfg]]
    fs_block, demo_ids = fewshot_block_from_pool(
        all_pool, k=5, prompt_fn=_format_prompt, gold_fn=_extract_gold)

    sizes = [sum(1 for ex in by_depth[c] if id(ex) not in demo_ids) for c in cfgs]
    caps = distribute_n(n, len(cfgs), sizes)

    print(f"  [ruletaker] Loaded depths: {cfgs}, 5-shot demos built")

    results = {}
    for cfg, cap in zip(cfgs, caps):
        pool = [ex for ex in by_depth[cfg] if id(ex) not in demo_ids]
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
        results[cfg] = {"accuracy": acc, "correct": correct, "total": total}
        print(f"  [ruletaker] {cfg}: {correct}/{total} = {acc:.1%}")

    total_correct = sum(r["correct"] for r in results.values())
    total_count = sum(r["total"] for r in results.values())
    results["overall"] = {"accuracy": total_correct / total_count if total_count > 0 else 0.0,
                          "correct": total_correct, "total": total_count}
    print(f"  [ruletaker] overall: {total_correct}/{total_count} = {results['overall']['accuracy']:.1%}")
    return results
