from typing import Dict

from ..registry import register
from ..scoring import score_choices, subsample, distribute_n

LABELS = [
    "left", "right", "above", "below",
    "upper-left", "upper-right", "lower-left", "lower-right",
    "overlap",
]
DEFAULT_N = 1000


def _load_stepgame(split="test"):
    from datasets import load_dataset
    try:
        return load_dataset("ZhengyanShi/StepGame", split=split)
    except Exception as e:
        print(f"  [stepgame] load failed: {e}")
        return None


def _format_prompt(ex) -> str:
    return (f"{ex['story']}\n\n"
            f"Question: {ex['question']}\n"
            f"Answer:")


def _build_few_shot_block(k=5) -> str:
    ds = _load_stepgame("train")
    if ds is None:
        return ""

    examples = []
    seen_labels = set()
    for ex in ds:
        lbl = ex.get("label", "").strip().lower()
        if lbl in seen_labels or lbl not in LABELS:
            continue
        seen_labels.add(lbl)
        examples.append(ex)
        if len(examples) >= k:
            break

    blocks = [f"{_format_prompt(ex)} {ex['label'].strip().lower()}" for ex in examples]
    return "\n\n".join(blocks) + "\n\n" if blocks else ""


@register("stepgame")
def eval_stepgame(model, tokenizer, use_chat: bool, n: int = None) -> Dict:
    n = n if n is not None else DEFAULT_N

    ds = _load_stepgame("test")
    if ds is None:
        return {"error": "StepGame load failed"}

    by_hops = {}
    for ex in ds:
        cfg = ex.get("config", "qa?")
        by_hops.setdefault(cfg, []).append(ex)

    print(f"  [stepgame] Loaded {len(ds)} examples")

    fs_block = _build_few_shot_block(k=5)

    def _hops(c):
        try:
            return int(c.replace("qa", ""))
        except Exception:
            return 99

    cfgs = sorted(by_hops.keys(), key=_hops)
    sizes = [len(by_hops[c]) for c in cfgs]
    caps = distribute_n(n, len(cfgs), sizes)

    results = {}
    for cfg, cap in zip(cfgs, caps):
        group = subsample(by_hops[cfg], cap)
        correct = 0
        total = 0

        for ex in group:
            prompt = fs_block + _format_prompt(ex)
            out = score_choices(model, tokenizer, prompt, LABELS, use_chat)
            gold = ex.get("label", "").strip().lower()

            if out["prediction"] == gold:
                correct += 1
            total += 1

        acc = correct / total if total > 0 else 0.0
        results[cfg] = {"accuracy": acc, "correct": correct, "total": total}
        print(f"  [stepgame] {cfg}: {correct}/{total} = {acc:.1%}")

    total_correct = sum(r["correct"] for r in results.values())
    total_count = sum(r["total"] for r in results.values())
    results["overall"] = {"accuracy": total_correct / total_count if total_count > 0 else 0.0,
                          "correct": total_correct, "total": total_count}
    print(f"  [stepgame] overall: {total_correct}/{total_count} = {results['overall']['accuracy']:.1%}")
    return results
