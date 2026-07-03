from typing import Dict

from ..registry import register
from ..scoring import score_choices, subsample

LABELS = ["A", "B", "C", "D"]
DEFAULT_N = 1000


def _load_logiqa():
    from datasets import load_dataset
    for repo in ("hails/agieval-logiqa-en",):
        try:
            return load_dataset(repo, split="test")
        except Exception:
            continue
    return None


def _format_prompt(ex) -> str:
    q = ex.get("query")
    if q and "the answer is" in q.lower():
        return q

    ctx = ex.get("context") or ex.get("passage", "")
    qtext = q or ex.get("question", "")
    opts = ex.get("options") or ex.get("choices", [])

    if isinstance(opts, str):
        try:
            import ast
            opts = ast.literal_eval(opts)
        except Exception:
            opts = []
    if isinstance(opts, dict):
        opts = [opts.get(k, "") for k in sorted(opts.keys())]
    while len(opts) < 4:
        opts.append("")

    return (f"Passage: {ctx}\nQuestion: {qtext}\nChoices:\n"
            f"A. {opts[0]}\nB. {opts[1]}\nC. {opts[2]}\nD. {opts[3]}\nAnswer:")


def _extract_gold(ex) -> str:
    g = ex.get("gold")
    if g is None:
        g = ex.get("correct_option")
    if g is None:
        g = ex.get("label")
    if g is None:
        g = ex.get("answer")

    if isinstance(g, str):
        try:
            import ast
            g = ast.literal_eval(g)
        except Exception:
            pass
    if isinstance(g, list) and g:
        g = g[0]
    if isinstance(g, int):
        return LABELS[g] if 0 <= g < 4 else None
    if isinstance(g, str):
        g = g.strip().upper()
        if g in LABELS:
            return g
        if g.isdigit() and int(g) < 4:
            return LABELS[int(g)]
    return None


@register("logiqa")
def eval_logiqa(model, tokenizer, use_chat: bool, n: int = None) -> Dict:
    n = n if n is not None else DEFAULT_N

    ds = _load_logiqa()
    if ds is None:
        return {"error": "LogiQA load failed"}

    print(f"  [logiqa] Loaded {len(ds)} test examples")

    correct = 0
    total = 0

    for ex in subsample(list(ds), n):
        gold = _extract_gold(ex)
        if gold is None:
            continue

        prompt = _format_prompt(ex)
        out = score_choices(model, tokenizer, prompt, LABELS, use_chat)
        pred = out["prediction"]

        if pred == gold:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0.0
    print(f"  [logiqa] overall: {correct}/{total} = {acc:.1%}")

    return {
        "accuracy": acc,
        "correct": correct,
        "total": total,
    }
