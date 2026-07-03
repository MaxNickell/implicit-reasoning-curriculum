from typing import Dict

from ..registry import register
from ..scoring import score_choices, subsample

LABELS = ["True", "False", "Uncertain"]
NUM_SHOTS = 8
DEFAULT_N = 1000

_FEW_SHOT_CACHE = None


def _load_folio(split: str = "validation"):
    from datasets import load_dataset
    for repo in ("yale-nlp/FOLIO", "tasksource/folio"):
        try:
            return load_dataset(repo, split=split)
        except Exception:
            continue
    return None


def _format_example(ex) -> str:
    prem = ex.get("premises", [])
    if isinstance(prem, list):
        prem_text = "\n".join(prem)
    else:
        prem_text = str(prem)
    return f"Premises:\n{prem_text}\nConclusion: {ex.get('conclusion', '')}\nAnswer:"


def _build_few_shot_block() -> str:
    global _FEW_SHOT_CACHE
    if _FEW_SHOT_CACHE is not None:
        return _FEW_SHOT_CACHE

    ds = _load_folio("train")
    if ds is None:
        _FEW_SHOT_CACHE = ""
        return ""

    demos = []
    seen = set()
    for ex in ds:
        lbl = str(ex.get("label", "")).strip().capitalize()
        lbl = lbl.replace("Unknown", "Uncertain")
        if lbl not in LABELS:
            continue
        if lbl in seen and len(seen) < 3:
            continue
        seen.add(lbl)

        prem = ex.get("premises", [])
        if isinstance(prem, list):
            prem_text = "\n".join(prem)
        else:
            prem_text = str(prem)
        conc = ex.get("conclusion", "")
        demos.append(f"Premises:\n{prem_text}\nConclusion: {conc}\nAnswer: {lbl}")
        if len(demos) >= NUM_SHOTS:
            break

    _FEW_SHOT_CACHE = "\n\n".join(demos) + "\n\n" if demos else ""
    return _FEW_SHOT_CACHE


@register("folio")
def eval_folio(model, tokenizer, use_chat: bool, n: int = None) -> Dict:
    n = n if n is not None else DEFAULT_N

    ds = _load_folio("validation")
    if ds is None:
        return {"error": "FOLIO load failed (try `huggingface-cli login`)"}

    print(f"  [folio] Loaded {len(ds)} validation examples")

    few_shot_block = _build_few_shot_block()
    print(f"  [folio] Few-shot block: {len(few_shot_block)} chars, {NUM_SHOTS} demos")

    correct = 0
    total = 0
    per_label = {lbl: [0, 0] for lbl in LABELS}

    for ex in subsample(list(ds), n):
        gold_raw = str(ex.get("label", "")).strip().capitalize()
        gold = gold_raw.replace("Unknown", "Uncertain")
        if gold not in LABELS:
            continue

        prompt = few_shot_block + _format_example(ex)
        out = score_choices(model, tokenizer, prompt, LABELS, use_chat)
        pred = out["prediction"]

        per_label[gold][1] += 1
        if pred == gold:
            correct += 1
            per_label[gold][0] += 1
        total += 1

    acc = correct / total if total > 0 else 0.0
    print(f"  [folio] overall: {correct}/{total} = {acc:.1%}")

    return {
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "per_label": {
            lbl: {"correct": c[0], "total": c[1], "accuracy": c[0] / c[1] if c[1] > 0 else 0.0}
            for lbl, c in per_label.items()
        },
    }
