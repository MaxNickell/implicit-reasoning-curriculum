"""FOLIO (Han et al. 2024): first-order-logic deduction over natural-language premises,
True / False / Uncertain. 8-shot with demos from the train split (the paper's protocol),
log-likelihood over the three labels, the whole validation split (203 items).

Dataset: HF yale-nlp/FOLIO (gated; accept the licence and log in to the Hub once), with
tasksource/folio as a fallback mirror.
"""
from typing import Dict

from ..registry import register, register_prepare
from ..scoring import eval_strata, subsample, item_hash

LABELS = ["True", "False", "Uncertain"]
K_SHOT = 8
DEFAULT_N = None   # the whole validation split


def load(split: str = "validation"):
    from datasets import load_dataset
    for repo in ("yale-nlp/FOLIO", "tasksource/folio"):
        try:
            return load_dataset(repo, split=split)
        except Exception:
            continue
    return None


def _premises(ex) -> str:
    prem = ex.get("premises", [])
    return "\n".join(prem) if isinstance(prem, list) else str(prem)


def fmt(ex) -> str:
    return f"Premises:\n{_premises(ex)}\nConclusion: {ex.get('conclusion', '')}\nAnswer:"


def gold_of(ex):
    g = str(ex.get("label", "")).strip().capitalize().replace("Unknown", "Uncertain")
    return g if g in LABELS else None


def item_id(ex):
    if ex.get("example_id") is not None:
        return ex["example_id"]
    return item_hash(ex.get("premises", ""), ex.get("conclusion", ""))


_FS_CACHE = None


def few_shot_block() -> str:
    """K demos from the train split, the first example of each label first, then in order."""
    global _FS_CACHE
    if _FS_CACHE is not None:
        return _FS_CACHE
    ds = load("train")
    if ds is None:
        print("  [folio] could not load the train split for demos; running 0-shot")
        _FS_CACHE = ""
        return _FS_CACHE
    demos, seen = [], set()
    for ex in ds:
        lbl = gold_of(ex)
        if lbl is None or lbl in seen and len(seen) < 3:
            continue
        seen.add(lbl)
        demos.append(f"Premises:\n{_premises(ex)}\nConclusion: {ex.get('conclusion', '')}\nAnswer: {lbl}")
        if len(demos) >= K_SHOT:
            break
    _FS_CACHE = ("\n\n".join(demos) + "\n\n") if demos else ""
    return _FS_CACHE


@register_prepare("folio")
def prepare(n=None) -> Dict:
    n = DEFAULT_N if n is None else n
    ds = load("validation")
    if ds is None:
        raise RuntimeError("FOLIO load failed (gated dataset: accept the licence on the Hub and log in)")
    print(f"  [folio] loaded {len(ds)} validation examples")
    fs = few_shot_block()
    items = subsample(list(ds), n)
    return dict(strata=[("overall", items)], labels=LABELS, fmt=fmt, gold_of=gold_of,
                item_id=item_id, demos_block=fs,
                cf_prompt=fs + fmt({"premises": ["N/A"], "conclusion": "N/A"}))


@register("folio")
def eval_folio(model, tokenizer, use_chat: bool, n: int = None) -> Dict:
    return eval_strata(model, tokenizer, use_chat, "folio", **prepare(n))
