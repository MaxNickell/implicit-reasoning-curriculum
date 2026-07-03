import os
import json
import glob
from typing import Dict, List, Optional, Tuple

from ..registry import register
from ..scoring import score_choices, subsample, distribute_n, fewshot_block_from_pool

LABELS = ["Yes", "No"]
DEFAULT_N = 1000


def _get_data_dir():
    return os.environ.get("DOWNSTREAM_DATA_DIR")


def _ensure_data() -> Optional[str]:
    import subprocess
    data_dir = _get_data_dir()
    if data_dir is None:
        return None

    base = os.path.join(data_dir, "multilogieval", "Multi-LogiEval")
    if os.path.isdir(os.path.join(base, "data")):
        return base

    parent = os.path.dirname(base)
    os.makedirs(parent, exist_ok=True)
    try:
        subprocess.check_call(
            ["git", "clone", "--depth=1",
             "https://github.com/Mihir3009/Multi-LogiEval.git", base],
            cwd=parent, timeout=120
        )
        return base
    except Exception as e:
        print(f"  [multilogieval] clone failed: {e}")
        return None


def _load_all() -> Optional[List[Tuple[str, str, str, List[Dict]]]]:
    base = _ensure_data()
    if base is None:
        return None

    out = []
    for ddir in sorted(glob.glob(os.path.join(base, "data", "d*_Data"))):
        depth = os.path.basename(ddir).split("_")[0]
        for logic_dir in sorted(os.listdir(ddir)):
            ldir = os.path.join(ddir, logic_dir)
            if not os.path.isdir(ldir):
                continue
            for fpath in sorted(glob.glob(os.path.join(ldir, "*.json"))):
                rule = os.path.splitext(os.path.basename(fpath))[0]
                try:
                    with open(fpath) as f:
                        items = json.load(f)
                    if isinstance(items, dict):
                        items = items.get("samples", []) or list(items.values())
                    out.append((depth, logic_dir, rule, items))
                except Exception:
                    pass

    return out if out else None


def _format_prompt(ex: Dict) -> str:
    ctx = ex.get("context", "")
    q = ex.get("question", "")
    return (f"Given the context and question, answer ONLY in 'yes' or 'no'.\n\n"
            f"Context: {ctx}\nQuestion: {q}\nAnswer:")


def _gold_fn(ex):
    g = str(ex.get("answer", "")).strip().lower()
    return g.capitalize() if g in ("yes", "no") else None


@register("multilogieval")
def eval_multilogieval(model, tokenizer, use_chat: bool, n: int = None) -> Dict:
    n = n if n is not None else DEFAULT_N

    data = _load_all()
    if data is None:
        return {"error": "Multi-LogiEval data not found."}

    bucket = {}
    for depth, logic, rule, items in data:
        bucket.setdefault((depth, logic), []).extend(items)

    keys = sorted(bucket.keys())

    # 3-shot demos balanced Yes/No
    all_pool = [ex for items in bucket.values() for ex in items]
    fs_block, demo_ids = fewshot_block_from_pool(
        all_pool, k=3, prompt_fn=_format_prompt, gold_fn=_gold_fn)

    sizes = [sum(1 for ex in bucket[k] if id(ex) not in demo_ids) for k in keys]
    caps = distribute_n(n, len(keys), sizes)

    print(f"  [multilogieval] Loaded {len(keys)} (depth, logic) buckets, 3-shot demos built")

    results = {}
    for (depth, logic), cap in zip(keys, caps):
        pool = [ex for ex in bucket[(depth, logic)] if id(ex) not in demo_ids]
        correct = 0
        total = 0

        for ex in subsample(pool, cap):
            gold = str(ex.get("answer", "")).strip().lower()
            if gold not in ("yes", "no"):
                continue
            prompt = fs_block + _format_prompt(ex)
            out = score_choices(model, tokenizer, prompt, LABELS, use_chat)
            pred = out["prediction"].lower()

            if pred == gold:
                correct += 1
            total += 1

        if total > 0:
            results[f"{depth}_{logic}"] = {"accuracy": correct / total, "correct": correct, "total": total}

    total_correct = sum(r["correct"] for r in results.values())
    total_count = sum(r["total"] for r in results.values())
    results["overall"] = {"accuracy": total_correct / total_count if total_count > 0 else 0.0,
                          "correct": total_correct, "total": total_count}
    print(f"  [multilogieval] overall: {total_correct}/{total_count} = {results['overall']['accuracy']:.1%}")
    return results
