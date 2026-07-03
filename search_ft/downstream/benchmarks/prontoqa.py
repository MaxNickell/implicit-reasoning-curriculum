import os
import json
import glob
import urllib.request
import zipfile
from typing import Dict, List, Optional, Tuple

from ..registry import register
from ..scoring import score_choices, subsample, distribute_n

LABELS = ["True", "False"]
DEFAULT_N = 1000
def _get_data_dir():
    return os.environ.get("DOWNSTREAM_DATA_DIR")


def _ensure_data() -> Optional[str]:
    if _get_data_dir() is None:
        return None
    base = os.path.join(_get_data_dir(), "prontoqa_ood")
    if os.path.isdir(base) and any(f.endswith(".json") for f in os.listdir(base)):
        return base

    os.makedirs(base, exist_ok=True)
    zip_path = os.path.join(base, "generated_ood_data.zip")
    url = "https://github.com/asaparov/prontoqa/raw/main/generated_ood_data.zip"
    try:
        print(f"  [prontoqa_ood] downloading {url} (~5MB)")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(base)
        os.remove(zip_path)
        return base
    except Exception as e:
        print(f"  [prontoqa_ood] download failed: {e}")
        return None


def _load_all() -> Optional[Dict[str, Dict]]:
    base = _ensure_data()
    if base is None:
        return None

    out = {}
    for fpath in sorted(glob.glob(os.path.join(base, "*.json"))):
        name = os.path.splitext(os.path.basename(fpath))[0]
        try:
            with open(fpath) as f:
                out[name] = json.load(f)
        except Exception:
            pass
    return out if out else None


def _derive_gold_label(query: str, chain: List[str]) -> str:
    if not chain or not query:
        return "False"

    target = query.strip()
    for prefix in ("Prove:", "prove:"):
        if target.startswith(prefix):
            target = target[len(prefix):].strip()
            break
    target = target.rstrip(".").strip().lower()
    last = chain[-1].strip().rstrip(".").strip().lower()
    return "True" if last == target else "False"


def _iter_examples(d: Dict) -> List[Tuple[str, str]]:
    examples = []
    for ex_key in d:
        ex = d[ex_key]

        in_context = []
        for k in sorted(ex.keys()):
            if k.startswith("in_context_example"):
                ic = ex[k]
                ic_label = _derive_gold_label(
                    ic.get("query", ""),
                    ic.get("chain_of_thought", [])
                )
                in_context.append(
                    f"Q: {ic['question']}\n{ic.get('query', '')}\n"
                    f"A: {ic_label}"
                )

        test = ex.get("test_example", {})
        prompt = "\n\n".join(in_context) + "\n\n" + \
                 f"Q: {test.get('question', '')}\n{test.get('query', '')}\nA:"
        gold_chain = test.get("chain_of_thought", [])
        gold_label = _derive_gold_label(test.get("query", ""), gold_chain)
        examples.append((prompt, gold_label))

    return examples


@register("prontoqa_ood")
def eval_prontoqa_ood(model, tokenizer, use_chat: bool, n: int = None) -> Dict:
    n = n if n is not None else DEFAULT_N

    variants = _load_all()
    if variants is None:
        return {"error": "PrOntoQA-OOD data not found. Pass --data_dir to the downstream runner."}

    vnames = sorted(variants.keys())
    sizes = [100] * len(vnames)  # approximate ~100 per variant
    caps = distribute_n(n, len(vnames), sizes)

    print(f"  [prontoqa_ood] Loaded {len(vnames)} variants, ~{caps[0]} samples/variant")

    results = {}
    for vname, cap in zip(vnames, caps):
        examples = _iter_examples(variants[vname])
        sampled = subsample(examples, cap)

        correct = 0
        total = 0

        for prompt, gold_label in sampled:
            out = score_choices(model, tokenizer, prompt, LABELS, use_chat)
            pred = out["prediction"]

            if pred == gold_label:
                correct += 1
            total += 1

        if total > 0:
            acc = correct / total
            results[vname] = {
                "accuracy": acc,
                "correct": correct,
                "total": total,
            }

    total_correct = sum(r["correct"] for r in results.values())
    total_count = sum(r["total"] for r in results.values())
    overall_acc = total_correct / total_count if total_count > 0 else 0.0

    results["overall"] = {
        "accuracy": overall_acc,
        "correct": total_correct,
        "total": total_count,
    }
    print(f"  [prontoqa_ood] overall: {total_correct}/{total_count} = {overall_acc:.1%}")

    return results
