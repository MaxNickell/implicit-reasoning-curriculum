import os
import json
import glob
import subprocess
from typing import Dict, List, Optional, Tuple, Set

from ..registry import register
from ..scoring import score_choices, subsample

BQA_LABELS = ["Yes", "No"]
DEFAULT_N = 1000


def _get_data_dir():
    return os.environ.get("DOWNSTREAM_DATA_DIR")


def _ensure_repo() -> Optional[str]:
    data_dir = _get_data_dir()
    if data_dir is None:
        return None

    base = os.path.join(data_dir, "logicbench", "LogicBench")
    if os.path.isdir(os.path.join(base, "data")):
        return base

    parent = os.path.dirname(base)
    os.makedirs(parent, exist_ok=True)
    try:
        subprocess.check_call(
            ["git", "clone", "--depth=1",
             "https://github.com/Mihir3009/LogicBench.git", base],
            cwd=parent, timeout=120
        )
        return base
    except Exception as e:
        print(f"  [logicbench] clone failed: {e}")
        return None


def _load_bqa() -> Optional[List[Dict]]:
    base = _ensure_repo()
    if base is None:
        return None

    pat = os.path.join(base, "data", "LogicBench(Eval)", "BQA", "*", "*", "data_instances.json")
    out = []
    for fpath in sorted(glob.glob(pat)):
        try:
            with open(fpath) as f:
                d = json.load(f)
        except Exception:
            continue

        for s in d.get("samples", []):
            out.append({
                "context": s.get("context", ""),
                "qa_pairs": s.get("qa_pairs", []),
            })

    print(f"  [logicbench] loaded {len(out)} samples")
    return out if out else None


def _yesno(s: str) -> Optional[str]:
    s = s.strip().lower()
    if s in ("yes", "true", "1"):
        return "yes"
    if s in ("no", "false", "0"):
        return "no"
    return None


def _format_bqa(ex: Dict) -> str:
    ctx = ex.get("context", "")
    q = ex.get("question", "")
    return (f"Context: {ctx}\n"
            f"Question: {q}\n"
            f"Answer the question ONLY in 'yes' or 'no'.\n"
            f"Answer:")


def _iter_bqa(items: List[Dict]):
    for ex in items:
        ctx = ex.get("context", "")
        for qp in ex.get("qa_pairs", []):
            yield {
                "context": ctx,
                "question": qp.get("question", ""),
                "answer": qp.get("answer", ""),
            }


_FS_CACHE = None

def _build_few_shot(items: List[Dict], k: int = 3) -> Tuple[str, Set]:
    global _FS_CACHE
    if _FS_CACHE is not None:
        return _FS_CACHE

    half = max(1, k // 2)
    picked = {"yes": [], "no": []}
    used_keys = set()

    for ex in _iter_bqa(items):
        gold = _yesno(str(ex.get("answer", "")))
        if not gold:
            continue
        max_for_label = k - half if gold == "no" else half
        if len(picked[gold]) >= max_for_label:
            continue
        if sum(len(v) for v in picked.values()) >= k:
            break
        picked[gold].append(ex)
        used_keys.add((ex["context"], ex["question"]))

    demos = []
    for ex in picked["yes"] + picked["no"]:
        demos.append(_format_bqa(ex) + " " + _yesno(str(ex.get("answer", ""))))

    _FS_CACHE = ("\n\n".join(demos) + "\n\n", used_keys)
    return _FS_CACHE


@register("logicbench")
def eval_logicbench(model, tokenizer, use_chat: bool, n: int = None) -> Dict:
    n = n if n is not None else DEFAULT_N

    items = _load_bqa()
    if items is None:
        return {"error": "LogicBench load failed."}

    fs_block, demo_keys = _build_few_shot(items, k=3)

    all_examples = [
        ex for ex in _iter_bqa(items)
        if (ex["context"], ex["question"]) not in demo_keys
        and _yesno(str(ex.get("answer", "")))
    ]
    sampled = subsample(all_examples, n)

    print(f"  [logicbench] {len(sampled)} examples (3-shot)")

    correct = 0
    total = 0

    for ex in sampled:
        gold = _yesno(str(ex.get("answer", "")))
        prompt = fs_block + _format_bqa(ex)
        out = score_choices(model, tokenizer, prompt, BQA_LABELS, use_chat)
        pred = out["prediction"].lower()

        if pred == gold:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0.0
    print(f"  [logicbench] overall: {correct}/{total} = {acc:.1%}")

    return {"accuracy": acc, "correct": correct, "total": total}
