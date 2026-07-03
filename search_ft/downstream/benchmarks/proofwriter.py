import json
import os
import urllib.request
import zipfile
from typing import Dict, List, Optional

from ..registry import register
from ..scoring import score_choices, subsample, distribute_n

PROOFWRITER_URL = "https://aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip"
DEFAULT_N = 1000
NUM_SHOTS = 8


def _get_data_dir():
    return os.environ.get("DOWNSTREAM_DATA_DIR")


def _find_proofwriter_raw() -> Optional[str]:
    data_dir = _get_data_dir()
    if data_dir is None:
        return None

    base = os.path.join(data_dir, "proofwriter_raw")
    p = os.path.join(base, "proofwriter-dataset-V2020.12.3")
    if os.path.isdir(os.path.join(p, "CWA")) and os.path.isdir(os.path.join(p, "OWA")):
        return p

    os.makedirs(base, exist_ok=True)
    zip_path = os.path.join(base, "proofwriter-dataset-V2020.12.3.zip")
    try:
        print(f"  [proofwriter] downloading {PROOFWRITER_URL} (~45MB)")
        urllib.request.urlretrieve(PROOFWRITER_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(base)
        os.remove(zip_path)
        if os.path.isdir(os.path.join(p, "CWA")):
            return p
    except Exception as e:
        print(f"  [proofwriter] download failed: {e}")

    return None


def _load_proofwriter_raw(mode: str, depth_dir: str = "depth-5") -> Optional[List[Dict]]:
    root = _find_proofwriter_raw()
    if root is None:
        return None
    path = os.path.join(root, mode, depth_dir, "meta-test.jsonl")
    if not os.path.isfile(path):
        return None

    examples = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            theory = rec.get("theory", "")
            for qid, q in rec.get("questions", {}).items():
                examples.append({
                    "theory": theory,
                    "question": q.get("question", ""),
                    "answer": q.get("answer"),
                    "QDep": q.get("QDep"),
                })
    return examples


def _normalize_answer(ans) -> Optional[str]:
    if ans is True:
        return "True"
    if ans is False:
        return "False"
    if isinstance(ans, str):
        return ans
    return None


def _format_owa(ex):
    return (f"Facts and rules:\n{ex['theory']}\n\n"
            f"Based only on the facts and rules above, is the following "
            f"statement True, False, or Unknown (meaning it can be neither "
            f"proven nor disproven)?\n{ex['question']}\n"
            f"Answer with only True, False, or Unknown.")


def _format_cwa(ex):
    return (f"Facts and rules:\n{ex['theory']}\n\n"
            f"Based only on the facts and rules above, is the following "
            f"statement true or false? Assume anything not provable is False.\n"
            f"{ex['question']}\nAnswer with only True or False.")


def _build_few_shot_block(mode: str, labels: List[str]) -> str:
    root = _find_proofwriter_raw()
    if root is None:
        return ""

    # Use depth-3 train data for demos (different from eval depth-5)
    p = None
    for depth_dir in ("depth-3", "depth-2", "depth-1"):
        for fname in ("meta-train.jsonl", "meta-test.jsonl"):
            cand = os.path.join(root, mode, depth_dir, fname)
            if os.path.isfile(cand):
                p = cand
                break
        if p:
            break
    if p is None:
        return ""

    examples = []
    label_counts = {lbl: 0 for lbl in labels}
    want_per_label = NUM_SHOTS // len(labels) + 1

    with open(p) as f:
        for line in f:
            if len(examples) >= NUM_SHOTS:
                break
            rec = json.loads(line)
            theory = rec.get("theory", "")
            for q in rec.get("questions", {}).values():
                ans = _normalize_answer(q.get("answer"))
                if ans not in labels:
                    continue
                if label_counts[ans] >= want_per_label:
                    continue
                label_counts[ans] += 1
                examples.append({"theory": theory, "question": q.get("question", ""), "answer": ans})
                if len(examples) >= NUM_SHOTS:
                    break

    fmt_fn = _format_owa if mode == "OWA" else _format_cwa
    blocks = [f"{fmt_fn(ex)} {ex['answer']}" for ex in examples[:NUM_SHOTS]]
    return "\n\n".join(blocks) + "\n\n" if blocks else ""


@register("proofwriter")
def eval_proofwriter(model, tokenizer, use_chat: bool, n: int = None) -> Dict:
    n = n if n is not None else DEFAULT_N

    examples = _load_proofwriter_raw("OWA", "depth-5")
    if examples is None:
        return {"error": "ProofWriter dataset not found."}

    print(f"  [proofwriter] Loaded OWA/depth-5: {len(examples)} questions")

    by_depth = {}
    for ex in examples:
        ans = _normalize_answer(ex["answer"])
        if ans not in ("True", "False", "Unknown"):
            continue
        ex["answer"] = ans
        depth = ex["QDep"]
        if depth is not None and depth <= 5:
            by_depth.setdefault(depth, []).append(ex)

    depths = sorted(by_depth.keys())
    sizes = [len(by_depth[d]) for d in depths]
    caps = distribute_n(n, len(depths), sizes)

    few_shot_block = _build_few_shot_block("OWA", ["True", "False", "Unknown"])
    choices = ["True", "False", "Unknown"]
    results = {}

    for depth, cap in zip(depths, caps):
        correct = 0
        total = 0

        for ex in subsample(by_depth[depth], cap):
            gold = ex["answer"]
            prompt = few_shot_block + _format_owa(ex)
            out = score_choices(model, tokenizer, prompt, choices, use_chat)

            if out["prediction"] == gold:
                correct += 1
            total += 1

        acc = correct / total if total > 0 else 0.0
        results[depth] = {"accuracy": acc, "correct": correct, "total": total}
        print(f"  [proofwriter] depth={depth}: {correct}/{total} = {acc:.1%}")

    total_correct = sum(r["correct"] for r in results.values())
    total_count = sum(r["total"] for r in results.values())
    results["overall"] = {"accuracy": total_correct / total_count if total_count > 0 else 0.0,
                          "correct": total_correct, "total": total_count}
    print(f"  [proofwriter] overall: {total_correct}/{total_count} = {results['overall']['accuracy']:.1%}")
    return results


@register("proofwriter_cwa")
def eval_proofwriter_cwa(model, tokenizer, use_chat: bool, n: int = None) -> Dict:
    n = n if n is not None else DEFAULT_N

    examples = _load_proofwriter_raw("CWA", "depth-5")
    if examples is None:
        return {"error": "ProofWriter dataset not found."}

    print(f"  [proofwriter_cwa] Loaded CWA/depth-5: {len(examples)} questions")

    by_depth = {}
    for ex in examples:
        ans = _normalize_answer(ex["answer"])
        if ans not in ("True", "False"):
            continue
        ex["answer"] = ans
        depth = ex["QDep"]
        if depth is not None and depth <= 5:
            by_depth.setdefault(depth, []).append(ex)

    depths = sorted(by_depth.keys())
    sizes = [len(by_depth[d]) for d in depths]
    caps = distribute_n(n, len(depths), sizes)

    few_shot_block = _build_few_shot_block("CWA", ["True", "False"])
    choices = ["True", "False"]
    results = {}

    for depth, cap in zip(depths, caps):
        correct = 0
        total = 0

        for ex in subsample(by_depth[depth], cap):
            gold = ex["answer"]
            prompt = few_shot_block + _format_cwa(ex)
            out = score_choices(model, tokenizer, prompt, choices, use_chat)

            if out["prediction"] == gold:
                correct += 1
            total += 1

        acc = correct / total if total > 0 else 0.0
        results[depth] = {"accuracy": acc, "correct": correct, "total": total}
        print(f"  [proofwriter_cwa] depth={depth}: {correct}/{total} = {acc:.1%}")

    total_correct = sum(r["correct"] for r in results.values())
    total_count = sum(r["total"] for r in results.values())
    results["overall"] = {"accuracy": total_correct / total_count if total_count > 0 else 0.0,
                          "correct": total_correct, "total": total_count}
    print(f"  [proofwriter_cwa] overall: {total_correct}/{total_count} = {results['overall']['accuracy']:.1%}")
    return results
