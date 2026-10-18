"""LogicBench-BQA (Parmar et al. 2024): binary yes/no questions over propositional,
first-order and non-monotonic inference rules, the LogicBench(Eval) BQA set. 3-shot direct
(no chain of thought), log-likelihood over 'yes' / 'no' in the lower case the prompt asks
for (the model obeys the prompt: ' yes' / ' no' carry the mass, ' Yes' / ' No' do not).

Data: the authors' repository github.com/Mihir3009/LogicBench, cloned on first use into
$DOWNSTREAM_DATA_DIR/logicbench/. The whole eval set is scored (1,520 question pairs, of
which 1,512 remain after removing every question on the three demo contexts).

Demos: a seeded shuffle over all question pairs, one demo per context, labels alternating
yes / no. Gold is about 66% 'no' (each context carries four questions, usually one yes), so
``majority_baseline`` is reported with the pooled accuracy, together with ``acc_yes``,
``acc_no`` and their mean ``balanced_accuracy`` (the quantities Parmar et al. report).
"""
import glob
import json
import os
import random
from typing import Dict, List, Optional

from ..registry import register, register_prepare
from ..datautil import require_data_dir, clone_repo
from ..scoring import eval_strata, subsample, item_hash, yesno

LABELS = ["yes", "no"]
K_SHOT = 3
DEFAULT_N = None   # the whole eval set
REPO = "https://github.com/Mihir3009/LogicBench.git"


def ensure_repo() -> str:
    dest = os.path.join(require_data_dir(), "logicbench", "LogicBench")
    out = clone_repo(REPO, dest, os.path.join("data", "**", "*.json"), "logicbench")
    if out is None:
        raise RuntimeError(f"LogicBench repository not available at {dest}")
    return out


def load_bqa() -> Optional[List[Dict]]:
    base = ensure_repo()
    pat = os.path.join(base, "data", "LogicBench(Eval)", "BQA", "*", "*", "data_instances.json")
    out = []
    for fpath in sorted(glob.glob(pat)):
        try:
            with open(fpath) as f:
                d = json.load(f)
        except Exception:
            continue
        for s in d.get("samples", []):
            out.append({"context": s.get("context", ""), "qa_pairs": s.get("qa_pairs", []),
                        "logic_type": d.get("type", ""), "axiom": d.get("axiom", "")})
    print(f"  [logicbench_bqa] loaded {len(out)} contexts")
    return out if out else None


def iter_bqa(items):
    for ex in items:
        ctx = ex.get("context", "")
        for qp in ex.get("qa_pairs", []):
            yield {"context": ctx, "question": qp.get("question", ""),
                   "answer": qp.get("answer", ""),
                   "logic_type": ex.get("logic_type", ""), "axiom": ex.get("axiom", "")}


def fmt(ex) -> str:
    return (f"Context: {ex.get('context', '')}\n"
            f"Question: {ex.get('question', '')}\n"
            f"Answer the question ONLY in 'yes' or 'no'.\n"
            f"Answer:")


def gold_of(ex):
    return yesno(str(ex.get("answer", ""))) or None


def few_shot_block(items, k=K_SHOT):
    """(demo block, set of (context, question) keys excluded from scoring)."""
    rng = random.Random(1234)
    pool = list(iter_bqa(items))
    rng.shuffle(pool)
    picked, used_ctx = [], set()
    want = ["yes", "no"] * k
    for target in want[:k]:
        for ex in pool:
            if gold_of(ex) == target and ex["context"] not in used_ctx:
                picked.append(ex)
                used_ctx.add(ex["context"])
                break
    used_keys = {(ex["context"], ex["question"]) for ex in pool if ex["context"] in used_ctx}
    demos = [fmt(ex) + " " + gold_of(ex) for ex in picked]
    print(f"  [logicbench_bqa] demos: {[gold_of(ex) for ex in picked]} from {len(used_ctx)} contexts; "
          f"{len(used_keys)} question pairs excluded from the scored pool")
    return ("\n\n".join(demos) + "\n\n") if demos else "", used_keys


@register_prepare("logicbench_bqa")
def prepare(n=None) -> Dict:
    n = DEFAULT_N if n is None else n
    items = load_bqa()
    if items is None:
        raise RuntimeError("LogicBench load failed")
    fs_block, demo_keys = few_shot_block(items)
    all_examples = [ex for ex in iter_bqa(items)
                    if (ex["context"], ex["question"]) not in demo_keys and gold_of(ex)]
    sampled = all_examples if (n is None or n >= len(all_examples)) else subsample(all_examples, n)
    print(f"  [logicbench_bqa] scored {len(sampled)} of {len(all_examples)} question pairs ({K_SHOT}-shot)")
    return dict(strata=[("overall", sampled)], labels=LABELS, fmt=fmt, gold_of=gold_of,
                item_id=lambda ex: item_hash(ex["context"], ex["question"]),
                demos_block=fs_block, cf_prompt=fs_block + fmt({"context": "N/A", "question": "N/A"}))


@register("logicbench_bqa")
def eval_logicbench_bqa(model, tokenizer, use_chat: bool, n: int = None) -> Dict:
    res = eval_strata(model, tokenizer, use_chat, "logicbench_bqa", **prepare(n))
    pc = res.get("per_class", {})
    acc_yes = pc.get("yes", {}).get("accuracy", 0.0)
    acc_no = pc.get("no", {}).get("accuracy", 0.0)
    res.update({"gold_counts": {l: pc.get(l, {}).get("total", 0) for l in LABELS},
                "acc_yes": acc_yes, "acc_no": acc_no,
                "balanced_accuracy": 0.5 * (acc_yes + acc_no)})
    print(f"  [logicbench_bqa] gold {res['gold_counts']}  majority {res['majority_baseline']:.1%}  "
          f"acc_yes {acc_yes:.1%}  acc_no {acc_no:.1%}")
    return res
