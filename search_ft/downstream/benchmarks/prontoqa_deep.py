"""PrOntoQA-Deep: the official PrOntoQA generator (Saparov & He 2023) driven to k in
{1, 2, 4, 8, 16, 32, 48, 64} hops, 500 items per k, the (query polarity x label) 2x2
balanced at 125 per cell per depth. 8-shot, log-likelihood over True / False.

The file is built on first use by generators/prontoqa_deep.py (deterministic; see its
docstring for what changes relative to the shipped generator and how every item is
verified). Demos are a fixed 8-item block at k=2 (4 True / 4 False) WITHOUT chain of
thought (Q + label only), shared by every stratum, so the prompt grows only through the
test question (about 1,000 words at k=64). Framing matches PrOntoQA-OOD: fictional
ModusPonens ontology, relevant distractors, random sentence order, "True or false:" query.

Reported per stratum besides accuracy: ``majority_baseline`` (50% by construction) and
``polarity_baseline``, the accuracy of the no-reasoning rule "query says 'is not' -> False,
else True" (50% by construction after the 2x2 balancing).
"""
import json
import os
from typing import Dict

from ..registry import register, register_prepare
from ..datautil import require_data_dir, run_generator
from ..scoring import eval_strata, subsample, distribute_n, item_hash

DEFAULT_N = 4000
LABELS = ["True", "False"]


def ensure_files():
    base = os.path.join(require_data_dir(), "prontoqa_deep")
    path = os.path.join(base, "prontoqa_deep.jsonl")
    demo_path = os.path.join(base, "prontoqa_deep_demos.jsonl")
    if not (os.path.isfile(path) and os.path.isfile(demo_path)):
        print("  [prontoqa_deep] building the fixed 4,000-item file with the official "
              "generator (first use only; clones github.com/asaparov/prontoqa)")
        run_generator("search_ft.downstream.generators.prontoqa_deep", "--out-dir", base)
    return path, demo_path


def _load_jsonl(p):
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def fmt(ex) -> str:
    return f"Q: {ex['question']}\n{ex['query']}\nA:"


@register_prepare("prontoqa_deep")
def prepare(n=None) -> Dict:
    n = DEFAULT_N if n is None else n
    path, demo_path = ensure_files()
    demos = _load_jsonl(demo_path)
    by_k = {}
    for ex in _load_jsonl(path):
        by_k.setdefault(int(ex["k_hop"]), []).append(ex)
    if not by_k or not demos:
        raise RuntimeError("prontoqa_deep: no items parsed")
    demo_block = "\n\n".join(f"Q: {d['question']}\n{d['query']}\nA: {d['label']}" for d in demos)
    ks = sorted(by_k)
    sizes = [len(by_k[k]) for k in ks]
    caps = distribute_n(n, len(ks), sizes)
    strata = [(f"k={k}", subsample(by_k[k], c)) for k, c in zip(ks, caps)]
    print(f"  [prontoqa_deep] k={ks}, available={sizes}, scored={[len(v) for _, v in strata]}, "
          f"demos={len(demos)} at k={demos[0]['k_hop']}")
    return dict(strata=strata, labels=LABELS, fmt=fmt,
                gold_of=lambda ex: str(ex["label"]).strip(),
                item_id=lambda ex: item_hash(ex["question"], ex["query"]),
                demos_block=demo_block + "\n\n",
                cf_prompt=demo_block + "\n\nQ: N/A\nN/A\nA:",
                extra_of=lambda ex, gold: {
                    "polarity_baseline": ("False" if " is not " in ex["query"] else "True") == gold})


@register("prontoqa_deep")
def eval_prontoqa_deep(model, tokenizer, use_chat: bool, n: int = None) -> Dict:
    return eval_strata(model, tokenizer, use_chat, "prontoqa_deep", **prepare(n))
