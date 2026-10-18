"""Web-of-Lies-Deep: BIG-Bench Hard's web_of_lies (Suzgun et al. 2022) regenerated with a
depth axis, k in {1, 2, 4, 8, 16, 32, 48, 64}, 500 items per k, Yes/No exactly 50/50 at
every k by construction. 3-shot, log-likelihood over Yes / No.

The file is built on first use by generators/web_of_lies_deep.py (seed 42, 65-name pool,
every item re-parsed and re-solved before writing). The demos are the three canonical BBH
exemplars for web_of_lies, verbatim from lm-eval-harness tasks/bbh/fewshot/web_of_lies.yaml
in its layout "Q: {input}\\nA: {target}"; zero-shot, the model's top next token was the
speaker's name rather than a label.

Caveat recorded with the benchmark: the task admits a surface shortcut (the answer is the
parity of the word "lies" with the root folded in), inherited from BBH's design, and all
arms saturate near ceiling at k=1 and at chance from k=4 on.
"""
import json
import os
from typing import Dict

from ..registry import register, register_prepare
from ..datautil import require_data_dir
from ..scoring import eval_strata, subsample, distribute_n, item_hash

DEFAULT_N = 4000
LABELS = ["Yes", "No"]

FEWSHOT = [
    ("Question: Fidel tells the truth. Jerry says Fidel tells the truth. Vina says Jerry tells "
     "the truth. Millicent says Vina lies. Raymond says Millicent lies. Does Raymond tell the truth?",
     "Yes"),
    ("Question: Kristian lies. Millie says Kristian lies. Maybelle says Millie tells the truth. "
     "Fidel says Maybelle lies. Leda says Fidel lies. Does Leda tell the truth?",
     "Yes"),
    ("Question: Kristian tells the truth. Michaela says Kristian lies. Raymond says Michaela "
     "tells the truth. Osvaldo says Raymond tells the truth. Jamey says Osvaldo tells the truth. "
     "Does Jamey tell the truth?",
     "No"),
]
FS_BLOCK = "".join(f"Q: {q}\nA: {a}\n\n" for q, a in FEWSHOT)


def ensure_file() -> str:
    path = os.path.join(require_data_dir(), "web_of_lies_deep", "web_of_lies_deep.jsonl")
    if not os.path.isfile(path):
        from ..generators.web_of_lies_deep import build
        print("  [web_of_lies_deep] building the fixed 4,000-item file (first use only)")
        build(path)
    return path


def fmt(ex) -> str:
    return f"Q: {ex['input']}\nA:"


@register_prepare("web_of_lies_deep")
def prepare(n=None) -> Dict:
    n = DEFAULT_N if n is None else n
    by_k = {}
    with open(ensure_file()) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            by_k.setdefault(int(ex["k_hop"]), []).append(ex)
    if not by_k:
        raise RuntimeError("web_of_lies_deep: no items parsed")
    ks = sorted(by_k)
    sizes = [len(by_k[k]) for k in ks]
    caps = distribute_n(n, len(ks), sizes)
    strata = [(f"k={k}", subsample(by_k[k], c)) for k, c in zip(ks, caps)]
    print(f"  [web_of_lies_deep] k={ks}, available={sizes}, scored={[len(v) for _, v in strata]}")
    return dict(strata=strata, labels=LABELS, fmt=fmt,
                gold_of=lambda ex: str(ex["target"]).strip(),
                item_id=lambda ex: item_hash(ex["input"]),
                demos_block=FS_BLOCK,
                cf_prompt=FS_BLOCK + "Q: Question: N/A Does N/A tell the truth?\nA:")


@register("web_of_lies_deep")
def eval_web_of_lies_deep(model, tokenizer, use_chat: bool, n: int = None) -> Dict:
    return eval_strata(model, tokenizer, use_chat, "web_of_lies_deep", **prepare(n))
