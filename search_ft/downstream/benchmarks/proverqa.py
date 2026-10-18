"""ProverQA (Qi et al. 2025): first-order-logic problems with distractors, the dev set at
three levels easy / medium / hard (proof lengths about 1-2 / 3-5 / 6-9), 500 items each.
Options are always A) True, B) False, C) Uncertain and are scored as the letter. 3-shot
with one demo per answer taken from `easy` and removed from scoring (1,497 items scored),
log-likelihood over A / B / C. Data: HF opendatalab/ProverQA, files dev/{easy,medium,hard}.json.
"""
import ast
import json
from typing import Dict

from ..registry import register, register_prepare
from ..scoring import eval_strata, subsample, distribute_n, balanced_demos

LABELS = ["A", "B", "C"]
LEVELS = ("easy", "medium", "hard")
DEFAULT_N = None   # the whole dev set


def load(level: str):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("opendatalab/ProverQA", f"dev/{level}.json", repo_type="dataset")
    with open(p) as f:
        d = json.load(f)
    for ex in d:
        if isinstance(ex["options"], str):
            ex["options"] = ast.literal_eval(ex["options"])
        ex["uid"] = f"{level}:{ex['id']}"     # ids restart at 0 in every level file
        assert [o.split(")")[0].strip() for o in ex["options"]] == LABELS, ex["options"]
        assert ex["answer"] in LABELS, ex["answer"]
    return d


def fmt(ex) -> str:
    return (f"{ex['context']}\n\n{ex['question']}\nOptions:\n" +
            "\n".join(ex["options"]) + "\nAnswer:")


CF = ("N/A\n\nBased on the above information, is the following statement true, false, or "
      "uncertain? N/A\nOptions:\nA) True\nB) False\nC) Uncertain\nAnswer:")


@register_prepare("proverqa")
def prepare(n=None) -> Dict:
    n = DEFAULT_N if n is None else n
    levels = {lv: load(lv) for lv in LEVELS}
    demos = balanced_demos(levels["easy"], lambda ex: ex["answer"], LABELS)
    demo_ids = {ex["uid"] for ex in demos}
    block = "".join(fmt(ex) + f" {ex['answer']}\n\n" for ex in demos)
    pools = [(lv, [ex for ex in levels[lv] if ex["uid"] not in demo_ids]) for lv in LEVELS]
    caps = distribute_n(n, len(pools), [len(p) for _, p in pools])
    strata = [(lv, subsample(p, c)) for (lv, p), c in zip(pools, caps)]
    print("  [proverqa] strata: " + " ".join(f"{k}={len(v)}" for k, v in strata) +
          f" | demos {[ex['answer'] for ex in demos]}")
    return dict(strata=strata, labels=LABELS, fmt=fmt, gold_of=lambda ex: ex["answer"],
                item_id=lambda ex: ex["uid"], demos_block=block, cf_prompt=block + CF)


@register("proverqa")
def eval_proverqa(model, tokenizer, use_chat: bool, n: int = None) -> Dict:
    return eval_strata(model, tokenizer, use_chat, "proverqa", **prepare(n))
