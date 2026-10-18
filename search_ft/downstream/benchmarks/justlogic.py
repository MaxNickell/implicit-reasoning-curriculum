"""JustLogic (Chen et al. 2025): deductive reasoning independent of world knowledge,
4,900 items, 700 at each depth 1-7, labels True / False / Uncertain. 3-shot with one demo
per label drawn from depth 1 and removed from scoring (4,897 items scored), log-likelihood
over the three labels. The paragraph is the given input and `question` is the statement
to judge (dataset card). Data: HF michaelchenkj/JustLogic.
"""
from typing import Dict

from ..registry import register, register_prepare
from ..scoring import eval_strata, subsample, distribute_n, balanced_demos

LABELS = ["True", "False", "Uncertain"]
DEFAULT_N = None   # the whole set
_NORM = {"TRUE": "True", "FALSE": "False", "UNCERTAIN": "Uncertain"}


def load():
    from datasets import load_dataset
    ds = list(load_dataset("michaelchenkj/JustLogic", split="train"))
    for ex in ds:
        ex["gold"] = _NORM[str(ex["label"]).strip().upper()]
    return ds


def fmt(ex) -> str:
    return (f"{ex['paragraph']}\n\nBased only on the passage above, is the following "
            f"statement true, false, or uncertain?\nStatement: {ex['question']}\nAnswer:")


CF = ("N/A\n\nBased only on the passage above, is the following statement true, false, or "
      "uncertain?\nStatement: N/A\nAnswer:")


@register_prepare("justlogic")
def prepare(n=None) -> Dict:
    n = DEFAULT_N if n is None else n
    ds = load()
    demos = balanced_demos([ex for ex in ds if str(ex["depth"]) == "1"], lambda ex: ex["gold"], LABELS)
    demo_ids = {ex["id"] for ex in demos}
    block = "".join(fmt(ex) + f" {ex['gold']}\n\n" for ex in demos)
    order = sorted({int(ex["depth"]) for ex in ds})
    pools = [(f"depth {d}", [ex for ex in ds if int(ex["depth"]) == d and ex["id"] not in demo_ids])
             for d in order]
    caps = distribute_n(n, len(pools), [len(p) for _, p in pools])
    strata = [(k, subsample(p, c)) for (k, p), c in zip(pools, caps)]
    print("  [justlogic] strata: " + " ".join(f"{k}={len(v)}" for k, v in strata) +
          f" | demos {[ex['gold'] for ex in demos]}")
    return dict(strata=strata, labels=LABELS, fmt=fmt, gold_of=lambda ex: ex["gold"],
                item_id=lambda ex: str(ex["id"]), demos_block=block, cf_prompt=block + CF)


@register("justlogic")
def eval_justlogic(model, tokenizer, use_chat: bool, n: int = None) -> Dict:
    return eval_strata(model, tokenizer, use_chat, "justlogic", **prepare(n))
