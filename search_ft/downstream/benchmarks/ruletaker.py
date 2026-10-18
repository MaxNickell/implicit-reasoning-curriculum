"""RuleTaker (Clark et al. 2020), the predecessor of ProofWriter: True / False entailment
over rules and facts. 5-shot with balanced demos drawn from the pool (every question on a
demo's theory is removed from scoring), log-likelihood over True / False.

Data: HF tasksource/ruletaker, split test, the five BASE partitions only (depth-0, -1, -2,
-3, -5). The depth-3ext and depth-3ext-NatLang configs are documented supersets of the base
partitions (keeping them double-counts) and NatLang is the paper's separate transfer test.
Strata are the partitions, i.e. the theory's MAXIMUM depth, not the per-question inference
depth, which tasksource does not carry; a depth-N stratum contains questions of every
depth <= N.
"""
from typing import Dict

from ..registry import register, register_prepare
from ..scoring import eval_strata, subsample, distribute_n, fewshot_block_from_pool, item_hash

LABELS = ["True", "False"]
K_SHOT = 5
DEFAULT_N = 1000
BASE_PARTITIONS = {"depth-0", "depth-1", "depth-2", "depth-3", "depth-5"}


def load_by_partition():
    from datasets import load_dataset
    ds = None
    for repo in ("tasksource/ruletaker", "hitachi-nlp/ruletaker"):
        try:
            ds = load_dataset(repo, split="test")
            break
        except Exception as e:
            print(f"  [ruletaker] {repo} load failed: {e}")
    if ds is None:
        return None
    cfg_field = "config" if "config" in ds.column_names else None
    by = {}
    for ex in ds:
        d = str(ex.get(cfg_field, "depth-?")) if cfg_field else "depth-?"
        if cfg_field and d not in BASE_PARTITIONS:
            continue
        by.setdefault(d, []).append(ex)
    return by


def fmt(ex) -> str:
    return (f"Rules and facts:\n{ex.get('context', '')}\n\n"
            f"Statement: {ex.get('question', '')}\n"
            f"Based only on the rules and facts above, is the statement true or false?\n"
            f"Answer:")


def gold_of(ex):
    lbl = str(ex.get("label", "")).strip().lower()
    if lbl in ("true", "entailment", "1"):
        return "True"
    if lbl in ("false", "not entailment", "not_entailment", "0"):
        return "False"
    return None


def item_id(ex):
    return item_hash(ex.get("config", ""), ex.get("context", ""), ex.get("question", ""))


@register_prepare("ruletaker")
def prepare(n=None) -> Dict:
    n = DEFAULT_N if n is None else n
    by = load_by_partition()
    if by is None:
        raise RuntimeError("RuleTaker load failed")
    cfgs = sorted(by.keys())
    all_pool = [ex for cfg in cfgs for ex in by[cfg]]
    fs_block, demo_ids = fewshot_block_from_pool(all_pool, k=K_SHOT, prompt_fn=fmt, gold_fn=gold_of)
    # exclude every question on a demo's THEORY, not just the demo item
    demo_ctx = {ex.get("context", "") for ex in all_pool if id(ex) in demo_ids}
    demo_ids = demo_ids | {id(ex) for ex in all_pool if ex.get("context", "") in demo_ctx}
    pools = [[ex for ex in by[c] if id(ex) not in demo_ids] for c in cfgs]
    caps = distribute_n(n, len(cfgs), [len(p) for p in pools])
    strata = [(c, subsample(p, cap)) for c, p, cap in zip(cfgs, pools, caps)]
    print(f"  [ruletaker] partitions {cfgs}, pool {[len(p) for p in pools]}, "
          f"scored {[len(v) for _, v in strata]} ({K_SHOT}-shot, {len(demo_ctx)} demo theories excluded)")
    return dict(strata=strata, labels=LABELS, fmt=fmt, gold_of=gold_of, item_id=item_id,
                demos_block=fs_block, cf_prompt=fs_block + fmt({"context": "N/A", "question": "N/A"}))


@register("ruletaker")
def eval_ruletaker(model, tokenizer, use_chat: bool, n: int = None) -> Dict:
    return eval_strata(model, tokenizer, use_chat, "ruletaker", **prepare(n))
