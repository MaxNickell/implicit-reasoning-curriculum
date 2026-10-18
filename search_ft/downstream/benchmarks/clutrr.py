"""CLUTRR (Sinha et al. 2019): kinship inference over short stories, stratified by the
number of relation hops k = 2..10. 5-shot with demos from the canonical TRAIN split,
log-likelihood over the 18 kinship labels that occur as gold in the released data
('husband', 'wife' and 'sister-in-law' never do).

Data: HF kendrivp/CLUTRR_v1_extracted, the only maintained mirror with every task split
intact. Its HF "test" split is the whole 70,631-row dump (train + validation + test, all
noise families), so rows are filtered to the canonical ``task_split == "test"`` and the
clean ``task_1`` family: 2,641 items over k=2..10. The 1,000-item cap is spread over the
nine hop counts.
"""
from typing import Dict

from ..registry import register, register_prepare
from ..scoring import eval_strata, subsample, distribute_n, fewshot_block_from_pool, item_hash

LABELS = [
    "aunt", "uncle", "father", "mother", "brother", "sister",
    "son", "daughter", "grandmother", "grandfather",
    "granddaughter", "grandson", "niece", "nephew",
    "father-in-law", "mother-in-law", "son-in-law", "daughter-in-law",
]
K_SHOT = 5
DEFAULT_N = 1000


def load(split: str = "test"):
    from datasets import load_dataset
    ds, repo = None, None
    for repo in ("kendrivp/CLUTRR_v1_extracted", "kliang5/CLUTRR_huggingface_dataset"):
        try:
            ds = load_dataset(repo, split="test")
            break
        except Exception:
            continue
    if ds is None:
        return None
    if "task_split" not in ds.column_names:
        print("  [clutrr] WARNING: no task_split column; using the dump unfiltered")
        return list(ds)
    out = [ex for ex in ds
           if str(ex.get("task_split", "")) == split
           and str(ex.get("task_name") or ex.get("task", "")).startswith("task_1")]
    print(f"  [clutrr] {repo}: {len(ds)} rows in dump -> {len(out)} clean task_1 rows with task_split={split!r}")
    return out


def extract_k(ex) -> int:
    t = ex.get("task_name") or ex.get("task", "")
    try:
        return int(t.split(".")[-1])
    except Exception:
        return 0


def fmt(ex) -> str:
    story = ex.get("clean_story") or ex.get("story", "")
    q = ex.get("query")
    if isinstance(q, (list, tuple)):
        a, b = q[0], q[1]
    elif isinstance(q, str) and "(" in q:
        import ast
        try:
            t = ast.literal_eval(q)
            a, b = t[0], t[1]
        except Exception:
            a, b = "?", "?"
    else:
        a, b = "?", "?"
    return f"{story}\nQuestion: How is {b} related to {a}?\nAnswer:"


def gold_of(ex):
    g = str(ex.get("target_text") or ex.get("target") or ex.get("answer") or "").strip().lower()
    return g or None


def _assign_item_ids(rows):
    """Dataset `id` (uuid), else sha1 of the raw fields. The mirror repeats about 1.4% of rows
    verbatim under the same id, so the k-th repeat becomes "<id>#k" in dataset order."""
    seen = {}
    for ex in rows:
        base = ex.get("id") or item_hash(ex.get("task_name") or ex.get("task", ""),
                                         ex.get("clean_story") or ex.get("story", ""),
                                         ex.get("query", ""))
        k = seen.get(base, 0) + 1
        seen[base] = k
        ex["_item_id"] = base if k == 1 else f"{base}#{k}"


@register_prepare("clutrr")
def prepare(n=None) -> Dict:
    n = DEFAULT_N if n is None else n
    rows = load("test")
    if rows is None:
        raise RuntimeError("CLUTRR load failed")
    _assign_item_ids(rows)
    by_k = {}
    for ex in rows:
        by_k.setdefault(extract_k(ex), []).append(ex)
    train_pool = [ex for ex in (load("train") or []) if extract_k(ex) > 0]
    all_pool = train_pool if train_pool else [ex for k in sorted(by_k) if k > 0 for ex in by_k[k]]
    print(f"  [clutrr] demo pool: {'train' if train_pool else 'test (fallback)'} split, {len(all_pool)} rows")
    fs_block, demo_ids = fewshot_block_from_pool(all_pool, k=K_SHOT, prompt_fn=fmt, gold_fn=gold_of)
    ks = [k for k in sorted(by_k) if k > 0]
    pools = [[ex for ex in by_k[k] if id(ex) not in demo_ids] for k in ks]
    caps = distribute_n(n, len(ks), [len(p) for p in pools])
    strata = [(f"k={k}", subsample(p, c)) for k, p, c in zip(ks, pools, caps)]
    print(f"  [clutrr] k={ks}, pool {[len(p) for p in pools]}, scored {[len(v) for _, v in strata]}")
    return dict(strata=strata, labels=LABELS, fmt=fmt, gold_of=gold_of,
                item_id=lambda ex: ex["_item_id"], demos_block=fs_block,
                cf_prompt=fs_block + fmt({"clean_story": "N/A", "query": "('A', 'B')"}))


@register("clutrr")
def eval_clutrr(model, tokenizer, use_chat: bool, n: int = None) -> Dict:
    return eval_strata(model, tokenizer, use_chat, "clutrr", **prepare(n))
