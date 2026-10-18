"""LogiQA (Liu et al. 2020), four-way multiple choice from civil-service exams, in the
AGIEval English rendering (HF hails/agieval-logiqa-en, split test, 651 items). 3-shot
with demos drawn from the test pool (distinct gold letters) and removed from scoring,
log-likelihood over the letters A / B / C / D on the remaining 648 items.

The AGIEval `query` field already ends with the cue line "A: Among A through D, the answer
is", which is used as is; under a chat template the whole cue line moves into the
assistant turn as one unit (see scoring.CUE_RE).
"""
from typing import Dict

from ..registry import register, register_prepare
from ..scoring import eval_strata, subsample, fewshot_block_from_pool, item_hash

LABELS = ["A", "B", "C", "D"]
K_SHOT = 3
DEFAULT_N = None   # the whole pool after removing the demos


def load():
    from datasets import load_dataset
    for repo in ("hails/agieval-logiqa-en",):
        try:
            return load_dataset(repo, split="test")
        except Exception:
            continue
    return None


def fmt(ex) -> str:
    q = ex.get("query")
    if q and "the answer is" in q.lower():
        return q   # canonical AGIEval format, no extra trailing space
    ctx = ex.get("context") or ex.get("passage", "")
    qtext = q or ex.get("question", "")
    opts = ex.get("options") or ex.get("choices", [])
    if isinstance(opts, str):
        try:
            import ast
            opts = ast.literal_eval(opts)
        except Exception:
            opts = []
    if isinstance(opts, dict):
        opts = [opts.get(k, "") for k in sorted(opts.keys())]
    while len(opts) < 4:
        opts.append("")
    return (f"Passage: {ctx}\nQuestion: {qtext}\nChoices:\n"
            f"A. {opts[0]}\nB. {opts[1]}\nC. {opts[2]}\nD. {opts[3]}\nAnswer:")


def gold_letter(ex):
    g = ex.get("gold")
    if g is None:
        g = ex.get("correct_option")
    if g is None:
        g = ex.get("label")
    if g is None:
        g = ex.get("answer")
    if isinstance(g, str):
        try:
            import ast
            g = ast.literal_eval(g)
        except Exception:
            pass
    if isinstance(g, list) and g:
        g = g[0]
    if isinstance(g, int):
        return "ABCD"[g] if 0 <= g < 4 else None
    if isinstance(g, str):
        g = g.strip().upper()
        if g in LABELS:
            return g
        if g.isdigit() and int(g) < 4:
            return "ABCD"[int(g)]
    return None


def item_id(ex):
    return item_hash(ex.get("query", ""), ex.get("context") or ex.get("passage", ""),
                     ex.get("question", ""), ex.get("options") or ex.get("choices", ""))


CF_EX = {"query": ("N/A\nQ: N/A Answer Choices: (A)N/A (B)N/A (C)N/A (D)N/A\n"
                   "A: Among A through D, the answer is")}


@register_prepare("logiqa")
def prepare(n=None) -> Dict:
    n = DEFAULT_N if n is None else n
    ds = load()
    if ds is None:
        raise RuntimeError("LogiQA load failed")
    all_pool = list(ds)
    fs_block, demo_ids = fewshot_block_from_pool(all_pool, k=K_SHOT, prompt_fn=fmt, gold_fn=gold_letter)
    pool = [ex for ex in all_pool if id(ex) not in demo_ids]
    print(f"  [logiqa] loaded {len(all_pool)} test examples; {K_SHOT} demos removed; scored {len(subsample(pool, n))}")
    return dict(strata=[("overall", subsample(pool, n))], labels=LABELS, fmt=fmt,
                gold_of=gold_letter, item_id=item_id, demos_block=fs_block,
                cf_prompt=fs_block + fmt(CF_EX))


@register("logiqa")
def eval_logiqa(model, tokenizer, use_chat: bool, n: int = None) -> Dict:
    return eval_strata(model, tokenizer, use_chat, "logiqa", **prepare(n))
