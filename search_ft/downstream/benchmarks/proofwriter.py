"""ProofWriter (Tafjord et al. 2021) in three forms, all zero-shot, log-likelihood over the
label set, stratified by question depth QDep:

  proofwriter      OWA, True / False / Unknown, depth-5 meta-test, class-balanced per depth
  proofwriter_cwa  CWA, True / False, depth-5 meta-test
  pararules        ParaRules = OWA with human-paraphrased rules (the OWA/NatLang meta-test),
                   same prompt, labels and class balancing as `proofwriter`; QDep 3, 4 and 5
                   are merged into one stratum because Unknown is rare past depth 2

Data: the AllenAI release proofwriter-dataset-V2020.12.3.zip, downloaded on first use into
$DOWNSTREAM_DATA_DIR/proofwriter_raw/. The depth-5 partition holds questions at QDep 0..5
over theories that need up to 5 hops.

Class balancing (OWA and ParaRules only). In the raw OWA data the Unknown share falls from
69% at QDep 0 to 3% at QDep 5 (unprovable questions carry the depth of the deepest failing
search), so a constant-Unknown predictor scores 66/54/34/22/7/4 by depth and a per-depth
curve on the raw mix is the class prior, not reasoning. Each depth stratum is therefore
sampled with equal True / False / Unknown counts (seeded), which makes the reported number
the balanced accuracy at that depth, with chance 33.3% everywhere. CWA is two-way and
balanced by construction (majority never above 53%), so it is not resampled.
"""
import json
import os
import random
from typing import Dict, List, Optional

from ..registry import register, register_prepare
from ..datautil import require_data_dir, download_zip
from ..scoring import eval_strata, subsample, distribute_n

URL = "https://aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip"
# The release extracts to 3.2 GB; the suite reads these three files (about 80 MB).
NEEDED = ("OWA/depth-5/meta-test.jsonl", "CWA/depth-5/meta-test.jsonl", "OWA/NatLang/meta-test.jsonl")
OWA_LABELS = ["True", "False", "Unknown"]
CWA_LABELS = ["True", "False"]
DEFAULT_N = 3000          # 500 per depth for proofwriter and proofwriter_cwa (the paper's n)
PARARULES_DEFAULT_N = None  # the whole balanced pool (4,566 items)


def find_raw() -> str:
    base = os.path.join(require_data_dir(), "proofwriter_raw")
    root = os.path.join(base, "proofwriter-dataset-V2020.12.3")
    if os.path.isdir(os.path.join(root, "CWA")) and os.path.isdir(os.path.join(root, "OWA")):
        return root
    ok = download_zip(URL, base, "proofwriter", "(~45MB)",
                      member_filter=lambda m: m.endswith(NEEDED) or m.endswith("README.md"))
    if ok and os.path.isdir(os.path.join(root, "OWA")):
        return root
    raise RuntimeError(f"ProofWriter data not found under {base}")


def normalize_answer(ans) -> Optional[str]:
    """Answers are JSON booleans for True/False and the string 'Unknown' under OWA."""
    if ans is True:
        return "True"
    if ans is False:
        return "False"
    if isinstance(ans, str):
        return ans
    return None


def load_raw(mode: str, partition: str = "depth-5") -> List[Dict]:
    """Flatten meta-test.jsonl of one partition into one record per question."""
    path = os.path.join(find_raw(), mode, partition, "meta-test.jsonl")
    examples = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            theory = rec.get("theory", "")
            for qid, q in rec.get("questions", {}).items():
                examples.append({
                    "id": f"{rec.get('id')}:{qid}",   # stable item id for --items-output
                    "theory": theory,
                    "question": q.get("question", ""),
                    "answer": normalize_answer(q.get("answer")),
                    "QDep": q.get("QDep"),
                })
    return examples


def format_owa(ex) -> str:
    return (f"Facts and rules:\n{ex['theory']}\n\n"
            f"Based only on the facts and rules above, is the following "
            f"statement True, False, or Unknown (meaning it can be neither "
            f"proven nor disproven)?\n{ex['question']}\n"
            f"Answer with only True, False, or Unknown.")


def format_cwa(ex) -> str:
    return (f"Facts and rules:\n{ex['theory']}\n\n"
            f"Based only on the facts and rules above, is the following "
            f"statement true or false? Assume anything not provable is False.\n"
            f"{ex['question']}\nAnswer with only True or False.")


def balance_classes(groups: Dict, keys: List, labels: List[str], rng: random.Random) -> None:
    """In place: every group keeps equal counts of each label (the size of its smallest
    class), drawn by one shared seeded RNG over the groups in ``keys`` order."""
    for k in keys:
        by_cls = {l: [x for x in groups[k] if x["answer"] == l] for l in labels}
        m = min(len(v) for v in by_cls.values())
        bal = []
        for l in labels:
            rng.shuffle(by_cls[l])
            bal.extend(by_cls[l][:m])
        groups[k] = bal


def _group_by_depth(examples, labels):
    by_depth = {}
    for ex in examples:
        if ex["answer"] not in labels or ex["QDep"] is None:
            continue
        by_depth.setdefault(ex["QDep"], []).append(ex)
    depths = [d for d in sorted(by_depth) if d <= 5]
    return by_depth, depths


@register_prepare("proofwriter")
def prepare_proofwriter(n=None) -> Dict:
    n = DEFAULT_N if n is None else n
    by_depth, depths = _group_by_depth(load_raw("OWA", "depth-5"), OWA_LABELS)
    balance_classes(by_depth, depths, OWA_LABELS, random.Random(42))
    sizes = [len(by_depth[d]) for d in depths]
    caps = distribute_n(n, len(depths), sizes)
    strata = [(f"depth {d}", subsample(by_depth[d], c)) for d, c in zip(depths, caps)]
    print("  [proofwriter] class-balanced per depth: " +
          " ".join(f"d{d}={len(by_depth[d])}" for d in depths) +
          " | scored: " + " ".join(f"d{d}={len(v)}" for (_, v), d in zip(strata, depths)))
    return dict(strata=strata, labels=OWA_LABELS, fmt=format_owa,
                gold_of=lambda ex: ex["answer"], item_id=lambda ex: ex["id"],
                cf_prompt=format_owa({"theory": "N/A", "question": "N/A"}))


@register_prepare("proofwriter_cwa")
def prepare_proofwriter_cwa(n=None) -> Dict:
    n = DEFAULT_N if n is None else n
    by_depth, depths = _group_by_depth(load_raw("CWA", "depth-5"), CWA_LABELS)
    sizes = [len(by_depth[d]) for d in depths]
    caps = distribute_n(n, len(depths), sizes)
    strata = [(f"depth {d}", subsample(by_depth[d], c)) for d, c in zip(depths, caps)]
    print("  [proofwriter_cwa] per depth: " +
          " ".join(f"d{d}={len(by_depth[d])}" for d in depths) +
          " | scored: " + " ".join(f"d{d}={len(v)}" for (_, v), d in zip(strata, depths)))
    return dict(strata=strata, labels=CWA_LABELS, fmt=format_cwa,
                gold_of=lambda ex: ex["answer"], item_id=lambda ex: ex["id"],
                cf_prompt=format_cwa({"theory": "N/A", "question": "N/A"}))


@register_prepare("pararules")
def prepare_pararules(n=None) -> Dict:
    """QDep 3, 4 and 5 are merged: Unknown is rare past depth 2 (72 / 6 / 1 at QDep 3 / 4 / 5),
    so balancing QDep 4-5 alone would leave 21 items, while 3-5 balanced gives 237."""
    n = PARARULES_DEFAULT_N if n is None else n
    by = {}
    for ex in load_raw("OWA", "NatLang"):
        if ex["answer"] not in OWA_LABELS or ex["QDep"] is None:
            continue
        key = "depth 3-5" if ex["QDep"] >= 3 else f"depth {ex['QDep']}"
        by.setdefault(key, []).append(ex)
    order = [k for k in ("depth 0", "depth 1", "depth 2", "depth 3-5") if k in by]
    balance_classes(by, order, OWA_LABELS, random.Random(42))
    caps = distribute_n(n, len(order), [len(by[k]) for k in order])
    strata = [(k, subsample(by[k], c)) for k, c in zip(order, caps)]
    print("  [pararules] class-balanced strata: " + " ".join(f"{k}={len(v)}" for k, v in strata))
    return dict(strata=strata, labels=OWA_LABELS, fmt=format_owa,
                gold_of=lambda ex: ex["answer"], item_id=lambda ex: ex["id"],
                cf_prompt=format_owa({"theory": "N/A", "question": "N/A"}))


@register("proofwriter")
def eval_proofwriter(model, tokenizer, use_chat: bool, n: int = None) -> Dict:
    return eval_strata(model, tokenizer, use_chat, "proofwriter", **prepare_proofwriter(n))


@register("proofwriter_cwa")
def eval_proofwriter_cwa(model, tokenizer, use_chat: bool, n: int = None) -> Dict:
    return eval_strata(model, tokenizer, use_chat, "proofwriter_cwa", **prepare_proofwriter_cwa(n))


@register("pararules")
def eval_pararules(model, tokenizer, use_chat: bool, n: int = None) -> Dict:
    return eval_strata(model, tokenizer, use_chat, "pararules", **prepare_pararules(n))
