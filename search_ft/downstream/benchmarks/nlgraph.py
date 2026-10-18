"""NLGraph (Wang et al. 2023): eight graph-reasoning tasks in natural language
(connectivity, cycle, shortest_path, flow, hamilton, matching, topology, GNN), scored on
generated text with the benchmark's own 4-shot direct prompts (each task's
prompt/k-shot-prompt.txt, no chain of thought) and a port of its per-task scorers
(evaluation/*.py: yes/no match, Hamiltonian-path validation, shortest-path weight,
max-flow value, matching size, topological order, GNN per-node embeddings with partial
credit). The 1,000-item cap is spread over the eight tasks (the test sets hold 1,000 items
in total, so every item is scored).

Data: the official repository github.com/Arthur-Heng/NLGraph, cloned on first use into
$DOWNSTREAM_DATA_DIR/nlgraph/. The generation budget is scoring.GEN_MAX_NEW_TOKENS (200 in
the paper; --gen-max-new-tokens in the runner). With a --cue-suffix (untrained chat
models) the prompt's own "A:" cue plus the suffix are prefilled in the assistant turn and
markdown asterisks are stripped before scoring; the default path leaves the whole prompt
in the user turn.

Caveat recorded with the benchmark: NLGraph is the one benchmark on which the curriculum
scored significantly BELOW the no-curriculum control in the paper's paired test.
"""
import json
import os
import re
import urllib.request
from typing import Dict, List, Optional, Tuple

from ..registry import register, register_prepare
from ..datautil import require_data_dir, clone_repo
from .. import scoring
from ..scoring import (generate, generate_from_prefix, build_prefix, subsample, distribute_n,
                       item_hash, log_item)

TASKS = ["connectivity", "cycle", "shortest_path", "flow", "hamilton", "matching", "topology", "GNN"]
DEFAULT_N = 1000
REPO = "https://github.com/Arthur-Heng/NLGraph.git"

_KSHOT_CACHE = {}


def ensure_repo() -> str:
    """Returns the path of the NLGraph/ data subdirectory inside the clone."""
    base = os.path.join(require_data_dir(), "nlgraph", "NLGraph")
    # a full checkout is 640 MB of graph files; the suite needs the test sets and the k-shot prompts
    out = clone_repo(REPO, base, os.path.join("NLGraph", "*", "test.json"), "nlgraph",
                     sparse=["/NLGraph/*/test.json", "/NLGraph/*/prompt/k-shot-prompt.txt",
                             "/README.md", "/LICENSE"])
    if out is None:
        raise RuntimeError(f"NLGraph repository not available at {base}")
    return os.path.join(base, "NLGraph")


def load_all() -> List[Dict]:
    base = ensure_repo()
    out = []
    for task in TASKS:
        fpath = os.path.join(base, task, "test.json")
        if not os.path.isfile(fpath):
            continue
        with open(fpath) as f:
            d = json.load(f)
        items = d.values() if isinstance(d, dict) else d
        for ex in items:
            out.append({"task": task, "question": ex.get("question", ""),
                        "answer": ex.get("answer", ""), "difficulty": ex.get("difficulty", "")})
    print(f"  [nlgraph] loaded {len(out)} examples across {len(TASKS)} tasks")
    return out


def kshot_prompt(task: str) -> str:
    """The canonical 4-shot direct prompt of a task (prompt/k-shot-prompt.txt)."""
    if task in _KSHOT_CACHE:
        return _KSHOT_CACHE[task]
    text = None
    p = os.path.join(ensure_repo(), task, "prompt", "k-shot-prompt.txt")
    if os.path.isfile(p):
        with open(p) as f:
            text = f.read()
    if text is None:
        url = f"https://raw.githubusercontent.com/Arthur-Heng/NLGraph/main/NLGraph/{task}/prompt/k-shot-prompt.txt"
        try:
            text = urllib.request.urlopen(url, timeout=20).read().decode("utf-8")
        except Exception as e:
            print(f"  [nlgraph:{task}] k-shot prompt fetch failed: {e}; falling back to 0-shot")
            text = ""
    if text and not text.endswith("\n\n"):
        text = text.rstrip("\n") + "\n\n"
    _KSHOT_CACHE[task] = text
    return text


def _parse_edges(question_text: str) -> List[Tuple[int, int]]:
    return [(int(a), int(b)) for a, b in re.findall(r"\((\d+)\s*,\s*(\d+)\)", question_text)]


def _parse_query_pair(question_text: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"node\s+(\d+).*?node\s+(\d+)", question_text)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _has_yes(text: str) -> bool:
    return "the answer is yes" in text or re.search(r"\byes\b", text[:50]) is not None


def _has_no(text: str) -> bool:
    return "the answer is no" in text or re.search(r"\b(no|not)\b", text[:50]) is not None


def score(task: str, pred_text: str, gold_text: str, question_text: str = "") -> float:
    """Per-task scoring following Arthur-Heng/NLGraph evaluation/*.py."""
    pl = pred_text.lower()
    gl = str(gold_text).lower()

    if task == "connectivity":
        gold_yes = "yes" in gl
        pair = _parse_query_pair(question_text)
        path_phrase = f"there is a path between node {pair[0]} and node {pair[1]}" if pair else ""
        if gold_yes:
            return 1.0 if (("the answer is yes" in pl) or (path_phrase and path_phrase in pl)) else 0.0
        return 1.0 if (("the answer is no" in pl) or
                       (path_phrase and path_phrase not in pl and "the answer is yes" not in pl)) else 0.0

    if task == "cycle":
        return 1.0 if (_has_yes(pl) if "yes" in gl else _has_no(pl)) else 0.0

    if task == "hamilton":
        gold_yes = "yes" in gl
        if not gold_yes:
            return 1.0 if _has_no(pl) else 0.0
        edges = _parse_edges(question_text)
        if not edges:
            return 0.0
        nodes = set()
        for u, v in edges:
            nodes.add(u)
            nodes.add(v)
        n = len(nodes)
        m = re.search(r"the path (?:can be|is)[:\s]*", pl)
        if not m:
            return 0.0
        seq_str = pl[m.end():m.end() + 400]
        path = [int(x) for x in re.findall(r"\d+", seq_str)][:n]
        if len(path) != n or len(set(path)) != n:
            return 0.0
        edge_set = set()
        for u, v in edges:
            edge_set.add((u, v))
            edge_set.add((v, u))
        for i in range(n - 1):
            if (path[i], path[i + 1]) not in edge_set:
                return 0.0
        return 1.0

    if task == "shortest_path":
        gold_w = re.search(r"weight of (\d+)", gl)
        pred_w = re.search(r"weight of (\d+)|total weight[: ]+(\d+)", pl)
        if gold_w and pred_w:
            return 1.0 if int(pred_w.group(1) or pred_w.group(2)) == int(gold_w.group(1)) else 0.0
        return 0.0

    if task == "flow":
        gold_v = re.search(r"is\s+(\d+)", gl)
        pred_v = re.search(r"is\s+(\d+)|flow[: ]+(\d+)", pl)
        if gold_v and pred_v:
            return 1.0 if int(pred_v.group(1) or pred_v.group(2)) == int(gold_v.group(1)) else 0.0
        return 0.0

    if task == "matching":
        gold_n = re.search(r"(\d+)\s+applicants", gl)
        pred_n = re.search(r"(\d+)\s+applicants", pl)
        return 1.0 if (gold_n and pred_n and int(pred_n.group(1)) == int(gold_n.group(1))) else 0.0

    if task == "topology":
        gold_seq = re.findall(r"\d+", gl)
        pred_seq = re.findall(r"\d+", pl[:500])
        return 1.0 if (gold_seq and pred_seq[:len(gold_seq)] == gold_seq) else 0.0

    if task == "GNN":
        gold_pairs = dict(re.findall(r"node\s*(\d+)\s*:\s*\[(\d+\s*,\s*\d+)\]", gl))
        pred_pairs = dict(re.findall(r"node\s*(\d+)\s*:\s*\[(\d+\s*,\s*\d+)\]", pl))
        if not gold_pairs:
            return 0.0
        norm = lambda s: s.replace(" ", "")
        correct = sum(1 for k, v in gold_pairs.items() if k in pred_pairs and norm(pred_pairs[k]) == norm(v))
        return correct / len(gold_pairs)

    return 0.0


@register_prepare("nlgraph")
def prepare(n=None) -> Dict:
    n = DEFAULT_N if n is None else n
    by_task = {}
    for ex in load_all():
        by_task.setdefault(ex["task"], []).append(ex)
    tasks = sorted(by_task)
    caps = distribute_n(n, len(tasks), [len(by_task[t]) for t in tasks])
    strata = [(t, subsample(by_task[t], c)) for t, c in zip(tasks, caps)]
    blocks = {t: kshot_prompt(t) for t in tasks}
    print("  [nlgraph] scored " + " ".join(f"{t}={len(v)}" for t, v in strata) +
          f" | 4-shot prompts {[len(blocks[t]) for t in tasks]} chars")
    return dict(strata=strata, blocks=blocks)


@register("nlgraph")
def eval_nlgraph(model, tokenizer, use_chat: bool, n: int = None) -> Dict:
    spec = prepare(n)
    max_new = scoring.GEN_MAX_NEW_TOKENS
    results = {}
    for task, items in spec["strata"]:
        fs_block = spec["blocks"][task]
        score_sum, total = 0.0, 0
        for ex in items:
            question, gold = ex["question"], ex["answer"]
            prompt = fs_block + question
            if use_chat and scoring.CUE_SUFFIX:
                pred = generate_from_prefix(model, tokenizer, build_prefix(tokenizer, prompt, use_chat),
                                            max_new_tokens=max_new).replace("*", "")
            else:
                pred = generate(model, tokenizer, prompt, use_chat, max_new_tokens=max_new)
            s = score(task, pred, gold, question)
            score_sum += s
            total += 1
            log_item("nlgraph", task, item_hash(task, question), gold, pred, s >= 1.0, "accuracy", score=s)
        avg = score_sum / total if total else 0.0
        results[task] = {"score_sum": score_sum, "total": total, "accuracy": avg}
        print(f"  [nlgraph] {task}: {score_sum:.1f}/{total} = {avg:.1%}")
    all_s = sum(v["score_sum"] for v in results.values())
    all_t = sum(v["total"] for v in results.values())
    results["overall"] = {"score_sum": all_s, "total": all_t, "accuracy": all_s / all_t if all_t else 0.0}
    return results
