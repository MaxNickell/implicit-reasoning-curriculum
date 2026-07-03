import os
import re
import json
import subprocess
import urllib.request
from typing import Dict, List, Optional, Tuple

from ..registry import register
from ..scoring import generate, subsample, distribute_n

TASKS = ["connectivity", "cycle", "shortest_path", "flow",
         "hamilton", "matching", "topology", "GNN"]

def _get_data_dir():
    return os.environ.get("DOWNSTREAM_DATA_DIR")
DEFAULT_N = 1000

_KSHOT_CACHE = {}


def _ensure_repo() -> Optional[str]:
    if _get_data_dir() is None:
        return None

    base = os.path.join(_get_data_dir(), "nlgraph", "NLGraph")
    if os.path.isdir(os.path.join(base, "NLGraph", "connectivity")):
        return os.path.join(base, "NLGraph")

    parent = os.path.dirname(base)
    os.makedirs(parent, exist_ok=True)
    try:
        subprocess.check_call(
            ["git", "clone", "--depth=1",
             "https://github.com/Arthur-Heng/NLGraph.git", base],
            cwd=parent, timeout=120
        )
        return os.path.join(base, "NLGraph")
    except Exception as e:
        print(f"  [nlgraph] clone failed: {e}")
        return None


def _load_all() -> Optional[List[Dict]]:
    base = _ensure_repo()
    if base is None:
        from datasets import load_dataset
        try:
            return list(load_dataset("tasksource/nlgraph", split="test"))
        except Exception as e:
            print(f"  [nlgraph] HF fallback load failed: {e}")
            return None

    out = []
    for task in TASKS:
        fpath = os.path.join(base, task, "test.json")
        if not os.path.isfile(fpath):
            continue
        try:
            with open(fpath) as f:
                d = json.load(f)
        except Exception as e:
            print(f"  [nlgraph:{task}] load failed: {e}")
            continue

        items = d.values() if isinstance(d, dict) else d
        for ex in items:
            out.append({
                "task": task,
                "question": ex.get("question", ""),
                "answer": ex.get("answer", ""),
                "difficulty": ex.get("difficulty", ""),
            })

    print(f"  [nlgraph] loaded {len(out)} examples across {len(TASKS)} tasks")
    return out


def _get_kshot_prompt(task: str) -> str:
    if task in _KSHOT_CACHE:
        return _KSHOT_CACHE[task]

    text = None
    base = _ensure_repo()
    if base is not None:
        p = os.path.join(base, task, "prompt", "k-shot-prompt.txt")
        if os.path.isfile(p):
            try:
                with open(p) as f:
                    text = f.read()
            except Exception:
                pass

    if text is None:
        url = (f"https://raw.githubusercontent.com/Arthur-Heng/NLGraph/main/"
               f"NLGraph/{task}/prompt/k-shot-prompt.txt")
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                text = resp.read().decode("utf-8")
        except Exception as e:
            print(f"  [nlgraph:{task}] k-shot prompt fetch failed: {e}; using 0-shot")
            text = ""

    if text and not text.endswith("\n\n"):
        text = text.rstrip("\n") + "\n\n"

    _KSHOT_CACHE[task] = text
    return text


def _parse_edges(question_text: str) -> List[Tuple[int, int]]:
    return [(int(a), int(b)) for a, b in
            re.findall(r"\((\d+)\s*,\s*(\d+)\)", question_text)]


def _parse_query_pair(question_text: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"node\s+(\d+).*?node\s+(\d+)", question_text)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _has_yes(text: str) -> bool:
    return "the answer is yes" in text or re.search(r"\byes\b", text[:50]) is not None


def _has_no(text: str) -> bool:
    return "the answer is no" in text or re.search(r"\b(no|not)\b", text[:50]) is not None


def _score_per_task(task: str, pred_text: str, gold_text: str, question_text: str = "") -> float:
    pl = pred_text.lower()
    gl = str(gold_text).lower()

    if task == "connectivity":
        gold_yes = "yes" in gl
        pair = _parse_query_pair(question_text)
        path_phrase = ""
        if pair:
            path_phrase = f"there is a path between node {pair[0]} and node {pair[1]}"
        if gold_yes:
            return 1.0 if ("the answer is yes" in pl or
                          (path_phrase and path_phrase in pl)) else 0.0
        return 1.0 if ("the answer is no" in pl or
                       (path_phrase and path_phrase not in pl and
                        "the answer is yes" not in pl)) else 0.0

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
            gw = int(gold_w.group(1))
            pw = int(pred_w.group(1) or pred_w.group(2))
            return 1.0 if pw == gw else 0.0
        return 0.0

    if task == "flow":
        gold_v = re.search(r"is\s+(\d+)", gl)
        pred_v = re.search(r"is\s+(\d+)|flow[: ]+(\d+)", pl)
        if gold_v and pred_v:
            gv = int(gold_v.group(1))
            pv = int(pred_v.group(1) or pred_v.group(2))
            return 1.0 if pv == gv else 0.0
        return 0.0

    if task == "matching":
        gold_n = re.search(r"(\d+)\s+applicants", gl)
        pred_n = re.search(r"(\d+)\s+applicants", pl)
        if gold_n and pred_n and int(pred_n.group(1)) == int(gold_n.group(1)):
            return 1.0
        return 0.0

    if task == "topology":
        gold_seq = re.findall(r"\d+", gl)
        pred_seq = re.findall(r"\d+", pl[:500])
        if gold_seq and pred_seq[:len(gold_seq)] == gold_seq:
            return 1.0
        return 0.0

    if task == "GNN":
        gold_pairs = dict(re.findall(r"node\s*(\d+)\s*:\s*\[(\d+\s*,\s*\d+)\]", gl))
        pred_pairs = dict(re.findall(r"node\s*(\d+)\s*:\s*\[(\d+\s*,\s*\d+)\]", pl))
        if not gold_pairs:
            return 0.0
        norm = lambda s: s.replace(" ", "")
        correct = sum(1 for k, v in gold_pairs.items()
                      if k in pred_pairs and norm(pred_pairs[k]) == norm(v))
        return correct / len(gold_pairs)

    return 0.0


@register("nlgraph_gen")
def eval_nlgraph_gen(model, tokenizer, use_chat: bool, n: int = None) -> Dict:
    n = n if n is not None else DEFAULT_N

    ds = _load_all()
    if ds is None:
        return {"error": "NLGraph load failed. Pass --data_dir to the downstream runner."}

    by_task = {}
    for ex in ds:
        by_task.setdefault(ex.get("task", "?"), []).append(ex)

    tasks = sorted(by_task.keys())
    sizes = [len(by_task[t]) for t in tasks]
    caps = distribute_n(n, len(tasks), sizes)

    results = {}
    for task, cap in zip(tasks, caps):
        group = subsample(by_task[task], cap)
        score_sum = 0.0
        total = 0
        fs_block = _get_kshot_prompt(task)

        for ex in group:
            question = ex.get("question", "")
            prompt = fs_block + question
            gold = ex.get("answer", "")
            pred = generate(model, tokenizer, prompt, use_chat, max_new_tokens=200)
            score = _score_per_task(task, pred, gold, question)
            score_sum += score
            total += 1

        avg = score_sum / total if total > 0 else 0.0
        results[task] = {"score_sum": score_sum, "total": total, "accuracy": avg}
        print(f"  [nlgraph_gen] {task}: {score_sum:.1f}/{total} = {avg:.1%}")

    all_s = sum(v["score_sum"] for v in results.values())
    all_t = sum(v["total"] for v in results.values())
    # Macro-average: mean of per-task accuracies (paper: "average of per-task scores")
    task_accs = [v["accuracy"] for v in results.values()]
    macro_avg = sum(task_accs) / len(task_accs) if task_accs else 0.0
    results["overall"] = {
        "accuracy": macro_avg,
        "total": all_t,
    }
    print(f"  [nlgraph_gen] overall (macro): {macro_avg:.1%} across {len(task_accs)} tasks")

    return results
