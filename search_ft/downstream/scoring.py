"""Shared scoring protocol and helpers for the downstream suite.

Final protocol of the paper's Table 4 (see docs/DOWNSTREAM.md):

  answer cue    "assistant"  In chat mode a trailing answer cue in the prompt ("Answer:", "A:",
                             "the answer is", LogiQA's "A: Among A through D, the answer is") is
                             moved out of the user turn and prefilled at the start of the
                             assistant turn, so a label is the model's natural next token.
  label variant "auto"       Score the surface form the model emits at the scored position: the
                             bare label ("Yes") at a chat turn boundary (prefix ends in a newline)
                             or after a "**" cue suffix, the space-prefixed label (" Yes") after a
                             prefilled cue or in completion mode. Scoring only " Yes" under a chat
                             template read tokens about 10 nats below the real candidates.
  length norm   "off"        Summed log-probability; no per-byte normalisation (dividing single-
                             token scores by byte length is a fixed per-label rescaling).
  cue suffix    ""           Optional text prefilled after the cue, e.g. "**" for an untrained
                             Qwen3 chat model that answers in markdown bold ("Answer: **True**").

Every label benchmark runs ONE forward pass per item and reads three predictions from it:
the log-likelihood argmax over the label set (the reported ``accuracy``), the greedy next
token mapped to a label (a format-following diagnostic), and the contextually calibrated
argmax (Zhao et al. 2021: per-label log-probs on a content-free prompt are subtracted first).
"""
import hashlib
import random
import re
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch

# ---------------------------------------------------------------------------
# Protocol settings. Set once per run through configure(); module-level so every
# benchmark function and the content-free baseline share them.
# ---------------------------------------------------------------------------
LABEL_VARIANTS = ("auto", "space", "bare", "max")
ANSWER_CUES = ("assistant", "user")
LENGTH_NORMS = ("off", "auto", "on")

LABEL_VARIANT = "auto"
ANSWER_CUE = "assistant"
CUE_SUFFIX = ""
LENGTH_NORM = "off"
GEN_MAX_NEW_TOKENS = 200   # generation budget of the generation-scored benchmark (NLGraph)

# The cue that is moved into the assistant turn. LogiQA's AGIEval cue line moves as ONE unit;
# matching only "the answer is" would leave "A: Among A through D," in the user turn.
CUE_RE = re.compile(
    r"\n?\s*(A:\s*Among A through D,\s*the answer is|(?:Answer|answer|A)\s*:|the answer is)\s*$")


def configure(label_variant: str = None, answer_cue: str = None, length_norm: str = None,
              cue_suffix: str = None, gen_max_new_tokens: int = None) -> None:
    global LABEL_VARIANT, ANSWER_CUE, LENGTH_NORM, CUE_SUFFIX, GEN_MAX_NEW_TOKENS
    if label_variant is not None:
        if label_variant not in LABEL_VARIANTS:
            raise ValueError(f"label_variant must be one of {LABEL_VARIANTS}")
        LABEL_VARIANT = label_variant
    if answer_cue is not None:
        if answer_cue not in ANSWER_CUES:
            raise ValueError(f"answer_cue must be one of {ANSWER_CUES}")
        ANSWER_CUE = answer_cue
    if length_norm is not None:
        if length_norm not in LENGTH_NORMS:
            raise ValueError(f"length_norm must be one of {LENGTH_NORMS}")
        LENGTH_NORM = length_norm
    if cue_suffix is not None:
        CUE_SUFFIX = cue_suffix
    if gen_max_new_tokens is not None:
        GEN_MAX_NEW_TOKENS = int(gen_max_new_tokens)


def protocol() -> Dict[str, object]:
    return {"label_variant": LABEL_VARIANT, "answer_cue": ANSWER_CUE,
            "length_norm": LENGTH_NORM, "cue_suffix": CUE_SUFFIX,
            "gen_max_new_tokens": GEN_MAX_NEW_TOKENS}


# ---------------------------------------------------------------------------
# Prompt -> scored prefix
# ---------------------------------------------------------------------------
def chat_format(tokenizer, user_content: str) -> str:
    """Apply the chat template (thinking disabled where the template supports it)."""
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True,
        )


def build_prefix(tokenizer, prompt: str, use_chat: bool) -> str:
    """Prompt -> the exact string the model is scored after.

    Completion mode: unchanged. Chat mode with answer cue "user": the whole prompt is the
    user turn. Chat mode with answer cue "assistant": a trailing cue is moved out of the user
    turn and prefilled after the generation prompt, e.g.
        ...<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\nAnswer:
    The prompt's own cue text is kept ("A:" stays "A:"), so the prefill matches the prompt's
    demos. With a cue suffix the suffix follows the cue ("Answer: **"); a prompt without a
    cue (zero-shot ProofWriter) gets "**Answer:" instead, which is the format the untrained
    model writes there ("**Answer: Unknown**").
    """
    if not use_chat:
        return prompt
    if ANSWER_CUE == "assistant":
        m = CUE_RE.search(prompt)
        if m:
            body = prompt[: m.start()].rstrip()
            out = chat_format(tokenizer, body) + m.group(1).strip()
            return out + (" " + CUE_SUFFIX if CUE_SUFFIX else "")
    return chat_format(tokenizer, prompt) + (CUE_SUFFIX + "Answer:" if CUE_SUFFIX else "")


# ---------------------------------------------------------------------------
# Log-likelihood scoring
# ---------------------------------------------------------------------------
@torch.no_grad()
def loglik_completion(model, tokenizer, prefix: str, completion: str) -> float:
    """log p(completion | prefix), robust to BPE merges at the boundary: sums the log-probs
    of the tokens from the first position where tokenize(prefix) and
    tokenize(prefix + completion) diverge (the lm-eval-harness loglikelihood trick)."""
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    full_ids = tokenizer.encode(prefix + completion, add_special_tokens=False)
    split = 0
    while split < len(prefix_ids) and split < len(full_ids) and prefix_ids[split] == full_ids[split]:
        split += 1
    if split == len(full_ids):
        return 0.0  # completion absorbed into the last prefix token
    inp = torch.tensor([full_ids], device=model.device)
    logits = model(inp).logits[0]
    log_probs = torch.log_softmax(logits.float(), dim=-1)  # bf16 logits quantise to ~0.02-0.08 nats
    total = 0.0
    for t in range(split, len(full_ids)):
        total += log_probs[t - 1, full_ids[t]].item()
    return total


def _case_forms(c: str) -> List[str]:
    """Surface spellings scored for one label; the best is kept. The prompt's wording decides
    whether the model writes ' true' or ' True', and the score should not depend on that.
    Single characters keep their exact case (lower-casing 'A' would score the article ' a')
    and are also scored in their parenthesised form ('(B'), which the model prefers after
    "the answer is"."""
    if len(c) <= 1:
        return [c, "(" + c]
    fs = [c]
    for alt in (c.lower(), c.capitalize()):
        if alt not in fs:
            fs.append(alt)
    return fs


@torch.no_grad()
def score_choices(model, tokenizer, prompt: str, choices: Sequence[str], use_chat: bool,
                  baseline_logprobs: Optional[Dict[str, float]] = None) -> Dict[str, object]:
    """One forward pass on ``prompt``; returns
        logprobs        {choice: log p(choice | prefix)} (best surface form per choice)
        loglik_pred     argmax over choices by log-prob (the reported metric)
        gen_pred        the model's greedy next token mapped to a choice, else "<other>"
        calibrated_pred argmax over (logprobs - baseline_logprobs), or None
        greedy_token    the decoded greedy token
        prediction      alias of loglik_pred
    """
    prefix = build_prefix(tokenizer, prompt, use_chat)

    lv = LABEL_VARIANT
    if lv == "auto":
        # At a chat turn boundary the prefix ends "...</think>\n\n" and the next token is the
        # bare label; after a prefilled cue or in completion mode it carries the space. After
        # a "**" cue suffix the label follows the markup directly.
        lv = "bare" if prefix.endswith(("\n", "**")) else "space"
    variants = {"space": [" "], "bare": [""], "max": [" ", ""]}[lv]
    cand = {c: [tokenizer.encode(v + f, add_special_tokens=False)
                for v in variants for f in _case_forms(c)]
            for c in choices}
    all_single = all(len(ids) == 1 for lst in cand.values() for ids in lst)

    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    inp = torch.tensor([prefix_ids], device=model.device)
    logits = model(inp).logits[0, -1, :]
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    if all_single:
        scores = {c: max(log_probs[ids[0]].item() for ids in lst) for c, lst in cand.items()}
    else:
        scores = {c: max(loglik_completion(model, tokenizer, prefix, v + f)
                         for v in variants for f in _case_forms(c))
                  for c in choices}

    # Length normalisation (lm-eval acc_norm). "off" is the paper's setting: dividing by byte
    # length is a fixed per-label rescaling that favours long labels (FOLIO degenerated to
    # the Uncertain base rate). Kept for reproduction of that withdrawn run only.
    if LENGTH_NORM == "on" or (LENGTH_NORM == "auto"
                               and len({len(lst[0]) for lst in cand.values()}) > 1):
        scores = {c: s / max(1, len(c.encode("utf-8"))) for c, s in scores.items()}

    loglik_pred = max(scores, key=scores.get)

    # Greedy prediction: the model's top-1 token, mapped to a label by the FIRST TOKEN ID of
    # the scored forms (a substring match mapped ' grandmother' to 'mother' and ' not' to
    # 'No'). A non-label top-1 counts as wrong, never falls back to the log-lik answer.
    greedy_id = int(torch.argmax(logits).item())
    greedy_token = tokenizer.decode([greedy_id])
    first_ids: Dict[int, set] = {}
    for c, lst in cand.items():
        for ids in lst:
            if ids:
                first_ids.setdefault(ids[0], set()).add(c)
    owners = first_ids.get(greedy_id, set())
    if len(owners) > 1:
        # Labels sharing a first piece (' father' for father / father-in-law, ' upper' for
        # upper-left / upper-right): take ONE more constrained greedy step and resolve by the
        # second piece. Without it such labels could never be credited by the greedy column.
        lg2 = model(torch.tensor([prefix_ids + [greedy_id]], device=model.device)).logits[0, -1, :]
        g2 = int(torch.argmax(lg2).item())
        resolved = set()
        for c in owners:
            for ids in cand[c]:
                if ids and ids[0] == greedy_id:
                    if len(ids) == 1:
                        # a complete single-token label wins only if the continuation is NOT
                        # the second piece of a longer sibling
                        if all(not (len(o) > 1 and o[0] == greedy_id and o[1] == g2)
                               for oc in owners if oc != c for o in cand[oc]):
                            resolved.add(c)
                    elif ids[1] == g2:
                        resolved.add(c)
        owners = resolved
    gen_pred = next(iter(owners)) if len(owners) == 1 else "<other>"

    calibrated_pred = None
    if baseline_logprobs is not None:
        calibrated = {c: scores[c] - baseline_logprobs.get(c, 0.0) for c in choices}
        calibrated_pred = max(calibrated, key=calibrated.get)

    return {"logprobs": scores, "loglik_pred": loglik_pred, "gen_pred": gen_pred,
            "calibrated_pred": calibrated_pred, "greedy_token": greedy_token,
            "prediction": loglik_pred}


def content_free_baseline(model, tokenizer, cf_prompt: str, choices: Sequence[str],
                          use_chat: bool) -> Dict[str, float]:
    """Per-label log-probs on a content-free prompt (task input replaced by "N/A"), the
    class prior subtracted by the calibrated prediction (Zhao et al. 2021)."""
    return score_choices(model, tokenizer, cf_prompt, choices, use_chat)["logprobs"]


# ---------------------------------------------------------------------------
# Generation (NLGraph is scored on generated text)
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate(model, tokenizer, prompt: str, use_chat: bool, max_new_tokens: int = 30) -> str:
    """Greedy continuation of the prompt (whole prompt in the user turn under a chat template)."""
    prefix = chat_format(tokenizer, prompt) if use_chat else prompt
    enc = tokenizer(prefix, return_tensors="pt").to(model.device)
    ids = enc["input_ids"]
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def generate_from_prefix(model, tokenizer, prefix: str, max_new_tokens: int = 30) -> str:
    """Greedy continuation of an exact prefix string (e.g. one built by build_prefix with the
    answer cue prefilled in the assistant turn)."""
    enc = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).to(model.device)
    ids = enc["input_ids"]
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Sampling and few-shot helpers
# ---------------------------------------------------------------------------
def yesno(pred: str) -> str:
    """'yes' / 'no' / '' from a label string, anchored at the start (a substring test mapped
    'Unknown' and 'None' to 'no')."""
    s = str(pred).lower().strip()
    if re.match(r"^(yes|true)\b", s):
        return "yes"
    if re.match(r"^(no|false)\b", s):
        return "no"
    return ""


def subsample(items: Sequence, cap: Optional[int], seed: int = 42) -> list:
    """Up to ``cap`` items, sampled with a fixed seed (order-preserving slices would be
    class-imbalanced: LogicBench files alternate yes/no, ProofWriter groups by question
    type). ``cap=None`` keeps everything."""
    if cap is None:
        return list(items)
    if cap <= 0:
        return []
    if len(items) <= cap:
        return list(items)
    return random.Random(seed).sample(list(items), cap)


def distribute_n(n: Optional[int], num_subsets: int, sizes: Optional[List[int]] = None) -> list:
    """Split a per-benchmark cap ``n`` across subsets: min(n // k, size) each, leftover from
    small subsets redistributed round-robin to those with headroom. ``n=None`` means the
    whole pool (requires ``sizes``)."""
    if num_subsets <= 0:
        return []
    if n is None:
        if sizes is None:
            return [None] * num_subsets
        return list(sizes)
    if sizes is None:
        base = n // num_subsets
        rem = n - base * num_subsets
        return [base + (1 if i < rem else 0) for i in range(num_subsets)]
    caps = [min(sizes[i], n // num_subsets) for i in range(num_subsets)]
    leftover = n - sum(caps)
    while leftover > 0:
        progress = False
        for i in range(num_subsets):
            if caps[i] < sizes[i]:
                caps[i] += 1
                leftover -= 1
                progress = True
                if leftover == 0:
                    break
        if not progress:
            break
    return caps


def fewshot_block_from_pool(pool: Sequence, k: int, prompt_fn: Callable, gold_fn: Callable,
                            seed: int = 1234, balance_by_gold: bool = True):
    """(demo block, set of id() of the picked examples). Demos are drawn from a seeded shuffle
    of ``pool`` with a true round-robin over gold values (bucket the shuffled pool by gold,
    cycle over the golds taking one each until k are picked; empty buckets are skipped), so a
    binary set gives 3/2 at k=5 and a four-way set gives distinct letters at k=3."""
    rng = random.Random(seed)
    shuffled = list(pool)
    rng.shuffle(shuffled)
    picked = []
    if balance_by_gold:
        buckets, order = {}, []
        for ex in shuffled:
            g = gold_fn(ex)
            if g is None:
                continue
            if g not in buckets:
                buckets[g] = []
                order.append(g)
            buckets[g].append(ex)
        while len(picked) < k and any(buckets[g] for g in order):
            for g in order:
                if buckets[g]:
                    picked.append((buckets[g].pop(0), g))
                    if len(picked) >= k:
                        break
    else:
        for ex in shuffled:
            g = gold_fn(ex)
            if g is None:
                continue
            picked.append((ex, g))
            if len(picked) >= k:
                break
    blocks = [f"{prompt_fn(ex)} {g}" for ex, g in picked]
    used = {id(ex) for ex, _ in picked}
    return "\n\n".join(blocks) + ("\n\n" if blocks else ""), used


def balanced_demos(items: Sequence, label_of: Callable, labels: Sequence[str],
                   per_label: int = 1, seed: int = 42) -> list:
    """``per_label`` items per label, interleaved label by label (A, B, C, A, B, C, ...) so no
    answer dominates and no label sits systematically last."""
    rng = random.Random(seed)
    by = {l: [x for x in items if label_of(x) == l] for l in labels}
    for l in labels:
        rng.shuffle(by[l])
        if len(by[l]) < per_label:
            raise ValueError(f"not enough demos for label {l!r}")
    return [by[l][i] for i in range(per_label) for l in labels]


# ---------------------------------------------------------------------------
# Per-item outcome log (the input of mcnemar.py)
# ---------------------------------------------------------------------------
# Off unless the runner enables it. Records are
#   {"item_id", "subset", "gold", "pred", "correct", "metric"} (+ "score" for NLGraph)
# one per reported accuracy column, so sum(correct)/len(records) per (subset, metric) equals
# the aggregate with the same numerator and denominator. item_id is derived ONLY from the
# dataset example (its own id where it has one, else sha1 over the raw fields), never from
# the model-formatted prompt or the demos, so two arms' files pair exactly.
ITEM_LOG: Optional[Dict[str, list]] = None


def enable_item_log() -> None:
    global ITEM_LOG
    ITEM_LOG = {}


def pop_item_log(bench: str) -> Optional[list]:
    if ITEM_LOG is None:
        return None
    return ITEM_LOG.pop(bench, [])


def item_hash(*parts) -> str:
    """sha1 over dataset-provided fields."""
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8", "surrogatepass"))
        h.update(b"\x1f")
    return h.hexdigest()


def log_item(bench, subset, item_id, gold, pred, correct, metric, **extra) -> None:
    if ITEM_LOG is None:
        return
    rec = {"item_id": str(item_id), "subset": str(subset),
           "gold": None if gold is None else str(gold),
           "pred": None if pred is None else str(pred),
           "correct": bool(correct), "metric": metric}
    rec.update(extra)
    ITEM_LOG.setdefault(bench, []).append(rec)


def log_item_fused(bench, subset, item_id, gold, ll_pred, gen_pred, cal_pred) -> None:
    """One record per reported column of the fused scorer; "accuracy" duplicates the
    log-likelihood column so the default metric of mcnemar.py pairs on every benchmark."""
    if ITEM_LOG is None:
        return
    log_item(bench, subset, item_id, gold, ll_pred, ll_pred == gold, "loglik_accuracy")
    log_item(bench, subset, item_id, gold, gen_pred, gen_pred == gold, "gen_accuracy")
    log_item(bench, subset, item_id, gold, cal_pred, cal_pred == gold, "calibrated_accuracy")
    log_item(bench, subset, item_id, gold, ll_pred, ll_pred == gold, "accuracy")


# ---------------------------------------------------------------------------
# Stratified evaluation shared by the thirteen label benchmarks
# ---------------------------------------------------------------------------
class _Counts:
    def __init__(self):
        self.total = 0
        self.ll = self.gen = self.cal = 0
        self.golds = Counter()
        self.per_class_ll = Counter()
        self.extras = Counter()

    def add(self, gold, out, extra=None):
        self.total += 1
        self.golds[gold] += 1
        if out["loglik_pred"] == gold:
            self.ll += 1
            self.per_class_ll[gold] += 1
        self.gen += out["gen_pred"] == gold
        self.cal += out["calibrated_pred"] == gold
        if extra:
            for k, v in extra.items():
                self.extras[k] += bool(v)

    def merge(self, other: "_Counts"):
        self.total += other.total
        self.ll += other.ll
        self.gen += other.gen
        self.cal += other.cal
        self.golds.update(other.golds)
        self.per_class_ll.update(other.per_class_ll)
        self.extras.update(other.extras)

    def summary(self) -> Dict[str, object]:
        t = self.total
        d = {"loglik_accuracy": self.ll / t if t else 0.0,
             "gen_accuracy": self.gen / t if t else 0.0,
             "calibrated_accuracy": self.cal / t if t else 0.0,
             "accuracy": self.ll / t if t else 0.0,
             "total": t,
             "majority_baseline": max(self.golds.values()) / t if t else 0.0,
             "per_class": {cls: {"correct": self.per_class_ll[cls], "total": c,
                                 "accuracy": self.per_class_ll[cls] / c if c else 0.0}
                           for cls, c in sorted(self.golds.items(), key=lambda kv: str(kv[0]))}}
        for k, v in self.extras.items():
            d[k] = v / t if t else 0.0
        return d


def eval_strata(model, tokenizer, use_chat: bool, bench: str, strata: Sequence,
                labels: Union[Sequence[str], Callable], fmt: Callable, gold_of: Callable,
                item_id: Callable, demos_block: str = "", cf_prompt: Optional[str] = None,
                extra_of: Optional[Callable] = None) -> Dict[str, object]:
    """Score ``strata`` = ordered [(name, items)] with the fused scorer.

    labels     the label set, or a callable ex -> labels for per-example choice sets
               (ZebraLogic); then the content-free baseline is recomputed per example.
    fmt        ex -> the item's prompt; the scored prompt is ``demos_block + fmt(ex)``.
    cf_prompt  the full content-free prompt (demos included) for calibration, or None.
    extra_of   optional ex, gold -> {name: bool}, averaged per stratum (PrOntoQA's
               polarity rule).
    Returns {stratum: {...}, "overall": {...}}; a single stratum named "overall" returns
    the flat dict. Every stratum carries loglik / gen / calibrated accuracy, ``accuracy``
    (= log-likelihood), ``total``, ``majority_baseline`` and ``per_class``.
    """
    per_example_labels = callable(labels)
    baseline = None
    if cf_prompt is not None and not per_example_labels:
        baseline = content_free_baseline(model, tokenizer, cf_prompt, labels, use_chat)
        print(f"  [{bench}] content-free baseline: " +
              " ".join(f"{l}={baseline[l]:.3f}" for l in labels))
    results, agg = {}, _Counts()
    for name, items in strata:
        cnt = _Counts()
        for ex in items:
            gold = gold_of(ex)
            choices = list(labels(ex)) if per_example_labels else list(labels)
            if gold is None or gold not in choices:
                continue
            bl = baseline
            if cf_prompt is not None and per_example_labels:
                bl = content_free_baseline(model, tokenizer, cf_prompt, choices, use_chat)
            out = score_choices(model, tokenizer, demos_block + fmt(ex), choices, use_chat,
                                baseline_logprobs=bl)
            cnt.add(gold, out, extra_of(ex, gold) if extra_of else None)
            log_item_fused(bench, name, item_id(ex), gold,
                           out["loglik_pred"], out["gen_pred"], out["calibrated_pred"])
        if cnt.total == 0:
            continue
        results[str(name)] = cnt.summary()
        agg.merge(cnt)
        s = results[str(name)]
        print(f"  [{bench}] {name}: ll={cnt.ll}/{cnt.total}={s['loglik_accuracy']:.1%}  "
              f"gen={s['gen_accuracy']:.1%}  cal={s['calibrated_accuracy']:.1%}  "
              f"(majority {s['majority_baseline']:.1%})")
    overall = agg.summary()
    if baseline is not None:
        overall["baseline_logprobs"] = baseline
    if len(strata) == 1 and str(strata[0][0]) == "overall":
        return overall
    results["overall"] = overall
    return results
