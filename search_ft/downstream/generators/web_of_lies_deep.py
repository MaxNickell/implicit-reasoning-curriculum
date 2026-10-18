#!/usr/bin/env python3
"""Web-of-Lies-Deep: BIG-Bench Hard's web_of_lies with a true, unbounded depth axis.

WHY THIS EXISTS
---------------
The published BBH task is a truth-value chain:

    Question: Sherrie tells the truth. Vernell says Sherrie tells the truth.
    Alexis says Vernell lies. Michaela says Alexis tells the truth.
    Elanor says Michaela tells the truth. Does Elanor tell the truth?

A root is asserted outright, then each speaker makes a claim about the previous speaker,
and the question asks the truth value of the last one. Resolving it means composing k
XNORs, where k is the number of "says" statements.

All 250 published items sit at EXACTLY k=4. There is no depth axis. The task generalises
to any k with no ambiguity, and chance is exactly 50% at every depth: the answer is
binary and, because flipping the root inverts the result, a generator can hold Yes/No at
50/50 for any k, so a depth trend cannot be an artifact of a drifting label prior.

SEMANTICS, verified against all 250 published items (250/250 exact):

    value(speaker) = (claim == value(referent))

where claim is True for "tells the truth" and False for "lies". This generator
back-solves: it fixes the target answer first, samples the k claims, then derives the root
value that yields that target. Yes/No is therefore balanced by construction.

FAITHFULNESS. Statements are emitted in chain order, as BBH does; they are NOT shuffled.
Shuffling would add a search problem on top of the composition problem, and the point of
this benchmark is to isolate depth of composition. Surface form and phrasing are taken
from the published set. The published set uses 47 distinct names; the pool is extended
with 18 more ordinary first names to 65 so that every chain up to k=64 uses distinct,
plain names (suffixed names such as "Delbert 2" never appear).

Every item is verified before writing: the chain is re-parsed from the emitted text and
re-evaluated from scratch, and the recovered answer must match the intended one. Items are
textually unique. The paper's file is seed 42, 500 items per k in {1,2,4,8,16,32,48,64}.

USAGE
    python -m search_ft.downstream.generators.web_of_lies_deep \
        [--out $DOWNSTREAM_DATA_DIR/web_of_lies_deep/web_of_lies_deep.jsonl]
"""
import argparse
import json
import os
import random
import re

KS = [1, 2, 4, 8, 16, 32, 48, 64]
PER_K = 500
SEED = 42

# The names used by BBH web_of_lies, extracted from the published test split (47 distinct),
# followed by the 2026-09-22 extension to 65 plain names.
NAMES = [
    "Alejandro", "Alexis", "Amberly", "Andree", "Antwan", "Audrie", "Bernita",
    "Christie", "Conception", "Crista", "Dallas", "Delbert", "Delfina", "Elanor",
    "Fidel", "Fletcher", "Gwenn", "Helene", "Inga", "Jamey", "Jaymie", "Jerry",
    "Jim", "Ka", "Kandi", "Kristian", "Leda", "Lorine", "Maybelle", "Michael",
    "Michaela", "Millicent", "Osvaldo", "Phoebe", "Raymond", "Rashida", "Ryan",
    "Shalonda", "Shaunda", "Sherrie", "Shenna", "Tamika", "Teressa", "Vernell",
    "Vina", "Willian", "Yoland", "Delbert", "Lorine", "Jamey",
    "Bernice", "Clarence", "Dolores", "Ernest", "Francine", "Gilbert", "Harriet",
    "Irving", "Josephine", "Kenneth", "Lucille", "Marvin", "Norma", "Otis",
    "Priscilla", "Quentin", "Rosalind", "Stanley", "Theodora", "Ulysses", "Vivian",
    "Wendell", "Xavier", "Yvonne", "Zelda",
]
NAMES = sorted(set(NAMES))
assert len(NAMES) >= 65, len(NAMES)

TRUTH, LIE = "tells the truth", "lies"


def name_pool(k, rng):
    """k+1 distinct names. Beyond the pool, suffix to keep them distinct (never reached
    with the 65-name pool and k <= 64)."""
    need = k + 1
    pool = NAMES[:]
    rng.shuffle(pool)
    if need <= len(pool):
        return pool[:need]
    out = pool[:]
    rep = 2
    while len(out) < need:
        extra = [f"{n} {rep}" for n in pool]
        rng.shuffle(extra)
        out.extend(extra[: need - len(out)])
        rep += 1
    return out[:need]


def make_item(k, rng, target):
    """One k-hop item with a forced answer: build the claims first, then back-solve the root.
    value(p_i) = (claim_i == value(p_{i-1})) is its own inverse in value(p_{i-1}), so walking
    the relation backwards from the target recovers the root that produces it."""
    people = name_pool(k, rng)
    claims = [rng.random() < 0.5 for _ in range(k)]   # True == "tells the truth"
    v = target
    for i in range(k - 1, -1, -1):
        v = (claims[i] == v)
    root_val = v
    parts = [f"{people[0]} {TRUTH if root_val else LIE}."]
    for i in range(k):
        parts.append(f"{people[i+1]} says {people[i]} {TRUTH if claims[i] else LIE}.")
    body = " ".join(parts)
    q = f"Does {people[k]} tell the truth?"
    return {
        "input": f"Question: {body} {q}",
        "target": "Yes" if target else "No",
        "k_hop": k,
        "_root": people[0], "_last": people[k], "_root_val": root_val,
    }


_ROOT = re.compile(r"(.+?) (tells the truth|lies)")
_SAYS = re.compile(r"(.+?) says (.+?) (tells the truth|lies)")


def verify(item):
    """Re-parse the emitted text and re-evaluate the chain from scratch.
    Returns (answer, steps) or None if the text does not parse as a linear chain."""
    body = item["input"][len("Question: "):].strip()
    qm = re.search(r"Does (.+?) tell the truth\?$", body)
    if not qm:
        return None
    stmts = body[: body.rindex("Does ")].strip()
    sents = [s.strip() for s in stmts.split(".") if s.strip()]
    m = _ROOT.fullmatch(sents[0].strip())
    if not m:
        return None
    cur, prev = (m.group(2) == TRUTH), m.group(1).strip()
    steps = 0
    for s in sents[1:]:
        sm = _SAYS.fullmatch(s.strip())
        if not sm:
            return None
        spk, ref, pol = sm.group(1).strip(), sm.group(2).strip(), sm.group(3)
        if ref != prev:                       # chain must be strictly linear
            return None
        cur = ((pol == TRUTH) == cur)
        prev = spk
        steps += 1
    if qm.group(1).strip() != prev:           # question must ask about the last speaker
        return None
    return ("Yes" if cur else "No"), steps


def build(out_path, ks=KS, n=PER_K, seed=SEED, verbose=True):
    """Write the item file. Deterministic in (ks, n, seed). Returns the item count."""
    rng = random.Random(seed)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    total = 0
    seen_inputs = set()
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        for k in ks:
            yes = no = 0
            for i in range(n):
                target = (i % 2 == 0)                 # exact 50/50 by construction
                for _attempt in range(1000):
                    it = make_item(k, rng, target)
                    if it["input"] not in seen_inputs:
                        break
                else:
                    raise RuntimeError(f"k={k}: could not draw a unique item after 1000 tries")
                seen_inputs.add(it["input"])
                got = verify(it)
                if got is None:
                    raise RuntimeError(f"k={k}: emitted item failed to re-parse")
                ans, steps = got
                if ans != it["target"]:
                    raise RuntimeError(f"k={k}: re-evaluated answer {ans} != intended {it['target']}")
                if steps != k:
                    raise RuntimeError(f"k={k}: re-parsed chain has {steps} steps")
                yes += ans == "Yes"
                no += ans == "No"
                f.write(json.dumps(it) + "\n")
                total += 1
            if verbose:
                print(f"  [web_of_lies_deep] k={k:<4} {n} items | Yes/No {yes}/{no} "
                      f"| last item {len(it['input']):>6} chars")
    os.replace(tmp, out_path)
    if verbose:
        print(f"  [web_of_lies_deep] wrote {total} items -> {out_path}")
    return total


def default_out_path():
    d = os.environ.get("DOWNSTREAM_DATA_DIR")
    if not d:
        raise SystemExit("--out is required when DOWNSTREAM_DATA_DIR is not set")
    return os.path.join(d, "web_of_lies_deep", "web_of_lies_deep.jsonl")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ks", type=int, nargs="+", default=KS)
    ap.add_argument("--n", type=int, default=PER_K, help="items per k (kept Yes/No balanced)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=None,
                    help="output jsonl (default $DOWNSTREAM_DATA_DIR/web_of_lies_deep/web_of_lies_deep.jsonl)")
    args = ap.parse_args()
    build(args.out or default_out_path(), args.ks, args.n, args.seed)


if __name__ == "__main__":
    main()
