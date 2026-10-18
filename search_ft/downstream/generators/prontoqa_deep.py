#!/usr/bin/env python
"""PrOntoQA-Deep: the official PrOntoQA generator at hops k in {1,2,4,8,16,32,48,64}.

WHY A NEW SET. The released PrOntoQA / PrOntoQA-OOD data stops at 4-5 hops, and the
official generator cannot go deeper as shipped: it samples a random ontology whose depth
is geometric (stop_probability = 1/(hops+2)), from a pool of 17 concept names and 13
property families, and then performs a RANDOM proof walk that stops at any node with a
property (weights 1 : 0.3 : 0.7 for up / positive / negated). The official runner just
retries until the walk happens to have exactly `1 + hops` steps. Measured success rate
with the shipped pools: 22% at 1 hop, 8% at 2 hops, 0/60 at 4+ hops.

WHAT CHANGES (three things, everything else is the official code path):
  1. `generate_theory` is replaced by a deterministic builder that emits exactly the
     structure the official sampler emits for ModusPonens with relevant distractors
     (max_child_count=1 makes the official ontology a chain already): a chain
     c_0 -> c_1 -> ... -> c_k where every c_i (i>=1) has one property (negated w.p. 1/3,
     as in generate_concept_properties) and one DISTRACTOR PARENT d_i carrying one
     property, plus the separate two-node distractor ontology. The chain length is
     exactly k+1 so the walk's start node is unique.
  2. The proof walk's `np.random.choice` is scripted to go up k-1 times and then ask
     about the property of c_1, which is the only sequence with exactly `1 + hops`
     steps; the official code then checks `num_steps == num_deduction_steps` itself.
  3. Larger vocabularies: synthetic "-pus" concept names (registered in the official
     Morphology table so they inflect and parse) and a pool of ~200 English adjectives,
     each used at most once per question, as in the original (families are deleted
     after use there). The 46 original adjectives are included in the pool.

Everything the model sees is rendered AND parse-verified by the official generator
(`parse_sentence` must return the source formula, or it raises). On top of that this
script re-derives every item from the returned FOL formulas: BFS over subsumption rules
from both premises must derive exactly one of {P, not P} for the queried property, the
label must agree, and the depth to the property-bearing concept must be exactly k.

Format: fictional ontology, ModusPonens, relevant distractors, random ordering,
no_adjectives, "True or false:" query. The (query polarity x label) 2x2 is balanced at
exactly n/4 per cell per k (balancing only the label marginal left the rule "query says
'is not' -> False, else True" at about 67%). A separate 8-item demo file at k=2 (4 True /
4 False) is written for the few-shot block.

The official generator (github.com/asaparov/prontoqa) is cloned on first use into
``$DOWNSTREAM_DATA_DIR/prontoqa_src/prontoqa`` at the pinned commit below. The paper's
files are seed 1000 + k per depth and seed 7 for the demos, 500 items per k.

Usage:
  python -m search_ft.downstream.generators.prontoqa_deep            # full: 500/k
  python -m search_ft.downstream.generators.prontoqa_deep --smoke    # 6/k, prints samples
"""
import argparse
import json
import os
import random
import re
import sys
import time

import numpy as np

PRONTOQA_REPO = "https://github.com/asaparov/prontoqa.git"
PRONTOQA_COMMIT = "0a6412b6fddf46324a1cb96e066dd7b3d89b87d6"   # 2025-09-09, the revision used
KS = [1, 2, 4, 8, 16, 32, 48, 64]
PER_K = 500          # items per k, exactly half True
DEMO_K = 2
N_DEMOS = 8          # 4 True / 4 False, official protocol is 8-shot

# ---------------------------------------------------------------------------
# vocabularies
# ---------------------------------------------------------------------------
_ONSETS = ["", "b", "bl", "br", "ch", "cl", "cr", "d", "dr", "f", "fl", "fr", "g", "gl",
           "gr", "h", "j", "k", "kl", "kr", "l", "m", "n", "p", "pl", "pr", "r", "s",
           "sh", "sk", "sl", "sm", "sn", "sp", "st", "str", "sw", "t", "th", "tr", "tw",
           "v", "w", "wr", "y", "z", "zh"]
_VOWELS = ["a", "e", "i", "o", "u"]
_CODAS = ["mpus", "rpus", "lpus"]

# The 13 official families flattened (46 adjectives) come first; the rest are common,
# single-token, non-noun English adjectives. Any that the official Morphology table
# lists as a noun, or that the parser reserves, is dropped at load time.
_OFFICIAL_ADJ = ["blue", "red", "brown", "orange", "small", "large", "metallic", "wooden",
                 "luminous", "liquid", "transparent", "opaque", "nervous", "happy",
                 "feisty", "shy", "bright", "dull", "sweet", "sour", "spicy", "bitter",
                 "floral", "fruity", "earthy", "hot", "cold", "temperate", "kind", "mean",
                 "angry", "amenable", "aggressive", "melodic", "muffled", "discordant",
                 "loud", "slow", "moderate", "fast", "windy", "sunny", "overcast", "rainy",
                 "snowy"]
_EXTRA_ADJ = ["purple", "green", "yellow", "pink", "grey", "black", "white", "golden",
              "silver", "crimson", "scarlet", "violet", "amber", "beige", "tiny", "huge",
              "narrow", "wide", "tall", "short", "thick", "thin", "heavy", "light",
              "dense", "hollow", "solid", "soft", "hard", "smooth", "rough", "sharp",
              "blunt", "sticky", "slippery", "fuzzy", "furry", "scaly", "feathered",
              "hairy", "bald", "wet", "dry", "damp", "moist", "humid", "arid", "frozen",
              "warm", "cool", "lukewarm", "tepid", "fragrant", "smelly", "musty",
              "fresh", "stale", "salty", "savory", "tangy", "creamy", "crunchy", "chewy",
              "crispy", "juicy", "tender", "tough", "ripe", "raw", "cooked", "burnt",
              "quiet", "noisy", "silent", "shrill", "hoarse", "rhythmic", "tuneful",
              "calm", "anxious", "cheerful", "gloomy", "grumpy", "jolly", "moody",
              "timid", "bold", "brave", "cowardly", "gentle", "fierce", "docile", "wild",
              "tame", "friendly", "hostile", "polite", "rude", "honest", "sneaky",
              "clever", "foolish", "wise", "silly", "serious", "playful", "lazy",
              "diligent", "clumsy", "graceful", "nimble", "sluggish", "swift", "quick",
              "hasty", "patient", "restless", "sleepy", "alert", "drowsy", "lively",
              "sturdy", "fragile", "brittle", "flexible", "rigid", "elastic", "springy",
              "shiny", "glossy", "matte", "dim", "dark", "pale", "vivid", "faded",
              "spotted", "striped", "speckled", "dotted", "plain", "fancy", "ornate",
              "rustic", "modern", "ancient", "young", "old", "new", "worn", "pristine",
              "tidy", "messy", "neat", "cluttered", "spacious", "cramped", "cozy",
              "drafty", "airy", "stuffy", "breezy", "misty", "foggy", "cloudy", "stormy",
              "icy", "muddy", "dusty", "sandy", "rocky", "grassy", "leafy", "thorny",
              "prickly", "spiky", "bumpy", "lumpy", "wrinkled", "curly", "wavy",
              "straight", "crooked", "round", "square", "oval", "flat", "steep",
              "shallow", "deep", "broad", "slender", "stout", "plump", "lean", "bony",
              "muscular", "feeble", "mighty", "puny", "robust", "frail", "hardy",
              "sickly", "healthy", "sleek", "scruffy", "shaggy", "sparse", "lush",
              "barren", "fertile", "hungry", "thirsty", "greedy", "generous", "stingy",
              "frugal", "lavish", "humble", "proud", "vain", "modest", "earnest",
              "witty", "dreary", "merry", "somber", "tranquil", "frantic"]
_RESERVED = {"and", "or", "a", "an", "each", "every", "no", "all", "that", "is", "are",
             "so", "since", "as", "however", "therefore", "because", "not", "true",
             "false", "prove"}


def ensure_prontoqa_source(data_dir):
    """Clone the official generator at the pinned commit; returns its directory."""
    from ..datautil import clone_repo
    dest = os.path.join(data_dir, "prontoqa_src", "prontoqa")
    out = clone_repo(PRONTOQA_REPO, dest, "run_experiment.py", "prontoqa_deep", commit=PRONTOQA_COMMIT)
    if out is None:
        raise RuntimeError(f"could not obtain the official PrOntoQA generator at {dest}")
    return out


def _import_generator(prontoqa_dir):
    """Import the official generator in place (it reads bad_patterns.txt from cwd and
    parses sys.argv at module level only under __main__, but is defensive anyway)."""
    os.chdir(prontoqa_dir)
    sys.path.insert(0, prontoqa_dir)
    argv = sys.argv
    sys.argv = ["x"]
    try:
        import run_experiment as R
    finally:
        sys.argv = argv
    return R


def _build_vocab(R):
    names = []
    for on in _ONSETS:
        for v in _VOWELS:
            for coda in _CODAS:
                w = on + v + coda
                if R.morphology.is_any_noun(w) or R.morphology.is_any_noun(w + "es"):
                    continue          # already an official noun (e.g. impus, worpus)
                if w in names:
                    continue
                names.append(w)
    for w in names:
        R.morphology.add_noun(w, w + "es")
    adjs = []
    for w in _OFFICIAL_ADJ + _EXTRA_ADJ:
        if w in adjs or w in _RESERVED or R.morphology.is_any_noun(w):
            continue
        adjs.append(w)
    return names, adjs


# ---------------------------------------------------------------------------
# chain ontology + scripted walk
# ---------------------------------------------------------------------------
class ChainBuilder:
    """Stands in for theory.generate_theory. Two call shapes reach it per question:
    the main ontology (stop_probability = 1/(D+1), require_properties False) and the
    two-node distractor ontology (stop_probability 0.0, require_properties True,
    exactly two names). Both are reproduced deterministically."""

    def __init__(self, R, adjectives):
        self.R = R
        self.all_adjs = adjectives
        self.rng = random.Random(0)
        self.D = None
        self.adjs = []
        self.script = []
        self.active = False

    def prepare(self, D, rng):
        self.D = D
        self.rng = rng
        self.adjs = self.all_adjs[:]
        rng.shuffle(self.adjs)
        # walk: D-2 subsumption hops upward, then the (single) property of c_1
        self.script = [0] * (D - 2) + [1]
        self.active = True

    def _add_prop(self, node):
        adj = self.adjs.pop()
        if self.rng.random() < 1.0 / 3.0:     # generate_concept_properties: 1 : 0.5
            node.negated_properties.append(adj)
        else:
            node.properties.append(adj)

    def __call__(self, names, families, config):
        R = self.R
        if config.require_properties and config.stop_probability == 0.0:
            assert len(names) == 2, len(names)
            root = R.OntologyNode(names[0], None)
            child = R.OntologyNode(names[1], root)
            self._add_prop(child)
            return [root]
        D = self.D
        take = lambda: names.pop(self.rng.randrange(len(names)))
        chain = [R.OntologyNode(take(), None)]      # c_0: no property (as official root)
        distractor_roots = []
        for _ in range(1, D):
            c = R.OntologyNode(take(), chain[-1])   # parents[0] = true parent
            self._add_prop(c)
            d = R.OntologyNode(take(), None)        # distractor parent, parents[1]
            c.parents.append(d)
            d.children.append(c)
            self._add_prop(d)
            chain.append(c)
            distractor_roots.append(d)
        return [chain[0]] + distractor_roots

    def install(self):
        R = self.R
        R.generate_theory = self                      # run_experiment did `from theory import *`
        orig = np.random.choice
        builder = self

        def scripted_choice(a, size=None, replace=True, p=None):
            if builder.active and p is not None and size is None and builder.script \
                    and isinstance(a, (int, np.integer)):
                idx = builder.script.pop(0)
                if idx >= a:
                    raise RuntimeError(f"scripted walk asked for index {idx} of {a}")
                return idx
            return orig(a, size=size, replace=replace, p=p)
        np.random.choice = scripted_choice


# ---------------------------------------------------------------------------
# independent verification from the FOL returned by the generator
# ---------------------------------------------------------------------------
_QUERY_RE = re.compile(r"^True or false: (\w+) is (not )?([a-z]+)\.$")


def verify(R, k, question_text, query, formulas, label):
    """Re-derive label and depth from the FOL formulas + premises."""
    fol = R.fol
    isn = R.morphology.is_noun
    edges = {}        # concept -> [concept]
    props = {}        # concept -> set((adj, sign))
    premises = []     # (entity, concept)
    for f in formulas:
        if type(f) == fol.FOLForAll and type(f.operand) == fol.FOLIfThen:
            ante, cons = f.operand.antecedent, f.operand.consequent
            assert type(ante) == fol.FOLFuncApplication and isn(ante.function), ante
            x = ante.function
            sign = True
            if type(cons) == fol.FOLNot:
                cons, sign = cons.operand, False
            assert type(cons) == fol.FOLFuncApplication, cons
            y = cons.function
            if isn(y):
                assert sign, "negated subsumption not expected"
                edges.setdefault(x, []).append(y)
            else:
                props.setdefault(x, set()).add((y, sign))
        elif type(f) == fol.FOLFuncApplication and len(f.args) == 1 \
                and type(f.args[0]) == fol.FOLConstant:
            premises.append((f.args[0].constant, f.function))
        else:
            raise AssertionError(f"unexpected formula {fol.fol_to_tptp(f)}")
    m = _QUERY_RE.match(query)
    assert m, query
    entity, neg, P = m.group(1), m.group(2) is not None, m.group(3)
    assert not isn(P), f"query about a concept, not a property: {query}"
    assert len(premises) == 2 and all(e == entity for e, _ in premises), premises
    # BFS from both premises
    dist = {}
    frontier = [(c, 0) for _, c in premises]
    for c, d0 in frontier:
        dist[c] = min(d0, dist.get(c, d0))
    i = 0
    while i < len(frontier):
        c, d0 = frontier[i]
        i += 1
        for y in edges.get(c, []):
            if y not in dist or dist[y] > d0 + 1:
                dist[y] = d0 + 1
                frontier.append((y, d0 + 1))
    derived = {}      # (adj, sign) -> min hops (rule applications)
    for c, d0 in dist.items():
        for (adj, sign) in props.get(c, ()):
            derived[(adj, sign)] = min(d0 + 1, derived.get((adj, sign), d0 + 1))
    pos, negd = (P, True) in derived, (P, False) in derived
    assert pos != negd, f"consistency: pos={pos} neg={negd} for {P}"
    literal_true = (P, not neg)              # the literal the query asserts
    expect = "True" if literal_true in derived else "False"
    assert expect == label, f"label {label} but derivation says {expect}: {query}"
    depth = derived[(P, True)] if pos else derived[(P, False)]
    assert depth == k, f"depth {depth} != k={k}"
    # the queried adjective appears on exactly two concepts: the reachable chain node
    # and the official distractor rule "Every <distractor_concept> is [not] P", which
    # carries the OPPOSITE sign on a concept unreachable from either premise
    carriers = {c: [s for a, s in props[c] if a == P] for c in props
                if any(a == P for a, _ in props[c])}
    assert len(carriers) == 2, carriers
    reach = [c for c in carriers if c in dist]
    unreach = [c for c in carriers if c not in dist]
    assert len(reach) == 1 and len(unreach) == 1, (reach, unreach)
    assert carriers[reach[0]] == [pos], carriers
    assert carriers[unreach[0]] == [not pos], carriers
    # sentence count matches formula count (each formula is rendered as one sentence)
    n_sent = question_text.count(". ") + 1
    assert n_sent == len(formulas), (n_sent, len(formulas))
    return depth


# ---------------------------------------------------------------------------
def generate_k(R, builder, names, k, n_items, seed, verbose=False):
    D = k + 1
    rng = random.Random(seed)
    random.seed(seed)
    np.random.seed(seed)
    # BALANCE THE 2x2 OF (query negated) x (label), not just the label marginal: each cell
    # gets n_items/4, so the polarity rule alone scores 50%.
    q, rem = divmod(n_items, 4)
    cells = [(neg, lab) for neg in (False, True) for lab in ("True", "False")]
    want = {c: q + (1 if i < rem else 0) for i, c in enumerate(cells)}
    got = {c: 0 for c in cells}
    items, tries, nones = [], 0, 0
    t0 = time.time()
    while any(got[c] < want[c] for c in cells):
        tries += 1
        builder.prepare(D, rng)
        pool = names[:]
        rng.shuffle(pool)
        out = R.generate_question(D, pool, formula_ordering="random", ontology="fictional",
                                  distractors="relevant", deduction_rule="ModusPonens",
                                  proofs_only=False, dfs_mode="none", proof_width=2,
                                  no_adjectives=True)
        builder.active = False
        if out[0] is None:
            nones += 1
            continue
        question_text, query, formulas, cot, label, _ = out
        assert not builder.script, "scripted walk not fully consumed"
        depth = verify(R, k, question_text, query, formulas, label)
        neg = " is not " in query
        cell = (neg, label)
        if got[cell] >= want[cell]:
            continue
        got[cell] += 1
        items.append({"k_hop": k, "question": question_text, "query": query,
                      "label": label, "query_negated": neg, "chain_of_thought": cot,
                      "n_sentences": len(formulas), "n_words": len(question_text.split()),
                      "depth_verified": depth, "seed": seed, "idx": len(items)})
    if verbose:
        w = [it["n_words"] for it in items]
        print(f"  [prontoqa_deep] k={k:2d}: {len(items)} items, {tries} calls ({nones} None), "
              f"words mean={np.mean(w):.0f} max={max(w)}, {time.time()-t0:.1f}s")
    return items


def build(out_dir, prontoqa_dir, ks=KS, per_k=PER_K, smoke=False):
    """Generate the item and demo files into ``out_dir``. Returns (n_items, n_demos)."""
    R = _import_generator(prontoqa_dir)
    names, adjs = _build_vocab(R)
    builder = ChainBuilder(R, adjs)
    builder.install()
    need_names = 3 * (max(ks) + 1) + 3
    need_adjs = 2 * max(ks) + 1
    print(f"  [prontoqa_deep] vocab: {len(names)} concept names (need {need_names}), "
          f"{len(adjs)} adjectives (need {need_adjs})")
    assert len(names) >= need_names and len(adjs) >= need_adjs

    per_k = 6 if smoke else per_k
    all_items = []
    for k in ks:
        items = generate_k(R, builder, names, k, per_k, seed=1000 + k, verbose=True)
        all_items.extend(items)
        if smoke:
            ex = items[0]
            print(f"    Q: {ex['question'][:400]}{' ...' if len(ex['question']) > 400 else ''}")
            print(f"    {ex['query']}  -> {ex['label']}")
            print(f"    CoT steps: {len(ex['chain_of_thought'])}, sentences: {ex['n_sentences']}")
    demos = generate_k(R, builder, names, DEMO_K, N_DEMOS, seed=7, verbose=True)
    if smoke:
        print("  [prontoqa_deep] smoke OK, nothing written")
        return len(all_items), len(demos)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "prontoqa_deep.jsonl")
    with open(out + ".tmp", "w") as f:
        for it in all_items:
            f.write(json.dumps(it) + "\n")
    os.replace(out + ".tmp", out)
    demo_out = os.path.join(out_dir, "prontoqa_deep_demos.jsonl")
    with open(demo_out + ".tmp", "w") as f:
        for it in demos:
            f.write(json.dumps(it) + "\n")
    os.replace(demo_out + ".tmp", demo_out)
    print(f"  [prontoqa_deep] wrote {len(all_items)} items -> {out}")
    print(f"  [prontoqa_deep] wrote {len(demos)} demos -> {demo_out}")
    return len(all_items), len(demos)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--per-k", type=int, default=PER_K)
    ap.add_argument("--ks", type=int, nargs="*", default=KS)
    ap.add_argument("--out-dir", default=None,
                    help="default $DOWNSTREAM_DATA_DIR/prontoqa_deep")
    ap.add_argument("--prontoqa-dir", default=None,
                    help="checkout of github.com/asaparov/prontoqa; default: cloned into "
                         "$DOWNSTREAM_DATA_DIR/prontoqa_src/prontoqa at the pinned commit")
    args = ap.parse_args()
    data_dir = os.environ.get("DOWNSTREAM_DATA_DIR")
    out_dir = args.out_dir or (os.path.join(data_dir, "prontoqa_deep") if data_dir else None)
    if out_dir is None and not args.smoke:
        raise SystemExit("--out-dir is required when DOWNSTREAM_DATA_DIR is not set")
    prontoqa_dir = args.prontoqa_dir
    if prontoqa_dir is None:
        if data_dir is None:
            raise SystemExit("--prontoqa-dir is required when DOWNSTREAM_DATA_DIR is not set")
        prontoqa_dir = ensure_prontoqa_source(data_dir)
    build(out_dir, prontoqa_dir, ks=args.ks, per_k=args.per_k, smoke=args.smoke)


if __name__ == "__main__":
    main()
