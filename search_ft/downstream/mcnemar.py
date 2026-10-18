#!/usr/bin/env python3
"""Paired significance tests between two models' per-item downstream outcomes.

Input: the JSON written by ``python -m search_ft.downstream.runner ... --items-output``, i.e.
    {alias: {benchmark: [{"item_id", "subset", "gold", "pred", "correct", "metric"}, ...]}}
(select a model from a multi-model file with FILE:alias). Within each benchmark the two models'
records are aligned by (item_id, metric); the item sets must be identical or the script stops
with an error that reports the unmatched counts.

Per benchmark with n paired items: n11 = both correct, n00 = both wrong, b = A correct & B wrong,
c = A wrong & B correct, m = b + c, k = min(b, c), X ~ Binomial(m, 1/2).
  acc_A = (n11 + b) / n,  acc_B = (n11 + c) / n,  diff = acc_A - acc_B = (b - c) / n
  McNemar exact two-sided  p = min(1, 2 P[X <= k])  (1 if m = 0); no continuity approximation,
                           identical to scipy.stats.binomtest(b, m, 0.5).pvalue
  mid-p                    p_mid = 2 P[X <= k] - P[X = k]  (= p - P[X = b] whenever b != c)
  naive two-proportion z   z = diff / sqrt(2 pbar (1 - pbar) / n), pbar = (acc_A + acc_B) / 2,
                           p_z = 2 (1 - Phi(|z|)); ignores the pairing, shown for comparison
  95% CI on diff           percentile interval over --boot resamples of the benchmark's items with
                           replacement (drawn as multinomial counts over A-only / B-only / tie,
                           exactly equivalent to resampling item indices)
Average row: mean over benchmarks of diff, CI from the same bootstrap with every benchmark
resampled independently per replicate, p_boot = min(1, 2 (1 + min(#{mean* <= 0}, #{mean* >= 0}))
/ (boot + 1)). Pooled row: all items of all benchmarks concatenated into one benchmark.

Usage:
    python -m search_ft.downstream.mcnemar --a items_curr48.json --b items_nocurr48.json
    python -m search_ft.downstream.mcnemar --selftest
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import binom, norm


class AlignmentError(ValueError):
    """The two models' item sets differ, or a record list is malformed."""


def mcnemar_exact(b, c):
    """(exact two-sided p, mid-p) from the discordant counts b, c."""
    m, k = b + c, min(b, c)
    if m == 0:
        return 1.0, 1.0
    cdf, pmf = float(binom.cdf(k, m, 0.5)), float(binom.pmf(k, m, 0.5))
    return min(1.0, 2.0 * cdf), 2.0 * cdf - pmf


def ztest_p(acc_a, acc_b, n):
    """Naive pooled two-proportion z-test that treats A and B as independent samples."""
    pbar = 0.5 * (acc_a + acc_b)
    se = np.sqrt(2.0 * pbar * (1.0 - pbar) / n)
    return 1.0 if se == 0 else float(2.0 * norm.sf(abs(acc_a - acc_b) / se))


def paired_stats(a, b):
    """Contingency counts, accuracies and p-values for boolean outcome arrays a (model A), b (model B)."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    n = int(a.size)
    n11, n00 = int((a & b).sum()), int((~a & ~b).sum())
    bb, cc = int((a & ~b).sum()), int((~a & b).sum())
    acc_a, acc_b = (n11 + bb) / n, (n11 + cc) / n
    p_exact, p_mid = mcnemar_exact(bb, cc)
    return {"n": n, "n11": n11, "n00": n00, "b": bb, "c": cc, "acc_A": acc_a, "acc_B": acc_b,
            "diff": acc_a - acc_b, "p_mcnemar": p_exact, "p_mid": p_mid, "p_z": ztest_p(acc_a, acc_b, n)}


def boot_diffs(st, rng, boot):
    """`boot` resampled diff values for one benchmark (items with replacement == multinomial counts)."""
    n = st["n"]
    probs = np.array([st["b"], st["c"], n - st["b"] - st["c"]], dtype=float) / n
    draws = rng.multinomial(n, probs, size=boot)
    return (draws[:, 0] - draws[:, 1]) / n


def ci95(x):
    return [float(v) for v in np.percentile(x, [2.5, 97.5])]


def load_model(spec):
    """-> (alias, {benchmark: [records]}) for FILE[:alias]; the whole spec is tried as a path first."""
    path, alias = (spec, None) if os.path.exists(spec) else spec.rpartition(":")[::2]
    if not os.path.exists(path):
        raise SystemExit(f"error: items file not found for '{spec}'")
    blob = json.load(open(path))
    if alias is None:
        if len(blob) != 1:
            raise SystemExit(f"error: {path} holds {len(blob)} models {sorted(blob)}; use FILE:alias")
        alias = next(iter(blob))
    if alias not in blob:
        raise SystemExit(f"error: alias '{alias}' not in {path}; available: {sorted(blob)}")
    return alias, blob[alias]


def choose_metric(requested, model_a, model_b):
    """Metric present in both models: --metric if given, else 'accuracy', else the single shared one."""
    ma, mb = ({r["metric"] for recs in m.values() for r in recs} for m in (model_a, model_b))
    shared, have = sorted(ma & mb), f"A has {sorted(ma)}, B has {sorted(mb)}"
    if requested is not None:
        if requested not in shared:
            raise SystemExit(f"error: metric {requested!r} is not present in both models; {have}")
        return requested
    if "accuracy" in shared:
        return "accuracy"
    if len(shared) == 1:
        return shared[0]
    raise SystemExit(f"error: no default metric (shared metrics {shared}; {have}); pass --metric")


def outcomes(recs, metric, where):
    """item_id -> correct for one model/benchmark restricted to `metric`; duplicate ids are an error."""
    out = {}
    for r in recs:
        if r["metric"] != metric:
            continue
        if r["item_id"] in out:
            raise AlignmentError(f"{where}: duplicate item_id {r['item_id']!r} for metric {metric!r}")
        out[r["item_id"]] = bool(r["correct"])
    return out


def align(recs_a, recs_b, metric, bench, alias_a, alias_b):
    """Boolean outcome arrays of A and B over the shared item ids; unmatched ids are a hard error."""
    oa = outcomes(recs_a, metric, f"{alias_a}/{bench}")
    ob = outcomes(recs_b, metric, f"{alias_b}/{bench}")
    only_a, only_b = sorted(set(oa) - set(ob), key=str), sorted(set(ob) - set(oa), key=str)
    if only_a or only_b:
        raise AlignmentError(
            f"{bench} (metric {metric!r}): item sets differ: {len(oa)} items in A={alias_a}, "
            f"{len(ob)} in B={alias_b}, {len(oa.keys() & ob.keys())} shared, {len(only_a)} only in A, "
            f"{len(only_b)} only in B. Examples only-A={only_a[:5]} only-B={only_b[:5]}")
    ids = sorted(oa, key=str)
    return np.array([oa[i] for i in ids], dtype=bool), np.array([ob[i] for i in ids], dtype=bool)


def compare(model_a, model_b, metric, benchmarks=None, boot=10000, seed=0):
    """Per-benchmark statistics plus the Average and Pooled rows, as a JSON-serialisable dict."""
    (alias_a, recs_a), (alias_b, recs_b) = model_a, model_b
    if benchmarks:
        missing = [k for k in benchmarks if k not in recs_a or k not in recs_b]
        if missing:
            raise SystemExit(f"error: benchmarks {missing} missing from at least one model")
        benches = list(benchmarks)
    else:
        benches = [k for k in recs_a if k in recs_b]
        odd = sorted(set(recs_a) ^ set(recs_b))
        if odd:
            print(f"warning: benchmarks present in only one model, skipped: {odd}", file=sys.stderr)
    rng = np.random.default_rng(seed)
    per, draws, all_a, all_b = {}, [], [], []
    for bench in benches:
        a, b = align(recs_a[bench], recs_b[bench], metric, bench, alias_a, alias_b)
        if a.size == 0:
            print(f"warning: {bench}: no records with metric {metric!r}, skipped", file=sys.stderr)
            continue
        st = paired_stats(a, b)
        d = boot_diffs(st, rng, boot)
        st["ci95"] = ci95(d)
        per[bench] = st
        draws.append(d); all_a.append(a); all_b.append(b)
    if not per:
        raise SystemExit("error: no benchmark with paired items")
    avg = np.mean(draws, axis=0)
    average = {"n_benchmarks": len(per), "diff": float(np.mean([s["diff"] for s in per.values()])),
               "acc_A": float(np.mean([s["acc_A"] for s in per.values()])),
               "acc_B": float(np.mean([s["acc_B"] for s in per.values()])), "ci95": ci95(avg),
               "p_boot": min(1.0, 2.0 * (1 + min(int((avg <= 0).sum()), int((avg >= 0).sum()))) / (boot + 1))}
    pooled = paired_stats(np.concatenate(all_a), np.concatenate(all_b))
    pooled["ci95"] = ci95(boot_diffs(pooled, rng, boot))
    return {"model_A": alias_a, "model_B": alias_b, "metric": metric, "boot": boot, "seed": seed,
            "benchmarks": per, "average": average, "pooled": pooled}


def render_md(res):
    head = ["benchmark", "n", "n11", "n00", "b (A>B)", "c (B>A)", "acc A", "acc B", "diff (pp)",
            "95% CI (pp)", "p McNemar", "p mid", "p z-naive"]
    fmt_p = lambda p: f"{p:.2e}" if p < 1e-4 else f"{p:.4f}"

    def row(name, s):
        cells = [name] + ([s["n"], s["n11"], s["n00"], s["b"], s["c"]] if "n" in s else ["-"] * 5)
        cells += [f"{100 * s['acc_A']:.2f}", f"{100 * s['acc_B']:.2f}", f"{100 * s['diff']:+.2f}",
                  f"[{100 * s['ci95'][0]:+.2f}, {100 * s['ci95'][1]:+.2f}]"]
        cells += [fmt_p(s["p_mcnemar"]), fmt_p(s["p_mid"]), fmt_p(s["p_z"])] if "p_mcnemar" in s else ["-"] * 3
        return "| " + " | ".join(str(c) for c in cells) + " |"

    avg = res["average"]
    lines = [f"A = {res['model_A']}, B = {res['model_B']}, metric = {res['metric']}, diff = acc_A - acc_B, "
             f"bootstrap {res['boot']} resamples (seed {res['seed']})", "",
             "| " + " | ".join(head) + " |", "|:---|" + "---:|" * (len(head) - 1)]
    lines += [row(k, s) for k, s in res["benchmarks"].items()]
    lines += [row(f"Average ({avg['n_benchmarks']} bench)", avg), row("Pooled", res["pooled"]), "",
              f"Average row: mean of per-benchmark diffs; paired bootstrap two-sided p = {fmt_p(avg['p_boot'])}."]
    return "\n".join(lines)


def selftest():
    from scipy.stats import binomtest
    assert mcnemar_exact(0, 0) == (1.0, 1.0)
    for b_, c_ in [(0, 5), (1, 0), (3, 7), (10, 4), (12, 12), (40, 25), (100, 130)]:
        (p, pm), m = mcnemar_exact(b_, c_), b_ + c_
        ref = binomtest(b_, m, 0.5).pvalue                      # (1) exact p == scipy binomtest
        assert abs(p - ref) < 1e-12, f"exact p mismatch at b={b_} c={c_}: {p} vs scipy {ref}"
        if b_ != c_:                                             # (2) mid-p == exact p - P[X = b]
            assert abs(pm - (p - binom.pmf(b_, m, 0.5))) < 1e-12, f"mid-p identity fails at b={b_} c={c_}"
        else:
            assert abs(pm - 1.0) < 1e-12
    # (3) same marginal accuracies (0.70 vs 0.66, n=2000) with nested vs independent error sets
    rng = np.random.default_rng(0)
    n, ka, kb = 2000, 1400, 1320
    a = np.zeros(n, bool); a[:ka] = True
    b_corr = np.zeros(n, bool); b_corr[:kb] = True                   # B's correct set inside A's
    b_ind = np.zeros(n, bool); b_ind[rng.permutation(n)[:kb]] = True  # independent of A
    s_corr, s_ind = paired_stats(a, b_corr), paired_stats(a, b_ind)
    assert s_corr["acc_A"] == s_ind["acc_A"] == 0.70 and s_corr["acc_B"] == s_ind["acc_B"] == 0.66
    assert s_corr["p_mcnemar"] < 1e-3 * s_corr["p_z"], (s_corr["p_mcnemar"], s_corr["p_z"])
    ratio = s_ind["p_mcnemar"] / s_ind["p_z"]
    assert 0.5 < ratio < 2.0, f"independent errors: McNemar/z p ratio {ratio:.3f}"
    # (4) unmatched item sets raise a clear error
    mk = lambda ids, corr: [{"item_id": i, "subset": "s", "gold": "x", "pred": "x", "correct": bool(c),
                             "metric": "accuracy"} for i, c in zip(ids, corr)]
    ra = mk([f"q{i}" for i in range(10)], [True] * 10)
    rb = mk([f"q{i}" for i in range(9)] + ["zzz"], [False] * 10)
    try:
        align(ra, rb, "accuracy", "toy", "A", "B")
        raise AssertionError("unmatched item sets did not raise")
    except AlignmentError as e:
        assert all(t in str(e) for t in ("9 shared", "1 only in A", "1 only in B", "q9", "zzz")), str(e)
    # end-to-end: Pooled counts are the per-benchmark sums, Average is the mean of diffs, CI brackets it
    ma = ("A", {"t1": mk(range(200), a[:200]), "t2": mk(range(300), a[800:1100])})
    mb = ("B", {"t1": mk(range(200), b_ind[:200]), "t2": mk(range(300), b_ind[800:1100])})
    res = compare(ma, mb, choose_metric(None, ma[1], mb[1]), boot=2000, seed=1)
    per = res["benchmarks"].values()
    assert all(res["pooled"][k] == sum(s[k] for s in per) for k in ("n", "n11", "n00", "b", "c"))
    assert abs(res["average"]["diff"] - np.mean([s["diff"] for s in per])) < 1e-12
    assert res["average"]["ci95"][0] <= res["average"]["diff"] <= res["average"]["ci95"][1]
    json.dumps(res); render_md(res)
    print(f"selftest: all checks passed (nested p={s_corr['p_mcnemar']:.2e} vs z {s_corr['p_z']:.4f}; "
          f"independent p={s_ind['p_mcnemar']:.4f} vs z {s_ind['p_z']:.4f})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", metavar="FILE[:alias]", help="model A items file (alias needed if it holds several models)")
    ap.add_argument("--b", metavar="FILE[:alias]", help="model B items file")
    ap.add_argument("--metric", default=None, help="metric name; default 'accuracy' if present, else the single shared one")
    ap.add_argument("--benchmarks", nargs="+", default=None, help="benchmarks to test, in this order (default: all shared)")
    ap.add_argument("--format", choices=["md", "json"], default="md")
    ap.add_argument("--boot", type=int, default=10000, help="bootstrap resamples (default 10000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true", help="run the synthetic-data checks and exit")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not (args.a and args.b):
        ap.error("--a and --b are required (unless --selftest)")
    ma, mb = load_model(args.a), load_model(args.b)
    try:
        res = compare(ma, mb, choose_metric(args.metric, ma[1], mb[1]), args.benchmarks, args.boot, args.seed)
    except AlignmentError as e:
        raise SystemExit(f"error: {e}")
    print(json.dumps(res, indent=2) if args.format == "json" else render_md(res))


if __name__ == "__main__":
    main()
