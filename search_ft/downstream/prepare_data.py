"""Download or build every benchmark's data without loading a model, and print the scored
pool sizes. Run this once on a node with network access before evaluating on a compute
node; it also generates the two fixed depth sets (PrOntoQA-Deep, Web-of-Lies-Deep).

    DOWNSTREAM_DATA_DIR=... python -m search_ft.downstream.prepare_data [--benchmarks ...] [--n N]
"""
import argparse
import os
import sys
import traceback

from ..config import PathsConfig
from .registry import PREPARERS, list_benchmarks
from . import benchmarks as _  # noqa: F401


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmarks", nargs="+", default=None, help="default: all")
    ap.add_argument("--n", type=int, default=None, help="per-benchmark cap (default: the paper's n)")
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()
    data_dir = args.data_dir or os.environ.get("DOWNSTREAM_DATA_DIR") or PathsConfig().downstream_data_dir
    os.environ["DOWNSTREAM_DATA_DIR"] = data_dir
    print(f"data dir: {data_dir}")
    names = args.benchmarks or list_benchmarks()
    failed = []
    for name in names:
        print(f"\n=== {name}")
        try:
            spec = PREPARERS[name](args.n)
            strata = spec["strata"]
            total = sum(len(v) for _, v in strata)
            print(f"  -> {total} items in {len(strata)} strata: " +
                  ", ".join(f"{k}={len(v)}" for k, v in strata))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            failed.append((name, str(e)))
    if failed:
        print("\nFAILED: " + "; ".join(f"{n}: {e}" for n, e in failed))
        sys.exit(1)
    print("\nall benchmarks prepared")


if __name__ == "__main__":
    main()
