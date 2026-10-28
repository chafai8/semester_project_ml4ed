#!/usr/bin/env python
"""How many rows must be dropped if the SAME rows have to be removed in every language (parallel data)?

For each flag column (NeedsReview, and NeedsReviewStrict if strict_flags.py was run) it counts the rows that are flagged
in AT LEAST ONE language of the parallel set. The headline numbers EXCLUDE Swahili (--exclude, default: sw), i.e. the
parallel set is fr, de, zh, el, id, te; Swahili is then handled on its own. The numbers with all 7 languages are printed
as well.

Usage:  python union_drop.py                       # reads work/final_v2
        python union_drop.py --final-dir work/final      # the first (unrepaired) run
        python union_drop.py --exclude sw te            # leave out more languages
"""
import argparse
from pathlib import Path

import pandas as pd

from common import LANGS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-dir", default="work/final_v2")
    ap.add_argument("--exclude", nargs="*", default=["sw"], help="languages left out of the parallel set")
    args = ap.parse_args()
    final = Path(args.final_dir)

    for fname in ("train.csv", "misconception_mapping.csv"):
        frames = {}
        for lang in LANGS:
            p = final / lang / fname
            if p.exists():
                frames[lang] = pd.read_csv(p)
        if not frames:
            print(f"{fname}: nothing found in {final}")
            continue
        n = len(next(iter(frames.values())))
        print(f"\n=== {fname} ({n} rows per language)")
        for col in ("NeedsReview", "NeedsReviewStrict"):
            if not all(col in df for df in frames.values()):
                continue
            flags = pd.DataFrame({lang: df[col].astype(bool).values for lang, df in frames.items()})
            par = flags.drop(columns=[c for c in args.exclude if c in flags.columns])
            drop_par = int(par.any(axis=1).sum())
            drop_all = int(flags.any(axis=1).sum())
            print(f"{col}:")
            print("  flagged per language:", {k: int(v) for k, v in flags.sum().items()})
            print(f"  >>> parallel set {list(par.columns)} (without {args.exclude}): drop {drop_par} of {n} rows "
                  f"({100 * drop_par / n:.1f}%), keep {n - drop_par}")
            print(f"      all {flags.shape[1]} languages: drop {drop_all} rows ({100 * drop_all / n:.1f}%), keep {n - drop_all}")
            print(f"      separate per-language filtering: drop {int(flags.values.sum())} of {n * flags.shape[1]} rows in total")
            others = [c for c in args.exclude if c in flags.columns]
            for c in others:
                print(f"      {c} on its own: drop {int(flags[c].sum())} of {n} rows")


if __name__ == "__main__":
    main()
