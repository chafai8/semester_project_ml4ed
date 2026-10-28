#!/usr/bin/env python
"""Step 5 (optional): add a stricter review column, `NeedsReviewStrict`, to the final CSVs.

`NeedsReview` (from assemble.py) is deliberately broad: ONE judge, ONE weak score or ONE soft heuristic is enough.
For Swahili that marks about half of the train rows. This script keeps only the strong evidence.

A source string counts as STRICTLY flagged when at least one of these holds:
  * an automatic check that points to a real defect fired:
        status_review, formula_changed, wrong_script, stray_ph          (from work/final_v2/auto_checks.csv)
  * BOTH judges call it major or critical                              (from work/judging_v2/scores.csv)
  * BOTH judges score it <= --low-score overall (default 2 on the 1-3 scale)
  * BOTH judges say the maths was changed

Left out on purpose (too noisy on their own): a single judge's opinion, judges disagreeing, copied_source, length_ratio.

A row of train.csv / misconception_mapping.csv gets NeedsReviewStrict = True if any of its text fields is strictly flagged.
The column is added IN PLACE to work/final_v2/<lang>/train.csv and misconception_mapping.csv, so run this after assemble.py
(re-running assemble.py rebuilds those files and removes the column; just run this script again).
It also writes work/final_v2/strict_summary.csv.

Usage (from the project folder, on your laptop; no GPU, no cluster):
  python strict_flags.py
  python strict_flags.py --langs sw           # only some languages
"""
import argparse
from pathlib import Path

import pandas as pd

from common import LANGS, MISC_TEXT_COL, TRAIN_TEXT_COLS, sid

HARD_AUTO_FLAGS = {"status_review", "formula_changed", "wrong_script", "stray_ph"}
BAD_SEVERITY = {"major", "critical"}


def strict_sids(lang, checks, scores, judges, low_score=2):
    """Set of source-string ids (for this language) with strong evidence of a defect."""
    bad = set()

    c = checks[checks["lang"] == lang]
    for s, fl in zip(c["sid"], c["flags"].fillna("")):
        if HARD_AUTO_FLAGS & set(x for x in fl.split(";") if x):
            bad.add(s)

    g = scores[scores["lang"] == lang]
    if len(judges) >= 2 and not g.empty:
        a, b = judges[:2]
        both_parsed = g[f"parse_ok__{a}"].fillna(False).astype(bool) & g[f"parse_ok__{b}"].fillna(False).astype(bool)
        sev = g[f"severity__{a}"].isin(BAD_SEVERITY) & g[f"severity__{b}"].isin(BAD_SEVERITY)
        low = (g[f"overall__{a}"] <= low_score) & (g[f"overall__{b}"] <= low_score)
        math = (g[f"math_intact__{a}"] == False) & (g[f"math_intact__{b}"] == False)  # noqa: E712
        bad |= set(g.loc[both_parsed & (sev | low | math), "sid"])
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--final-dir", default="work/final_v2")
    ap.add_argument("--scores", default="work/judging_v2/scores.csv")
    ap.add_argument("--judges", nargs="+", default=["qwen3_32b", "gptoss_120b"])
    ap.add_argument("--low-score", type=int, default=2)
    ap.add_argument("--langs", nargs="+", default=list(LANGS))
    args = ap.parse_args()

    final = Path(args.final_dir)
    checks = pd.read_csv(final / "auto_checks.csv")
    scores = pd.read_csv(args.scores)
    tr0 = pd.read_csv(Path(args.data_dir) / "train.csv")
    mm0 = pd.read_csv(Path(args.data_dir) / "misconception_mapping.csv")

    summary = []
    for lang in args.langs:
        d = final / lang
        if not (d / "train.csv").exists():
            print(f"{lang}: no assembled files in {d} - run assemble.py first")
            continue
        bad = strict_sids(lang, checks, scores, args.judges, args.low_score)

        def is_bad(v):
            return (not pd.isna(v)) and sid(str(v)) in bad

        # the ids are computed from the ORIGINAL English values, row by row (same order as the final files)
        tr_flag = pd.Series(False, index=tr0.index)
        for col in TRAIN_TEXT_COLS:
            tr_flag |= tr0[col].map(is_bad)
        mm_flag = mm0[MISC_TEXT_COL].map(is_bad)

        tr = pd.read_csv(d / "train.csv")
        mm = pd.read_csv(d / "misconception_mapping.csv")
        if len(tr) != len(tr0) or len(mm) != len(mm0):
            raise SystemExit(f"{lang}: row counts differ from the original data - run assemble.py again")
        tr["NeedsReviewStrict"] = tr_flag.values
        mm["NeedsReviewStrict"] = mm_flag.values
        tr.to_csv(d / "train.csv", index=False)
        mm.to_csv(d / "misconception_mapping.csv", index=False)

        row = {"lang": lang,
               "strict_flagged_strings": len(bad),
               "train_rows_strict": int(tr_flag.sum()), "train_rows": len(tr),
               "train_rows_broad": int(tr["NeedsReview"].sum()) if "NeedsReview" in tr else None,
               "misc_rows_strict": int(mm_flag.sum()), "misc_rows": len(mm),
               "misc_rows_broad": int(mm["NeedsReview"].sum()) if "NeedsReview" in mm else None}
        summary.append(row)
        print(f"{lang}: strictly flagged strings {len(bad)} | train rows {row['train_rows_strict']}/{row['train_rows']} "
              f"(broad: {row['train_rows_broad']}) | misconception rows {row['misc_rows_strict']}/{row['misc_rows']} "
              f"(broad: {row['misc_rows_broad']})")

    if summary:
        pd.DataFrame(summary).to_csv(final / "strict_summary.csv", index=False)
        print(f"\nwrote {final / 'strict_summary.csv'}")


if __name__ == "__main__":
    main()
