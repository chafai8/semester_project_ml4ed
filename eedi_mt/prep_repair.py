#!/usr/bin/env python
"""Repair loop, step R0 (laptop): list the flagged strings that should be translated again.

Reads the results of the first run (work/final/auto_checks.csv from assemble.py, work/judging/scores.csv from
aggregate_judgments.py, data/train.csv) and writes work/repair/todo.jsonl, one row per (language, string):
  lang, sid, field, source (English), translation (the current one), flags, issues (the old judges' comments),
  context (for answer options: the question and all four options, so that an ambiguous word like "straight" can be
  disambiguated)

A string is selected if it has at least one automatic/judge flag other than the ones in --skip-flags
(default: copied_source and length_ratio, which are mostly harmless: codes like "IXOO", very short answers).

Usage (from the project folder):
  python prep_repair.py
  python prep_repair.py --langs fr de            # only some languages
Then: .\\rcp\\sync_scratch.ps1 -Direction up, and run the repair steps (see README, section 4).
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from common import LANGS, TRAIN_TEXT_COLS, FIELD_OF_COL, read_jsonl, sid, write_jsonl

ANSWER_COLS = [c for c in TRAIN_TEXT_COLS if FIELD_OF_COL[c] == "answer"]


def build_context(tr0):
    """sid(answer text) -> {"question", "options"} from the first train row where that text is an answer."""
    ctx = {}
    for _, row in tr0.iterrows():
        opts = {c[len("Answer")]: ("" if pd.isna(row[c]) else str(row[c])) for c in ANSWER_COLS}
        q = "" if pd.isna(row["QuestionText"]) else str(row["QuestionText"])
        for c in ANSWER_COLS:
            v = row[c]
            if pd.isna(v):
                continue
            ctx.setdefault(sid(str(v)), {"question": q, "options": opts})
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks", default="work/final/auto_checks.csv")
    ap.add_argument("--scores", default="work/judging/scores.csv", help="old judges' scores (for their comments)")
    ap.add_argument("--source", default="work/source_strings.jsonl")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="work/repair/todo.jsonl")
    ap.add_argument("--langs", nargs="+", default=list(LANGS))
    ap.add_argument("--skip-flags", nargs="*", default=["copied_source", "length_ratio"])
    args = ap.parse_args()

    checks = pd.read_csv(args.checks)
    checks = checks[checks["lang"].isin(args.langs)]
    src = {r["sid"]: r for r in read_jsonl(args.source)}
    ctx = build_context(pd.read_csv(Path(args.data_dir) / "train.csv"))

    issues = {}
    if Path(args.scores).exists():
        sc = pd.read_csv(args.scores)
        cols = [c for c in sc.columns if c.startswith("issue_summary__")]
        for _, r in sc.iterrows():
            texts = [str(r[c]).strip() for c in cols if isinstance(r[c], str) and r[c].strip()]
            if texts:
                issues[(r["lang"], r["sid"])] = texts
    else:
        print(f"note: {args.scores} not found, the old judges' comments will not be passed on")

    skip = set(args.skip_flags)
    rows, by_flag = [], Counter()
    for _, r in checks.iterrows():
        flags = [f for f in str(r["flags"]).split(";") if f] if isinstance(r["flags"], str) else []
        if not (set(flags) - skip):
            continue
        s = r["sid"]
        field = src[s]["primary"]
        rows.append({"lang": r["lang"], "sid": s, "field": field, "source": src[s]["text"],
                     "translation": r["translation"], "flags": flags,
                     "issues": issues.get((r["lang"], s), []),
                     "context": ctx.get(s) if field == "answer" else None})
        by_flag.update(set(flags) - skip)

    write_jsonl(args.out, rows)
    print(f"selected {len(rows)} strings -> {args.out}")
    print(pd.Series(Counter(r["lang"] for r in rows)).reindex(args.langs).fillna(0).astype(int).to_string())
    print("by flag:", dict(by_flag))
    print(f"with context (answer options): {sum(1 for r in rows if r['context'])}")


if __name__ == "__main__":
    main()
