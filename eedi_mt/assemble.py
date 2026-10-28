#!/usr/bin/env python
"""Step 4: rebuild translated train.csv / misconception_mapping.csv per language + automatic checks.

For every language L it writes <out-dir>/L/train.csv and <out-dir>/L/misconception_mapping.csv with the
SAME columns/ids as the originals (text columns replaced by the translation) plus a boolean column
`NeedsReview` (any text field of that row/entry was flagged).  Strings that need no translation
(pure math) are copied verbatim.

It also writes <out-dir>/auto_checks.csv: one row per (lang, sid) with automatic flags:
  status_review   translation pipeline could not produce a validated string
  formula_changed LaTeX / table / image syntax differs from the source
  wrong_script    (zh/el/te) too few letters in the expected script
  length_ratio    translation/source length outside [0.35, 4.5] (strings > 20 chars)
  copied_source   translation identical to the English source although it contains words
  stray_ph        a placeholder marker leaked into the output
  judge_flag      flagged by the LLM judges (if --judge-flags is given)

Usage:
  python assemble.py --langs fr de zh el id sw te --translations-dir work/translations --out-dir work/final \
                     [--judge-flags work/judging/flagged.csv]
"""
import argparse
import re
from pathlib import Path

import pandas as pd

from common import (LANGS, MISC_TEXT_COL, TRAIN_TEXT_COLS, formula_problems, read_jsonl, script_ratio,
                    sid)

# Chinese needs far fewer characters than English, so a correct translation is often well under 0.35 of the source length
LEN_MIN = {"zh": 0.10}

STRAY = re.compile(r"\[\[\s*\d+\s*\]\]|<\s*x\s*\d+\s*/?\s*>")


def auto_flags(src, rec, lang):
    tr = rec["translation"]
    flags = []
    if rec["status"] == "review":
        flags.append("status_review")
    if formula_problems(src, tr):
        flags.append("formula_changed")
    ratio = script_ratio(tr, lang)
    if ratio is not None and ratio < 0.5:
        flags.append("wrong_script")
    if len(src) > 20:
        lr = len(tr) / max(1, len(src))
        if not (LEN_MIN.get(lang, 0.35) <= lr <= 4.5):
            flags.append("length_ratio")
    if tr.strip() == src.strip() and len(src.split()) >= 3:
        flags.append("copied_source")
    if STRAY.search(tr):
        flags.append("stray_ph")
    return flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--source", default="work/source_strings.jsonl")
    ap.add_argument("--translations-dir", default="work/translations")
    ap.add_argument("--out-dir", default="work/final")
    ap.add_argument("--langs", nargs="+", default=list(LANGS))
    ap.add_argument("--judge-flags", default=None, help="flagged.csv from aggregate_judgments.py")
    args = ap.parse_args()

    out = Path(args.out_dir)
    src_by_sid = {r["sid"]: r for r in read_jsonl(args.source)}
    tr0 = pd.read_csv(Path(args.data_dir) / "train.csv")
    mm0 = pd.read_csv(Path(args.data_dir) / "misconception_mapping.csv")

    judge_flagged = set()
    if args.judge_flags and Path(args.judge_flags).exists():
        jf = pd.read_csv(args.judge_flags)
        judge_flagged = set(zip(jf["lang"], jf["sid"]))

    checks = []
    for lang in args.langs:
        recs = {r["sid"]: r for r in read_jsonl(Path(args.translations_dir) / f"{lang}.jsonl")}
        flagged = set()
        for s, rec in recs.items():
            fl = auto_flags(src_by_sid[s]["text"], rec, lang)
            if (lang, s) in judge_flagged:
                fl.append("judge_flag")
            if fl:
                flagged.add(s)
            checks.append({"lang": lang, "sid": s, "status": rec["status"], "mode": rec["mode"],
                           "flags": ";".join(fl), "source": src_by_sid[s]["text"],
                           "translation": rec["translation"]})

        def conv(v):
            if pd.isna(v):
                return v, False
            v = str(v)
            s = sid(v)
            rec = recs.get(s)
            if rec is None:
                missing = src_by_sid.get(s, {}).get("needs_translation", False)
                return v, bool(missing)          # pure-math string copied verbatim, or not translated yet
            return rec["translation"], s in flagged

        tr = tr0.copy()
        review = pd.Series(False, index=tr.index)
        for col in TRAIN_TEXT_COLS:
            pairs = tr0[col].map(conv)
            tr[col] = pairs.map(lambda p: p[0])
            review |= pairs.map(lambda p: p[1])
        tr["NeedsReview"] = review

        mm = mm0.copy()
        pairs = mm0[MISC_TEXT_COL].map(conv)
        mm[MISC_TEXT_COL] = pairs.map(lambda p: p[0])
        mm["NeedsReview"] = pairs.map(lambda p: p[1])

        d = out / lang
        d.mkdir(parents=True, exist_ok=True)
        tr.to_csv(d / "train.csv", index=False)
        mm.to_csv(d / "misconception_mapping.csv", index=False)
        print(f"{lang}: {len(recs)} translated strings | rows needing review: "
              f"train {int(review.sum())}/{len(tr)}, misconceptions {int(mm['NeedsReview'].sum())}/{len(mm)}")

    pd.DataFrame(checks).to_csv(out / "auto_checks.csv", index=False)
    df = pd.DataFrame(checks)
    if len(df):
        df["any_flag"] = df["flags"] != ""
        print("\nflag rate by language:")
        print(df.groupby("lang")["any_flag"].mean().round(3).to_string())


if __name__ == "__main__":
    main()