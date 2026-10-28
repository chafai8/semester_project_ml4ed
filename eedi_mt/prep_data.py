#!/usr/bin/env python
"""Step 1: extract every unique English string from the two Eedi CSVs.

Outputs (in --out-dir):
  source_strings.jsonl  one row per unique string: sid, text, fields, primary, needs_translation, flags
  pilot_ids.json        a stratified pilot sample (list of sids) that deliberately includes the
                        awkward cases: tables, image descriptions, \\text{words} in math, display math,
                        and the longest strings (to catch truncation).

Strings with no natural-language word left after protecting LaTeX (e.g. "\\( 29.157 \\)") are marked
needs_translation=False and are copied verbatim into every language.

Usage:
  python prep_data.py --data-dir data --out-dir work --pilot 150
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from common import (FIELD_OF_COL, FIELD_PRIORITY, MISC_TEXT_COL, TRAIN_TEXT_COLS,
                    _TEXT_IN_MATH, needs_translation, protect, sid, write_jsonl)


def primary_field(fields):
    for f in FIELD_PRIORITY:
        if f in fields:
            return f
    return sorted(fields)[0]


def build_pilot(recs, n, seed):
    rng = random.Random(seed)
    cand = [r for r in recs if r["needs_translation"]]
    chosen = {}

    def take(pred, k):
        pool = [r for r in cand if pred(r) and r["sid"] not in chosen]
        rng.shuffle(pool)
        for r in pool[:k]:
            chosen[r["sid"]] = r

    k = max(6, n // 25)
    take(lambda r: r["flags"]["tabular"], k)
    take(lambda r: r["flags"]["image"], k)
    take(lambda r: r["flags"]["text_in_math"], k)
    take(lambda r: r["flags"]["display_math"], k)
    for r in sorted(cand, key=lambda r: -len(r["text"]))[:max(4, n // 40)]:
        chosen.setdefault(r["sid"], r)

    shares = {"question": 0.45, "answer": 0.20, "misconception": 0.20, "construct": 0.10, "subject": 0.05}
    for f, share in shares.items():
        have = sum(1 for r in chosen.values() if r["primary"] == f)
        take(lambda r, f=f: r["primary"] == f, max(0, round(n * share) - have))
    return list(chosen)[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="work")
    ap.add_argument("--pilot", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data, out = Path(args.data_dir), Path(args.out_dir)
    tr = pd.read_csv(data / "train.csv")
    mm = pd.read_csv(data / "misconception_mapping.csv")

    fields: dict[str, set] = {}
    for col in TRAIN_TEXT_COLS:
        for v in tr[col].dropna().astype(str):
            fields.setdefault(v, set()).add(FIELD_OF_COL[col])
    for v in mm[MISC_TEXT_COL].dropna().astype(str):
        fields.setdefault(v, set()).add(FIELD_OF_COL[MISC_TEXT_COL])

    recs = []
    for text, fs in fields.items():
        _, spans = protect(text, "br")
        recs.append({
            "sid": sid(text),
            "text": text,
            "fields": sorted(fs),
            "primary": primary_field(fs),
            "needs_translation": needs_translation(text),
            "n_spans": len(spans),
            "flags": {
                "tabular": "\\begin{tabular}" in text,
                "image": "![" in text,
                "display_math": "\\[" in text,
                "text_in_math": bool(_TEXT_IN_MATH.search(text)),
                "newline": "\n" in text,
            },
        })
    recs.sort(key=lambda r: r["sid"])
    assert len({r["sid"] for r in recs}) == len(recs), "sid collision (extremely unlikely) - widen sid()"

    write_jsonl(out / "source_strings.jsonl", recs)
    pilot = build_pilot(recs, args.pilot, args.seed)
    (out / "pilot_ids.json").write_text(json.dumps(pilot))

    need = [r for r in recs if r["needs_translation"]]
    print(f"unique strings            : {len(recs)}")
    print(f"  need translation        : {len(need)}")
    print(f"  copied verbatim (math)  : {len(recs) - len(need)}")
    print("  needing translation, by primary field:", dict(Counter(r["primary"] for r in need)))
    for flag in ("tabular", "image", "display_math", "text_in_math", "newline"):
        print(f"  with {flag:<13}: {sum(r['flags'][flag] for r in need)}")
    chars = sum(len(r["text"]) for r in need)
    print(f"  characters to translate : {chars}  (~{chars // 3} source tokens per language)")
    print(f"pilot sample              : {len(pilot)} strings -> {out / 'pilot_ids.json'}")


if __name__ == "__main__":
    main()
