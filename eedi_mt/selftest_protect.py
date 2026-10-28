#!/usr/bin/env python
"""Offline self-test of the LaTeX / table / image protection (CPU only, seconds, no model, no GPU).

For every string that will be translated it checks that
  1. protect() -> restore() gives back the ORIGINAL string exactly (nothing lost or reordered),
  2. every placeholder 1..n appears exactly once in the protected text,
  3. the formula signature of the original is unchanged by the round trip,
  4. a few typical model "damages" are detected (placeholder dropped / duplicated / spaced out).
It also prints a few protected examples so you can eyeball them.

Run (after prep_data.py):   python selftest_protect.py --source work/source_strings.jsonl
"""
import argparse
import random
from collections import Counter

from common import (formula_problems, has_words, ph_ids, protect, read_jsonl, restore, split_ws)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="work/source_strings.jsonl")
    ap.add_argument("--show", type=int, default=4, help="number of protected examples to print")
    args = ap.parse_args()

    recs = [r for r in read_jsonl(args.source) if r["needs_translation"]]
    print(f"{len(recs)} strings that need translation")
    bad = Counter()
    examples = []
    n_spans = Counter()
    for r in recs:
        text = r["text"]
        lead, core, trail = split_ws(text)
        prot, spans = protect(core, "br")
        n_spans[min(len(spans), 10)] += 1

        back, probs = restore(prot, spans, "br")
        if lead + back + trail != text:
            bad["roundtrip_differs"] += 1
            examples.append(("roundtrip_differs", text, prot))
        if probs:
            bad["restore_problems"] += 1
        if sorted(ph_ids(prot, "br")) != list(range(1, len(spans) + 1)):
            bad["placeholder_ids"] += 1
            examples.append(("placeholder_ids", text, prot))
        if formula_problems(text, lead + back + trail):
            bad["formula_signature"] += 1
        if not has_words(prot, "br"):
            bad["no_words_left"] += 1

        # simulated damage must be detected
        if spans:
            dropped = prot.replace("[[1]]", "", 1)
            if not restore(dropped, spans, "br")[1]:
                bad["undetected_drop"] += 1
            doubled = prot + " [[1]]"
            if not restore(doubled, spans, "br")[1]:
                bad["undetected_duplicate"] += 1
            spaced = prot.replace("[[1]]", "[[ 1 ]]", 1)
            if restore(spaced, spans, "br")[1]:
                bad["spaced_placeholder_not_tolerated"] += 1

    print("placeholders per string (10 = 10 or more):", dict(sorted(n_spans.items())))
    if bad:
        print("PROBLEMS:", dict(bad))
        for kind, text, prot in examples[:5]:
            print(f"\n--- {kind}\nORIGINAL : {text!r}\nPROTECTED: {prot!r}")
    else:
        print("ALL CHECKS PASSED")

    rng = random.Random(1)
    wanted = [("table", lambda t: "\\begin{tabular}" in t), ("image", lambda t: "![" in t),
              ("text-in-math", lambda t: "\\text" in t and "\\(" in t), ("display math", lambda t: "\\[" in t)]
    for label, pred in wanted[:args.show]:
        pool = [r for r in recs if pred(r["text"])]
        if pool:
            r = rng.choice(pool)
            prot, spans = protect(split_ws(r["text"])[1], "br")
            print(f"\n=== example: {label} ({len(spans)} placeholders)\n{prot}")


if __name__ == "__main__":
    main()
