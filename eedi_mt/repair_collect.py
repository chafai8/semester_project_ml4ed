#!/usr/bin/env python
"""Repair loop, step R3 (CPU, cluster or laptop): merge the original translation (k=0) and all candidates for judging.

Reads work/repair/todo.jsonl, work/repair/cands_tg/<lang>.jsonl, work/repair/cands_pe/<lang>.jsonl and writes
work/repair/cand_all/<lang>.jsonl with {"sid": "<sid>~<k>", "lang", "translation", "mode"}; k=0 is the current translation.
judge_vllm.py then judges these exactly like normal translations.
"""
import argparse
from pathlib import Path

from common import LANGS, read_jsonl, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repair-dir", default="work/repair")
    ap.add_argument("--langs", nargs="+", default=list(LANGS))
    args = ap.parse_args()
    rd = Path(args.repair_dir)
    todo = read_jsonl(rd / "todo.jsonl")
    for lang in args.langs:
        rows = [r for r in todo if r["lang"] == lang]
        if not rows:
            continue
        cands = {}
        for sub in ("cands_tg", "cands_pe"):
            for c in read_jsonl(rd / sub / f"{lang}.jsonl"):
                if c["k"] > 0 and c["translation"].strip():
                    cands.setdefault(c["sid"], []).append(c)
        out, n_c, none = [], 0, 0
        for r in rows:
            out.append({"sid": f"{r['sid']}~0", "lang": lang, "translation": r["translation"], "mode": "original"})
            seen = {r["translation"].strip()}
            added = 0
            for c in sorted(cands.get(r["sid"], []), key=lambda c: c["k"]):
                if c["translation"].strip() in seen:
                    continue
                seen.add(c["translation"].strip())
                out.append({"sid": f"{r['sid']}~{c['k']}", "lang": lang, "translation": c["translation"], "mode": c["mode"]})
                added += 1
            n_c += added
            none += added == 0
        write_jsonl(rd / "cand_all" / f"{lang}.jsonl", out)
        print(f"{lang}: {len(rows)} strings, {n_c} candidates, {none} strings without any candidate -> {len(out)} rows to judge")


if __name__ == "__main__":
    main()
