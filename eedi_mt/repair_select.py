#!/usr/bin/env python
"""Repair loop, step R5 (CPU): pick the best candidate per flagged string and write work/translations_v2.

Input : work/repair/judging/<judge>/<lang>.jsonl  (judges on the candidates k=0..)  + work/repair/cand_all + work/translations
Rule  : both judges must have parsed a candidate for it to count.  Quality key (higher is better)
            (-#judges saying the maths changed, -#judges saying major/critical, min overall, sum overall)
        A candidate replaces the current translation only if its key is STRICTLY better than k=0's and it passes the
        automatic guards (candidate_problems).  Ties between candidates: the lower k wins (TranslateGemma before the post-edit).
Output: work/translations_v2/<lang>.jsonl   all records of work/translations; replaced ones get status "ok",
                                             mode "repaired:<mode>", problems [], and the old text in "previous"
        work/repair/selection_report.csv    one row per string: chosen k, mode, keys before/after
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

from common import LANGS, read_jsonl, write_jsonl
from repair_common import candidate_problems

BAD = {"major", "critical"}


def key(js):
    """js: list of judge records of one candidate (all parsed)."""
    return (-sum(1 for j in js if j["math_intact"] is False),
            -sum(1 for j in js if j["severity"] in BAD),
            min(j["overall"] for j in js),
            sum(j["overall"] for j in js))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repair-dir", default="work/repair")
    ap.add_argument("--translations-dir", default="work/translations")
    ap.add_argument("--out-dir", default="work/translations_v2")
    ap.add_argument("--source", default="work/source_strings.jsonl")
    ap.add_argument("--judges", nargs="+", default=["qwen3_32b", "gptoss_120b"])
    ap.add_argument("--langs", nargs="+", default=list(LANGS))
    args = ap.parse_args()
    rd = Path(args.repair_dir)
    src = {r["sid"]: r["text"] for r in read_jsonl(args.source)}
    report, stats = [], []

    for lang in args.langs:
        old = read_jsonl(Path(args.translations_dir) / f"{lang}.jsonl")
        if not old:
            continue
        cand = {c["sid"]: c for c in read_jsonl(rd / "cand_all" / f"{lang}.jsonl")}
        per = {}                      # full sid -> {judge: record}
        for j in args.judges:
            for r in read_jsonl(rd / "judging" / j / f"{lang}.jsonl"):
                if r.get("parse_ok"):
                    per.setdefault(r["sid"], {})[j] = r
        groups = {}
        for full in cand:
            b, k = full.split("~", 1)
            groups.setdefault(b, []).append(int(k))
        chosen = {}
        c = Counter()
        for b, ks in groups.items():
            def kk(k):
                d = per.get(f"{b}~{k}", {})
                return key([d[j] for j in args.judges]) if all(j in d for j in args.judges) else None
            k0 = kk(0)
            base_key = k0 if k0 is not None else (-9, -9, 0, 0)
            best_k, best_key = 0, base_key
            for k in sorted(x for x in ks if x != 0):
                kv = kk(k)
                t = cand[f"{b}~{k}"]["translation"]
                if kv is None or candidate_problems(src[b], t, lang):
                    continue
                if kv > best_key:
                    best_k, best_key = k, kv
            if best_k != 0:
                chosen[b] = cand[f"{b}~{best_k}"]
            c["replaced" if best_k else "kept_original"] += 1
            report.append({"lang": lang, "sid": b, "chosen_k": best_k, "mode": cand[f"{b}~{best_k}"]["mode"],
                           "key_before": base_key, "key_after": best_key,
                           "n_candidates": len(ks) - 1, "original_judged": k0 is not None})
        out = []
        for r in old:
            if r["sid"] in chosen:
                ch = chosen[r["sid"]]
                r = {**r, "previous": r["translation"], "translation": ch["translation"], "status": "ok",
                     "mode": f"repaired:{ch['mode']}", "problems": []}
            out.append(r)
        write_jsonl(Path(args.out_dir) / f"{lang}.jsonl", out)
        stats.append({"lang": lang, "flagged_strings": len(groups), "replaced": c["replaced"], "kept_original": c["kept_original"],
                      "pct_replaced": round(100 * c["replaced"] / max(1, len(groups)), 1)})
    if report:
        with open(rd / "selection_report.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(report[0]))
            w.writeheader()
            w.writerows(report)
    print(f"{'lang':5s}{'flagged':>9s}{'replaced':>10s}{'kept':>7s}{'%repl':>8s}")
    for r in stats:
        print(f"{r['lang']:5s}{r['flagged_strings']:>9d}{r['replaced']:>10d}{r['kept_original']:>7d}{r['pct_replaced']:>8.1f}")
    print(f"\nwrote {args.out_dir}/<lang>.jsonl and {rd / 'selection_report.csv'}")
    print("Languages that were not repaired were not written: copy work/translations/<lang>.jsonl there by hand if needed.")


if __name__ == "__main__":
    main()
