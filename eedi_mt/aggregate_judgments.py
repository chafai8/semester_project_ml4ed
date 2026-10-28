#!/usr/bin/env python
"""Step 3b: combine the judges' scores, flag weak translations, and sample items for human spot checks.

The judges score on a 1-3 scale (3 = good, 2 = noticeable problems, 1 = serious/unusable).
A (lang, sid) is FLAGGED if any of
  - any judge gives overall <= --low-score (default 2)
  - any judge reports severity major/critical
  - any judge says math_intact == false
  - the judges disagree by >= --disagree points on overall (default 2, i.e. a 1 against a 3)
  - a judge's answer could not be parsed (so nobody has verified it)
  - the translation pipeline marked it "review"

Outputs in --out-dir:
  scores.csv                 one row per (lang, sid) with both judges' scores
  summary_by_lang.csv        mean scores, % major/critical, % flagged, judge agreement per language
  flagged.csv                the flagged items with source, translation and both judges' comments
  human_review_sample.csv    OPTIONAL, only written if you pass --human-flagged / --human-random > 0
                             (off by default: no human reviewers are available for this project)
  report.md                  the summary table as markdown

Usage:
  python aggregate_judgments.py --judges qwen3_32b gptoss_120b \
         --judging-dir work/judging_v2 --translations-dir work/translations_v2
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import LANGS, read_jsonl


def qwk(a, b, k=3):
    """Quadratic weighted kappa between two integer rating vectors in 1..k."""
    a = np.asarray(a, int) - 1
    b = np.asarray(b, int) - 1
    if len(a) < 2:
        return float("nan")
    O = np.zeros((k, k))
    for i, j in zip(a, b):
        O[i, j] += 1
    W = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    den = (W * E).sum()
    return float(1 - (W * O).sum() / den) if den > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="+", required=True)
    ap.add_argument("--langs", nargs="+", default=list(LANGS))
    ap.add_argument("--judging-dir", default="work/judging_v2")
    ap.add_argument("--translations-dir", default="work/translations_v2")
    ap.add_argument("--scale-max", type=int, default=3, help="top of the judges' scale (3 since the 1-3 rubric)")
    ap.add_argument("--low-score", type=int, default=2, help="flag a judge's overall score <= this")
    ap.add_argument("--disagree", type=int, default=2, help="flag judges that differ by >= this on overall")
    ap.add_argument("--source", default="work/source_strings.jsonl")
    ap.add_argument("--out-dir", default=None, help="default: <judging-dir>")
    ap.add_argument("--human-flagged", type=int, default=0, help="optional: flagged items per language for a human sheet")
    ap.add_argument("--human-random", type=int, default=0, help="optional: unflagged items per language for a human sheet")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out_dir or args.judging_dir)
    out.mkdir(parents=True, exist_ok=True)

    src = {r["sid"]: r for r in read_jsonl(args.source)}
    frames = []
    for lang in args.langs:
        tr = pd.DataFrame(read_jsonl(Path(args.translations_dir) / f"{lang}.jsonl"))
        if tr.empty:
            continue
        df = tr[["sid", "lang", "status", "mode", "translation"]].copy()
        for j in args.judges:
            rows = read_jsonl(Path(args.judging_dir) / j / f"{lang}.jsonl")
            jd = pd.DataFrame(rows)
            if jd.empty:
                continue
            jd = jd.drop(columns=["lang", "judge"]).add_suffix(f"__{j}").rename(columns={f"sid__{j}": "sid"})
            df = df.merge(jd, on="sid", how="inner")
        frames.append(df)
    if not frames:
        raise SystemExit("no judged data found")
    df = pd.concat(frames, ignore_index=True)
    df["source"] = df["sid"].map(lambda s: src[s]["text"])
    df["field"] = df["sid"].map(lambda s: src[s]["primary"])

    # ---- flags -----------------------------------------------------------
    sev_bad = {"major", "critical"}
    flag = pd.Series(False, index=df.index)
    reasons = pd.Series("", index=df.index)

    def add(mask, why):
        nonlocal flag, reasons
        flag |= mask
        reasons = reasons.where(~mask, reasons + why + ";")

    for j in args.judges:
        ok = df[f"parse_ok__{j}"].fillna(False).astype(bool)
        add(~ok, f"{j}:unparsed")
        add(ok & (df[f"overall__{j}"] <= args.low_score), f"{j}:overall<={args.low_score}")
        add(ok & df[f"severity__{j}"].isin(sev_bad), f"{j}:{'major/critical'}")
        add(ok & (df[f"math_intact__{j}"] == False), f"{j}:math_changed")  # noqa: E712
    if len(args.judges) >= 2:
        a, b = args.judges[:2]
        add((df[f"overall__{a}"] - df[f"overall__{b}"]).abs() >= args.disagree, "judges_disagree")
    add(df["status"] == "review", "pipeline_review")
    df["flagged"] = flag
    df["flag_reasons"] = reasons
    df.to_csv(out / "scores.csv", index=False)

    # ---- summary ---------------------------------------------------------
    rows = []
    for lang, g in df.groupby("lang"):
        row = {"lang": lang, "n": len(g), "pct_flagged": round(100 * g["flagged"].mean(), 1),
               "pct_pipeline_review": round(100 * (g["status"] == "review").mean(), 1)}
        for j in args.judges:
            row[f"overall_{j}"] = round(g[f"overall__{j}"].mean(), 2)
            row[f"acc_{j}"] = round(g[f"accuracy__{j}"].mean(), 2)
            row[f"pct_major_crit_{j}"] = round(100 * g[f"severity__{j}"].isin(sev_bad).mean(), 1)
        if len(args.judges) >= 2:
            a, b = args.judges[:2]
            ok = g[[f"overall__{a}", f"overall__{b}"]].dropna()
            row["qwk_overall"] = round(qwk(ok.iloc[:, 0], ok.iloc[:, 1], k=args.scale_max), 3)
            row["spearman_overall"] = round(ok.iloc[:, 0].corr(ok.iloc[:, 1], method="spearman"), 3)
        rows.append(row)
    summ = pd.DataFrame(rows).sort_values("lang")
    summ.to_csv(out / "summary_by_lang.csv", index=False)
    try:
        report = summ.to_markdown(index=False)      # needs the optional `tabulate` package
    except ImportError:
        report = summ.to_string(index=False)
    (out / "report.md").write_text(report)
    print(summ.to_string(index=False))

    # ---- flagged + human sample -------------------------------------------
    keep = ["lang", "sid", "field", "source", "translation", "status", "flag_reasons"]
    for j in args.judges:
        keep += [f"overall__{j}", f"severity__{j}", f"issue_summary__{j}"]
    df[df["flagged"]][keep].to_csv(out / "flagged.csv", index=False)

    if args.human_flagged or args.human_random:      # optional: off by default (no human reviewers available)
        sample = []
        for lang, g in df.groupby("lang"):
            f = g[g["flagged"]].sample(n=min(args.human_flagged, int(g["flagged"].sum())), random_state=args.seed)
            u = g[~g["flagged"]]
            u = u.sample(n=min(args.human_random, len(u)), random_state=args.seed)
            sample += [f.assign(sample_kind="flagged"), u.assign(sample_kind="unflagged")]
        hs = pd.concat(sample, ignore_index=True)[["sample_kind"] + keep]
        hs["human_score_1to5"] = ""
        hs["human_comment"] = ""
        hs.to_csv(out / "human_review_sample.csv", index=False)
    print(f"\nflagged: {int(df['flagged'].sum())}/{len(df)} -> {out / 'flagged.csv'}")


if __name__ == "__main__":
    main()
