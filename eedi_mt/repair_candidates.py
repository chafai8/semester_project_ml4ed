#!/usr/bin/env python
"""Repair loop, step R1 (cluster, TranslateGemma, 1 GPU): extra translation candidates for the flagged strings.

For every string in work/repair/todo.jsonl it produces several DIFFERENT, VALID translations:
  * placeholder mode, sampled (T=0.8, --n-ph samples)      formulas protected as [[n]]
  * raw mode, sampled (T=0.8, --n-raw samples)             formulas left in the text, checked by formula signature
  * hinted greedy: if a rule in repair_hints.json changes the English source, that rewritten source is translated
A candidate is kept only if it is non-empty, not truncated, formula-identical to the ORIGINAL source (candidate_problems),
in the right script and different from the current translation and from the other candidates.

Output: work/repair/cands_tg/<lang>.jsonl  {"sid", "lang", "k", "mode", "translation"}  (k = 1, 2, ...)   (resumable)
"""
import argparse
import time
from pathlib import Path

from common import LANGS, SRC_CANDIDATES, append_jsonl, protect, read_jsonl, restore, split_ws
from repair_common import apply_hints, candidate_problems, load_hints
from translate_vllm import Gen, Renderer, resolve_code, _basic_problems
from common import formula_problems


def ph_samples(gen, texts, src, tgt, n, temp, style="br"):
    """Per text: list of (translation, mode) that restore cleanly (formula check against the text itself)."""
    prepared = []
    for t in texts:
        lead, core, trail = split_ws(t)
        ptxt, spans = protect(core, style)
        prepared.append((lead, trail, spans, core))
    gens = gen([protect(p_[3], style)[0] for p_ in prepared], src, tgt, n=n, temperature=temp)
    out = []
    for (lead, trail, spans, core), cands in zip(prepared, gens):
        keep = []
        for raw, fin in cands:
            if _basic_problems(raw, fin):
                continue
            restored, p2 = restore(raw, spans, style)
            if p2 or formula_problems(core, restored):
                continue
            keep.append((lead + restored + trail, "tg_ph"))
        out.append(keep)
    return out


def raw_samples(gen, texts, src, tgt, n, temp):
    prepared = [split_ws(t) for t in texts]
    gens = gen([p_[1] for p_ in prepared], src, tgt, n=n, temperature=temp)
    out = []
    for (lead, core, trail), cands in zip(prepared, gens):
        keep = []
        for raw, fin in cands:
            if _basic_problems(raw, fin) or formula_problems(core, raw):
                continue
            keep.append((lead + raw + trail, "tg_raw"))
        out.append(keep)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/translategemma-27b-it")
    ap.add_argument("--todo", default="work/repair/todo.jsonl")
    ap.add_argument("--out-dir", default="work/repair/cands_tg")
    ap.add_argument("--hints", default="repair_hints.json")
    ap.add_argument("--langs", nargs="+", default=list(LANGS))
    ap.add_argument("--n-ph", type=int, default=6)
    ap.add_argument("--n-raw", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-cands", type=int, default=8, help="keep at most this many distinct candidates per string")
    ap.add_argument("--chunk", type=int, default=800)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    todo = read_jsonl(args.todo)
    if not todo:
        raise SystemExit(f"{args.todo} is empty or missing - run prep_repair.py on the laptop and upload work/repair")
    hints = load_hints(args.hints)
    rend = Renderer(args.model)
    src_code = resolve_code(rend, SRC_CANDIDATES, "fr", is_target=False)
    gen = Gen(args, rend)

    for lang in args.langs:
        rows = [r for r in todo if r["lang"] == lang]
        if not rows:
            continue
        code = resolve_code(rend, LANGS[lang]["candidates"], src_code, is_target=True)
        path = Path(args.out_dir) / f"{lang}.jsonl"
        done = {r["sid"] for r in read_jsonl(path)}
        rows = [r for r in rows if r["sid"] not in done]
        print(f"\n=== {lang} (code={code}): {len(rows)} strings to do, {len(done)} candidate rows already on disk", flush=True)
        t0 = time.time()
        for c0 in range(0, len(rows), args.chunk):
            batch = rows[c0:c0 + args.chunk]
            srcs = [r["source"] for r in batch]
            ph = ph_samples(gen, srcs, src_code, code, args.n_ph, args.temperature)
            rw = raw_samples(gen, srcs, src_code, code, args.n_raw, args.temperature)
            hinted = [apply_hints(s, lang, hints) for s in srcs]
            hi = [i for i, (h, s) in enumerate(zip(hinted, srcs)) if h != s]
            hres = {}
            if hi:
                for i, k in zip(hi, ph_samples(gen, [hinted[i] for i in hi], src_code, code, 1, 0.0)):
                    hres[i] = [(t, "tg_hint") for t, _ in k]
            out = []
            for i, r in enumerate(batch):
                seen = {r["translation"].strip()}
                cands = []
                for t, mode in hres.get(i, []) + ph[i] + rw[i]:
                    if t.strip() in seen or candidate_problems(r["source"], t, lang):
                        continue
                    seen.add(t.strip())
                    cands.append((t, mode))
                    if len(cands) >= args.max_cands:
                        break
                out += [{"sid": r["sid"], "lang": lang, "k": k, "mode": m, "translation": t}
                        for k, (t, m) in enumerate(cands, 1)]
                if not cands:       # marker so that the string counts as done on resume
                    out.append({"sid": r["sid"], "lang": lang, "k": 0, "mode": "none", "translation": ""})
            append_jsonl(path, out)
            got = sum(1 for o in out if o["k"] > 0)
            print(f"  chunk {c0 // args.chunk + 1}: {len(batch)} strings -> {got} candidates "
                  f"({sum(1 for o in out if o['k'] == 0)} without) [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
    import os, sys
    print("== candidates finished; exiting", flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
