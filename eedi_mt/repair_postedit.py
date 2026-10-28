#!/usr/bin/env python
"""Repair loop, step R2 (cluster, Qwen3-32B, 2 GPUs): post-edit the current translation of each flagged string.

The model sees the English source, the current translation, the problems found (automatic flags + the first judges'
comments) and, for answer options, the question with all four options (so an ambiguous phrase is read in context).
It must return ONLY a corrected translation, with every LaTeX/table/image span unchanged.  Candidates that change a formula,
leave a placeholder, or are in the wrong script are discarded (candidate_problems), as are identical ones.

Per string: one greedy edit plus --n-sample sampled edits (T=0.7).
Output: work/repair/cands_pe/<lang>.jsonl  {"sid","lang","k","mode":"postedit","translation"}  with k = 101, 102, ...
"""
import argparse
import time
from pathlib import Path

from common import FIELD_DESC, LANGS, append_jsonl, read_jsonl, split_ws
from repair_common import candidate_problems

PROMPT = """You are a professional {lang} translator and a school-mathematics teacher. Below is an English source text from a diagnostic mathematics question dataset ({field_desc}) and its current {lang} translation, which has problems.

[SOURCE - English]
{src}

[CURRENT TRANSLATION - {lang}]
{tr}

Problems found: {problems}
{context}
Write an improved {lang} translation of the SOURCE.
Rules:
- Keep every LaTeX formula (\\( ... \\), \\[ ... \\]), table (\\begin{{tabular}} ... \\\\ \\hline) and markdown image ![...]() exactly as in the source, character for character. Translate only the natural-language words.
- Never fix the mathematics: wrong answers and misconceptions are intentional and must stay wrong in the same way. Keep numbers, comparisons and negations exactly.
- Use the wording of a {lang} school mathematics textbook. Names, units and symbols may stay.
- Do not add explanations, quotes or notes.
Reply with ONLY the translation."""


def context_text(r):
    c = r.get("context")
    if not c:
        return ""
    opts = "\n".join(f"  {k}. {v}" for k, v in c["options"].items())
    return f"\nThis text is one answer option of this question (English):\nQuestion: {c['question']}\nOptions:\n{opts}\n"


def problems_text(r):
    parts = []
    if r["flags"]:
        parts.append("automatic checks: " + ", ".join(r["flags"]))
    for t in r.get("issues") or []:
        parts.append(f"reviewer: {t}")
    return "; ".join(parts) if parts else "judged as low quality"


def clean(text):
    t = text.strip()
    if t.startswith("```") and t.endswith("```"):
        t = t.strip("`").strip()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--todo", default="work/repair/todo.jsonl")
    ap.add_argument("--out-dir", default="work/repair/cands_pe")
    ap.add_argument("--langs", nargs="+", default=list(LANGS))
    ap.add_argument("--n-sample", type=int, default=2)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--max-model-len", type=int, default=6144)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    todo = read_jsonl(args.todo)
    if not todo:
        raise SystemExit(f"{args.todo} is empty or missing - run prep_repair.py on the laptop and upload work/repair")

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, dtype="bfloat16", tensor_parallel_size=args.tp, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, enable_prefix_caching=True, seed=args.seed)
    ctk = {"enable_thinking": False}
    greedy = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    sampled = SamplingParams(temperature=0.7, top_p=0.95, n=args.n_sample, max_tokens=args.max_tokens)

    for lang in args.langs:
        rows = [r for r in todo if r["lang"] == lang]
        if not rows:
            continue
        path = Path(args.out_dir) / f"{lang}.jsonl"
        done = {r["sid"] for r in read_jsonl(path)}
        rows = [r for r in rows if r["sid"] not in done]
        print(f"\n=== post-edit {lang}: {len(rows)} strings, {len(done)} already done", flush=True)
        t0 = time.time()
        for c0 in range(0, len(rows), args.chunk):
            batch = rows[c0:c0 + args.chunk]
            convs = [[{"role": "user", "content": PROMPT.format(
                lang=LANGS[lang]["name"], field_desc=FIELD_DESC[r["field"]], src=r["source"], tr=r["translation"],
                problems=problems_text(r), context=context_text(r))}] for r in batch]
            g = [o.outputs[0].text for o in llm.chat(convs, greedy, use_tqdm=True, chat_template_kwargs=ctk)]
            s = [[c.text for c in o.outputs] for o in llm.chat(convs, sampled, use_tqdm=True, chat_template_kwargs=ctk)]
            out = []
            for r, gt, st in zip(batch, g, s):
                lead, _, trail = split_ws(r["source"])
                seen = {r["translation"].strip()}
                k = 100
                for t in [gt] + st:
                    t = lead + clean(t) + trail
                    if t.strip() in seen or candidate_problems(r["source"], t, lang):
                        continue
                    seen.add(t.strip())
                    k += 1
                    out.append({"sid": r["sid"], "lang": lang, "k": k, "mode": "postedit", "translation": t})
                if k == 100:
                    out.append({"sid": r["sid"], "lang": lang, "k": 0, "mode": "none", "translation": ""})
            append_jsonl(path, out)
            print(f"  chunk {c0 // args.chunk + 1}: {len(batch)} strings -> {sum(1 for o in out if o['k'] > 0)} valid edits "
                  f"[{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
    import os, sys
    print("== post-edit finished; exiting", flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
