#!/usr/bin/env python
"""Step 2: translate the unique English strings with google/translategemma-27b-it served by vLLM (offline).

TranslateGemma has a fixed chat template (roles user/assistant only; the user content is ONE entry
with type/text/source_lang_code/target_lang_code), so we cannot add instructions such as
"do not touch the placeholders".  Instead every string goes through up to four passes and the first
valid result wins:

  1. placeholder : LaTeX/tables/image-syntax replaced by [[n]] (or <xn/>), greedy decoding
  2. raw         : original string, validated by comparing the multiset of LaTeX spans
  3. lines       : (multi-line strings only) translate line by line, falling back to the English line
  4. sampled     : placeholder mode again with temperature 0.5, 4 samples, first valid one wins
Strings that still fail get status "review" (kept, but flagged for the judges / manual check).

Resumable: results are appended to <out-dir>/<lang>.jsonl chunk by chunk; re-running skips finished sids.

Examples
  # CPU-only sanity check of the language codes (no GPU, no vLLM needed):
  python translate_vllm.py --check-langs

  # pilot on 1 GPU:
  python translate_vllm.py --langs fr de zh el id sw te --pilot-ids work/pilot_ids.json \
         --out-dir work/pilot/translations

  # full run, 1 x 80GB GPU is enough (27B bf16 ~ 54GB).  For 40GB GPUs use --tp 2.
  python translate_vllm.py --langs fr de zh el id sw te --out-dir work/translations
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

from common import (LANGS, SRC_CANDIDATES, append_jsonl, formula_problems, has_words, ph_ids,
                    protect, read_jsonl, restore, split_ws)


# --------------------------------------------------------------------------- #
# Prompt rendering (uses the model's own chat template -> exact official format)
# --------------------------------------------------------------------------- #
def make_messages(text, src, tgt):
    return [{"role": "user",
             "content": [{"type": "text", "source_lang_code": src,
                          "target_lang_code": tgt, "text": text}]}]


class Renderer:
    def __init__(self, model):
        from transformers import AutoProcessor
        self.proc = AutoProcessor.from_pretrained(model)
        self.tok = self.proc.tokenizer

    def prompt_str(self, text, src, tgt):
        return self.proc.apply_chat_template(make_messages(text, src, tgt),
                                             tokenize=False, add_generation_prompt=True)

    def ids(self, text, src, tgt):
        # The rendered string already contains <bos>, so do NOT let the tokenizer add another one.
        ids = self.tok(self.prompt_str(text, src, tgt), add_special_tokens=False)["input_ids"]
        bos = self.tok.bos_token_id
        if bos is not None and (not ids or ids[0] != bos):
            ids = [bos] + ids
        return ids


def resolve_code(rend, candidates, other, is_target):
    last = None
    for c in candidates:
        try:
            if is_target:
                rend.prompt_str("Hello", other, c)
            else:
                rend.prompt_str("Hello", c, other)
            return c
        except Exception as e:  # unsupported codes raise when the template is applied
            last = e
    raise RuntimeError(f"none of {candidates} accepted by the chat template (last error: {last})")


# --------------------------------------------------------------------------- #
# vLLM wrapper
# --------------------------------------------------------------------------- #
class Gen:
    def __init__(self, args, rend):
        from vllm import LLM, SamplingParams
        self.SP = SamplingParams
        self.rend = rend
        self.max_new = args.max_new_tokens
        self.llm = LLM(
            model=args.model,
            dtype="bfloat16",
            tensor_parallel_size=args.tp,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem,
            limit_mm_per_prompt={"image": 0},   # text only -> saves memory
            enable_prefix_caching=True,
            seed=args.seed,
        )

    def __call__(self, texts, src, tgt, n=1, temperature=0.0):
        prompts = [{"prompt_token_ids": self.rend.ids(t, src, tgt)} for t in texts]
        sp = self.SP(temperature=temperature, top_p=0.95 if temperature > 0 else 1.0, n=n,
                     max_tokens=self.max_new, skip_special_tokens=True)
        outs = self.llm.generate(prompts, sp, use_tqdm=True)
        return [[(c.text.strip(), c.finish_reason) for c in o.outputs] for o in outs]


# --------------------------------------------------------------------------- #
# The four passes.  Each takes a list of source strings and returns, per string,
# {"translation": str, "problems": [..]}   (problems == [] means valid)
# --------------------------------------------------------------------------- #
def _basic_problems(raw, fin):
    p = []
    if fin == "length":
        p.append("truncated")
    if not raw:
        p.append("empty")
    return p


def pass_placeholder(gen, texts, src, tgt, style, n=1, temperature=0.0):
    prepared = []
    for t in texts:
        lead, core, trail = split_ws(t)
        ptxt, spans = protect(core, style)
        prepared.append((lead, trail, ptxt, spans, core))
    gens = gen([p[2] for p in prepared], src, tgt, n=n, temperature=temperature)
    results = []
    for (lead, trail, ptxt, spans, core), cands in zip(prepared, gens):
        first = None
        for raw, fin in cands:
            problems = _basic_problems(raw, fin)
            restored, p2 = restore(raw, spans, style)
            problems += p2
            if not p2:
                problems += formula_problems(core, restored)    # e.g. a dropped "&" in a table
            res = {"translation": lead + restored + trail, "problems": problems}
            first = first or res
            if not problems:
                first = res
                break
        results.append(first)
    return results


def pass_raw(gen, texts, src, tgt):
    prepared = [split_ws(t) for t in texts]
    gens = gen([p[1] for p in prepared], src, tgt)
    results = []
    for t, (lead, core, trail), cands in zip(texts, prepared, gens):
        raw, fin = cands[0]
        problems = _basic_problems(raw, fin) + formula_problems(core, raw)
        results.append({"translation": lead + raw + trail, "problems": problems})
    return results


def pass_lines(gen, texts, src, tgt, style):
    """Translate each text line separately (placeholders are global, so multi-line math stays intact)."""
    prepared, jobs = [], []
    for ti, t in enumerate(texts):
        lead, core, trail = split_ws(t)
        ptxt, spans = protect(core, style)
        lines = ptxt.split("\n")
        prepared.append((lead, trail, spans, lines))
        for li, line in enumerate(lines):
            if has_words(line, style):
                jobs.append((ti, li))
    gens = gen([split_ws(prepared[ti][3][li])[1] for ti, li in jobs], src, tgt) if jobs else []
    failed = {ti: [] for ti in range(len(texts))}
    for (ti, li), cands in zip(jobs, gens):
        raw, fin = cands[0]
        lead_l, core_l, trail_l = split_ws(prepared[ti][3][li])
        problems = _basic_problems(raw, fin)
        if sorted(ph_ids(raw, style)) != sorted(ph_ids(core_l, style)):
            problems.append("line_ph_mismatch")
        if problems:
            failed[ti].append(f"line{li}:{','.join(problems)}")   # keep the English line
        else:
            prepared[ti][3][li] = lead_l + raw + trail_l
    results = []
    for ti, (lead, trail, spans, lines) in enumerate(prepared):
        restored, p2 = restore("\n".join(lines), spans, style)
        results.append({"translation": lead + restored + trail, "problems": failed[ti] + p2})
    return results


# --------------------------------------------------------------------------- #
def run_chunk(gen, texts, src, tgt, style):
    n = len(texts)
    final = [None] * n
    best = [None] * n            # (rank, mode, result) fallback if everything fails
    rank = {"placeholder": 0, "sampled": 0, "raw": 1, "lines": 2}
    pending = list(range(n))

    passes = [
        ("placeholder", lambda ts: pass_placeholder(gen, ts, src, tgt, style), lambda t: True),
        ("raw", lambda ts: pass_raw(gen, ts, src, tgt), lambda t: True),
        ("lines", lambda ts: pass_lines(gen, ts, src, tgt, style), lambda t: "\n" in t.strip()),
        ("sampled", lambda ts: pass_placeholder(gen, ts, src, tgt, style, n=4, temperature=0.5),
         lambda t: True),
    ]
    for mode, fn, applicable in passes:
        idx = [i for i in pending if applicable(texts[i])]
        if not idx:
            continue
        for i, res in zip(idx, fn([texts[i] for i in idx])):
            if not res["problems"]:
                final[i] = {"status": "ok", "mode": mode, **res}
            elif best[i] is None or rank[mode] > best[i][0]:
                best[i] = (rank[mode], mode, res)
        pending = [i for i in pending if final[i] is None]

    for i in pending:
        _, mode, res = best[i] if best[i] else (0, "none", {"translation": texts[i], "problems": ["no_attempt"]})
        final[i] = {"status": "review", "mode": mode, **res}
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/translategemma-27b-it")
    ap.add_argument("--source", default="work/source_strings.jsonl")
    ap.add_argument("--out-dir", default="work/translations")
    ap.add_argument("--langs", nargs="+", default=list(LANGS))
    ap.add_argument("--pilot-ids", default=None, help="json list of sids; translate only these")
    ap.add_argument("--limit", type=int, default=None, help="debug: only the first N strings")
    ap.add_argument("--chunk", type=int, default=1500, help="strings per checkpoint")
    ap.add_argument("--ph-style", choices=["br", "xml"], default="br")
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size (1 on an 80GB GPU)")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check-langs", action="store_true",
                    help="only check which language codes the chat template accepts, then exit")
    args = ap.parse_args()

    rend = Renderer(args.model)
    src_code = resolve_code(rend, SRC_CANDIDATES, "fr", is_target=False)
    codes = {}
    for lang in args.langs:
        try:
            codes[lang] = resolve_code(rend, LANGS[lang]["candidates"], src_code, is_target=True)
        except RuntimeError as e:
            print(f"[{lang}] {e}")
    print(f"source code: {src_code} | target codes: {codes}")
    missing = [l for l in args.langs if l not in codes]
    if args.check_langs:
        print("OK" if not missing else f"UNSUPPORTED: {missing}")
        return
    if missing:
        raise SystemExit(f"unsupported languages {missing}; fix LANGS candidates in common.py")

    recs = [r for r in read_jsonl(args.source) if r["needs_translation"]]
    if args.pilot_ids:
        keep = set(json.loads(Path(args.pilot_ids).read_text()))
        recs = [r for r in recs if r["sid"] in keep]
    if args.limit:
        recs = recs[:args.limit]
    # long strings last so the first chunks finish (and checkpoint) fast
    recs.sort(key=lambda r: len(r["text"]))
    print(f"{len(recs)} strings x {len(codes)} languages")

    gen = Gen(args, rend)
    out_dir = Path(args.out_dir)
    for lang, code in codes.items():
        path = out_dir / f"{lang}.jsonl"
        done = {r["sid"] for r in read_jsonl(path)}
        todo = [r for r in recs if r["sid"] not in done]
        print(f"\n=== {lang} ({LANGS[lang]['name']}, code={code}): {len(todo)} to do, {len(done)} already done")
        t0 = time.time()
        for k in range(0, len(todo), args.chunk):
            batch = todo[k:k + args.chunk]
            res = run_chunk(gen, [r["text"] for r in batch], src_code, code, args.ph_style)
            rows = []
            for r, o in zip(batch, res):
                rows.append({"sid": r["sid"], "lang": lang, "code": code, "status": o["status"],
                             "mode": o["mode"], "translation": o["translation"],
                             "problems": o["problems"]})
            append_jsonl(path, rows)
            c = Counter((r["status"], r["mode"]) for r in rows)
            print(f"  chunk {k // args.chunk + 1}: {dict(c)}  [{time.time() - t0:.0f}s elapsed]")
        (out_dir / f"run_meta_{lang}.json").write_text(json.dumps({
            "model": args.model, "lang": lang, "target_code": code, "source_code": src_code,
            "ph_style": args.ph_style, "max_new_tokens": args.max_new_tokens,
        }, indent=1))


if __name__ == "__main__":
    main()
    # All results are on disk. vLLM's engine shutdown can hang and keep the GPU reserved, so leave immediately.
    import os, sys
    print("== all languages finished; exiting", flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
