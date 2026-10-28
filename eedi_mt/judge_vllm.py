#!/usr/bin/env python
"""Step 3: LLM-as-judge for the machine translations (run once per judge model).

Each (source, translation) pair is scored with an MQM-style rubric on a 1-3 scale and returned as JSON:
  accuracy, fluency, terminology (1-3), math_intact (bool), severity (none|minor|major|critical),
  overall (1-3), issue_summary (short English text).

Results: <out-dir>/<judge-name>/<lang>.jsonl  (resumable).

The records in --translations-dir/<lang>.jsonl need {"sid", "translation"}.  A sid may carry a candidate suffix
"<sid>~<k>" (used by the repair loop to judge several candidate translations of one source string): the source
text is looked up with the part before "~".

Judges used in this project (one judge per job; each job loads ONE model, so give it just the GPUs it needs)
  # judge A: Qwen3-32B, 2 x 80GB.  Thinking is switched off through the chat template.
  python judge_vllm.py --judge-name qwen3_32b --model Qwen/Qwen3-32B --tp 2 \
         --translations-dir work/translations_v2 --out-dir work/judging_v2

  # judge B: gpt-oss-120b (OpenAI, MXFP4, ~65GB; a reasoning model).  1 x 80GB if it fits, otherwise --tp 2.
  # Its answer is "analysis ... final {json}", so it needs a larger --max-tokens, low reasoning effort,
  # and the special tokens kept so that the final channel can be found.
  python judge_vllm.py --judge-name gptoss_120b --model openai/gpt-oss-120b --tp 1 --dtype auto \
         --chat-kwargs '{"reasoning_effort": "low"}' --max-tokens 1500 --keep-special \
         --translations-dir work/translations_v2 --out-dir work/judging_v2
"""
import argparse
import json
import time
from pathlib import Path

from common import FIELD_DESC, LANGS, append_jsonl, read_jsonl

SCALE_MAX = 3

PROMPT = """You are a professional translator and a school-mathematics teacher, fluent in English and {lang}.
Evaluate the machine translation below. It comes from a diagnostic mathematics question dataset.
Text type: {field_desc}.

[SOURCE - English]
{src}

[TRANSLATION - {lang}]
{tr}

Evaluation rules:
- Judge the translation only against the source. Never "fix" the mathematics; wrong answers and misconceptions are intentional and must stay wrong in the same way.
- LaTeX (\\( ... \\), \\[ ... \\]), tables (\\begin{{tabular}}, \\hline, \\\\) and markdown images ![...]() must stay unchanged; only the natural-language words are translated. Set math_intact to false if any formula, number, symbol or table structure was altered, dropped or added.
- Numbers, operators, comparisons (more/less than, at least), negations and quantifiers must be preserved exactly. Any change of these is at least "major".
- Terminology must match what a {lang}-language school mathematics textbook would use.
- Untranslated English words that should have been translated are an error (names, units and symbols may stay).

Scores are integers 1-3: 3 = good (no problem, or only trivial ones), 2 = noticeable problems but still understandable, 1 = serious problems or unusable.
- accuracy: is the full meaning preserved, with nothing added or omitted?
- fluency: is it grammatical and natural {lang}?
- terminology: are the mathematical terms correct and standard?
- severity of the worst error (MQM): none | minor (meaning intact) | major (meaning changed/misleading or wrong term) | critical (opposite meaning, or the question/answer becomes wrong or unusable).
- overall: your overall quality score. Keep it consistent with the severity: critical means overall 1, major means overall at most 2, and overall 3 means no major or critical problem.

Reply with ONLY a JSON object, no other text:
{{"accuracy": <1-3>, "fluency": <1-3>, "terminology": <1-3>, "math_intact": <true|false>, "severity": "<none|minor|major|critical>", "overall": <1-3>, "issue_summary": "<max 25 words in English, empty if no issue>"}}"""

KEYS = ("accuracy", "fluency", "terminology", "overall")
SEVERITIES = {"none", "minor", "major", "critical"}
# gpt-oss (harmony format) answers "analysis ... final <answer>"; depending on skip_special_tokens the final channel
# shows up as one of these markers.  Everything before the last marker is reasoning and may contain braces.
FINAL_MARKERS = ("<|channel|>final<|message|>", "assistantfinal")


def _json_objects(text):
    """Yield every balanced {...} substring of text that json.loads accepts, in order of appearance."""
    n = len(text)
    for i in range(n):
        if text[i] != "{":
            continue
        depth, in_str, esc = 0, False, False
        for j in range(i, n):
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        yield json.loads(text[i:j + 1])
                    except ValueError:
                        pass
                    break


def parse(text):
    for marker in FINAL_MARKERS:
        if marker in text:
            text = text.rsplit(marker, 1)[1]
            break
    found = [d for d in _json_objects(text) if isinstance(d, dict) and "overall" in d]
    if not found:
        return None
    d = found[-1]                      # the last object is the answer; earlier ones may be quoted examples
    try:
        out = {k: int(d[k]) for k in KEYS}
        if not all(1 <= v <= SCALE_MAX for v in out.values()):
            return None
        sev = str(d["severity"]).strip().lower()
        if sev not in SEVERITIES:
            return None
        out["severity"] = sev
        mi = d["math_intact"]
        out["math_intact"] = (mi.strip().lower() == "true") if isinstance(mi, str) else bool(mi)
        out["issue_summary"] = str(d.get("issue_summary", ""))[:300]
        return out
    except (ValueError, KeyError, TypeError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-name", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--langs", nargs="+", default=list(LANGS))
    ap.add_argument("--source", default="work/source_strings.jsonl")
    ap.add_argument("--translations-dir", default="work/translations_v2")
    ap.add_argument("--out-dir", default="work/judging_v2")
    ap.add_argument("--pilot-ids", default=None)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16", help="bfloat16 (Qwen) or auto (gpt-oss ships MXFP4 weights)")
    ap.add_argument("--quantization", default=None, help="e.g. fp8 (vLLM dynamic quantisation)")
    ap.add_argument("--disable-mm", action="store_true", help="for multimodal checkpoints")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, default=300, help="answer length; reasoning models need far more")
    ap.add_argument("--chat-kwargs", default='{"enable_thinking": false}',
                    help='JSON passed to the chat template. Qwen3: {"enable_thinking": false}; '
                         'gpt-oss: {"reasoning_effort": "low"}')
    ap.add_argument("--keep-special", action="store_true",
                    help="keep special tokens in the output (needed to find the final channel of gpt-oss)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    kw = {}
    if args.quantization:
        kw["quantization"] = args.quantization
    if args.disable_mm:
        kw["limit_mm_per_prompt"] = {"image": 0}
    llm = LLM(model=args.model, dtype=args.dtype, tensor_parallel_size=args.tp,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
              enable_prefix_caching=True, seed=args.seed, **kw)
    sp_kw = {"max_tokens": args.max_tokens, "skip_special_tokens": not args.keep_special}
    greedy = SamplingParams(temperature=0.0, **sp_kw)
    retry = SamplingParams(temperature=0.5, top_p=0.95, **sp_kw)
    ctk = json.loads(args.chat_kwargs)

    src = {r["sid"]: r for r in read_jsonl(args.source)}
    keep = set(json.loads(Path(args.pilot_ids).read_text())) if args.pilot_ids else None

    def base(s):
        return s.split("~", 1)[0]

    def run(convs, sp):
        outs = llm.chat(convs, sp, use_tqdm=True, chat_template_kwargs=ctk)
        return [o.outputs[0].text for o in outs]

    for lang in args.langs:
        lname = LANGS[lang]["name"]
        path = Path(args.out_dir) / args.judge_name / f"{lang}.jsonl"
        done = {r["sid"] for r in read_jsonl(path)}
        trs = [r for r in read_jsonl(Path(args.translations_dir) / f"{lang}.jsonl")
               if r["sid"] not in done and (keep is None or base(r["sid"]) in keep)]
        print(f"\n=== judge {args.judge_name} | {lang}: {len(trs)} to judge, {len(done)} done")
        t0 = time.time()
        for k in range(0, len(trs), args.chunk):
            batch = trs[k:k + args.chunk]
            convs = [[{"role": "user", "content": PROMPT.format(
                lang=lname, field_desc=FIELD_DESC[src[base(r["sid"])]["primary"]],
                src=src[base(r["sid"])]["text"], tr=r["translation"])}] for r in batch]
            texts = run(convs, greedy)
            parsed = [parse(t) for t in texts]
            bad = [i for i, p in enumerate(parsed) if p is None]
            for attempt in range(2):                    # re-ask with sampling for unparsable answers
                if not bad:
                    break
                again = run([convs[i] for i in bad], retry)
                for i, t in zip(bad, again):
                    parsed[i] = parse(t)
                bad = [i for i in bad if parsed[i] is None]
            rows = []
            for r, p in zip(batch, parsed):
                row = {"sid": r["sid"], "lang": lang, "judge": args.judge_name, "parse_ok": p is not None}
                row.update(p or {})
                rows.append(row)
            append_jsonl(path, rows)
            print(f"  chunk {k // args.chunk + 1}: {len(rows) - len(bad)}/{len(rows)} parsed "
                  f"[{time.time() - t0:.0f}s elapsed]")


if __name__ == "__main__":
    main()
    # Results are on disk; skip vLLM's (sometimes hanging) engine shutdown so the GPUs are released at once.
    import os, sys
    print("== judging finished; exiting", flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
