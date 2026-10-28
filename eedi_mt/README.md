# Eedi multilingual translation + LLM-judge pipeline

Part of the semester project **"Cross-Lingual On-Policy Self-Distillation for Educational Tasks"**.

The English Eedi "mining misconceptions in mathematics" data (`train.csv`, `misconception_mapping.csv`) was translated into
**French, German, Chinese (Simplified), Greek, Indonesian, Swahili and Telugu** with `google/translategemma-27b-it`.
Every translation is judged by two local LLM judges and by automatic checks. There is no human review in this project.

**Two runs.** *Run 1 (v1)* used the judges Qwen3-32B and Gemma-3-27B on a 1-5 scale; its results are in section 3 and in
`work/final/`. *Run 2 (v2)* (section 4) re-translates the strings flagged in run 1 (several candidates, the best one chosen
by the judges), then judges everything again with the **new judges Qwen3-32B and GPT-OSS-120B on a 1-3 scale**; its output
goes to `work/translations_v2`, `work/judging_v2` and `work/final_v2`, and it ends with the drop-out count for the parallel
set without Swahili.
Everything heavy ran on the EPFL RCP cluster (Run:ai, A100 80GB); the light steps run on the laptop.

## Contents
1. [Pipeline at a glance](#1-pipeline-at-a-glance)
2. [Key design choices](#2-key-design-choices) (formula protection, the 4 passes, `review`, judges, flags)
3. [Results](#3-results-run-1-judges-qwen3-32b--gemma-3-27b-scale-1-5) (run 1, and run 2 in 3.6)
4. [Flagged data: what it means and how we plan to address it](#4-flagged-data-what-it-means-and-how-we-plan-to-address-it)
5. [Using the data](#5-using-the-data)
6. [Reproducing the pipeline](#6-reproducing-the-pipeline)
7. [Files and their roles](#7-files-and-their-roles)
8. [The cluster setup](#8-the-cluster-setup)
9. [Cleaning up the cluster](#9-cleaning-up-the-cluster)
10. [Limits: how far to trust the results](#10-limits-how-far-to-trust-the-results)

---

## 1. Pipeline at a glance

```
data/train.csv, misconception_mapping.csv
        |  prep_data.py                     (laptop)  unique strings, pilot sample
        v
work/source_strings.jsonl  (9,496 unique strings; 6,252 need translation, 3,244 are pure maths and are copied)
        |  translate_vllm.py                (cluster, 1 GPU, ~23 min)
        v
work/translations/<lang>.jsonl   (7 languages x 6,252 = 43,764 strings, each with status ok / review)
        |  judge_vllm.py x2                 (cluster, Qwen 2 GPUs + Gemma 1 GPU at the same time, ~2 h)
        v
work/judging/<judge>/<lang>.jsonl
        |  aggregate_judgments.py           (laptop)  scores.csv, summary_by_lang.csv, flagged.csv
        |  assemble.py                      (laptop)  translated CSVs + automatic checks + NeedsReview
        |  strict_flags.py                  (laptop)  NeedsReviewStrict
        v
work/final/<lang>/train.csv, misconception_mapping.csv, and work/final/auto_checks.csv
```

Run 2 (repair + new judges) continues from there:
```
work/final/auto_checks.csv + work/judging/scores.csv
        |  prep_repair.py                   (laptop)  work/repair/todo.jsonl = the flagged strings
        |  repair_candidates.py (1 GPU) + repair_postedit.py (2 GPUs)        new candidates
        |  repair_collect.py -> repair_judge_qwen (2) + repair_judge_gptoss (1) -> repair_select.py
        v
work/translations_v2/<lang>.jsonl           (best candidate per flagged string; others unchanged)
        |  judge_qwen (2 GPUs) + judge_gptoss (1 GPU), 1-3 rubric
        v
work/judging_v2  ->  aggregate_judgments.py, assemble.py, strict_flags.py, union_drop.py  ->  work/final_v2
```

What is translated: `ConstructName, SubjectName, QuestionText, AnswerAText..AnswerDText` (train.csv) and
`MisconceptionName` (misconception_mapping.csv). IDs, `CorrectAnswer` and the misconception-id columns are untouched.
Chinese uses the code `zh-Hans`, which is Simplified Chinese, so no conversion step is needed.

A pilot (150 strings x 7 languages, stratified) was run first. It cost under an hour and showed that the judging worked
(scores spread out, almost all flags real) before the expensive full run.

## 2. Key design choices

### 2.1 Protecting formulas, tables and images
The questions contain LaTeX (`\( ... \)`, `\[ ... \]`), tables (`\begin{tabular}`, `\hline`, `\\`, `&`) and markdown images.
A translation model will happily "translate" or rewrite these. So before translating, `common.py` replaces every such span
with a numbered placeholder such as `[[1]]` (adjacent spans merge), and puts the originals back afterwards. The words
inside `\text { for }` and the alt text of `![description]()` images are not protected, because they must be translated.
A placeholder must come back exactly once, and a formula must be identical to the source.
`selftest_protect.py` checks the protect/restore round trip on all 9,496 real strings (passed).

### 2.2 The four passes (retry ladder)
Translating in placeholder mode usually works (about 95% of strings on the first try), but the model sometimes drops,
duplicates or reorders a placeholder. Instead of accepting that, each failing string goes down a ladder; the first pass
that produces a string that passes validation wins, and later passes only run for the strings that are still failing.

Example: `What is \( 4 \% \) of \( 200 ? \)`

| Pass | What it does | When it is used |
|---|---|---|
| 1. **placeholders** (temperature 0) | translate `What is [[1]] of [[2]] ?`, check every placeholder appears exactly once, restore the formulas | every string |
| 2. **raw LaTeX** (temperature 0) | translate the original text with the LaTeX left in place, accept only if the formulas are identical to the source ("formula signature" check) | strings that failed pass 1 |
| 3. **line by line** (temperature 0) | split a multi-line string and translate each line on its own; a line that still fails stays in English | only multi-line strings that are still failing |
| 4. **sampling** (temperature 0.5, 4 samples) | placeholder mode again with randomness; the first valid sample wins | strings that are still failing |

Why this order: passes 1 and 2 are the most faithful and deterministic, so they come first. Pass 3 can keep the maths intact
at the cost of leaving a line in English, so it comes late. Pass 4 is random, so it is the last resort.
In the pilot about 5 of 150 strings per language needed more than pass 1 and 1 or 2 ended in `review`.

### 2.3 What `review` means
A string gets status `review` when **none of the four passes produced a translation that passed validation**.
It is not dropped and not blank: the pipeline keeps its **best failed attempt** (preferring pass 3 over 2 over 1 and 4),
and marks it. If the best attempt came from pass 3, one or more lines may still be in English. If nothing was ever attempted,
the English source is used (this basically never happens). These strings are never passed off as good: they reach the
judges' flag list, `assemble.py` (`status_review`), `NeedsReview` and `NeedsReviewStrict`.

### 2.3.1 Why the flag exists at all
The judges are lenient on exactly these cases (they gave 4s and 5s to half-English questions and rewritten LaTeX), so the
pipeline's own `review` marker is a necessary independent signal.

### 2.4 The judges and the rubric
**Run 2 (current):** **Qwen3-32B** (independent of the translator, 119 languages) and **GPT-OSS-120B** (OpenAI, MXFP4, a
reasoning model run at low reasoning effort; a different family from both the translator and Qwen). Scores are **1-3**
(3 good, 2 noticeable problems but understandable, 1 serious or unusable) and the prompt tells the judge to keep
`overall` consistent with `severity` (critical = 1, major = at most 2, 3 = no major/critical problem).
*Run 1 used Qwen3-32B and Gemma-3-27B (same family as TranslateGemma, so it may favour its style) on a 1-5 scale.*
Both get the same prompt (`PROMPT` in `judge_vllm.py`): act as a professional translator and school-maths teacher,
judge only against the source, never "fix" the maths (wrong answers and misconceptions are intentional), keep LaTeX
unchanged, preserve numbers, operators, negations and quantifiers, use textbook terminology, and treat untranslated
English as an error. The answer is JSON only:

| Field | Meaning |
|---|---|
| `accuracy`, `fluency`, `terminology` (1-3; run 1: 1-5) | meaning preserved / natural language / correct maths terms |
| `math_intact` (true/false) | formulas, numbers, tables unchanged |
| `severity` | worst error: none, minor, major (meaning changed or wrong term), critical (opposite meaning, or unusable) |
| `overall` (1-3; run 1: 1-5) | the judge's own overall score (it is **not** computed from the other fields) |
| `issue_summary` | up to 25 words in English |

Decoding: temperature 0 (greedy), seed 0; temperature 0.5 only for a retry when the answer was not valid JSON.
Models run with vLLM (`--tp 2` for Qwen; `--tp 1`, `--dtype auto`, `reasoning_effort: low`, `--max-tokens 1500` for GPT-OSS,
whose answer is "reasoning ... final {json}": the parser takes the JSON after the last final-channel marker).

### 2.5 What gets flagged
A string is flagged (`flagged.csv`) when a judge says `overall <= 3` (run 1) / `<= 2` (run 2, 1-3 scale), `severity` major or critical, `math_intact` false,
or its answer could not be parsed; when the two judges differ by 2 points or more (run 2: a 1 against a 3); or when the pipeline marked it `review`.
`assemble.py` adds six mechanical checks (the **automatic checks**):

| Flag | Fires when |
|---|---|
| `status_review` | the pipeline could not validate the string (section 2.3) |
| `formula_changed` | the LaTeX / table / image syntax differs from the source |
| `wrong_script` | for zh, el, te: too few letters in the expected script (text left in English) |
| `length_ratio` | translation/source length outside [0.35, 4.5] for sources over 20 characters (zh: minimum 0.10, since Chinese is much shorter than English) |
| `copied_source` | translation identical to the English source and at least 3 words |
| `stray_ph` | a `[[n]]` marker leaked into the output |
| `judge_flag` | the string is in the judges' `flagged.csv` |

## 3. Results (run 1: judges Qwen3-32B + Gemma-3-27B, scale 1-5)

### 3.1 Judges (full run, 6,252 strings per language)
| Language | Qwen3-32B overall | Gemma-3-27B overall | strings flagged | pipeline `review` | QWK between judges |
|---|---|---|---|---|---|
| French | 4.88 | 4.86 | 5.3% | 0.3% | 0.62 |
| German | 4.88 | 4.88 | 4.9% | 0.3% | 0.68 |
| Chinese (Simplified) | 4.88 | 4.85 | 7.2% | 1.6% | 0.62 |
| Greek | 4.88 | 4.88 | 5.2% | 0.1% | 0.62 |
| Indonesian | 4.91 | 4.90 | 3.8% | 0.1% | 0.66 |
| **Swahili** | **4.76** | **4.60** | **13.9%** | 0.1% | 0.46 |
| Telugu | 4.90 | 4.87 | 4.8% | 0.2% | 0.55 |

Scores are 1-5. In total 2,820 of 43,764 strings (6.4%) are flagged by the judge criteria. QWK (quadratic weighted kappa) is
the agreement between the two judges on `overall`: 0.46-0.68 is moderate. It is pulled down because almost all scores are
5, and it is lowest for Swahili and Telugu, where the judges are least sure.

### 3.2 Automatic checks (strings per flag)
| Flag | de | el | fr | id | sw | te | zh |
|---|---|---|---|---|---|---|---|
| judge_flag | 309 | 322 | 331 | 237 | 868 | 301 | 452 |
| formula_changed | 23 | 14 | 29 | 5 | 11 | 6 | 101 |
| status_review | 21 | 9 | 16 | 7 | 6 | 11 | 103 |
| copied_source | 27 | 10 | 13 | 17 | 18 | 10 | 13 |
| wrong_script | 0 | 13 | 0 | 0 | 0 | 23 | 48 |

### 3.3 Rows of the final CSVs marked `NeedsReview` (broad rule)
| Language | train.csv (of 1,869 rows) | misconception_mapping.csv (of 2,587 rows) |
|---|---|---|
| French | 343 | 145 |
| German | 378 | 123 |
| Chinese | 402 | 154 |
| Greek | 426 | 128 |
| Indonesian | 162 | 108 |
| Swahili | 952 | 379 |
| Telugu | 296 | 133 |
| **Total** | **2,959 of 13,083 (22.6%)** | **1,170 of 18,109 (6.5%)** |

A `train.csv` row has 7 text fields, so a 5% string flag rate marks about 18% of the rows.

### 3.4 Verdict
- **French, German, Greek, Indonesian, Telugu and Chinese: good.** About 95% of strings pass every check and both judges.
- **Swahili: usable but clearly weaker** (about 7.5% of strings are major or critical for both judges). Use `NeedsReviewStrict`.
- **Chinese** has the most damaged maths markup (101 `formula_changed`, 103 `review`), a small share of its strings.
- Chinese first showed 45% flagged because of the length check (Chinese needs far fewer characters); with its own limit it is 7.8%.

### 3.5 Typical errors found
Most errors come from **short strings with no context**, so the model picks the wrong sense of a word:
"They are straight" (about lines) became "heterosexual" (fr, de, zh) or "honest people" (id, sw); "Work out:" became "exercise"
(fr, de, zh, id); "Co-Interior" (angles) is wrong in fr, de, zh, el; "Protractor" became "compass" (el); "obtuse" became "round" (te).
There are also real maths-wording errors: French "of" translated as "divided by", a Chinese "5th number" question turned into
a "second number" question, a Swahili number that lost its "thirty". These recur across languages, so a handful of source
strings account for a large share of the damage.

### 3.6 Results of run 2 (repair + judges Qwen3-32B and GPT-OSS-120B, scale 1-3)

**Repair of the 2,905 strings flagged in run 1** (`work/repair/selection_report.csv`). A candidate replaced the old translation
only if both judges rated it strictly better and it passed the formula, script and placeholder checks:

| Language | Flagged | Replaced | Kept | % replaced |
|---|---|---|---|---|
| fr | 340 | 265 | 75 | 77.9 |
| de | 313 | 260 | 53 | 83.1 |
| zh | 489 | 293 | 196 | 59.9 |
| el | 335 | 260 | 75 | 77.6 |
| id | 237 | 202 | 35 | 85.2 |
| sw | 872 | 713 | 159 | 81.8 |
| te | 319 | 239 | 80 | 74.9 |

"Replaced" means the judges preferred the new candidate, not that it is free of problems.

**Judges on the full run** (6,252 strings per language, `work/judging_v2/summary_by_lang.csv`):

| Language | % flagged (broad) | Qwen overall | GPT-OSS overall | Qwen % major/critical | GPT-OSS % major/critical | QWK | Spearman |
|---|---|---|---|---|---|---|---|
| fr | 15.1 | 2.96 | 2.86 | 1.4 | 8.4 | 0.264 | 0.309 |
| de | 15.7 | 2.96 | 2.85 | 1.0 | 8.4 | 0.228 | 0.269 |
| zh | 15.2 | 2.97 | 2.85 | 1.2 | 9.5 | 0.254 | 0.313 |
| el | 16.8 | 2.97 | 2.84 | 0.9 | 10.2 | 0.182 | 0.248 |
| id | 10.4 | 2.98 | 2.90 | 0.4 | 5.7 | 0.186 | 0.225 |
| sw | 49.1 | 2.92 | 2.48 | 1.8 | 37.0 | 0.112 | 0.208 |
| te | 25.1 | 2.98 | 2.75 | 0.4 | 19.3 | 0.074 | 0.128 |

Qwen3-32B is close to the ceiling everywhere (about 2.97 of 3, at most 1.8% major/critical), while GPT-OSS-120B is much stricter and
separates the languages: Swahili 2.48 with 37% major/critical, Telugu 2.75 with 19%, the others 6-10%. The two judges agree only weakly
(QWK 0.07-0.26, lower than the 0.46-0.68 of run 1), mostly because Qwen gives almost everything a 3. So most flags come from GPT-OSS,
and the strict rule (which needs both judges, or a hard automatic defect) is the safer reading. The language ranking (sw worst,
then te, then el/de/fr/zh, id best) is consistent between the two runs and between the judges.

**Rows to drop for the parallel set without Swahili** (fr, de, zh, el, id, te; `python union_drop.py`, 1,869 train rows and 2,587
misconception rows per language):

| Rule | Train rows dropped | Misconception rows dropped |
|---|---|---|
| `NeedsReview` (broad) | 1,747 (93.5%) | 1,525 (58.9%) |
| `NeedsReviewStrict` | 434 (23.2%), keep 1,435 | 277 (10.7%), keep 2,310 |

With all 7 languages: strict drops 675 train rows (36.1%) and 458 misconception rows (17.7%); broad drops 98.7% and 77.3%.
Swahili alone: strict flags 338 train and 225 misconception rows. Strict rows per language (train): fr 99, de 150, zh 180, el 74,
id 41, sw 338, te 77. Filtering each language separately would drop 959 of 13,083 train rows and 546 of 18,109 misconception rows.

**Reading.** The broad rule is not usable as a drop rule: on the 1-3 scale one judge giving a 2 ("noticeable problems") is enough, and
a row is dropped if any of its six fields is flagged in any of six languages. These numbers also cannot be compared with run 1,
because the judges and the scale changed. **Use `NeedsReviewStrict`.** What is not measured yet is the improvement caused by the
repair itself: the old translations were never scored by the new judges. That needs one more judging run (about 1.5 GPU-hours)
of `work/translations` with the new judges, then a comparison of the strict counts.

## 4. Flagged data: what it means and how we plan to address it

### 4.1 What a flag is (and is not)
A flag marks a string for a look; it is not a verdict. Each judge also has false alarms (for example the meaningless code
`IXOO`, correctly copied, scored as a failure), and each is blind to some real defects (half-English text, rewritten LaTeX).
That is why the automatic checks and the pipeline's `review` marker exist next to the judges.

### 4.2 The columns in the final files
- **`NeedsReview`** (broad, from `assemble.py`): True if any text field of the row has any flag.
- **`NeedsReviewStrict`** (from `strict_flags.py`): True only on strong evidence: a hard automatic check (`status_review`,
  `formula_changed`, `wrong_script`, `stray_ph`), **or both judges** call it major/critical, **or both** score it 3 or lower,
  **or both** say the maths was changed. Single-judge opinions, disagreement, `copied_source` and `length_ratio` are ignored.

### 4.3 What it costs to drop flagged rows (run 1 numbers; run 2 numbers are in 3.6)
- **Filtering each language separately:** 2,959 train rows and 1,170 misconception rows in total (table 3.3).
- **Parallel data (the same rows removed in every language):** at least 952 train rows (Swahili alone) and at most all 1,869.
  If flags in different languages were independent it would be about 85% of the train rows (70% without Swahili) and about
  38% of the misconception rows. Flags overlap, so the real figure is lower. `python union_drop.py` gives the exact numbers.
  Dropping the union would wreck the train set, so for parallel data prefer `NeedsReviewStrict`, leaving Swahili out of the
  parallel set, or the repair loop below.

### 4.4 Run 2: the repair loop and the new judges
No process can guarantee zero flags, because the flags come from noisy checks; what it can do is reduce them. Run 2 works
only on the strings flagged in run 1 (any automatic flag except `copied_source` / `length_ratio` alone; `prep_repair.py`):
1. **Candidates per flagged string** (all validated against the original source: same formulas, right script, no leaked placeholder):
   - TranslateGemma samples (`repair_candidates.py`): 6 placeholder-mode and 4 raw-mode samples at T=0.8, plus one translation of a
     clearer English rewrite when a rule in `repair_hints.json` applies (for example "Work out" -> "Calculate"; edit that file freely).
   - Qwen3-32B post-edit (`repair_postedit.py`): sees the source, the current translation, the automatic flags, the first judges'
     comments and, for answer options, the question with all four options; one greedy plus two sampled edits.
2. **Judge every candidate, and the current translation as candidate 0, with the new judges** (`repair_judge_*`).
3. **Select** (`repair_select.py`): both judges must have parsed the candidate; the quality key is (judges saying the maths changed,
   judges saying major/critical, lowest `overall`, sum of `overall`); a candidate replaces the current translation only if its
   key is **strictly better** and it passes the automatic guards. Otherwise the original stays (and stays flagged if it was).
   Result: `work/translations_v2`, `work/repair/selection_report.csv`.
4. **Judge all of `translations_v2` once** with the new judges (`judge_qwen`, `judge_gptoss`), then aggregate, assemble, add
   `NeedsReviewStrict` and count the drop-out: `python union_drop.py` (parallel set **without Swahili**; the all-language and the
   Swahili-only numbers are printed too).

Baseline to compare with (run 1, independence estimate): about 70% of the train rows and 27% of the misconception rows dropped
for the parallel set without Swahili (85% / 38% with all seven); the exact run-1 figure is `python union_drop.py --final-dir work/final`.
Limits: the judges grade their own repairs and share blind spots, Swahili and Telugu improve least, a floor of judge false
alarms remains, and new judges on a new 1-3 scale will flag a different set than run 1, so run-1 and run-2 flag rates are
not directly comparable.

### 4.5 Judges
Gemma-3 shares its base with the translator, so it may be biased towards its style; run 2 replaces it with GPT-OSS-120B.
GPT-OSS-120B has not been used on this task yet: run `check_gptoss` first (it loads the model and checks that the answer parser
reads its output). Still untried: a dedicated translation-quality model (xCOMET-XL or MetricX-24) as a third signal; it is
independent of both LLMs but blind to LaTeX.

## 5. Using the data
Run-2 files are in `work\final_v2\` (same layout), judge scores in `work\judging_v2\`, and the repair record in
`work\repair\` (`selection_report.csv`, `todo.jsonl`, candidates). Run-1 final files: `work\final\<lang>\train.csv` and `work\final\<lang>\misconception_mapping.csv` for `fr, de, zh, el, id, sw, te`.
They have the same columns and ids as the originals; pure-maths strings are copied verbatim. Extra columns: `NeedsReview`
and `NeedsReviewStrict` (section 4.2). Also: `work\final\auto_checks.csv` (every string with its flags, English source and
translation), `work\judging\flagged.csv` (judge-flagged strings with the judges' explanations), `work\judging\scores.csv`
(both judges' scores per string) and `work\final\strict_summary.csv`.

Suggested use: keep all rows and check that your results hold when you drop `NeedsReview` rows (use `NeedsReviewStrict` for
Swahili).

## 6. Reproducing the pipeline
Run from the project folder in PowerShell (prefix `powershell -ExecutionPolicy Bypass -File` for the `.ps1` scripts).
Laptop setup: `pip install -U transformers pillow sentencepiece huggingface_hub pandas numpy tabulate`, `hf auth login`,
and accept the license of `google/translategemma-27b-it` on huggingface.co (`openai/gpt-oss-120b` is Apache-2.0, normally not gated).
`runai` (logged in, default project `ml4ed-chafai`) and `kubectl` must be on PATH, on the EPFL network or VPN.

**Run 1** (done, with the earlier scripts): `prep_data.py`, `selftest_protect.py`, `sync_scratch.ps1 -Direction up -SetToken`,
`download_models`, `check_gpu`, then translation and the two judges; results are in `work/`.

**Run 2** (repair + new judges). Wait for `Succeeded` (`runai list jobs -p ml4ed-chafai`) before the next line; lines marked `+` can run at the same time.
```powershell
# 1. laptop: list the flagged strings (needs work/final/auto_checks.csv and work/judging/scores.csv from run 1)
python prep_repair.py

# 2. laptop -> cluster (re-store the token if you cleaned the cluster in section 9), then fetch the new model (~65 GB, CPU job)
.\rcp\sync_scratch.ps1 -Direction up -SetToken
.\rcp\submit_job.ps1 -Step download_models                  # also re-fetches the other models if they were deleted
.\rcp\submit_job.ps1 -Step check_gptoss                     # must end with CHECK_GPTOSS PASSED, otherwise see section 8

# 3. repair (3 GPUs in total)
.\rcp\submit_job.ps1 -Step repair_candidates                # 1 GPU  +
.\rcp\submit_job.ps1 -Step repair_postedit                  # 2 GPUs
.\rcp\submit_job.ps1 -Step repair_collect                   # CPU
.\rcp\submit_job.ps1 -Step repair_judge_qwen                # 2 GPUs +
.\rcp\submit_job.ps1 -Step repair_judge_gptoss              # 1 GPU
.\rcp\submit_job.ps1 -Step repair_select                    # CPU -> work/translations_v2

# 4. judge all of translations_v2 with the new judges
.\rcp\submit_job.ps1 -Step judge_qwen                       # 2 GPUs +
.\rcp\submit_job.ps1 -Step judge_gptoss                     # 1 GPU

# 5. cluster -> laptop, then the laptop steps
.\rcp\sync_scratch.ps1 -Direction down
python aggregate_judgments.py --judges qwen3_32b gptoss_120b --judging-dir work/judging_v2 --translations-dir work/translations_v2
python assemble.py --translations-dir work/translations_v2 --out-dir work/final_v2 --judge-flags work/judging_v2/flagged.csv
python strict_flags.py
python union_drop.py                                          # rows to drop for the parallel set WITHOUT Swahili (headline line)
```
Add `-Dry` to `submit_job.ps1` to print the exact `runai submit` command without submitting. Jobs resume from the chunks
already finished. Close Excel (or any program with files from `work\` open) before `-Direction down`: Windows cannot
overwrite open files. Re-running `assemble.py` rebuilds the CSVs and removes `NeedsReviewStrict`; run `strict_flags.py` again.

## 7. Files and their roles
| File | Role | Runs on |
|---|---|---|
| `common.py` | language list, field definitions, placeholder protection and restoration, formula checks | both |
| `prep_data.py` | unique strings to translate (`work/source_strings.jsonl`), stratified pilot sample (`work/pilot_ids.json`) | laptop |
| `selftest_protect.py` | offline round-trip test of the LaTeX protection on all real strings | laptop |
| `translate_vllm.py` | TranslateGemma translation with vLLM, 4-pass retry, resumable, exits at once when done | cluster GPU |
| `judge_vllm.py` | one LLM judge with vLLM (1-3 rubric prompt, JSON output, handles Qwen and GPT-OSS answers, also judges repair candidates `sid~k`), resumable | cluster GPU |
| `prep_repair.py` | lists the flagged strings of run 1 with source, translation, flags, old judge comments and answer-option context (`work/repair/todo.jsonl`) | laptop |
| `repair_hints.json` | optional English rewrites used to get one extra candidate for known ambiguous phrases (editable) | both |
| `repair_common.py` | shared validity check for candidates (formulas, script, leaked placeholders) and hint helpers | cluster |
| `repair_candidates.py` | TranslateGemma sampling: several valid candidates per flagged string | cluster 1 GPU |
| `repair_postedit.py` | Qwen3-32B post-edit with source, flags, judge comments and context | cluster 2 GPUs |
| `repair_collect.py` | merges the current translation (k=0) and all candidates for judging | CPU |
| `repair_select.py` | picks the best candidate by the judges' scores and writes `work/translations_v2` | CPU |
| `aggregate_judgments.py` | combines both judges: `scores.csv`, `summary_by_lang.csv`, `flagged.csv`, `report.md`, QWK/Spearman | laptop |
| `assemble.py` | rebuilds the translated CSVs per language, adds `NeedsReview`, writes `auto_checks.csv` | laptop |
| `strict_flags.py` | adds the stricter `NeedsReviewStrict` column and `strict_summary.csv` | laptop |
| `union_drop.py` | counts the rows to drop if the same rows are removed in every language; headline = parallel set without Swahili (`--exclude`) | laptop |
| `rcp/config.ps1` | project, volume, image, paths, run-as user/group (edit here) | laptop |
| `rcp/sync_scratch.ps1` | copies code up / results down through a CPU-only helper job; stores the HF token | laptop |
| `rcp/submit_job.ps1` | submits one pipeline step as a cluster job; GPU counts per step are set here | laptop |
| `rcp/run_step.sh` | what runs inside the container for each step (GPU guard, caches, logging) | cluster |
| `rcp/probe_cluster.ps1`, `rcp/RCP_CONNECT.md` | the read-only probe used to learn the cluster, and how the connection works | laptop |

Generated folders: `work/` (`source_strings.jsonl`, `pilot_ids.json`, `translations/`, `judging/`, `pilot/`, `final/` = run 1; `repair/`, `translations_v2/`, `judging_v2/`, `final_v2/` = run 2) and
`logs/` (job logs from the cluster).

## 8. The cluster setup
Details are in `rcp/RCP_CONNECT.md`.
- Project `ml4ed-chafai` has **3 guaranteed GPUs**. Jobs that name no GPU pool land on the default pool = **A100 80GB**.
- GPU plan (run 2): TranslateGemma candidates 1 GPU + Qwen post-edit 2 GPUs together; Qwen judge 2 GPUs + GPT-OSS judge 1 GPU together
  (for the candidates and again for the full re-judging). GPT-OSS-120B (MXFP4, ~63 GB of weights) is meant to fit one 80 GB A100.
  If `check_gptoss` runs out of memory, set `$GptossTp = 2` in `rcp/config.ps1` and run the two judges one after the other
  (2 + 2 GPUs exceed the quota of 3); if it fails inside the attention kernels on the A100, try `$VllmAttn = "TRITON_ATTN"`. Model downloads run in a CPU-only job.
  Every GPU job checks the GPU type first, has no automatic retry (`--backoff-limit 0`), and ends as soon as its work is saved
  (vLLM's own shutdown could hang and keep the GPU reserved).
- Files live on `ml4ed-scratch`, the lab's shared 10 TB scratch, mounted at `/scratch`. Our folder is `/scratch/chafai/eedi_mt`
  (`code/`, `hf_cache/`, `cache/`, `logs/`, `.hf_token`).
- The volume only admits user 222948 / group 11159. The Windows `runai` cannot fill in your IDs for `--run-as-user`, so jobs get
  them explicitly with `--run-as-uid 222948 --run-as-gid 11159`.
- Image `vllm/vllm-openai:v0.31.0-cu129`. The image's `torchcodec` (audio/video decoding, unused here) needs CUDA 13 libraries and
  crashes the vLLM import, so `run_step.sh` puts a placeholder `torchcodec` package on the path that makes vLLM treat it as not installed.
- `runai delete job` is asynchronous: wait until a helper job has disappeared before submitting one with the same name
  (`sync_scratch.ps1` does this).

## 9. Cleaning up the cluster
Make sure `work\final_v2\`, `work\translations_v2\`, `work\judging_v2\` and `work\repair\` (and the run-1 folders) are on your laptop, then:
```powershell
.\rcp\sync_scratch.ps1 -Direction down -KeepHelper
kubectl exec eedi-sync-0-0 -n runai-ml4ed-chafai -- sh -c "rm -rf /scratch/chafai/eedi_mt/hf_cache /scratch/chafai/eedi_mt/cache /scratch/chafai/eedi_mt/.hf_token && du -sh /scratch/chafai/eedi_mt"
runai delete job eedi-sync -p ml4ed-chafai
```
This frees about 175 GB on the lab's volume and removes the stored Hugging Face token. Consider revoking that token on
huggingface.co if it is not needed elsewhere. Keep the models if you still want to re-translate or re-judge anything
(run 2 needs TranslateGemma, Qwen3-32B and GPT-OSS-120B).

## 10. Limits: how far to trust the results
- **Two judges, weak agreement in run 2** (QWK 0.07-0.26 against 0.46-0.68 in run 1, lowest for Telugu and Swahili): Qwen3-32B scores almost everything 3, GPT-OSS-120B is much stricter. Treat flags as a list to inspect, and use the strict rule. Treat flags as a list to inspect, not as verdicts.
- **Judge blind spot:** judges read mostly for meaning and gave 5s and 4s to half-English strings and rewritten LaTeX. The automatic checks and
  the pipeline's `review` marker cover this.
- **Weakest languages:** judge reliability is lowest for Swahili and Telugu, so their flag rates are probably underestimates.
- **Judges grade the repairs they select:** candidates are picked by the same judges that later score them. GPT-OSS-120B is untested on this task.
- **No human review:** nothing guarantees cultural or pedagogical fit, only meaning, maths integrity and fluency as seen by the two models.
- **Run 1 files are unrepaired**; run 2 files contain the repaired choices, and strings where no candidate was better keep their flags.
- **The size of the repair's effect is not measured:** run 2 was judged with new judges and a new scale, and the old translations were not re-scored with them (see 3.6).
