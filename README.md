# ML4ED Semester project

Semester project **"Cross-Lingual On-Policy Self-Distillation for Educational Tasks"** (EPFL, ML4ED).

This repository holds the pipeline that translates the Eedi "Mining Misconceptions in Mathematics" data into
French, German, Simplified Chinese, Greek, Indonesian, Swahili and Telugu, judges the translations with LLM judges
and automatic checks, and repairs the flagged strings.

**Start here: [`eedi_mt/README.md`](eedi_mt/README.md)** (pipeline, key choices, results of both runs, how to reproduce).

## What is in the repository
- `eedi_mt/`: code (`*.py`), cluster scripts (`rcp/`), the full README.
- `eedi_mt/work/final_v2/<lang>/`: the **final translated datasets** (`train.csv`, `misconception_mapping.csv`) with the
  `NeedsReview` and `NeedsReviewStrict` columns, plus `auto_checks.csv` and `strict_summary.csv`. Use `NeedsReviewStrict`.
- `eedi_mt/work/judging*/summary_by_lang.csv` and `eedi_mt/work/repair/selection_report.csv`: summary tables of the judging and the repair.

## What is not
- The original Eedi files (`eedi_mt/data/`): download them from Kaggle (see `eedi_mt/data/README.md`).
- Large intermediate files (per-string judge scores, candidates, raw translations) and cluster logs; they can be regenerated with the pipeline.
