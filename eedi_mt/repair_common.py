"""Shared helpers for the repair loop (prep_repair / repair_candidates / repair_postedit / repair_collect / repair_select)."""
import json
import re
from pathlib import Path

from common import formula_problems, script_ratio

STRAY = re.compile(r"\[\[\s*\d+\s*\]\]|<\s*x\s*\d+\s*/?\s*>")


def candidate_problems(src: str, tr: str, lang: str) -> list[str]:
    """Hard defects of a candidate translation of `src` (empty list = acceptable)."""
    p = []
    if not tr.strip():
        return ["empty"]
    p += formula_problems(src, tr)
    if STRAY.search(tr):
        p.append("stray_ph")
    r = script_ratio(tr, lang)
    if r is not None and r < 0.5:
        p.append("wrong_script")
    return p


def load_hints(path="repair_hints.json"):
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for h in json.loads(p.read_text(encoding="utf-8")).get("hints", []):
        out.append((re.compile(h["pattern"]), h["replace"], h.get("langs")))
    return out


def apply_hints(text: str, lang: str, hints) -> str:
    for rx, rep, langs in hints:
        if langs is None or lang in langs:
            text = rx.sub(rep, text)
    return text
