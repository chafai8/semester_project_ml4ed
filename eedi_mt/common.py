"""Shared helpers for the Eedi multilingual translation + judging pipeline.

Key idea: the Eedi strings mix natural language with LaTeX (\\( .. \\), \\[ .. \\]),
LaTeX tables (\\begin{tabular}, \\hline, \\\\) and markdown image descriptions
(![alt text]()).  Only the natural-language parts should be translated.  We
therefore replace every non-translatable span by a numbered placeholder before
translation, and put the original spans back afterwards (after verifying that
every placeholder survived exactly once).

Things handled on purpose (found by inspecting train.csv):
  * `\\text { for }` / `\\text { Speed }` words inside math  -> the word itself is
    left translatable, only the surrounding LaTeX is protected.
  * `![alt text]()` images -> the alt text is translated (it carries real
    information because the image itself is not available), the markdown
    syntax is protected.
  * `<` characters in the data are inequalities inside math, NOT html, so there
    is deliberately no html handling.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

# --------------------------------------------------------------------------- #
# Dataset schema
# --------------------------------------------------------------------------- #
TRAIN_TEXT_COLS = [
    "ConstructName", "SubjectName", "QuestionText",
    "AnswerAText", "AnswerBText", "AnswerCText", "AnswerDText",
]
MISC_TEXT_COL = "MisconceptionName"

FIELD_OF_COL = {
    "ConstructName": "construct",
    "SubjectName": "subject",
    "QuestionText": "question",
    "AnswerAText": "answer",
    "AnswerBText": "answer",
    "AnswerCText": "answer",
    "AnswerDText": "answer",
    "MisconceptionName": "misconception",
}
FIELD_PRIORITY = ["question", "misconception", "construct", "subject", "answer"]

FIELD_DESC = {
    "question": "the stem of a school mathematics multiple-choice question",
    "answer": ("one answer option of a multiple-choice maths question "
               "(it may be a deliberately WRONG distractor; do not penalise it for being wrong)"),
    "misconception": ("a short description of an INCORRECT belief a student may hold. "
                      "The translation must keep the belief exactly as stated "
                      "(it must not be corrected, softened or generalised)"),
    "construct": "the title of a mathematics skill / learning objective",
    "subject": "the name of a mathematics topic category",
}

# --------------------------------------------------------------------------- #
# Languages.  `candidates` = language codes tried in order; the first one that the
# TranslateGemma chat template accepts is used (see `translate_vllm.py --check-langs`).
# --------------------------------------------------------------------------- #
LANGS = {
    "fr": {"name": "French", "candidates": ["fr", "fr-FR", "fr_FR"]},
    "de": {"name": "German", "candidates": ["de", "de-DE", "de_DE"]},
    # Verified with `translate_vllm.py --check-langs`: the chat template rejects zh-CN but accepts
    # zh-Hans (= Simplified Chinese), so that is what gets used.
    "zh": {"name": "Chinese (Simplified)", "candidates": ["zh-Hans", "zh-CN", "zh_CN"]},
    "el": {"name": "Greek", "candidates": ["el", "el-GR", "el_GR"]},
    "id": {"name": "Indonesian", "candidates": ["id", "id-ID", "id_ID"]},
    "sw": {"name": "Swahili", "candidates": ["sw", "sw-KE", "sw_KE"]},
    "te": {"name": "Telugu", "candidates": ["te", "te-IN", "te_IN"]},
}
SRC_CANDIDATES = ["en", "en-US", "en_US"]

# Unicode ranges used by the "did it come out in the right script" auto-check.
SCRIPT_RANGES = {
    "zh": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
    "el": [(0x0370, 0x03FF), (0x1F00, 0x1FFF)],
    "te": [(0x0C00, 0x0C7F)],
}


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def sid(text: str) -> str:
    """Stable id for a source string (exact text, no normalisation)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def read_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_ws(text: str):
    """(leading whitespace, core, trailing whitespace)."""
    core = text.strip()
    if not core:
        return text, "", ""
    i = text.find(core)
    return text[:i], core, text[i + len(core):]


# --------------------------------------------------------------------------- #
# Protecting non-translatable spans
# --------------------------------------------------------------------------- #
_TEXT_IN_MATH = re.compile(r"(\\text\s*\{)([^{}]*[^\W\d_]{3,}[^{}]*)(\})")
_BLANK_TEXT = re.compile(r"(\\text\s*\{)[^{}]*(\})")

_TOKEN = re.compile(
    r"(?P<img>!\[(?P<alt>[^\]]*)\]\((?P<url>[^)]*)\))"
    r"|(?P<disp>\\\[.*?\\\])"
    r"|(?P<inl>\\\(.*?\\\))"
    r"|(?P<dol>\$[^$\n]*\\[A-Za-z]+[^$\n]*\$)"
    r"|(?P<env>\\begin\{[A-Za-z*]+\}(?:\{[^{}]*\})?|\\end\{[A-Za-z*]+\}|\\hline|\\cline\{[^{}]*\}|\\\\)",
    re.S,
)

_MERGE_GAP = re.compile(r"[\s&]*")

_PH_RE = {
    # tolerant of full-width / CJK brackets and stray spaces the model might produce
    "br": re.compile(r"[\[［【〔]\s*[\[［【〔]\s*(\d+)\s*[\]］】〕]\s*[\]］】〕]"),
    "xml": re.compile(r"[<＜]\s*x\s*(\d+)\s*/?\s*[>＞]", re.I),
}
_WORD = re.compile(r"[^\W\d_]{3,}")


def _ph(i: int, style: str) -> str:
    return f"[[{i}]]" if style == "br" else f"<x{i}/>"


def ph_ids(text: str, style: str = "br") -> list[int]:
    return [int(x) for x in _PH_RE[style].findall(text)]


def has_words(protected_text: str, style: str = "br") -> bool:
    """True if anything translatable (a 3+ letter word) is left outside placeholders."""
    return bool(_WORD.search(_PH_RE[style].sub(" ", protected_text)))


def protect(text: str, style: str = "br"):
    """Replace non-translatable spans by numbered placeholders.

    Returns (protected_text, spans) where spans[i-1] is the original text of placeholder i.
    Adjacent protectable spans separated only by whitespace are merged into ONE placeholder
    (fewer placeholders = fewer ways for the model to break them).
    """
    spans: list[str] = []

    def add(s: str) -> str:
        spans.append(s)
        return _ph(len(spans), style)

    out: list[str] = []
    pending = ""
    last = 0

    def flush():
        nonlocal pending
        if pending:
            out.append(add(pending))
            pending = ""

    for m in _TOKEN.finditer(text):
        gap = text[last:m.start()]
        is_img = m.group("img") is not None
        s = m.group(0)
        is_math = any(m.group(g) is not None for g in ("disp", "inl", "dol"))
        has_txt = is_math and _TEXT_IN_MATH.search(s) is not None
        standalone = is_img or has_txt  # these never merge with neighbours

        if pending and _MERGE_GAP.fullmatch(gap) and not standalone:
            pending += gap          # whitespace and table column separators (&) between spans stay protected
        else:
            flush()
            out.append(gap)

        if is_img:
            alt = m.group("alt")
            if alt.strip():
                out.append(add("!["))
                out.append(alt)
                out.append(add(s[2 + len(alt):]))
            else:
                out.append(add(s))
        elif has_txt:
            pos = 0
            for t in _TEXT_IN_MATH.finditer(s):
                out.append(add(s[pos:t.end(1)]))
                out.append(t.group(2))        # the translatable words inside \text{ }
                pos = t.start(3)
            out.append(add(s[pos:]))
        else:
            pending += s
        last = m.end()

    flush()
    out.append(text[last:])
    return "".join(out), spans


def needs_translation(text: str) -> bool:
    protected, _ = protect(text, "br")
    return has_words(protected, "br")


def restore(out: str, spans: list[str], style: str = "br", expected=None):
    """Put the original spans back.  Returns (text, problems).

    `expected` = list of placeholder ids that must each occur (default: all, once).
    """
    rx = _PH_RE[style]
    got = Counter(int(x) for x in rx.findall(out))
    exp = Counter(expected if expected is not None else range(1, len(spans) + 1))
    problems = []
    missing = sorted((exp - got).elements())
    extra = sorted((got - exp).elements())
    if missing:
        problems.append(f"missing_ph:{missing[:6]}")
    if extra:
        problems.append(f"extra_ph:{extra[:6]}")

    def _sub(m):
        i = int(m.group(1))
        return spans[i - 1] if 1 <= i <= len(spans) else m.group(0)

    return rx.sub(_sub, out), problems


# --------------------------------------------------------------------------- #
# Formula integrity check (used for un-protected translation and as an auto-check)
# --------------------------------------------------------------------------- #
def formula_signature(text: str) -> Counter:
    """Multiset of LaTeX/markdown-syntax spans, ignoring whitespace and the *content* of \\text{}."""
    sig = []
    for m in _TOKEN.finditer(text):
        if m.group("img") is not None:
            sig.append("IMG:" + m.group("url"))
            continue
        s = _BLANK_TEXT.sub(r"\1\2", m.group(0))
        sig.append(re.sub(r"\s+", "", s))
    return Counter(sig)


def formula_problems(src: str, out: str) -> list[str]:
    a, b = formula_signature(src), formula_signature(out)
    problems = []
    if src.count("&") != out.count("&"):          # table column separators live between the protected spans
        problems.append(f"ampersand_count:{src.count('&')}->{out.count('&')}")
    lost = sorted((a - b).elements())
    added = sorted((b - a).elements())
    if lost:
        problems.append(f"formula_lost:{[x[:30] for x in lost[:3]]}")
    if added:
        problems.append(f"formula_added:{[x[:30] for x in added[:3]]}")
    return problems


def script_ratio(text: str, lang: str) -> float | None:
    """Share of letters (outside protected spans) that are in the expected script, if checkable."""
    ranges = SCRIPT_RANGES.get(lang)
    if not ranges:
        return None
    protected, _ = protect(text, "br")
    rem = _PH_RE["br"].sub(" ", protected)
    letters = [c for c in rem if c.isalpha()]
    if len(letters) < 8:
        return None
    hit = sum(any(lo <= ord(c) <= hi for lo, hi in ranges) for c in letters)
    return hit / len(letters)
