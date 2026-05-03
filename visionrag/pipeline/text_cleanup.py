"""OCR noise cleanup for textbook page text.

PageIndex / PyMuPDF text extraction routinely emits:
  • Hyphenated line breaks ("magne-\ntism" → should be "magnetism").
  • Stray control chars and form-feeds (\x0c) from the PDF stream.
  • Repeated running headers / footers (chapter title, page numbers).
  • Smart quotes / en-dashes / em-dashes that confuse embedders.
  • Orphan single-digit lines that are just page numbers.

These artefacts survive into `search_text`, bloat the embedding input, and
confuse downstream LLMs. We normalise conservatively — the goal is to keep
every letter of the textbook while removing the layout noise.

The cleanup is intentionally regex-only (no LLM) so it is cheap, deterministic,
and auditable. It runs once per page during the merge step.
"""

from __future__ import annotations

import re
import unicodedata

_UNICODE_MAP = {
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-",
    "\u2026": "...",
    "\u00a0": " ",
}

_HYPHEN_BREAK_RE = re.compile(r"(\w)-\s*\n\s*(?=[a-z])")
_PAGE_NUMBER_LINE_RE = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)
_PAGE_GLUE_RE = re.compile(r"^(\d{2,4})(?=\d+\.\d+)", re.MULTILINE)
_LETTER_GAP_RE = re.compile(
    r"(?m)^((?:\d+(?:\.\d+)+\s+)?)([A-Z])\s+(?=[A-Z]{2,}\b)"
)
_TITLE_LETTER_GAP_RE = re.compile(r"^([A-Z])\s+(?=[A-Z]{2,}\b)")
_CAPS_STUTTER_RE = re.compile(r"([A-Z]{2,10})\1{2,}")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def _collapse_token_repeats(line: str) -> str:
    tokens = line.split(" ")
    out: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        j = i
        while j + 1 < n and tokens[j + 1] == tokens[i] and tokens[i] != "":
            j += 1
        run = j - i + 1
        if run >= 3:
            out.append(tokens[i])
        else:
            out.extend(tokens[i : j + 1])
        i = j + 1
    return " ".join(out)


def clean_ocr_text(text: str) -> str:
    """Run the full cleanup chain. Safe to call on empty / None input."""
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    for bad, good in _UNICODE_MAP.items():
        text = text.replace(bad, good)

    text = _CONTROL_RE.sub("", text)
    text = _HYPHEN_BREAK_RE.sub(r"\1", text)
    text = _PAGE_NUMBER_LINE_RE.sub("", text)
    text = _PAGE_GLUE_RE.sub("", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _CAPS_STUTTER_RE.sub(r"\1", text)
    text = "\n".join(_collapse_token_repeats(line) for line in text.split("\n"))
    text = _LETTER_GAP_RE.sub(r"\1\2", text)

    return text.strip()


def clean_title(title: str) -> str:
    """Aggressive cleanup tuned for chunk titles / section headings."""
    if not title:
        return ""
    cleaned = clean_ocr_text(title)
    cleaned = _TITLE_LETTER_GAP_RE.sub(r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def strip_running_header(text: str, header_probe: str, min_hits: int = 2) -> str:
    """Remove a repeated running-header line."""
    if not text or not header_probe:
        return text
    probe = header_probe.strip().lower()
    if not probe:
        return text
    kept = []
    hits = 0
    for line in text.split("\n"):
        if line.strip().lower() == probe:
            hits += 1
            if hits <= min_hits:
                kept.append(line)
            continue
        kept.append(line)
    return "\n".join(kept)
