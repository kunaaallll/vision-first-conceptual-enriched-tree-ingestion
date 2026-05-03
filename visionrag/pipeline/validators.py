"""Three layers of validation for the vision-first ingestion pipeline.

  FormulaValidator    — per-formula checks before a formula hits the DB
  VisionValidator     — per-page checks after vision extraction
  EnrichmentValidator — per-node checks before the enricher commits a chunk
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

logger = logging.getLogger(__name__)

_MIN_PROSE_CHARS = 50
_MAX_DEFINITION_CHARS = 500

_DOLLAR_RE = re.compile(r"(?<!\\)\$")


def validate_formula(f: dict) -> tuple[bool, str]:
    """Return (is_valid, reason). Reason is empty on success."""
    if not isinstance(f, dict):
        return False, "not a dict"
    latex = f.get("latex")
    if not isinstance(latex, str) or not latex.strip():
        return False, "empty latex"
    if _DOLLAR_RE.search(latex):
        return False, "contains unescaped $ delimiters"
    if latex.count("{") != latex.count("}"):
        return False, "unbalanced braces"
    begins = re.findall(r"\\begin\{([^}]+)\}", latex)
    ends = re.findall(r"\\end\{([^}]+)\}", latex)
    if sorted(begins) != sorted(ends):
        return False, "mismatched \\begin/\\end environments"

    has_symbolic = bool(re.search(r"[A-Za-z]", latex))
    variables = f.get("variables") or []
    if has_symbolic and not variables:
        return False, "symbolic formula has no variable explanations"
    for v in variables:
        if not isinstance(v, dict):
            return False, "variable entry is not a dict"
        if not (v.get("symbol") or "").strip():
            return False, "variable missing symbol"
        if not (v.get("meaning") or "").strip():
            return False, "variable missing meaning"
    return True, ""


def filter_formulas(formulas: Iterable[dict]) -> tuple[list[dict], list[str]]:
    """Keep only valid formulas; return (kept, rejection_reasons)."""
    kept: list[dict] = []
    reasons: list[str] = []
    for f in formulas or []:
        ok, reason = validate_formula(f)
        if ok:
            kept.append(f)
        else:
            latex_preview = (f.get("latex") if isinstance(f, dict) else "") or ""
            reasons.append(f"dropped formula ({reason}): {latex_preview[:80]}")
    return kept, reasons


def validate_page(page: dict) -> dict:
    """Annotate a merged page record with quality flags."""
    text = (page.get("text") or "").strip()
    page["low_prose"] = len(text) < _MIN_PROSE_CHARS
    has_visuals = bool(
        (page.get("diagrams") or []) or (page.get("tables") or []) or
        (page.get("graphs") or [])
    )
    page["figure_only"] = page["low_prose"] and has_visuals

    kept, reasons = filter_formulas(page.get("formulas") or [])
    page["formulas"] = kept
    page["bad_formula_count"] = len(reasons)
    for r in reasons:
        logger.warning("page %s: %s", page.get("page_number"), r)
    return page


def validate_enriched_node(parent: dict, children: list[dict]) -> list[str]:
    """Structural checks on a parent + its children before DB commit."""
    problems: list[str] = []
    p_start = parent.get("page_start")
    p_end = parent.get("page_end")
    if p_start is None or p_end is None or p_start > p_end:
        problems.append(f"invalid parent range {p_start}..{p_end}")

    for i, c in enumerate(children):
        cs, ce = c.get("page_start"), c.get("page_end")
        if cs is None or ce is None:
            problems.append(f"child[{i}]: missing page range")
            continue
        if cs < p_start or ce > p_end or cs > ce:
            problems.append(
                f"child[{i}]: range {cs}..{ce} outside parent {p_start}..{p_end}"
            )

    if children:
        parent_len = len(parent.get("content") or "")
        child_len = sum(len(c.get("content") or "") for c in children)
        if parent_len > 0:
            ratio = child_len / parent_len
            if ratio < 0.85 or ratio > 1.15:
                problems.append(
                    f"child coverage off-band: children={child_len} parent={parent_len} "
                    f"ratio={ratio:.2f}"
                )

    rs = parent.get("rich_summary") or {}
    for d in rs.get("definitions") or []:
        if isinstance(d, str) and len(d) > _MAX_DEFINITION_CHARS:
            problems.append(
                f"definition exceeds {_MAX_DEFINITION_CHARS} chars (likely hallucinated paragraph)"
            )
            break

    return problems
