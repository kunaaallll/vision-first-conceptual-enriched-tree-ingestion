"""Feature-flagged, threshold-gated concept splitter.

Splits one enriched parent node into several concept-level child chunks
ONLY when the node is genuinely multi-topic. Small, single-idea nodes pay
no LLM cost. The decision is made by `should_split()`; the split itself
is delegated to an LLM because concept boundaries are semantic, not
structural.

Output contract — the LLM must return spans that:
  1. are contiguous (each start = previous end + 1, modulo whitespace),
  2. are non-overlapping,
  3. jointly cover 100% of the input characters.

If the contract is violated we abort the split and keep the parent as a
flat leaf. Losing content at the retrieval layer is far worse than
returning a too-coarse chunk.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_EXAMPLE_RE = re.compile(r"(?m)^\s*(Example\s+\d+(?:\.\d+)?)\b[^\n]*")
_VALID_ROLES = {"definition", "explanation", "formula", "example", "derivation"}

_MIN_CONTENT_CHARS = 800
_LONG_CONTENT_CHARS = 3000
_LONG_PAGE_SPAN = 2
_MANY_CONCEPTS = 4
_DISTINCT_FORMULA_VARS = 2


def should_split(parent: dict) -> tuple[bool, str]:
    """Return (should_split, reason). Reason is a short tag for logging."""
    content = parent.get("content") or ""
    if len(content) < _MIN_CONTENT_CHARS:
        return False, "too-short"

    page_span = (parent.get("page_end", 0) or 0) - (parent.get("page_start", 0) or 0)
    if page_span >= _LONG_PAGE_SPAN:
        return True, f"wide-span({page_span + 1}p)"

    if len(content) >= _LONG_CONTENT_CHARS:
        return True, f"long-content({len(content)})"

    rich = parent.get("rich_summary") or {}
    concepts = rich.get("key_concepts") or []
    if len(concepts) >= _MANY_CONCEPTS:
        return True, f"many-concepts({len(concepts)})"

    formulas = parent.get("formulas") or []
    primary_symbols: set[str] = set()
    for f in formulas:
        for v in f.get("variables") or []:
            sym = (v.get("symbol") or "").strip()
            if sym:
                primary_symbols.add(sym[:1])
                break
    if len(primary_symbols) >= _DISTINCT_FORMULA_VARS and len(formulas) >= 2:
        return True, f"formula-hetero({len(primary_symbols)})"

    return False, "below-threshold"


_SPLIT_PROMPT_TEMPLATE = """You are segmenting a section of an NCERT-style physics textbook into CHILD CHUNKS for retrieval.

Each child chunk must be tagged with a ROLE so the retrieval layer can route queries.

The input text has EXACTLY {total_len} characters (0-indexed). Your spans MUST cover all {total_len} characters exactly — no gaps, no overlaps.

Allowed roles:
- "definition"   — formal statement of a concept, law, or rule
- "explanation"  — physical reasoning, intuition, "why" discussion
- "formula"      — derivation or enumeration of equations with variables
- "example"      — a labelled Example N.M block (problem + solution)
- "derivation"   — step-by-step mathematical proof

Return STRICT JSON in this shape:

{{
  "spans": [
    {{
      "title": "short concept title (≤ 10 words)",
      "role": "definition",
      "start_char": 0,
      "end_char": <INT>,
      "keywords": ["term1", "term2"]
    }},
    {{
      "title": "next chunk title",
      "role": "explanation",
      "start_char": <INT>,
      "end_char": {total_len},
      "keywords": ["..."]
    }}
  ]
}}

HARD rules — the pipeline REJECTS the split if any rule is violated:
- spans[0].start_char MUST be 0.
- spans[-1].end_char MUST equal {total_len}.
- For each i > 0: spans[i].start_char MUST equal spans[i-1].end_char (contiguous, no gaps).
- Use 2 to 5 spans. Never 1 (no-op) or more than 5 (too fragmented).
- Each span should be ≥ 250 characters unless the whole text is very short.
- Each span's `role` MUST be one of the allowed roles listed above.
- Do NOT invent content. Titles come only from terms appearing in the span.
- Output ONLY the JSON object."""


def _extract_example_ranges(text: str) -> list[tuple[int, int, str]]:
    matches = list(_EXAMPLE_RE.finditer(text))
    if not matches:
        return []
    out: list[tuple[int, int, str]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        label = m.group(1).strip()
        while start < end and text[start] in " \t\n":
            start += 1
        if end - start >= 80:
            out.append((start, end, label))
    return out


def _paragraph_fallback_spans(text: str) -> list[dict]:
    n = len(text)
    if n < 800:
        return []
    breaks = [m.end() for m in re.finditer(r"\n\s*\n", text)]
    if len(breaks) < 1:
        return []
    target_n = 3 if n >= 2400 else 2
    target_size = n / target_n
    cuts: list[int] = [0]
    for k in range(1, target_n):
        ideal = int(k * target_size)
        nearest = min(breaks, key=lambda b: abs(b - ideal))
        if nearest not in cuts:
            cuts.append(nearest)
    cuts.append(n)
    cuts = sorted(set(cuts))
    if len(cuts) < 3:
        return []
    spans = []
    for i in range(len(cuts) - 1):
        s, e = cuts[i], cuts[i + 1]
        if e - s < 200:
            continue
        spans.append({
            "start_char": s,
            "end_char": e,
            "role": "explanation",
            "title": f"Part {i + 1}",
            "keywords": [],
        })
    if len(spans) < 2:
        return []
    spans[0]["start_char"] = 0
    spans[-1]["end_char"] = n
    return spans


def _validate_spans(spans: list[dict], total_len: int) -> tuple[bool, str]:
    if not isinstance(spans, list) or not spans:
        return False, "no spans"
    if len(spans) < 2:
        return False, "single span (no-op)"
    if len(spans) > 5:
        return False, "too many spans"
    if spans[0].get("start_char") != 0:
        return False, f"first span does not start at 0 (got {spans[0].get('start_char')})"
    if spans[-1].get("end_char") != total_len:
        return False, f"last span ends at {spans[-1].get('end_char')}, expected {total_len}"
    prev_end = 0
    for i, s in enumerate(spans):
        st, en = s.get("start_char"), s.get("end_char")
        if not isinstance(st, int) or not isinstance(en, int) or st >= en:
            return False, f"span[{i}] invalid range {st}..{en}"
        if st != prev_end:
            return False, f"span[{i}] start {st} != previous end {prev_end}"
        prev_end = en
    return True, ""


async def split_parent_into_concepts(
    parent: dict,
    client: AsyncOpenAI,
    model: str,
) -> list[dict]:
    """Return a list of child chunk dicts typed by role. Empty list = no split."""
    content = parent.get("content") or ""
    if not content.strip():
        return []

    page_offsets: list[tuple[int, int, int]] = parent.get("_page_char_offsets") or []
    example_ranges = _extract_example_ranges(content)

    pieces: list[tuple[int, int, str, str | None]] = []
    cursor = 0
    for ex_start, ex_end, label in example_ranges:
        if ex_start > cursor:
            pieces.append((cursor, ex_start, "prose", None))
        pieces.append((ex_start, ex_end, "example", label))
        cursor = ex_end
    if cursor < len(content):
        pieces.append((cursor, len(content), "prose", None))

    children: list[dict] = []
    child_idx = 0

    for piece_start, piece_end, kind, label in pieces:
        if kind == "example":
            child_idx += 1
            span_content = content[piece_start:piece_end].strip()
            if not span_content:
                continue
            ps, pe = _resolve_page_range(
                page_offsets, piece_start, piece_end,
                parent["page_start"], parent["page_end"],
            )
            children.append(_build_child(
                parent, child_idx, ps, pe, span_content,
                title=label or f"{parent.get('title','')} — Example",
                role="example",
                keywords=[label] if label else [],
            ))
            continue

        piece_text = content[piece_start:piece_end]
        if len(piece_text.strip()) < 400:
            child_idx += 1
            ps, pe = _resolve_page_range(
                page_offsets, piece_start, piece_end,
                parent["page_start"], parent["page_end"],
            )
            children.append(_build_child(
                parent, child_idx, ps, pe, piece_text.strip(),
                title=f"{parent.get('title','')} — prose",
                role="explanation",
                keywords=[],
            ))
            continue

        sub_spans = await _llm_split_prose(piece_text, client, model, parent)
        if not sub_spans:
            sub_spans = _paragraph_fallback_spans(piece_text)
        if not sub_spans:
            sub_spans = [{
                "start_char": 0,
                "end_char": len(piece_text),
                "role": "explanation",
                "title": f"{parent.get('title','')} — prose",
                "keywords": [],
            }]

        for span in sub_spans:
            st = piece_start + span["start_char"]
            en = piece_start + span["end_char"]
            span_content = content[st:en].strip()
            if not span_content:
                continue
            child_idx += 1
            ps, pe = _resolve_page_range(
                page_offsets, st, en, parent["page_start"], parent["page_end"],
            )
            role = span.get("role") if span.get("role") in _VALID_ROLES else "explanation"
            children.append(_build_child(
                parent, child_idx, ps, pe, span_content,
                title=(span.get("title") or "").strip() or f"{parent.get('title','')} (part {child_idx})",
                role=role,
                keywords=span.get("keywords") or [],
            ))

    if len(children) < 2:
        return []

    return children


async def _llm_split_prose(
    prose: str,
    client: AsyncOpenAI,
    model: str,
    parent: dict,
) -> list[dict]:
    prompt = _SPLIT_PROMPT_TEMPLATE.format(total_len=len(prose))
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": prose},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
    except Exception as exc:
        logger.warning(
            "concept split: model error on node %s (%s) — will try paragraph fallback",
            parent.get("node_id", "?"), exc,
        )
        return []

    spans = data.get("spans") or []
    ok, reason = _validate_spans(spans, len(prose))
    if not ok:
        logger.warning(
            "concept split: LLM spans invalid on node %s (%s) — will try paragraph fallback",
            parent.get("node_id", "?"), reason,
        )
        return []
    return spans


def _build_child(
    parent: dict,
    idx: int,
    page_start: int,
    page_end: int,
    span_content: str,
    *,
    title: str,
    role: str,
    keywords: list[str],
) -> dict:
    return {
        "node_id": f"{parent.get('node_id', '')}#c{idx}",
        "section_id": parent.get("section_id", ""),
        "title": title,
        "page_start": page_start,
        "page_end": page_end,
        "content": span_content,
        "keywords": keywords,
        "role": role,
        "formulas": [
            f for f in (parent.get("formulas") or [])
            if (f.get("latex") or "") and (f["latex"] in span_content)
        ],
        "tables": [],
        "pages_data": [],
        "parent_node_id": parent.get("node_id", ""),
    }


def _resolve_page_range(
    offsets: list[tuple[int, int, int]],
    char_start: int,
    char_end: int,
    fallback_start: int,
    fallback_end: int,
) -> tuple[int, int]:
    if not offsets:
        return fallback_start, fallback_end
    ps = fallback_start
    pe = fallback_end
    for page_num, cs, ce in offsets:
        if cs <= char_start < ce:
            ps = page_num
        if cs < char_end <= ce:
            pe = page_num
    if ps > pe:
        ps, pe = pe, ps
    return ps, pe
