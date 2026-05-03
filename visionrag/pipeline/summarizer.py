"""Per-chunk rich summarizer.

For each topic chunk, runs a focused LLM pass that returns a compact,
retrieval-friendly JSON structure: definitions, key concepts, formulas (as
clean LaTeX), worked examples and exam keywords.

The serialised summary is folded back into `chunk["search_text"]` BEFORE
embedding — this is the biggest lever for retrieval precision on exam-style
queries, because the embedding now represents the section's concepts
explicitly rather than raw OCR noise.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from openai import AsyncOpenAI

from visionrag.config import get_settings

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = """You are preparing retrieval-ready summaries for an NCERT-style high-school science textbook section.

Given the raw text of ONE section, produce STRICT JSON with this exact shape:

{
  "section_number": "e.g. 5.2   — or null if none present",
  "headline": "one-sentence academic description of what this section teaches",
  "definitions": ["…", "…"],
  "key_concepts": ["concise phrase", "concise phrase"],
  "formulas": [
    {
      "name": "<human label, e.g. 'Ohm's law', or '' if none>",
      "latex": "V = IR",
      "variables": [{"symbol": "V", "meaning": "potential difference"}],
      "context": "<1-sentence description of when this formula applies>",
      "describes": "<short one-phrase description for quick retrieval>"
    }
  ],
  "worked_examples": ["one-sentence summary of each example in the section"],
  "exam_keywords": ["terms an exam question would use"]
}

Rules:
- Use ONLY content that is present in the input text. Do NOT invent anything.
- Write formulas in clean LaTeX WITHOUT the surrounding $ delimiters.
- Every formula variable used in the LaTeX must have a meaning entry; if you cannot explain a symbol, omit the formula rather than guess.
- Keep each definition to a single NCERT-style sentence.
- `exam_keywords` should be the 5–15 terms most likely to appear in an exam question on this section.
- Cap lists: ≤6 definitions, ≤10 key_concepts, ≤10 formulas, ≤5 worked_examples, ≤15 exam_keywords.
- If a field has no content, return an empty list (or null for section_number / headline).
- Output ONLY the JSON object — no prose, no markdown fence.
"""


def _serialise_for_embedding(rich: dict[str, Any]) -> str:
    """Flatten the rich summary into a labeled string for the embedding layer."""
    parts: list[str] = []
    if rich.get("section_number"):
        parts.append(f"SECTION: {rich['section_number']}")
    if rich.get("headline"):
        parts.append(f"OVERVIEW: {rich['headline']}")
    for d in rich.get("definitions") or []:
        parts.append(f"DEFINITION: {d}")
    for kc in rich.get("key_concepts") or []:
        parts.append(f"CONCEPT: {kc}")
    for f in rich.get("formulas") or []:
        if not isinstance(f, dict):
            continue
        latex = f.get("latex") or ""
        if not latex:
            continue
        name = f.get("name") or ""
        describes = f.get("describes") or f.get("context") or ""
        header = f"FORMULA: {latex}"
        if name:
            header = f"FORMULA ({name}): {latex}"
        if describes:
            header = f"{header} — {describes}"
        parts.append(header)
        for v in f.get("variables") or []:
            if not isinstance(v, dict):
                continue
            sym, meaning = v.get("symbol", ""), v.get("meaning", "")
            if sym and meaning:
                parts.append(f"  VAR: {sym} = {meaning}")
    for ex in rich.get("worked_examples") or []:
        parts.append(f"EXAMPLE: {ex}")
    kws = rich.get("exam_keywords") or []
    if kws:
        parts.append("KEYWORDS: " + ", ".join(kws))
    return "\n".join(parts)


def _canonical_formulas(
    summarizer_formulas: list[dict],
    vision_formulas: list[dict],
) -> list[dict]:
    """Merge summarizer formulas with vision formulas, de-duplicating on LaTeX."""

    def _norm(latex: str) -> str:
        return "".join((latex or "").split())

    index: dict[str, dict] = {}
    for f in vision_formulas or []:
        if not isinstance(f, dict):
            continue
        key = _norm(f.get("latex") or "")
        if not key:
            continue
        index[key] = {
            "name": f.get("name") or "",
            "latex": f.get("latex") or "",
            "variables": f.get("variables") or [],
            "context": f.get("context") or "",
            "describes": f.get("describes") or "",
        }
    for f in summarizer_formulas or []:
        if not isinstance(f, dict):
            continue
        key = _norm(f.get("latex") or "")
        if not key:
            continue
        slot = index.get(key)
        if slot is None:
            index[key] = {
                "name": f.get("name") or "",
                "latex": f.get("latex") or "",
                "variables": f.get("variables") or [],
                "context": f.get("context") or "",
                "describes": f.get("describes") or "",
            }
        else:
            if not slot["name"]:
                slot["name"] = f.get("name") or ""
            if not slot["variables"]:
                slot["variables"] = f.get("variables") or []
            if not slot["context"]:
                slot["context"] = f.get("context") or ""
            if not slot["describes"]:
                slot["describes"] = f.get("describes") or ""
    return list(index.values())


async def _summarise_one(
    chunk: dict,
    client: AsyncOpenAI,
    model: str,
    max_body_chars: int = 12000,
) -> dict:
    """Run the summarizer on one chunk. Fail-safe on errors."""
    body = (chunk.get("search_text") or "")[:max_body_chars]
    if not body.strip():
        return chunk

    topic_hint = chunk.get("topic") or ""
    user_msg = (
        f"Section title: {topic_hint}\n\n"
        f"Raw section text (may contain OCR noise):\n---\n{body}\n---"
    )

    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SUMMARY_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        rich = json.loads(raw)
    except Exception as exc:
        logger.warning("summarizer: skipping chunk %s (%s)", chunk.get("topic", "?"), exc)
        return chunk

    rich["formulas"] = _canonical_formulas(
        rich.get("formulas") or [], chunk.get("formulas") or []
    )

    chunk["rich_summary"] = rich
    chunk["formulas"] = rich["formulas"]
    chunk["exam_keywords"] = rich.get("exam_keywords") or []
    if rich.get("section_number"):
        chunk["section_id"] = rich["section_number"]

    serialised = _serialise_for_embedding(rich)
    if serialised:
        chunk["search_text"] = serialised + "\n\n" + (chunk.get("search_text") or "")
        chunk["topic_scope"] = rich.get("headline") or chunk.get("topic_scope", "")
    return chunk


async def summarise_chunks(
    chunks: list[dict],
    client: AsyncOpenAI,
    concurrency: int = 5,
) -> list[dict]:
    """Run the summarizer over all chunks with bounded concurrency."""
    settings = get_settings()
    model = getattr(settings, "summary_model", None) or settings.llm_model

    sem = asyncio.Semaphore(max(1, concurrency))

    async def _run(c: dict) -> dict:
        async with sem:
            return await _summarise_one(c, client, model)

    return await asyncio.gather(*(_run(c) for c in chunks))
