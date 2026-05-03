"""Vision-first enrichment: map vision page data onto tree nodes.

Input:
    flat_leaves (from tree upload), page_records (from DB)

Output:
    list[EnrichedNode] where EnrichedNode = {parent, children: [...]}

Contract:
  • Vision prose is the CANONICAL content.
  • Every leaf in the tree becomes one parent chunk.
  • Orphan pages (not covered by any leaf) are swept into a synthetic
    "UNMAPPED_PAGES" parent so content is never silently lost.
  • Overlapping leaf ranges are trimmed deterministically so a page
    belongs to exactly one parent.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any

from openai import AsyncOpenAI

from visionrag.pipeline.concept_splitter import should_split, split_parent_into_concepts
from visionrag.pipeline.summarizer import _canonical_formulas, _summarise_one
from visionrag.pipeline.text_cleanup import clean_ocr_text, clean_title
from visionrag.pipeline.validators import validate_enriched_node
from visionrag.config import get_settings

logger = logging.getLogger(__name__)

ORPHAN_NODE_ID = "UNMAPPED_PAGES"

_HEADING_RE = re.compile(
    r"^\s*(\d+(?:\.\d+){0,3})\s+([A-Z][A-Z0-9\s'&,\-()]{3,}?)(?:\n|$)",
    re.MULTILINE,
)


def _extract_leading_heading(text: str) -> tuple[str, str] | None:
    if not text:
        return None
    head = text[:300]
    m = _HEADING_RE.match(head.lstrip())
    if not m:
        return None
    sec = m.group(1).strip()
    rest = m.group(2).strip()
    if not any(c.isalpha() for c in rest):
        return None
    return sec, f"{sec} {rest}"


def _correct_leaves_with_vision(
    leaves: list[dict],
    page_records: dict[int, dict],
) -> tuple[list[dict], list[str]]:
    """Merge tree leaves with vision-detected section headings."""
    warnings: list[str] = []

    vision_at: dict[int, tuple[str, str]] = {}
    for p_num, rec in page_records.items():
        h = _extract_leading_heading(rec.get("text") or "")
        if h:
            vision_at[p_num] = h

    tree_claim: dict[int, dict] = {}
    for leaf in leaves:
        for p in range(leaf["start_index"], leaf["end_index"] + 1):
            if p not in tree_claim:
                tree_claim[p] = leaf

    all_pages = sorted(page_records.keys())
    final: dict[int, tuple[str, str, dict | None]] = {}
    current_vision: tuple[str, str] | None = None

    for p in all_pages:
        v = vision_at.get(p)
        tl = tree_claim.get(p)
        tl_sec = (tl.get("section_id") or "").strip() if tl else ""
        tl_title = (tl.get("title") or "").strip() if tl else ""

        if v:
            v_sec, v_title = v
            if tl_sec and (tl_sec == v_sec or tl_sec.startswith(v_sec + ".")):
                final[p] = (tl_sec, tl_title, tl)
            else:
                if tl_sec:
                    warnings.append(
                        f"p{p}: vision heading '{v_title}' overrides tree "
                        f"leaf '{tl_title}' (sec={tl_sec})"
                    )
                final[p] = (v_sec, v_title, None)
            current_vision = v
            continue

        if tl_sec:
            final[p] = (tl_sec, tl_title, tl)
            continue

        if tl and tl_title.lower().startswith("example") and current_vision:
            final[p] = (current_vision[0], current_vision[1], None)
            warnings.append(f"p{p}: absorbed '{tl_title}' into current section '{current_vision[1]}'")
            continue

        if tl:
            final[p] = ("", tl_title, tl)
            continue

        if current_vision:
            final[p] = (current_vision[0], current_vision[1], None)
            continue

        final[p] = ("", "", None)

    corrected: list[dict] = []
    i = 0
    while i < len(all_pages):
        p = all_pages[i]
        sec, title, _ = final[p]
        j = i + 1
        while j < len(all_pages):
            sj, tj, _ = final[all_pages[j]]
            if sj == sec and tj == title:
                j += 1
            else:
                break
        p_end = all_pages[j - 1]

        group_tl: dict | None = None
        for pp in range(p, p_end + 1):
            _, _, tl = final.get(pp, ("", "", None))
            if tl and (tl.get("section_id") or "").strip() == sec:
                group_tl = tl
                break

        if sec:
            if group_tl:
                new_leaf = dict(group_tl)
                new_leaf["start_index"] = p
                new_leaf["end_index"] = p_end
                if sec in (group_tl.get("title") or "") and len(group_tl.get("title") or "") >= len(title):
                    new_leaf["title"] = group_tl["title"]
                else:
                    new_leaf["title"] = title
            else:
                new_leaf = {
                    "node_id": f"VISION#{sec}",
                    "title": title or sec,
                    "section_id": sec,
                    "start_index": p,
                    "end_index": p_end,
                    "summary": "",
                }
        else:
            if group_tl is None:
                i = j
                continue
            new_leaf = dict(group_tl)
            new_leaf["start_index"] = p
            new_leaf["end_index"] = p_end

        corrected.append(new_leaf)
        i = j

    return corrected, warnings


def _deoverlap_ranges(leaves: list[dict]) -> list[tuple[int, int]]:
    """Produce disjoint (start, end) pairs for a flat list of leaf nodes."""
    indexed = [(i, n["start_index"], n["end_index"]) for i, n in enumerate(leaves)]
    indexed.sort(key=lambda t: (t[1], -t[2]))

    trimmed: dict[int, tuple[int, int]] = {}
    last_end = -1
    for i, s, e in indexed:
        new_start = max(s, last_end + 1)
        if new_start > e:
            trimmed[i] = (s, s - 1)
        else:
            trimmed[i] = (new_start, e)
            last_end = e

    return [trimmed[i] for i in range(len(leaves))]


def _assemble_content(pages: list[dict]) -> tuple[str, list[tuple[int, int, int]]]:
    chunks: list[str] = []
    offsets: list[tuple[int, int, int]] = []
    cursor = 0
    for p in pages:
        text = clean_ocr_text((p.get("text") or "").strip())
        if not text:
            continue
        start = cursor
        chunks.append(text)
        cursor += len(text)
        offsets.append((p["page_number"], start, cursor))
        if p is not pages[-1]:
            cursor += 2
    combined = "\n\n".join(chunks)
    return combined, offsets


def _collect_formulas(pages: list[dict]) -> list[dict]:
    return _canonical_formulas([], [f for p in pages for f in p.get("formulas", [])])


def _collect_keywords(pages: list[dict]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for p in pages:
        for k in (p.get("keywords") or []):
            key = (k or "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(k.strip())
    return out


async def _build_parent(
    leaf: dict,
    pages_in_range: list[dict],
    openai_client: AsyncOpenAI,
    summary_model: str,
) -> dict:
    combined_content, offsets = _assemble_content(pages_in_range)
    formulas = _collect_formulas(pages_in_range)
    keywords = _collect_keywords(pages_in_range)

    title = clean_title(leaf.get("title") or "")
    section_id = (leaf.get("section_id") or "").strip()

    parent = {
        "node_id": leaf.get("node_id") or "",
        "title": title,
        "section_id": section_id,
        "topic": f"{section_id} {title}".strip() if section_id else title,
        "page_start": leaf["start_index"],
        "page_end": leaf["end_index"],
        "content": combined_content,
        "search_text": combined_content,
        "formulas": formulas,
        "tables": [t for p in pages_in_range for t in p.get("tables", [])],
        "keywords": keywords,
        "pages_data": pages_in_range,
        "_page_char_offsets": offsets,
        "topic_scope": clean_ocr_text(leaf.get("summary") or ""),
        "related_topics": [],
        "exclude_keywords": [],
        "is_parent": True,
    }

    parent = await _summarise_one(parent, openai_client, summary_model)
    parent["content"] = combined_content
    return parent


async def enrich_nodes(
    flat_leaves: list[dict],
    page_records: dict[int, dict],
    total_pages: int,
    openai_client: AsyncOpenAI,
    *,
    enable_concept_split: bool | None = None,
) -> tuple[list[dict], list[str]]:
    """Main entry. Returns (enriched_parents_with_children, warnings).

    Each returned dict has the shape:
        {"parent": {<parent chunk fields>}, "children": [<child chunk fields>, ...]}
    """
    settings = get_settings()
    summary_model = getattr(settings, "summary_model", None) or settings.llm_model
    split_enabled = (
        enable_concept_split
        if enable_concept_split is not None
        else getattr(settings, "enable_concept_split", True)
    )

    warnings: list[str] = []
    corrected_leaves, correction_warnings = _correct_leaves_with_vision(flat_leaves, page_records)
    warnings.extend(correction_warnings)
    trimmed = _deoverlap_ranges(corrected_leaves)

    covered_pages: set[int] = set()
    enriched: list[dict] = []

    for leaf, (start, end) in zip(corrected_leaves, trimmed):
        if start is None or end is None or start > end:
            warnings.append(f"leaf '{leaf.get('title', '')}' consumed by a sibling — skipped")
            continue

        pages = [page_records[p] for p in range(start, end + 1) if p in page_records]
        covered_pages.update(p for p in range(start, end + 1) if p in page_records)

        if not pages:
            warnings.append(f"leaf '{leaf.get('title', '')}' ({start}-{end}) has no vision coverage")

        parent = await _build_parent(leaf, pages, openai_client, summary_model)
        parent["page_start"] = start
        parent["page_end"] = end

        children: list[dict] = []
        if split_enabled and pages:
            do_split, reason = should_split(parent)
            if do_split:
                logger.info("concept split: node=%s reason=%s", parent.get("node_id", "?"), reason)
                children = await split_parent_into_concepts(parent, openai_client, summary_model)
                if children:
                    logger.info("concept split: node=%s produced %d children", parent.get("node_id", "?"), len(children))

        problems = validate_enriched_node(parent, children)
        for p in problems:
            warnings.append(f"node '{parent.get('title', '?')}': {p}")
            logger.warning("enricher: node=%s %s", parent.get("node_id", "?"), p)

        parent.pop("_page_char_offsets", None)
        enriched.append({"parent": parent, "children": children})

    # Orphan sweep
    orphan_pages = [page_records[p] for p in sorted(page_records.keys()) if p not in covered_pages]
    if orphan_pages:
        logger.info("enricher: %d orphan pages will land in %s", len(orphan_pages), ORPHAN_NODE_ID)
        orphan_leaf = {
            "node_id": ORPHAN_NODE_ID,
            "title": "Unmapped pages",
            "section_id": "",
            "start_index": orphan_pages[0]["page_number"],
            "end_index": orphan_pages[-1]["page_number"],
            "summary": (
                "Pages not covered by any node in the uploaded tree. "
                "Stored so their content remains retrievable."
            ),
        }
        orphan_parent = await _build_parent(orphan_leaf, orphan_pages, openai_client, summary_model)
        orphan_parent["page_start"] = orphan_pages[0]["page_number"]
        orphan_parent["page_end"] = orphan_pages[-1]["page_number"]
        orphan_parent.pop("_page_char_offsets", None)
        enriched.append({"parent": orphan_parent, "children": []})

    return enriched, warnings
