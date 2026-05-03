"""Multimodal merge: combines tree structure text with vision extraction results.

Vision model is authoritative for prose text, formulas, derivations, tables,
diagrams, graphs, and chemical equations.
"""

from visionrag.pipeline.text_cleanup import clean_ocr_text


def merge_page(
    page_num: int,
    pi_page: dict,
    vision_page: dict,
    node_summary: str = "",
) -> dict:
    """Merge tree and vision outputs for a single page into unified schema.

    Args:
        page_num: 1-indexed page number.
        pi_page: Tree page data with keys: text, headings, node_ids, summaries.
        vision_page: Vision extraction with keys: formulas, derivations, tables, etc.
        node_summary: Concatenated summaries from tree nodes covering this page.

    Returns:
        Unified page dict.
    """
    vision_prose = (vision_page.get("prose_text") or "").strip()
    pi_text = pi_page.get("text", "") or ""
    if vision_prose and len(vision_prose) >= 0.5 * len(pi_text.strip()):
        text = clean_ocr_text(vision_prose)
    else:
        text = clean_ocr_text(pi_text)

    headings = pi_page.get("headings", [])
    vision_topics = vision_page.get("topics_detected", [])
    section_ids = vision_page.get("section_identifiers", []) or []
    key_terms = vision_page.get("key_terms", []) or []

    topics = vision_topics if vision_topics else headings

    formulas = vision_page.get("formulas", [])
    derivations = vision_page.get("derivations", [])
    tables = vision_page.get("tables", [])
    diagrams = vision_page.get("diagrams", [])
    graphs = vision_page.get("graphs", [])
    chemical_equations = vision_page.get("chemical_equations", [])

    def _formula_str(f: dict) -> str:
        parts = [
            f.get("name") or "",
            f.get("latex") or "",
            f.get("context") or f.get("description") or "",
        ]
        for v in f.get("variables", []) or []:
            sym = (v or {}).get("symbol", "")
            meaning = (v or {}).get("meaning", "")
            if sym or meaning:
                parts.append(f"{sym}: {meaning}")
        return " ".join(p for p in parts if p)

    formula_text = " ".join(_formula_str(f) for f in formulas)
    table_text = " ".join(
        " ".join(str(cell) for row in t.get("rows", []) for cell in row)
        for t in tables
    )
    diagram_text = " ".join(d.get("description", "") for d in diagrams)
    graph_text = " ".join(
        f"{g.get('title', '')} {g.get('meaning', '')}" for g in graphs
    )
    chem_text = " ".join(
        f"{c.get('equation', '')} {c.get('description', '')}" for c in chemical_equations
    )

    search_text = " ".join(
        filter(None, [
            text,
            formula_text,
            table_text,
            diagram_text,
            graph_text,
            chem_text,
            node_summary,
            " ".join(topics),
            " ".join(key_terms),
            " ".join(section_ids),
        ])
    )

    _seen: set[str] = set()
    keywords: list[str] = []
    for token in list(key_terms) + list(topics):
        if not token:
            continue
        k = token.strip().lower()
        if k and k not in _seen:
            _seen.add(k)
            keywords.append(token.strip())

    summary = node_summary or text[:500]

    return {
        "page_number": page_num,
        "topics": topics,
        "text": text,
        "formulas": formulas,
        "derivations": derivations,
        "tables": tables,
        "diagrams": diagrams,
        "graphs": graphs,
        "chemical_equations": chemical_equations,
        "summary": summary,
        "keywords": keywords,
        "section_identifiers": section_ids,
        "search_text": search_text,
    }


def get_node_summary_for_page(flat_nodes: list[dict], page_num: int) -> str:
    """Return the summary of the narrowest tree node that covers this page."""
    best_span: int | None = None
    best_summary = ""
    for node in flat_nodes:
        start = node.get("start_index")
        end = node.get("end_index")
        if start is None or end is None:
            continue
        if not (start <= page_num <= end):
            continue
        span = end - start
        if best_span is None or span < best_span:
            best_span = span
            best_summary = node.get("summary") or ""
    return best_summary


def merge_all_pages(
    pi_pages: dict[int, dict],
    vision_pages: dict[int, dict],
    flat_nodes: list[dict],
    total_pages: int,
) -> dict[int, dict]:
    """Merge all pages from tree and vision extraction.

    Args:
        pi_pages: Tree per-page data keyed by page number.
        vision_pages: Vision extraction results keyed by page number.
        flat_nodes: Flat list of tree nodes.
        total_pages: Total number of pages in the document.

    Returns:
        Dict mapping page number to merged page record.
    """
    merged = {}
    for page_num in range(1, total_pages + 1):
        pi_page = pi_pages.get(page_num, {"text": "", "headings": [], "node_ids": [], "summaries": []})
        vision_page = vision_pages.get(page_num, {
            "formulas": [], "derivations": [], "tables": [],
            "diagrams": [], "graphs": [], "chemical_equations": [],
            "topics_detected": [],
        })
        node_summary = get_node_summary_for_page(flat_nodes, page_num)
        merged[page_num] = merge_page(page_num, pi_page, vision_page, node_summary)
    return merged
