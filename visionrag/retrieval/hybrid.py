"""Hybrid retrieval combining semantic search, keyword search, and section-scoped search.

Uses Reciprocal Rank Fusion (RRF) to merge ranked lists from:
  1. Section-scoped search  — direct section number / title matching (structural lane)
  2. Semantic search        — pgvector cosine similarity
  3. BM25 keyword search    — PostgreSQL full-text
  4. Modality-specific      — formula_embedding or table_embedding (when query warrants)
"""

import re
from collections import defaultdict

from openai import AsyncOpenAI
from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from visionrag.pipeline.embedder import generate_embedding
from visionrag.retrieval.bm25 import bm25_search
from visionrag.retrieval.semantic import semantic_search
from visionrag.config import get_settings

_SECTION_RE = re.compile(r"\b(\d+(?:\.\d+){1,3})\b")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']{2,}")

_STOPWORDS = frozenset([
    "the", "and", "for", "with", "that", "this", "from", "into", "onto",
    "about", "are", "was", "were", "has", "have", "had", "been", "its",
    "out", "over", "under", "between", "among", "what", "which", "why",
    "how", "when", "where", "who", "does", "did", "doing", "explain",
    "define", "definition", "describe", "discuss", "example", "examples",
])

_FORMULA_KEYWORDS = frozenset([
    "formula", "equation", "derive", "derivation", "proof",
    "calculate", "compute", "solve", "integral", "differential",
    "theorem", "lemma", "expression",
])

_TABLE_KEYWORDS = frozenset([
    "table", "comparison", "compare", "list", "values",
    "properties", "data", "chart",
])

_COMPREHENSIVE_PHRASES = (
    "explain in depth", "explain in detail", "cover all", "all key concepts",
    "key concepts", "key theorems", "worked examples", "in depth", "in detail",
    "everything about", "complete overview", "full explanation", "overview of",
)


def _query_content_tokens(query: str) -> list[str]:
    raw = _WORD_RE.findall((query or "").lower())
    noise = _STOPWORDS | _FORMULA_KEYWORDS | _TABLE_KEYWORDS
    seen: set[str] = set()
    out: list[str] = []
    for w in raw:
        if w in noise or w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out


def detect_query_type(query: str) -> str:
    """Classify a query into: 'comprehensive', 'formula', 'table', 'derivation', 'general'."""
    q = query.lower()
    words = set(q.split())
    if any(phrase in q for phrase in _COMPREHENSIVE_PHRASES):
        return "comprehensive"
    if words & {"derive", "derivation", "step-by-step", "steps", "proof", "prove"}:
        return "derivation"
    if words & _FORMULA_KEYWORDS:
        return "formula"
    if words & _TABLE_KEYWORDS:
        return "table"
    return "general"


def _extract_section_numbers(query: str) -> list[str]:
    return _SECTION_RE.findall(query or "")


async def section_scoped_search(
    query: str,
    document_id: str,
    db: AsyncSession,
    k: int = 10,
) -> list[dict]:
    """Return chunks whose section number OR topic name explicitly matches the query."""
    section_ids = _extract_section_numbers(query)
    q_lower = (query or "").lower()
    content_tokens = _query_content_tokens(query)
    name_probes: list[str] = [q_lower] if len(q_lower) >= 4 else []

    if not section_ids and not content_tokens and not name_probes:
        return []

    clauses: list[str] = []
    params: dict = {"doc_id": document_id, "k": k}

    for i, s in enumerate(section_ids):
        key = f"sec{i}"
        clauses.append(
            f"(section_id = :{key}_exact "
            f"OR node_id LIKE :{key}_a "
            f"OR topic LIKE :{key}_b "
            f"OR topic LIKE :{key}_c "
            f"OR LEFT(search_text, 200) LIKE :{key}_d "
            f"OR LEFT(search_text, 200) LIKE :{key}_e)"
        )
        params[f"{key}_exact"] = s
        params[f"{key}_a"] = f"{s}%"
        params[f"{key}_b"] = f"{s} %"
        params[f"{key}_c"] = f"{s}.%"
        params[f"{key}_d"] = f"%{s} %"
        params[f"{key}_e"] = f"%{s}\n%"

    for i, probe in enumerate(name_probes):
        key = f"name{i}"
        clauses.append(
            f"(LOWER(topic) LIKE :{key} OR LOWER(COALESCE(title, '')) LIKE :{key})"
        )
        params[key] = f"%{probe}%"

    token_score_terms: list[str] = []
    for i, tok in enumerate(content_tokens):
        key = f"tok{i}"
        params[key] = f"%{tok}%"
        clauses.append(
            f"(LOWER(COALESCE(title, '')) LIKE :{key} "
            f"OR LOWER(COALESCE(topic, '')) LIKE :{key})"
        )
        token_score_terms.append(
            f"(CASE WHEN LOWER(COALESCE(title, '')) LIKE :{key} "
            f"OR LOWER(COALESCE(topic, '')) LIKE :{key} THEN 1 ELSE 0 END)"
        )
    token_score_sql = " + ".join(token_score_terms) if token_score_terms else "0"

    where = " OR ".join(clauses)
    sql = sql_text(f"""
        SELECT
            id, node_id, topic, topic_scope, related_topics,
            page_start, page_end, search_text,
            formulas, tables, pages_data,
            is_parent, parent_id, title, section_id,
            ({token_score_sql}) AS token_overlap
        FROM topic_chunks
        WHERE document_id = :doc_id
          AND ({where})
        ORDER BY token_overlap DESC, page_start ASC
        LIMIT :k
    """)

    rows = (await db.execute(sql, params)).fetchall()
    return [
        {
            "id": str(r[0]),
            "node_id": r[1],
            "topic": r[2],
            "topic_scope": r[3],
            "related_topics": r[4] or [],
            "page_start": r[5],
            "page_end": r[6],
            "search_text": r[7],
            "formulas": r[8] or [],
            "tables": r[9] or [],
            "pages_data": r[10] or [],
            "is_parent": r[11],
            "parent_id": str(r[12]) if r[12] is not None else None,
            "title": r[13] or r[2],
            "section_id": r[14] or "",
            "score": 1.0,
        }
        for r in rows
    ]


async def fetch_all_parent_chunks(document_id: str, db: AsyncSession) -> list[dict]:
    """Return every parent chunk for a document ordered by page."""
    sql = sql_text(
        """
        SELECT id, node_id, topic, topic_scope, related_topics,
               page_start, page_end, search_text, content,
               formulas, tables, pages_data,
               is_parent, parent_id, title, section_id, rich_summary
        FROM topic_chunks
        WHERE document_id = :doc_id
          AND (is_parent = TRUE OR is_parent IS NULL)
          AND (pipeline_version = 'v2' OR pipeline_version IS NULL)
        ORDER BY page_start ASC, page_end ASC
        """
    )
    rows = (await db.execute(sql, {"doc_id": document_id})).fetchall()
    return [
        {
            "id": str(r[0]),
            "node_id": r[1],
            "topic": r[2],
            "topic_scope": r[3],
            "related_topics": r[4] or [],
            "page_start": r[5],
            "page_end": r[6],
            "search_text": r[7],
            "content": r[8],
            "formulas": r[9] or [],
            "tables": r[10] or [],
            "pages_data": r[11] or [],
            "is_parent": r[12],
            "parent_id": str(r[13]) if r[13] is not None else None,
            "title": r[14] or r[2],
            "section_id": r[15] or "",
            "rich_summary": r[16],
            "score": 1.0,
        }
        for r in rows
    ]


async def _load_parent_map(chunk_ids: list[str], db: AsyncSession) -> dict[str, dict]:
    if not chunk_ids:
        return {}
    sql = sql_text(
        """
        SELECT c.id, p.id, p.title, p.section_id, p.page_start, p.page_end, p.topic
        FROM topic_chunks c
        LEFT JOIN topic_chunks p ON p.id = c.parent_id
        WHERE c.id = ANY(CAST(:ids AS uuid[]))
        """
    )
    rows = (await db.execute(sql, {"ids": chunk_ids})).fetchall()
    out: dict[str, dict] = {}
    for r in rows:
        cid, pid, ptitle, psec, pstart, pend, ptopic = r
        if pid is None:
            continue
        out[str(cid)] = {
            "id": str(pid),
            "title": ptitle or ptopic or "",
            "section_id": psec or "",
            "page_start": pstart,
            "page_end": pend,
        }
    return out


async def hybrid_retrieve(
    query: str,
    document_id: str,
    db: AsyncSession,
    openai_client: AsyncOpenAI,
    top_k: int = 10,
    query_type: str | None = None,
) -> list[dict]:
    """Run hybrid retrieval combining semantic + BM25 + section-scoped search."""
    if query_type is None:
        query_type = detect_query_type(query)

    query_embedding = await generate_embedding(query, openai_client)

    sem_results = await semantic_search(query_embedding, document_id, db, k=top_k)
    bm25_results = await bm25_search(query, document_id, db, k=top_k)
    section_results = await section_scoped_search(query, document_id, db, k=top_k)

    all_lists = [section_results, sem_results, bm25_results] if section_results \
        else [sem_results, bm25_results]

    if query_type == "formula":
        formula_results = await semantic_search(
            query_embedding, document_id, db, k=top_k, column="formula_embedding"
        )
        all_lists.append(formula_results)
    elif query_type == "table":
        table_results = await semantic_search(
            query_embedding, document_id, db, k=top_k, column="table_embedding"
        )
        all_lists.append(table_results)

    settings = get_settings()
    fused = reciprocal_rank_fusion(all_lists, k=settings.rrf_k)

    parent_map = await _load_parent_map([c["id"] for c in fused], db)
    surviving_parent_ids = {
        p["id"] for cid, p in parent_map.items() if cid in {c["id"] for c in fused}
    }
    deduped: list[dict] = []
    for chunk in fused:
        if chunk.get("is_parent") is True and chunk["id"] in surviving_parent_ids:
            continue
        if chunk["id"] in parent_map:
            chunk["parent"] = parent_map[chunk["id"]]
        deduped.append(chunk)

    return deduped[:top_k]


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF score for document d = sum over lists L of 1 / (k + rank_L(d))
    """
    scores: dict[str, float] = defaultdict(float)
    id_to_chunk: dict[str, dict] = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list):
            chunk_id = chunk["id"]
            scores[chunk_id] += 1.0 / (k + rank + 1)
            id_to_chunk[chunk_id] = chunk

    for chunk_id, chunk in id_to_chunk.items():
        chunk["score"] = scores[chunk_id]

    return sorted(id_to_chunk.values(), key=lambda c: c["score"], reverse=True)
