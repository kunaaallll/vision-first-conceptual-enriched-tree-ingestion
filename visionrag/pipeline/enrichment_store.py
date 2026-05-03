"""Embedding + persistence for vision-first enrichment output.

Given the list of {parent, children} dicts produced by enrich_nodes:
  1. generates the three embeddings (search/formula/table) for every chunk;
  2. deletes any prior v2 chunks for the document (idempotent re-enrichment);
  3. inserts parents first (so children can reference parent ids);
  4. inserts children with parent_id.
"""

from __future__ import annotations

import asyncio
import json
import logging
from uuid import uuid4

from openai import AsyncOpenAI
from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from visionrag.pipeline.embedder import generate_embedding

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "v2"


def _formula_embedding_text(formulas: list[dict]) -> str:
    parts: list[str] = []
    for f in formulas or []:
        if not isinstance(f, dict):
            continue
        parts.append(f.get("name") or "")
        parts.append(f.get("latex") or "")
        parts.append(f.get("context") or "")
        parts.append(f.get("describes") or "")
        for v in f.get("variables") or []:
            sym, meaning = v.get("symbol", ""), v.get("meaning", "")
            if sym or meaning:
                parts.append(f"{sym}: {meaning}")
    return " ".join(p for p in parts if p)


def _table_embedding_text(tables: list[dict]) -> str:
    parts: list[str] = []
    for t in tables or []:
        parts.append(t.get("caption") or "")
        for row in t.get("rows", []) or []:
            parts.extend(str(cell) for cell in row)
    return " ".join(p for p in parts if p)


async def _embed_chunk(chunk: dict, client: AsyncOpenAI) -> dict:
    search_text = chunk.get("search_text") or chunk.get("content") or ""
    formula_text = _formula_embedding_text(chunk.get("formulas") or []) or search_text
    table_text = _table_embedding_text(chunk.get("tables") or []) or search_text

    search_emb, formula_emb, table_emb = await asyncio.gather(
        generate_embedding(search_text, client),
        generate_embedding(formula_text, client),
        generate_embedding(table_text, client),
    )
    chunk["search_embedding"] = search_emb
    chunk["formula_embedding"] = formula_emb
    chunk["table_embedding"] = table_emb
    return chunk


def _merge_keywords(chunk: dict) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for bucket in [list(chunk.get("exam_keywords") or []), list(chunk.get("keywords") or [])]:
        for k in bucket:
            key = (k or "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(k.strip())
    return out


async def _insert_chunk(
    db: AsyncSession,
    document_id: str,
    chunk: dict,
    *,
    is_parent: bool,
    parent_id: str | None,
) -> str:
    chunk_id = str(uuid4())
    rich_summary = chunk.get("rich_summary") or {}
    section_id = chunk.get("section_id") or ""
    keywords = _merge_keywords(chunk)

    await db.execute(
        sql_text(
            """
            INSERT INTO topic_chunks (
                id, document_id, node_id, topic, topic_scope,
                related_topics, exclude_keywords,
                page_start, page_end, search_text,
                formulas, tables, pages_data,
                keywords, section_id, rich_summary,
                parent_id, is_parent, title, content, pipeline_version, role,
                search_embedding, formula_embedding, table_embedding
            ) VALUES (
                :id, :document_id, :node_id, :topic, :topic_scope,
                :related_topics, :exclude_keywords,
                :page_start, :page_end, :search_text,
                :formulas, :tables, :pages_data,
                :keywords, :section_id, :rich_summary,
                :parent_id, :is_parent, :title, :content, :pipeline_version, :role,
                CAST(:search_embedding AS vector),
                CAST(:formula_embedding AS vector),
                CAST(:table_embedding  AS vector)
            )
            """
        ),
        {
            "id": chunk_id,
            "document_id": document_id,
            "node_id": chunk.get("node_id", ""),
            "topic": chunk.get("topic", "") or chunk.get("title", ""),
            "topic_scope": chunk.get("topic_scope", ""),
            "related_topics": chunk.get("related_topics", []),
            "exclude_keywords": chunk.get("exclude_keywords", []),
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "search_text": chunk.get("search_text") or chunk.get("content", ""),
            "formulas": json.dumps(chunk.get("formulas", [])),
            "tables": json.dumps(chunk.get("tables", [])),
            "pages_data": json.dumps(chunk.get("pages_data", [])),
            "keywords": keywords,
            "section_id": section_id,
            "rich_summary": json.dumps(rich_summary),
            "parent_id": parent_id,
            "is_parent": is_parent,
            "title": chunk.get("title", ""),
            "content": chunk.get("content", ""),
            "pipeline_version": PIPELINE_VERSION,
            "role": chunk.get("role") if not is_parent else None,
            "search_embedding": str(chunk["search_embedding"]),
            "formula_embedding": str(chunk["formula_embedding"]),
            "table_embedding": str(chunk["table_embedding"]),
        },
    )
    return chunk_id


async def store_enriched(
    document_id: str,
    enriched: list[dict],
    openai_client: AsyncOpenAI,
    db: AsyncSession,
) -> tuple[int, int]:
    """Embed + persist all enriched parents and their children.

    Returns (parents_written, children_written). Idempotent: deletes prior
    v2 rows for this document before inserting.
    """
    await db.execute(
        sql_text("DELETE FROM topic_chunks WHERE document_id = :id AND pipeline_version = :ver"),
        {"id": document_id, "ver": PIPELINE_VERSION},
    )

    all_chunks: list[tuple[dict, bool, str | None]] = []
    for item in enriched:
        all_chunks.append((item["parent"], True, None))
        for child in item.get("children", []):
            all_chunks.append((child, False, None))

    await asyncio.gather(*[_embed_chunk(c, openai_client) for c, _, _ in all_chunks])

    parents_written = 0
    children_written = 0
    for item in enriched:
        parent = item["parent"]
        parent_id = await _insert_chunk(db, document_id, parent, is_parent=True, parent_id=None)
        parents_written += 1
        for child in item.get("children", []):
            await _insert_chunk(db, document_id, child, is_parent=False, parent_id=parent_id)
            children_written += 1

    await db.commit()
    logger.info(
        "enrichment store: doc=%s parents=%d children=%d",
        document_id, parents_written, children_written,
    )
    return parents_written, children_written
