"""Keyword search on search_text using PostgreSQL full-text search.

Uses ts_rank + plainto_tsquery for BM25-like keyword matching.
No external BM25 library needed — the corpus is indexed in Postgres.
"""

from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession


async def bm25_search(
    query: str,
    document_id: str,
    db: AsyncSession,
    k: int = 10,
) -> list[dict]:
    """Keyword search over topic_chunks.search_text using PostgreSQL full-text."""
    result = await db.execute(
        sql_text("""
            SELECT
                id, node_id, topic, topic_scope, related_topics,
                page_start, page_end, search_text,
                formulas, tables, pages_data,
                is_parent, parent_id, title, section_id,
                ts_rank(
                    to_tsvector('english', COALESCE(search_text, '')),
                    plainto_tsquery('english', :query)
                ) AS score
            FROM topic_chunks
            WHERE document_id = :doc_id
              AND to_tsvector('english', COALESCE(search_text, ''))
                  @@ plainto_tsquery('english', :query)
            ORDER BY score DESC
            LIMIT :k
        """),
        {"query": query, "doc_id": document_id, "k": k},
    )
    rows = result.fetchall()

    return [
        {
            "id": str(row[0]),
            "node_id": row[1],
            "topic": row[2],
            "topic_scope": row[3],
            "related_topics": row[4] or [],
            "page_start": row[5],
            "page_end": row[6],
            "search_text": row[7],
            "formulas": row[8] or [],
            "tables": row[9] or [],
            "pages_data": row[10] or [],
            "is_parent": row[11],
            "parent_id": str(row[12]) if row[12] is not None else None,
            "title": row[13] or row[2],
            "section_id": row[14] or "",
            "score": float(row[15]),
        }
        for row in rows
    ]
