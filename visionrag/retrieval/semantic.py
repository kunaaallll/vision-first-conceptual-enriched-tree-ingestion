"""pgvector cosine similarity search over precomputed embeddings.

Supports querying against three embedding columns:
- search_embedding  (default, general semantic)
- formula_embedding (formula-focused)
- table_embedding   (table-focused)
"""

from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

_VALID_COLUMNS = {"search_embedding", "formula_embedding", "table_embedding"}


async def semantic_search(
    query_embedding: list[float],
    document_id: str,
    db: AsyncSession,
    k: int = 10,
    column: str = "search_embedding",
) -> list[dict]:
    """Search topic_chunks by cosine similarity on the given embedding column."""
    if column not in _VALID_COLUMNS:
        raise ValueError(f"Invalid column '{column}'. Must be one of {_VALID_COLUMNS}")

    query = sql_text(f"""
        SELECT
            id, node_id, topic, topic_scope, related_topics,
            page_start, page_end, search_text,
            formulas, tables, pages_data,
            is_parent, parent_id, title, section_id,
            1 - ({column} <=> CAST(:embedding AS vector)) AS score
        FROM topic_chunks
        WHERE document_id = :doc_id
          AND {column} IS NOT NULL
        ORDER BY {column} <=> CAST(:embedding AS vector)
        LIMIT :k
    """)

    result = await db.execute(query, {
        "embedding": str(query_embedding),
        "doc_id": document_id,
        "k": k,
    })
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
