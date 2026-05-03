"""Embedding generation using OpenAI text-embedding-3-large.

Generates three embeddings per topic chunk:
1. search_embedding  — from full search_text (primary retrieval)
2. formula_embedding — from concatenated formula LaTeX + descriptions
3. table_embedding   — from concatenated table content
"""

import asyncio
import logging

from openai import AsyncOpenAI

from visionrag.config import get_settings

logger = logging.getLogger(__name__)


async def generate_embedding(
    text: str,
    client: AsyncOpenAI,
    model: str | None = None,
) -> list[float]:
    """Generate a single embedding vector."""
    settings = get_settings()
    embedding_model = model or settings.embedding_model

    if not text.strip():
        return [0.0] * 1536

    response = await client.embeddings.create(
        model=embedding_model,
        input=text,
    )
    return response.data[0].embedding


async def embed_chunk(chunk: dict, client: AsyncOpenAI) -> dict:
    """Generate all three embeddings for a topic chunk (in-place)."""

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

    formula_str = " ".join(_formula_str(f) for f in chunk.get("formulas", []))
    table_str = " ".join(
        f"{t.get('caption', '')} {' '.join(str(c) for r in t.get('rows', []) for c in r)}"
        for t in chunk.get("tables", [])
    )

    search_text = chunk.get("search_text", "")
    formula_input = formula_str.strip() or search_text
    table_input = table_str.strip() or search_text

    search_emb, formula_emb, table_emb = await asyncio.gather(
        generate_embedding(search_text, client),
        generate_embedding(formula_input, client),
        generate_embedding(table_input, client),
    )

    chunk["search_embedding"] = search_emb
    chunk["formula_embedding"] = formula_emb
    chunk["table_embedding"] = table_emb
    return chunk
