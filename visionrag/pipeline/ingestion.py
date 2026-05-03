"""Ingestion orchestrator — vision-first pipeline.

Pipeline stages:
    upload → render pages → vision extract → merge (per-page records) → VISION_DONE

The remaining stages (tree upload, enrichment, embedding) are triggered by
separate API calls so ingestion never blocks waiting for a tree structure.
"""

import asyncio
import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from uuid import uuid4

from openai import AsyncOpenAI
from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from visionrag.db import get_session_factory
from visionrag.pipeline.vision_extractor import extract_all_pages_vision
from visionrag.pipeline.merger import merge_all_pages
from visionrag.pipeline.validators import validate_page
from visionrag.config import get_settings
from visionrag.utils.pdf_utils import render_all_pages, pdf_page_count

logger = logging.getLogger(__name__)


class IngestionStatus(str, Enum):
    PENDING = "pending"
    VISION_DONE = "vision_done"
    TREE_UPLOADED = "tree_uploaded"
    ENRICHED = "enriched"
    COMPLETE = "complete"
    FAILED = "failed"


async def store_document(file) -> tuple[str, str]:
    """Save uploaded PDF to disk and create a document record.

    Returns (document_id, pdf_path).
    """
    settings = get_settings()
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    doc_id = str(uuid4())
    pdf_path = upload_dir / f"{doc_id}.pdf"

    content = await file.read()
    pdf_path.write_bytes(content)

    factory = get_session_factory()
    async with factory() as db:
        page_count = pdf_page_count(str(pdf_path))
        await db.execute(
            sql_text("""
                INSERT INTO documents (id, filename, pdf_path, status, page_count, created_at, updated_at)
                VALUES (:id, :filename, :pdf_path, :status, :page_count, :created_at, :updated_at)
            """),
            {
                "id": doc_id,
                "filename": file.filename or "unknown.pdf",
                "pdf_path": str(pdf_path),
                "status": IngestionStatus.PENDING.value,
                "page_count": page_count,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            },
        )
        await db.commit()

    logger.info(f"Stored document {doc_id}: {file.filename} ({page_count} pages)")
    return doc_id, str(pdf_path)


async def _update_status(db: AsyncSession, doc_id: str, status: IngestionStatus) -> None:
    await db.execute(
        sql_text("UPDATE documents SET status = :status, updated_at = :now WHERE id = :id"),
        {"status": status.value, "now": datetime.utcnow(), "id": doc_id},
    )
    await db.commit()
    logger.info(f"Document {doc_id}: status -> {status.value}")


async def _save_page_records(db: AsyncSession, doc_id: str, merged_pages: dict) -> None:
    for page_num, page_data in merged_pages.items():
        await db.execute(
            sql_text("""
                INSERT INTO page_records (
                    document_id, page_number, topics, text,
                    formulas, derivations, tables, diagrams, graphs,
                    chemical_equations, summary, keywords, search_text,
                    section_identifiers
                ) VALUES (
                    :doc_id, :page_number, :topics, :text,
                    :formulas, :derivations, :tables, :diagrams, :graphs,
                    :chemical_equations, :summary, :keywords, :search_text,
                    :section_identifiers
                )
                ON CONFLICT (document_id, page_number) DO UPDATE SET
                    topics = EXCLUDED.topics,
                    text = EXCLUDED.text,
                    formulas = EXCLUDED.formulas,
                    derivations = EXCLUDED.derivations,
                    tables = EXCLUDED.tables,
                    diagrams = EXCLUDED.diagrams,
                    graphs = EXCLUDED.graphs,
                    chemical_equations = EXCLUDED.chemical_equations,
                    summary = EXCLUDED.summary,
                    keywords = EXCLUDED.keywords,
                    search_text = EXCLUDED.search_text,
                    section_identifiers = EXCLUDED.section_identifiers
            """),
            {
                "doc_id": doc_id,
                "page_number": page_num,
                "topics": page_data.get("topics", []),
                "text": page_data.get("text", ""),
                "formulas": json.dumps(page_data.get("formulas", [])),
                "derivations": json.dumps(page_data.get("derivations", [])),
                "tables": json.dumps(page_data.get("tables", [])),
                "diagrams": json.dumps(page_data.get("diagrams", [])),
                "graphs": json.dumps(page_data.get("graphs", [])),
                "chemical_equations": json.dumps(page_data.get("chemical_equations", [])),
                "summary": page_data.get("summary", ""),
                "keywords": page_data.get("keywords", []),
                "search_text": page_data.get("search_text", ""),
                "section_identifiers": page_data.get("section_identifiers", []),
            },
        )
    await db.commit()


async def run_pipeline(doc_id: str, pdf_path: str) -> None:
    """Run the vision-first ingestion pipeline (stops at VISION_DONE).

    Resumable: if vision already completed for this document we skip the
    image rendering + vision calls.
    """
    settings = get_settings()
    factory = get_session_factory()
    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

    try:
        async with factory() as db:
            row = await db.execute(
                sql_text("SELECT status, page_count FROM documents WHERE id = :id"),
                {"id": doc_id},
            )
            result = row.fetchone()
            current_status = result[0] if result else IngestionStatus.PENDING.value
            total_pages = result[1] if result else 0

        if current_status in (
            IngestionStatus.VISION_DONE.value,
            IngestionStatus.ENRICHED.value,
            IngestionStatus.COMPLETE.value,
        ):
            logger.info("Document %s: vision already done (status=%s) — skipping", doc_id, current_status)
            return

        logger.info("Document %s: rendering pages", doc_id)
        images_dir = Path(settings.upload_dir) / f"{doc_id}_images"
        image_paths = render_all_pages(pdf_path, str(images_dir))
        if not total_pages:
            total_pages = len(image_paths)

        logger.info("Document %s: running vision extraction on %d pages", doc_id, len(image_paths))
        vision_data = await extract_all_pages_vision(image_paths, openai_client)

        logger.info("Document %s: building per-page merged records", doc_id)
        merged_pages = merge_all_pages(
            pi_pages={},
            vision_pages=vision_data,
            flat_nodes=[],
            total_pages=total_pages,
        )
        for pn in list(merged_pages.keys()):
            merged_pages[pn] = validate_page(merged_pages[pn])

        async with factory() as db:
            await _save_page_records(db, doc_id, merged_pages)
            await db.execute(
                sql_text(
                    "UPDATE documents SET vision_done_at = :now, status = :status, updated_at = :now WHERE id = :id"
                ),
                {"id": doc_id, "now": datetime.utcnow(), "status": IngestionStatus.VISION_DONE.value},
            )
            await db.commit()

        logger.info("Document %s: ingestion complete — awaiting tree upload", doc_id)

    except Exception as e:
        logger.exception("Document %s: pipeline failed: %s", doc_id, e)
        async with factory() as db:
            await _update_status(db, doc_id, IngestionStatus.FAILED)
        raise


async def load_page_records(doc_id: str) -> dict[int, dict]:
    """Load merged page records from the database."""
    factory = get_session_factory()
    async with factory() as db:
        rows = await db.execute(
            sql_text("""
                SELECT page_number, topics, text, formulas, derivations,
                       tables, diagrams, graphs, chemical_equations,
                       summary, keywords, search_text, section_identifiers
                FROM page_records WHERE document_id = :doc_id
                ORDER BY page_number
            """),
            {"doc_id": doc_id},
        )
        pages = {}
        for row in rows.fetchall():
            pages[row[0]] = {
                "page_number": row[0],
                "topics": row[1] or [],
                "text": row[2] or "",
                "formulas": row[3] or [],
                "derivations": row[4] or [],
                "tables": row[5] or [],
                "diagrams": row[6] or [],
                "graphs": row[7] or [],
                "chemical_equations": row[8] or [],
                "summary": row[9] or "",
                "keywords": row[10] or [],
                "search_text": row[11] or "",
                "section_identifiers": row[12] or [],
            }
        return pages
