"""VisionRAG FastAPI application.

Endpoints:
  POST /documents/upload            — upload a PDF, start ingestion
  GET  /documents/{id}/status       — poll ingestion status
  POST /documents/{id}/tree         — upload section tree (JSON)
  POST /documents/{id}/enrich       — run enrichment + embedding
  POST /documents/{id}/search       — hybrid retrieval search
  GET  /health                      — liveness probe
"""

import json
import logging
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from pydantic import BaseModel
from sqlalchemy import text as sql_text

from visionrag.config import get_settings
from visionrag.db import get_session_factory
from visionrag.pipeline.ingestion import run_pipeline, store_document, load_page_records
from visionrag.pipeline.enricher import enrich_nodes
from visionrag.pipeline.enrichment_store import store_enriched
from visionrag.retrieval.hybrid import hybrid_retrieve, detect_query_type, fetch_all_parent_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="VisionRAG", version="1.0.0")

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Upload + Ingestion
# ---------------------------------------------------------------------------

@app.post("/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Upload a PDF and kick off vision-first ingestion in the background."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    doc_id, pdf_path = await store_document(file)
    background_tasks.add_task(run_pipeline, doc_id, pdf_path)

    return {"document_id": doc_id, "status": "ingestion_started"}


@app.get("/documents/{document_id}/status")
async def get_document_status(document_id: str):
    """Poll the current ingestion/enrichment status of a document."""
    factory = get_session_factory()
    async with factory() as db:
        row = (
            await db.execute(
                sql_text("SELECT status, page_count, filename FROM documents WHERE id = :id"),
                {"id": document_id},
            )
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Document not found")

    return {"document_id": document_id, "status": row[0], "page_count": row[1], "filename": row[2]}


# ---------------------------------------------------------------------------
# Tree Upload
# ---------------------------------------------------------------------------

class TreeUploadRequest(BaseModel):
    tree: dict | list
    flat_nodes: list[dict]
    doc_name: str = ""
    doc_description: str = ""


@app.post("/documents/{document_id}/tree")
async def upload_tree(document_id: str, body: TreeUploadRequest):
    """Upload the section hierarchy tree for a document.

    The tree is a JSON structure where each node has:
      - node_id: unique identifier
      - title: section title
      - section_id: dotted section number (e.g. "5.2")
      - start_index: first page (1-indexed, inclusive)
      - end_index: last page (1-indexed, inclusive)
      - summary: optional summary text
      - nodes: list of child nodes (for nested structure)

    flat_nodes is a flat list of all nodes (including nested ones).
    """
    factory = get_session_factory()
    async with factory() as db:
        doc = (
            await db.execute(
                sql_text("SELECT status FROM documents WHERE id = :id"),
                {"id": document_id},
            )
        ).fetchone()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        if doc[0] not in ("vision_done", "tree_uploaded", "enriched", "complete"):
            raise HTTPException(
                status_code=400,
                detail=f"Document must be in vision_done status first (current: {doc[0]})",
            )

        # Store the tree
        await db.execute(
            sql_text("""
                INSERT INTO pageindex_trees (document_id, tree, flat_nodes)
                VALUES (:doc_id, :tree, :flat_nodes)
                ON CONFLICT DO NOTHING
            """),
            {
                "doc_id": document_id,
                "tree": json.dumps(body.tree if isinstance(body.tree, dict) else {"nodes": body.tree}),
                "flat_nodes": json.dumps(body.flat_nodes),
            },
        )
        await db.execute(
            sql_text(
                "UPDATE documents SET status = 'tree_uploaded', updated_at = :now WHERE id = :id"
            ),
            {"id": document_id, "now": datetime.utcnow()},
        )
        await db.commit()

    return {
        "document_id": document_id,
        "status": "tree_uploaded",
        "leaves": len(body.flat_nodes),
    }


# ---------------------------------------------------------------------------
# Enrich
# ---------------------------------------------------------------------------

async def _run_enrich(document_id: str, enable_concept_split: bool) -> None:
    factory = get_session_factory()
    settings = get_settings()

    async with factory() as db:
        tree_row = (
            await db.execute(
                sql_text("SELECT flat_nodes FROM pageindex_trees WHERE document_id = :id"),
                {"id": document_id},
            )
        ).fetchone()
        if not tree_row:
            logger.error("enrich: doc=%s has no tree — aborting", document_id)
            return
        flat_nodes = tree_row[0]
        if isinstance(flat_nodes, str):
            flat_nodes = json.loads(flat_nodes)

        doc_row = (
            await db.execute(
                sql_text("SELECT page_count FROM documents WHERE id = :id"),
                {"id": document_id},
            )
        ).fetchone()
        total_pages = doc_row[0] if doc_row else 0

    page_records = await load_page_records(document_id)
    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    enriched, warnings = await enrich_nodes(
        flat_nodes,
        page_records,
        total_pages,
        openai_client,
        enable_concept_split=enable_concept_split,
    )

    async with factory() as db:
        parents, children = await store_enriched(document_id, enriched, openai_client, db)
        await db.execute(
            sql_text(
                "UPDATE documents SET enriched_at = :now, pipeline_version = 'v2', "
                "status = 'complete', updated_at = :now WHERE id = :id"
            ),
            {"id": document_id, "now": datetime.utcnow()},
        )
        await db.commit()

    logger.info("enrich: doc=%s done — parents=%d children=%d warnings=%d",
                document_id, parents, children, len(warnings))


@app.post("/documents/{document_id}/enrich")
async def enrich_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    enable_concept_split: bool = True,
):
    """Trigger enrichment + embedding for a document that has a tree uploaded."""
    factory = get_session_factory()
    async with factory() as db:
        doc = (
            await db.execute(
                sql_text("SELECT status FROM documents WHERE id = :id"),
                {"id": document_id},
            )
        ).fetchone()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        page_count = (
            await db.execute(
                sql_text("SELECT COUNT(*) FROM page_records WHERE document_id = :id"),
                {"id": document_id},
            )
        ).scalar()
        if not page_count:
            raise HTTPException(
                status_code=400,
                detail="Vision extraction not complete — wait for ingestion to finish",
            )

    background_tasks.add_task(_run_enrich, document_id, enable_concept_split)
    return {"document_id": document_id, "status": "enrichment_started"}


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    query_type: str | None = None  # auto-detected if None


@app.post("/documents/{document_id}/search")
async def search(document_id: str, body: SearchRequest):
    """Hybrid retrieval search over an enriched document."""
    settings = get_settings()
    factory = get_session_factory()
    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

    async with factory() as db:
        doc = (
            await db.execute(
                sql_text("SELECT status FROM documents WHERE id = :id"),
                {"id": document_id},
            )
        ).fetchone()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        if doc[0] not in ("complete", "enriched"):
            raise HTTPException(
                status_code=400,
                detail=f"Document not yet enriched (status: {doc[0]})",
            )

        detected_type = detect_query_type(body.query)
        query_type = body.query_type or detected_type

        if query_type == "comprehensive":
            results = await fetch_all_parent_chunks(document_id, db)
        else:
            results = await hybrid_retrieve(
                body.query,
                document_id,
                db,
                openai_client,
                top_k=body.top_k,
                query_type=query_type,
            )

    return {
        "query": body.query,
        "query_type": query_type,
        "document_id": document_id,
        "results": results,
        "count": len(results),
    }
