# VisionRAG — Vision-First Multimodal RAG Pipeline Using PageIndex Structured tree

A production-ready RAG (Retrieval-Augmented Generation) pipeline for textbook PDFs. Uses GPT-4o Vision to extract formulas, tables, diagrams, and prose from every page, then builds a hybrid search index (semantic + BM25 + section-scoped) backed by PostgreSQL + pgvector.

## How It Works

```
PDF Upload
    ↓
Render pages → JPEG images (PyMuPDF)
    ↓
GPT-4o Vision extraction per page
  → prose_text, formulas {name, latex, variables[]}, tables,
     derivations, diagrams, graphs, key_terms, section_identifiers
    ↓
Per-page merge + validation → page_records table
    ↓
[VISION_DONE — ingestion stops here]
    ↓
Upload section tree (JSON hierarchy)
    ↓
Enrichment:
  1. Vision-first leaf correction (vision headings override tree)
  2. Concept splitting (LLM segments multi-topic nodes into role-typed children)
  3. Rich summarization (definitions, exam keywords, canonical formulas)
  4. Triple embedding: search / formula / table (text-embedding-3-large)
    ↓
[COMPLETE — ready for search]
    ↓
Hybrid retrieval:
  Section-scoped + Semantic (pgvector cosine) + BM25 (pg full-text)
  → Reciprocal Rank Fusion → Top-K results
```

## Quick Start

```bash
# 1. Clone
git clone https://github.com/your-org/visionrag
cd visionrag

# 2. Configure
cp .env.example .env
# Edit .env — set OPENAI_API_KEY at minimum

# 3. Start
docker compose up -d

# 4. Check health
curl http://localhost:8000/health
```

## API Reference

### Upload a PDF

```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@your_textbook.pdf"
# → {"document_id": "...", "status": "ingestion_started"}
```

### Poll status

```bash
curl http://localhost:8000/documents/{document_id}/status
# → {"status": "vision_done", "page_count": 18, ...}
```

### Upload section tree

After ingestion reaches `vision_done`, upload your section hierarchy using PageIndex tree(You can get it from official pageIndex chat):

```bash
curl -X POST http://localhost:8000/documents/{document_id}/tree \
  -H "Content-Type: application/json" \
  -d '{
    "flat_nodes": [
      {
        "node_id": "5.1",
        "title": "Introduction to Magnetism",
        "section_id": "5.1",
        "start_index": 1,
        "end_index": 3,
        "summary": "Overview of magnetic phenomena"
      },
      {
        "node_id": "5.2",
        "title": "Bar Magnet",
        "section_id": "5.2",
        "start_index": 4,
        "end_index": 8,
        "summary": "Properties and field lines of bar magnets"
      }
    ],
    "tree": {"nodes": []}
  }'
```

Each node in `flat_nodes`:
| Field | Required | Description |
|---|---|---|
| `node_id` | yes | Unique ID (e.g. "5.1") |
| `title` | yes | Section title |
| `section_id` | yes | Dotted number (e.g. "5.2.1") |
| `start_index` | yes | First page (1-indexed, inclusive) |
| `end_index` | yes | Last page (1-indexed, inclusive) |
| `summary` | no | Brief description of section content |

### Run enrichment

```bash
curl -X POST http://localhost:8000/documents/{document_id}/enrich
# → {"status": "enrichment_started"}
# Wait for status to become "complete"
```

### Search

```bash
curl -X POST http://localhost:8000/documents/{document_id}/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the formula for magnetic flux density?", "top_k": 5}'
```

Query types (auto-detected):
- `formula` — routes to formula embeddings: *"derive Coulomb's law"*
- `table` — routes to table embeddings: *"compare diamagnetic and paramagnetic materials"*
- `comprehensive` — returns all sections in page order: *"explain everything in chapter 5"*
- `derivation` — emphasis on step-by-step proofs
- `general` — default hybrid search

## Architecture

### Pipeline modules

| Module | Responsibility |
|---|---|
| `vision_extractor.py` | GPT-4o vision calls with structured JSON output + caching |
| `merger.py` | Merges vision output with tree structure per page |
| `enricher.py` | Maps pages to tree nodes, vision-first leaf correction |
| `concept_splitter.py` | LLM-based concept segmentation into role-typed children |
| `summarizer.py` | Rich section summaries (definitions, formulas, exam keywords) |
| `embedder.py` | Triple embedding generation (search / formula / table) |
| `enrichment_store.py` | Embeds and persists enriched chunks to PostgreSQL |
| `validators.py` | Formula validation, page quality flags, node integrity checks |
| `text_cleanup.py` | OCR noise removal (hyphenation, stutter, page-number artefacts) |

### Retrieval modules

| Module | Responsibility |
|---|---|
| `retrieval/hybrid.py` | RRF fusion of semantic + BM25 + section-scoped results |
| `retrieval/semantic.py` | pgvector cosine similarity (3 embedding columns) |
| `retrieval/bm25.py` | PostgreSQL full-text ts_rank search |

## Configuration

All settings are loaded from environment variables (or `.env` file):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required** |
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL connection string |
| `VISION_MODEL` | `gpt-4o` | Model for page image extraction |
| `EMBEDDING_MODEL` | `text-embedding-3-large` | Embedding model (1536 dims) |
| `LLM_MODEL` | `gpt-4o-mini` | Model for concept splitting |
| `SUMMARY_MODEL` | `gpt-4o-mini` | Model for rich summarization |
| `VISION_CONCURRENCY` | `5` | Max parallel vision API calls |
| `ENABLE_CONCEPT_SPLIT` | `true` | Enable LLM-based concept splitting |
| `UPLOAD_DIR` | `uploads` | Where PDFs and page images are stored |
| `VISION_CACHE_DIR` | `.vision_cache` | Vision result cache (SHA256-keyed) |

## Requirements

- Docker + Docker Compose
- An OpenAI API key with access to `gpt-4o` and `text-embedding-3-large`

Running locally without Docker:
```bash
pip install -r requirements.txt
# Start PostgreSQL with pgvector separately, then:
uvicorn api.main:app --reload
```

## License

MIT
