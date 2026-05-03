-- VisionRAG schema
-- Run once against a fresh PostgreSQL database that has pgvector installed.
-- $ psql -U postgres -d visionrag -f schema.sql

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- -----------------------------------------------------------------------
-- documents: one row per uploaded PDF
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename     TEXT NOT NULL,
    pdf_path     TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending',
    page_count   INT  NOT NULL DEFAULT 0,
    -- JSON tree uploaded by the user that defines section structure
    pageindex_data JSONB,
    vision_done_at TIMESTAMPTZ,
    enriched_at    TIMESTAMPTZ,
    pipeline_version TEXT DEFAULT 'v2',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- -----------------------------------------------------------------------
-- pageindex_trees: flattened section tree (uploaded separately)
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pageindex_trees (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id  UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tree         JSONB,
    flat_nodes   JSONB,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_pageindex_trees_document_id ON pageindex_trees(document_id);

-- -----------------------------------------------------------------------
-- page_records: per-page vision extraction output (intermediate)
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS page_records (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id         UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number         INT  NOT NULL,
    topics              TEXT[],
    text                TEXT,
    formulas            JSONB DEFAULT '[]',
    derivations         JSONB DEFAULT '[]',
    tables              JSONB DEFAULT '[]',
    diagrams            JSONB DEFAULT '[]',
    graphs              JSONB DEFAULT '[]',
    chemical_equations  JSONB DEFAULT '[]',
    summary             TEXT,
    keywords            TEXT[],
    search_text         TEXT,
    section_identifiers TEXT[],
    UNIQUE(document_id, page_number)
);

CREATE INDEX IF NOT EXISTS ix_page_records_document_id ON page_records(document_id);

-- -----------------------------------------------------------------------
-- topic_chunks: final retrieval corpus with pgvector embeddings
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS topic_chunks (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id      UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    node_id          TEXT,
    topic            TEXT,
    title            TEXT,
    section_id       TEXT,
    topic_scope      TEXT,
    related_topics   TEXT[],
    exclude_keywords TEXT[],
    content          TEXT,
    search_text      TEXT,
    formulas         JSONB DEFAULT '[]',
    tables           JSONB DEFAULT '[]',
    pages_data       JSONB DEFAULT '[]',
    keywords         TEXT[],
    rich_summary     JSONB,
    page_start       INT,
    page_end         INT,
    -- Parent/child relationship
    parent_id        UUID REFERENCES topic_chunks(id) ON DELETE CASCADE,
    is_parent        BOOLEAN DEFAULT TRUE,
    role             TEXT,   -- definition / explanation / formula / example / derivation
    pipeline_version TEXT DEFAULT 'v2',
    -- Three pgvector embeddings (text-embedding-3-large = 1536 dims)
    search_embedding  VECTOR(1536),
    formula_embedding VECTOR(1536),
    table_embedding   VECTOR(1536),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_topic_chunks_document_id ON topic_chunks(document_id);
CREATE INDEX IF NOT EXISTS ix_topic_chunks_parent_id   ON topic_chunks(parent_id);
CREATE INDEX IF NOT EXISTS ix_topic_chunks_section_id  ON topic_chunks(section_id);

-- pgvector HNSW indexes for fast ANN search
CREATE INDEX IF NOT EXISTS ix_topic_chunks_search_vec
    ON topic_chunks USING hnsw (search_embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS ix_topic_chunks_formula_vec
    ON topic_chunks USING hnsw (formula_embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS ix_topic_chunks_table_vec
    ON topic_chunks USING hnsw (table_embedding vector_cosine_ops);

-- Full-text index for BM25 keyword search
CREATE INDEX IF NOT EXISTS ix_topic_chunks_fts
    ON topic_chunks USING gin(to_tsvector('english', COALESCE(search_text, '')));

-- Trigram index for fuzzy matching
CREATE INDEX IF NOT EXISTS ix_topic_chunks_trgm
    ON topic_chunks USING gin(search_text gin_trgm_ops);
