-- Poker Hand Review Agent — vector store schema.
--
-- Two tables: documents (one row per ingested file) and chunks (the
-- retrievable unit, embedding included directly). A separate embeddings
-- table would only help if you planned to store multiple embedding models
-- side by side — this project uses one model, so that split isn't earning
-- its keep. Keep it simple.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    title       TEXT NOT NULL,
    source      TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id  BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INT NOT NULL,
    content      TEXT NOT NULL,
    embedding    vector(768) NOT NULL,
    UNIQUE (document_id, chunk_index)
);

-- 768 dims = nomic-embed-text (EMBEDDING_MODEL in .env). Changing the
-- embedding model means changing this and re-ingesting: vectors from two
-- models aren't comparable even at equal dimension.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks
    USING hnsw (embedding vector_cosine_ops);

-- For keyword search (the full-text half of hybrid_search).
CREATE INDEX IF NOT EXISTS chunks_content_fts
    ON chunks
    USING gin (to_tsvector('english', content));