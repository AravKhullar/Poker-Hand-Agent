"""Retrieval layer — dense vector search, Postgres full-text (BM25-style)
search, and Reciprocal Rank Fusion of the two."""

from pgvector import Vector
from pydantic import BaseModel

from pokeragent.db import fetch_all
from pokeragent.llm import embeddings

LIMIT = 10


class RetrievedChunk(BaseModel):
    chunk_id: int
    content: str
    similarity: float


def embed_query(query: str) -> list[float]:
    """Embed a query string with the configured embedding model."""
    return embeddings.embed_query(query)


def dense_search(query: str, k: int = LIMIT) -> list[RetrievedChunk]:
    """Nearest-neighbour search over chunk embeddings by cosine distance."""
    query_vector = Vector(embed_query(query))
    rows = fetch_all(
        "SELECT id, content, 1-(embedding <=> %s) AS similarity "
        "FROM chunks "
        "ORDER BY embedding <=> %s "
        "LIMIT %s",
        (query_vector, query_vector, k)
    )
    return [
        RetrievedChunk(chunk_id=row[0], content=row[1], similarity=row[2])
        for row in rows
    ]


def bm25_search(query: str, k: int = LIMIT) -> list[RetrievedChunk]:
    """Keyword ranking over chunk content (Postgres ts_rank)."""
    rows = fetch_all(
        "SELECT id, content, "
        "ts_rank(to_tsvector('english', content), plainto_tsquery('english', %s)) AS rank "
        "FROM chunks "
        "WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s) "
        "ORDER BY rank DESC "
        "LIMIT %s",
        (query, query, k),
    )
    return [
        RetrievedChunk(chunk_id=row[0], content=row[1], similarity=row[2])
        for row in rows
    ]


def hybrid_search(query: str, k: int = LIMIT) -> list[RetrievedChunk]:
    """Fuse dense + keyword results with Reciprocal Rank Fusion."""
    RRF_K = 60          # standard RRF constant
    POOL = max(k * 3, 20)  # pull a deeper pool from each method before fusing

    dense = dense_search(query, k=POOL)
    sparse = bm25_search(query, k=POOL)

    # accumulate RRF score per chunk, keeping the chunk object around
    scores: dict[int, float] = {}
    chunks: dict[int, RetrievedChunk] = {}
    for ranked_list in (dense, sparse):
        for rank, chunk in enumerate(ranked_list):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (RRF_K + rank)
            chunks[chunk.chunk_id] = chunk

    top_ids = sorted(scores, key=scores.get, reverse=True)[:k]
    return [
        RetrievedChunk(
            chunk_id=cid, content=chunks[cid].content, similarity=scores[cid]
        )
        for cid in top_ids
    ]