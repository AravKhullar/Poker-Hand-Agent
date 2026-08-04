"""Retrieval layer — hybrid BM25 + dense vector search with reranking."""

from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
import os
from pgvector import Vector
from pokeragent.db import fetch_all
from dotenv import load_dotenv

LIMIT = 8
load_dotenv()

embeddings = OpenAIEmbeddings(
        model = os.getenv("EMBEDDING_MODEL"),
        api_key = os.getenv("OPENROUTER_API_KEY"),
        base_url = os.getenv("OPENROUTER_BASE_URL"),
    )

class RetrievedChunk(BaseModel):
    chunk_id: int
    content: str
    similarity: float


def embed_query(query: str) -> list[float]:
    """Embed a query string with the configured embedding model."""
    return embeddings.embed_query(query)


def dense_search(query: str, k: int = LIMIT) -> list[RetrievedChunk]:
    """Nearest-neighbour search over the embeddings table by cosine distance."""
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
    """Keyword/BM25 ranking over chunk content.

    TODO(you): implement via rank-bm25 or Postgres full-text search.
    """
    raise NotImplementedError


def hybrid_search(query: str, k: int = LIMIT) -> list[RetrievedChunk]:
    """Fuse dense + BM25 (e.g. Reciprocal Rank Fusion), optionally rerank.

    TODO(you): combine the two ranked lists and return the top-k.
    """
    raise NotImplementedError