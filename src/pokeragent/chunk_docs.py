"""Chunk and embed data/corpus/*.txt into Postgres + pgvector.

Usage: uv run python -m pokeragent.chunk_docs_db
"""
import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

from pokeragent.db import execute, fetch_one, fetch_all

load_dotenv()

CORPUS_DIR = "data/corpus"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

embeddings = OpenAIEmbeddings(
        model = os.getenv("EMBEDDING_MODEL"),
        api_key = os.getenv("OPENROUTER_API_KEY"),
        base_url = os.getenv("OPENROUTER_BASE_URL"),
    )

def main() -> None:
    already_ingested = {row[0] for row in fetch_all("SELECT source FROM documents")}
    for path in sorted(Path(CORPUS_DIR).glob("*.txt")):
        if str(path) in already_ingested:
            print(f"skipping {path.name} (already ingested)")
            continue
        _ingest_file(path)


def _ingest_file(path: Path) -> None:
    doc_id = fetch_one(
        "INSERT INTO documents (title, source) VALUES (%s, %s) RETURNING id",
        (path.stem, str(path)),
    )[0]

    chunks = _chunk_text(path.read_text(encoding="utf-8"))
    vectors = _embed_chunks(chunks)

    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        execute(
            "INSERT INTO chunks (document_id, chunk_index, content, embedding) "
            "VALUES (%s, %s, %s, %s)",
            (doc_id, i, chunk, vector),
        )
    print(f"ingested {path.name}")


def _chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        length_function = len,
    )
    return splitter.split_text(text)


def _embed_chunks(chunks: list[str]) -> list[list[float]]:
    return embeddings.embed_documents(chunks)


if __name__ == "__main__":
    main()