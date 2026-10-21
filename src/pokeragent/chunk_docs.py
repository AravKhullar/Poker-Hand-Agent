"""Chunk and embed data/corpus/*.txt into Postgres + pgvector.

Usage: uv run python -m pokeragent.chunk_docs
"""
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from pokeragent.db import execute, fetch_all, fetch_one
from pokeragent.llm import embeddings

CORPUS_DIR = "data/corpus"
CHUNK_SIZE = 750
CHUNK_OVERLAP = 150


def main() -> None:
    already_ingested = {row[0] for row in fetch_all("SELECT source FROM documents")}
    for path in sorted(Path(CORPUS_DIR).rglob("*.txt")):
        if path.as_posix() in already_ingested:
            print(f"skipping {path.name} (already ingested)")
            continue
        _ingest_file(path)


def _ingest_file(path: Path) -> None:
    # Embed first: a failed embedding call must not leave a document row
    # behind, or the next run would skip this file as "already ingested".
    chunks = _chunk_text(path.read_text(encoding="utf-8"))
    vectors = _embed_chunks(chunks)

    doc_id = fetch_one(
        "INSERT INTO documents (title, source) VALUES (%s, %s) RETURNING id",
        (path.stem, path.as_posix()),
    )[0]

    for i, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
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