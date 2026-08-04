from pokeragent.chunk_docs import _chunk_text


def test_chunk_text_splits_long_input():
    chunks = _chunk_text("word " * 500)
    assert len(chunks) > 1


def test_chunk_text_short_input_stays_one_chunk():
    chunks = _chunk_text("A short paragraph about pot odds.")
    assert len(chunks) == 1


def test_chunk_text_no_empty_chunks():
    chunks = _chunk_text("word " * 500)
    assert all(c.strip() for c in chunks)