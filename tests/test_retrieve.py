import pytest

from pokeragent.retrieval import dense_search


@pytest.mark.integration
def test_dense_search_returns_sorted_relevant_results():
    results = dense_search("pot odds calculation")
    assert len(results) > 0
    similarities = [r.similarity for r in results]
    assert similarities == sorted(similarities, reverse=True)