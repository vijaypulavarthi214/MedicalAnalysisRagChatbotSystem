from types import SimpleNamespace

import pytest

from app.errors import RerankError
from app.models.schemas import HybridSearchResult
from app.retrieval.reranker import rerank


def _candidate(chunk_id, text):
    return HybridSearchResult(
        chunk_id=chunk_id,
        text=text,
        section_title="MEDICATIONS",
        page_start=1,
        page_end=1,
        semantic_score=0.5,
        lexical_score=0.5,
        combined_score=0.5,
    )


class FakeCohereClient:
    def __init__(self, results):
        self._results = results

    def rerank(self, model, query, documents, top_n):
        return SimpleNamespace(results=self._results)


class FailingCohereClient:
    def rerank(self, model, query, documents, top_n):
        raise RuntimeError("boom")


def test_rerank_maps_cohere_indices_back_to_metadata():
    candidates = [_candidate("a", "text a"), _candidate("b", "text b"), _candidate("c", "text c")]
    fake_results = [
        SimpleNamespace(index=2, relevance_score=0.9),
        SimpleNamespace(index=0, relevance_score=0.4),
    ]
    client = FakeCohereClient(fake_results)

    reranked = rerank("query", candidates, client, model="rerank-v3.5", top_n=2)

    assert len(reranked) == 2
    assert reranked[0].chunk_id == "c"
    assert reranked[0].relevance_score == 0.9
    assert reranked[1].chunk_id == "a"


def test_rerank_raises_rerank_error_on_cohere_failure():
    candidates = [_candidate("a", "text a")]
    with pytest.raises(RerankError):
        rerank("query", candidates, FailingCohereClient(), model="rerank-v3.5", top_n=3)


def test_rerank_empty_candidates_returns_empty_list():
    assert rerank("query", [], FailingCohereClient(), model="rerank-v3.5") == []
