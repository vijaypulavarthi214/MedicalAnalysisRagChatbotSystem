import tempfile
import uuid

import chromadb
import pytest

from app.models.schemas import Chunk
from app.retrieval import hybrid_search as hybrid_search_module
from app.retrieval.chroma_store import ChromaStore
from app.retrieval.hybrid_search import hybrid_search


class FakeEmbeddingService:
    def __init__(self, query_vector):
        self._query_vector = query_vector

    def embed_query(self, text):
        return self._query_vector


def _chunk(document_id, text, order_index=0):
    return Chunk(
        chunk_id=str(uuid.uuid4()),
        document_id=document_id,
        section_id=str(uuid.uuid4()),
        section_title="MEDICATIONS",
        page_start=1,
        page_end=1,
        order_index=order_index,
        text=text,
        token_count=len(text.split()),
    )


@pytest.fixture(autouse=True)
def clear_cache():
    hybrid_search_module._bm25_cache.clear()
    yield
    hybrid_search_module._bm25_cache.clear()


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmp_dir:
        client = chromadb.PersistentClient(path=tmp_dir)
        yield ChromaStore(client)


def test_hybrid_search_combines_semantic_and_lexical_and_respects_top_k(store):
    chunk_a = _chunk("doc-1", "patient takes lisinopril for hypertension", order_index=0)
    chunk_b = _chunk("doc-1", "patient reports no known drug allergies", order_index=1)
    chunk_c = _chunk("doc-1", "vitals were stable during the visit", order_index=2)
    store.add_chunks(
        [chunk_a, chunk_b, chunk_c],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
    )
    embedding_service = FakeEmbeddingService([1.0, 0.0])

    results = hybrid_search("doc-1", "lisinopril hypertension", embedding_service, store, top_k=2, alpha=0.7)

    assert len(results) == 2
    assert results[0].chunk_id == chunk_a.chunk_id
    assert results[0].combined_score >= results[1].combined_score


def test_hybrid_search_scoped_to_document(store):
    chunk_1 = _chunk("doc-1", "doc one content about diabetes", order_index=0)
    chunk_2 = _chunk("doc-2", "doc two content about diabetes", order_index=0)
    store.add_chunks([chunk_1], embeddings=[[1.0, 0.0]])
    store.add_chunks([chunk_2], embeddings=[[1.0, 0.0]])
    embedding_service = FakeEmbeddingService([1.0, 0.0])

    results = hybrid_search("doc-1", "diabetes", embedding_service, store, top_k=10)

    assert len(results) == 1
    assert results[0].chunk_id == chunk_1.chunk_id
