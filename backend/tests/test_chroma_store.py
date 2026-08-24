import tempfile
import uuid

import chromadb
import pytest

from app.models.schemas import Chunk
from app.retrieval.chroma_store import ChromaStore


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmp_dir:
        client = chromadb.PersistentClient(path=tmp_dir)
        yield ChromaStore(client)


def _chunk(document_id, text, section_title="MEDICATIONS", page=1, order_index=0):
    return Chunk(
        chunk_id=str(uuid.uuid4()),
        document_id=document_id,
        section_id=str(uuid.uuid4()),
        section_title=section_title,
        page_start=page,
        page_end=page,
        order_index=order_index,
        text=text,
        token_count=len(text.split()),
    )


def test_add_and_query_returns_matching_chunks(store):
    chunks = [_chunk("doc-1", "patient takes lisinopril daily"), _chunk("doc-1", "patient has no known allergies")]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    store.add_chunks(chunks, embeddings)

    result = store.query("doc-1", query_embedding=[1.0, 0.0], top_k=2)

    assert len(result.ids) == 2
    assert chunks[0].chunk_id in result.ids


def test_query_filters_by_document_id(store):
    store.add_chunks([_chunk("doc-1", "doc one content")], [[1.0, 0.0]])
    store.add_chunks([_chunk("doc-2", "doc two content")], [[1.0, 0.0]])

    result = store.query("doc-1", query_embedding=[1.0, 0.0], top_k=10)

    assert len(result.ids) == 1
    assert result.metadatas[0]["document_id"] == "doc-1"


def test_delete_document_removes_only_that_document(store):
    store.add_chunks([_chunk("doc-1", "one")], [[1.0, 0.0]])
    store.add_chunks([_chunk("doc-2", "two")], [[0.0, 1.0]])

    store.delete_document("doc-1")

    assert store.list_documents() == {"doc-2": 1}


def test_get_document_chunks_returns_text_and_metadata(store):
    chunk = _chunk("doc-1", "some chunk text", section_title="ASSESSMENT", page=3)
    store.add_chunks([chunk], [[1.0, 0.0]])

    result = store.get_document_chunks("doc-1")

    assert len(result) == 1
    assert result[0]["chunk_id"] == chunk.chunk_id
    assert result[0]["text"] == "some chunk text"
    assert result[0]["section_title"] == "ASSESSMENT"
    assert result[0]["page_start"] == 3


def test_list_documents_counts_chunks_per_document(store):
    store.add_chunks([_chunk("doc-1", "a"), _chunk("doc-1", "b")], [[1.0, 0.0], [0.0, 1.0]])
    store.add_chunks([_chunk("doc-2", "c")], [[1.0, 1.0]])

    assert store.list_documents() == {"doc-1": 2, "doc-2": 1}
