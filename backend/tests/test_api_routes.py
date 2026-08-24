import tempfile
from types import SimpleNamespace

import chromadb
import fitz
import pytest
from fastapi.testclient import TestClient

from app.api import routes_upload
from app.api.deps import get_chroma_store, get_rag_pipeline
from app.config import Settings, get_settings
from app.main import app
from app.rag.pipeline import RagPipeline
from app.retrieval import hybrid_search as hybrid_search_module
from app.retrieval.chroma_store import ChromaStore


def _build_pdf_bytes(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class FakeEmbeddingService:
    def embed_texts(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


class FakeCohereClient:
    def rerank(self, model, query, documents, top_n):
        results = [SimpleNamespace(index=i, relevance_score=1.0 - 0.1 * i) for i in range(min(top_n, len(documents)))]
        return SimpleNamespace(results=results)


class FakeGroqClient:
    def __init__(self, answer_text="Lisinopril 10mg daily."):
        self._answer_text = answer_text
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._answer_text), finish_reason="stop")]
        )


@pytest.fixture(autouse=True)
def clear_state():
    routes_upload._DOCUMENT_REGISTRY.clear()
    hybrid_search_module._bm25_cache.clear()
    yield
    routes_upload._DOCUMENT_REGISTRY.clear()
    hybrid_search_module._bm25_cache.clear()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("MAX_FILE_SIZE_MB", "1")
    get_settings.cache_clear()
    with tempfile.TemporaryDirectory() as tmp_dir:
        chroma_client = chromadb.PersistentClient(path=tmp_dir)
        store = ChromaStore(chroma_client)
        settings = Settings(
            groq_api_key="key",
            groq_model="llama-3.3-70b-versatile",
            cohere_api_key="cohere-key",
            max_file_size_mb=1,
        )

        def _override_pipeline():
            return RagPipeline(store, FakeEmbeddingService(), FakeCohereClient(), FakeGroqClient(), settings)

        def _override_chroma_store() -> ChromaStore:
            return store

        app.dependency_overrides[get_rag_pipeline] = _override_pipeline
        app.dependency_overrides[get_chroma_store] = _override_chroma_store
        with TestClient(app) as test_client:
            yield test_client
        app.dependency_overrides.clear()
    get_settings.cache_clear()


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_upload_rejects_non_pdf(client):
    response = client.post("/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 422


def test_upload_rejects_oversized_file(client):
    big_content = b"%PDF-1.4\n" + b"0" * (2 * 1024 * 1024)
    response = client.post("/upload", files={"file": ("report.pdf", big_content, "application/pdf")})
    assert response.status_code == 413


def test_upload_accepts_valid_pdf_and_returns_document_id(client):
    pdf_bytes = _build_pdf_bytes("MEDICATIONS\nLisinopril 10mg daily")
    response = client.post("/upload", files={"file": ("report.pdf", pdf_bytes, "application/pdf")})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["chunk_count"] >= 1
    assert body["filename"] == "report.pdf"


def test_query_returns_404_for_unknown_document(client):
    response = client.post("/query", json={"question": "What medication?", "document_id": "nonexistent"})
    assert response.status_code == 404


def test_query_returns_grounded_answer(client):
    pdf_bytes = _build_pdf_bytes("MEDICATIONS\nLisinopril 10mg daily")
    upload_response = client.post("/upload", files={"file": ("report.pdf", pdf_bytes, "application/pdf")})
    document_id = upload_response.json()["document_id"]

    response = client.post("/query", json={"question": "What medication?", "document_id": document_id})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Lisinopril 10mg daily."
    assert len(body["sources"]) >= 1


def test_list_documents_and_delete(client):
    pdf_bytes = _build_pdf_bytes("MEDICATIONS\nLisinopril 10mg daily")
    upload_response = client.post("/upload", files={"file": ("report.pdf", pdf_bytes, "application/pdf")})
    document_id = upload_response.json()["document_id"]

    list_response = client.get("/documents")
    assert list_response.status_code == 200
    assert any(doc["document_id"] == document_id for doc in list_response.json())

    delete_response = client.delete(f"/documents/{document_id}")
    assert delete_response.status_code == 200

    list_after_delete = client.get("/documents")
    assert all(doc["document_id"] != document_id for doc in list_after_delete.json())


def test_delete_unknown_document_returns_404(client):
    response = client.delete("/documents/nonexistent")
    assert response.status_code == 404
