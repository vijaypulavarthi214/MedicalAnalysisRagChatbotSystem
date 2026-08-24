import tempfile
from types import SimpleNamespace

import chromadb
import fitz
import pytest

from app.config import Settings
from app.llm.groq_client import NOT_FOUND_MESSAGE
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


def _settings(**overrides):
    defaults = dict(
        groq_api_key="key",
        groq_model="llama-3.3-70b-versatile",
        cohere_api_key="cohere-key",
        cohere_rerank_model="rerank-v3.5",
        hybrid_alpha=0.7,
        debug_rag=False,
    )
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.fixture(autouse=True)
def clear_bm25_cache():
    hybrid_search_module._bm25_cache.clear()
    yield
    hybrid_search_module._bm25_cache.clear()


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmp_dir:
        client = chromadb.PersistentClient(path=tmp_dir)
        yield ChromaStore(client)


def test_ingest_document_produces_chunks(store):
    pipeline = RagPipeline(store, FakeEmbeddingService(), FakeCohereClient(), FakeGroqClient(), _settings())

    result = pipeline.ingest_document(_build_pdf_bytes("MEDICATIONS\nLisinopril 10mg daily"), "report.pdf")

    assert result.chunk_count >= 1
    assert result.page_count == 1
    assert result.status == "ready"


@pytest.mark.asyncio
async def test_answer_query_returns_grounded_answer_with_sources(store):
    pipeline = RagPipeline(store, FakeEmbeddingService(), FakeCohereClient(), FakeGroqClient(), _settings())
    upload = pipeline.ingest_document(_build_pdf_bytes("MEDICATIONS\nLisinopril 10mg daily"), "report.pdf")

    response = await pipeline.answer_query("What medication?", upload.document_id)

    assert response.answer == "Lisinopril 10mg daily."
    assert len(response.sources) >= 1
    assert response.debug is None


@pytest.mark.asyncio
async def test_answer_query_populates_debug_when_flag_enabled(store):
    settings = _settings(debug_rag=True)
    pipeline = RagPipeline(store, FakeEmbeddingService(), FakeCohereClient(), FakeGroqClient(), settings)
    upload = pipeline.ingest_document(_build_pdf_bytes("MEDICATIONS\nLisinopril 10mg daily"), "report.pdf")

    response = await pipeline.answer_query("What medication?", upload.document_id)

    assert response.debug is not None
    assert "retrieval" in response.debug.latency_ms


@pytest.mark.asyncio
async def test_answer_query_raises_config_error_when_api_key_unset(store):
    settings = _settings(groq_api_key="")
    pipeline = RagPipeline(store, FakeEmbeddingService(), FakeCohereClient(), FakeGroqClient(), settings)

    from app.errors import ConfigError

    with pytest.raises(ConfigError):
        await pipeline.answer_query("question", "doc-1")


@pytest.mark.asyncio
async def test_answer_query_returns_not_found_when_no_hybrid_results(store):
    pipeline = RagPipeline(store, FakeEmbeddingService(), FakeCohereClient(), FakeGroqClient(), _settings())

    response = await pipeline.answer_query("question", "nonexistent-doc")

    assert response.answer == NOT_FOUND_MESSAGE
    assert response.sources == []
