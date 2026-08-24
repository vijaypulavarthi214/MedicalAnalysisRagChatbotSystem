import time
import uuid

import groq

from app.config import Settings
from app.embeddings.embedding_service import EmbeddingService
from app.errors import ConfigError, RetrievalError
from app.ingestion.chunker import chunk_sections
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.structure_parser import parse_sections
from app.llm.groq_client import NOT_FOUND_MESSAGE, generate_answer
from app.models.schemas import DebugInfo, QueryResponse, SourceCitation, UploadResponse
from app.retrieval.chroma_store import ChromaStore
from app.retrieval.hybrid_search import hybrid_search, invalidate_lexical_cache
from app.retrieval.reranker import rerank


class RagPipeline:
    def __init__(
        self,
        chroma_store: ChromaStore,
        embedding_service: EmbeddingService,
        cohere_client,
        groq_client: groq.AsyncGroq,
        settings: Settings,
    ):
        self._chroma_store = chroma_store
        self._embedding_service = embedding_service
        self._cohere_client = cohere_client
        self._groq_client = groq_client
        self._settings = settings

    def ingest_document(self, pdf_bytes: bytes, filename: str) -> UploadResponse:
        document_id = str(uuid.uuid4())
        pages = load_pdf(pdf_bytes)
        sections = parse_sections(pages)
        chunks = chunk_sections(sections, document_id)
        if not chunks:
            raise RetrievalError("No extractable text found in document")

        embeddings = self._embedding_service.embed_texts([c.text for c in chunks])
        self._chroma_store.add_chunks(chunks, embeddings)
        invalidate_lexical_cache(document_id)

        return UploadResponse(
            document_id=document_id,
            filename=filename,
            chunk_count=len(chunks),
            page_count=len(pages),
            status="ready",
        )

    async def answer_query(self, question: str, document_id: str) -> QueryResponse:
        if not self._settings.groq_api_key:
            raise ConfigError("GROQ_API_KEY is not set.")

        latency: dict[str, float] = {}

        start = time.monotonic()
        hybrid_results = hybrid_search(
            document_id,
            question,
            self._embedding_service,
            self._chroma_store,
            top_k=10,
            alpha=self._settings.hybrid_alpha,
        )
        latency["retrieval"] = time.monotonic() - start

        if not hybrid_results:
            return QueryResponse(answer=NOT_FOUND_MESSAGE, sources=[], document_id=document_id, debug=None)

        start = time.monotonic()
        reranked = rerank(
            question,
            hybrid_results,
            self._cohere_client,
            model=self._settings.cohere_rerank_model,
            top_n=3,
        )
        latency["rerank"] = time.monotonic() - start

        start = time.monotonic()
        llm_answer = await generate_answer(
            self._groq_client,
            model=self._settings.groq_model,
            question=question,
            chunks=reranked,
        )
        latency["generation"] = time.monotonic() - start

        sources = (
            []
            if not llm_answer.grounded
            else [
                SourceCitation(
                    section_title=c.section_title,
                    page_start=c.page_start,
                    page_end=c.page_end,
                    relevance_score=c.relevance_score,
                    excerpt=c.text[:200],
                )
                for c in reranked
            ]
        )

        debug = None
        if self._settings.debug_rag:
            debug = DebugInfo(hybrid_results=hybrid_results, reranked_results=reranked, latency_ms=latency)

        return QueryResponse(
            answer=llm_answer.answer_text,
            sources=sources,
            document_id=document_id,
            debug=debug,
        )
