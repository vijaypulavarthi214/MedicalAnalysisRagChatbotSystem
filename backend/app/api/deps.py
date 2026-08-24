from fastapi import Request

from app.config import get_settings
from app.embeddings.embedding_service import get_embedding_service
from app.rag.pipeline import RagPipeline
from app.retrieval.chroma_store import ChromaStore, get_persistent_chroma_store


def get_chroma_store() -> ChromaStore:
    return get_persistent_chroma_store()


def get_rag_pipeline(request: Request) -> RagPipeline:
    settings = get_settings()
    return RagPipeline(
        chroma_store=get_chroma_store(),
        embedding_service=get_embedding_service(),
        cohere_client=request.app.state.cohere_client,
        groq_client=request.app.state.groq_client,
        settings=settings,
    )
