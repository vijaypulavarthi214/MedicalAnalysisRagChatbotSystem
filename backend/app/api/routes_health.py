from fastapi import APIRouter

from app.config import get_settings
from app.embeddings.embedding_service import get_embedding_service
from app.retrieval.chroma_store import get_persistent_chroma_store

router = APIRouter()


@router.get("/health")
def health():
    settings = get_settings()
    embedding_ok = True
    try:
        get_embedding_service()
    except Exception:
        embedding_ok = False

    chroma_ok = True
    try:
        get_persistent_chroma_store()
    except Exception:
        chroma_ok = False

    return {
        "status": "ok",
        "chroma": chroma_ok,
        "embedding_model_loaded": embedding_ok,
        "groq_configured": bool(settings.groq_api_key),
    }
