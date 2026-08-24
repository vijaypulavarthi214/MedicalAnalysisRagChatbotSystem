from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_chroma_store
from app.api.routes_upload import _DOCUMENT_REGISTRY
from app.models.schemas import DocumentSummary
from app.retrieval.chroma_store import ChromaStore
from app.retrieval.hybrid_search import invalidate_lexical_cache

router = APIRouter()


@router.get("/documents", response_model=list[DocumentSummary])
def list_documents() -> list[DocumentSummary]:
    return list(_DOCUMENT_REGISTRY.values())


@router.delete("/documents/{document_id}")
def delete_document(document_id: str, chroma_store: ChromaStore = Depends(get_chroma_store)):
    if document_id not in _DOCUMENT_REGISTRY:
        raise HTTPException(status_code=404, detail="Document not found")

    chroma_store.delete_document(document_id)
    invalidate_lexical_cache(document_id)
    del _DOCUMENT_REGISTRY[document_id]
    return {"status": "deleted", "document_id": document_id}
