from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_rag_pipeline
from app.api.routes_upload import _DOCUMENT_REGISTRY
from app.errors import ConfigError, GenerationError, RerankError, RetrievalError
from app.logging_config import get_logger
from app.models.schemas import QueryRequest, QueryResponse
from app.rag.pipeline import RagPipeline

router = APIRouter()
logger = get_logger(__name__)


@router.post("/query", response_model=QueryResponse)
async def query_document(
    payload: QueryRequest,
    pipeline: RagPipeline = Depends(get_rag_pipeline),
) -> QueryResponse:
    if payload.document_id not in _DOCUMENT_REGISTRY:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        return await pipeline.answer_query(payload.question, payload.document_id)
    except ConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RerankError as exc:
        logger.warning("rerank_failed error_type=%s", type(exc).__name__)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except GenerationError as exc:
        logger.warning("generation_failed error_type=%s", type(exc).__name__)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except RetrievalError as exc:
        logger.warning("retrieval_failed error_type=%s", type(exc).__name__)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
