from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.deps import get_rag_pipeline
from app.config import get_settings
from app.errors import IngestionError, RetrievalError
from app.logging_config import get_logger
from app.models.schemas import DocumentSummary, UploadResponse
from app.rag.pipeline import RagPipeline
from app.utils.sanitize import sanitize_filename

router = APIRouter()
logger = get_logger(__name__)

# In-memory document registry for this prototype — chunk data lives in Chroma,
# this just tracks upload metadata for /documents listing. Not persisted
# across restarts; acceptable for a single-instance deployment.
_DOCUMENT_REGISTRY: dict[str, DocumentSummary] = {}


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    pipeline: RagPipeline = Depends(get_rag_pipeline),
) -> UploadResponse:
    settings = get_settings()

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF files are supported")

    content = await file.read()
    if len(content) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=413, detail=f"File exceeds the {settings.max_file_size_mb}MB limit"
        )

    safe_filename = sanitize_filename(file.filename)

    try:
        result = pipeline.ingest_document(content, safe_filename)
    except (IngestionError, RetrievalError) as exc:
        logger.warning("ingestion_failed error_type=%s", type(exc).__name__)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    _DOCUMENT_REGISTRY[result.document_id] = DocumentSummary(
        document_id=result.document_id,
        filename=result.filename,
        chunk_count=result.chunk_count,
        page_count=result.page_count,
        uploaded_at=datetime.now(timezone.utc).isoformat(),
    )
    logger.info(
        "document_ingested document_id=%s chunks=%d pages=%d",
        result.document_id,
        result.chunk_count,
        result.page_count,
    )
    return result
