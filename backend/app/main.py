from contextlib import asynccontextmanager

import cohere
import groq
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_documents import router as documents_router
from app.api.routes_health import router as health_router
from app.api.routes_query import router as query_router
from app.api.routes_upload import router as upload_router
from app.config import get_settings
from app.embeddings.embedding_service import get_embedding_service
from app.logging_config import configure_logging, get_logger
from app.retrieval.chroma_store import get_persistent_chroma_store

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()
    get_embedding_service()
    get_persistent_chroma_store()
    app.state.cohere_client = cohere.ClientV2(api_key=settings.cohere_api_key)
    app.state.groq_client = groq.AsyncGroq(api_key=settings.groq_api_key)
    logger.info("startup_complete")
    yield
    await app.state.groq_client.close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Medical Document RAG API", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(upload_router)
    app.include_router(documents_router)
    app.include_router(query_router)

    return app


app = create_app()
