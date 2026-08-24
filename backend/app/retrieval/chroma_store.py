from functools import lru_cache

import chromadb
from chromadb.api import ClientAPI

from app.config import get_settings
from app.models.schemas import Chunk, ChromaQueryResult

_COLLECTION_NAME = "medical_chunks"


class ChromaStore:
    """Single Chroma collection shared by all documents, filtered by the
    `document_id` metadata field rather than one collection per document.
    """

    def __init__(self, client: ClientAPI):
        self._client = client
        self._collection = client.get_or_create_collection(_COLLECTION_NAME)

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "document_id": c.document_id,
                    "section_id": c.section_id,
                    "section_title": c.section_title,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "order_index": c.order_index,
                }
                for c in chunks
            ],
        )

    def query(self, document_id: str, query_embedding: list[float], top_k: int) -> ChromaQueryResult:
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"document_id": document_id},
        )
        return ChromaQueryResult(
            ids=result["ids"][0] if result["ids"] else [],
            documents=result["documents"][0] if result["documents"] else [],
            metadatas=result["metadatas"][0] if result["metadatas"] else [],
            distances=result["distances"][0] if result["distances"] else [],
        )

    def get_document_chunks(self, document_id: str) -> list[dict]:
        result = self._collection.get(where={"document_id": document_id}, include=["documents", "metadatas"])
        chunks = []
        for chunk_id, text, meta in zip(result["ids"], result["documents"], result["metadatas"]):
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "section_title": meta["section_title"],
                    "page_start": meta["page_start"],
                    "page_end": meta["page_end"],
                }
            )
        return chunks

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})

    def list_documents(self) -> dict[str, int]:
        """Returns {document_id: chunk_count} for all documents in the store."""
        result = self._collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for meta in result["metadatas"]:
            doc_id = meta["document_id"]
            counts[doc_id] = counts.get(doc_id, 0) + 1
        return counts


@lru_cache
def get_persistent_chroma_store() -> ChromaStore:
    settings = get_settings()
    client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
    return ChromaStore(client)
