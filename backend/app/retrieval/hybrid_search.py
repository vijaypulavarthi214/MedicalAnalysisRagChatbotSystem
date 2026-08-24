from app.embeddings.embedding_service import EmbeddingService
from app.models.schemas import HybridSearchResult
from app.retrieval.chroma_store import ChromaStore
from app.retrieval.lexical_search import LexicalSearchIndex

_bm25_cache: dict[str, LexicalSearchIndex] = {}


def invalidate_lexical_cache(document_id: str) -> None:
    _bm25_cache.pop(document_id, None)


def _get_lexical_index(document_id: str, chroma_store: ChromaStore) -> LexicalSearchIndex:
    if document_id not in _bm25_cache:
        chunks = chroma_store.get_document_chunks(document_id)
        _bm25_cache[document_id] = LexicalSearchIndex(chunks)
    return _bm25_cache[document_id]


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi == lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def hybrid_search(
    document_id: str,
    query: str,
    embedding_service: EmbeddingService,
    chroma_store: ChromaStore,
    top_k: int = 10,
    alpha: float = 0.7,
) -> list[HybridSearchResult]:
    """Fuse semantic (Chroma) and lexical (BM25) candidates into a single
    ranked list, weighted by `alpha` (semantic weight) vs `1 - alpha`
    (lexical weight)."""
    fetch_k = max(top_k * 2, top_k)

    query_embedding = embedding_service.embed_query(query)
    semantic_result = chroma_store.query(document_id, query_embedding, top_k=fetch_k)
    semantic_raw = {
        chunk_id: 1.0 - distance
        for chunk_id, distance in zip(semantic_result.ids, semantic_result.distances)
    }
    metadata_by_id = {
        chunk_id: meta for chunk_id, meta in zip(semantic_result.ids, semantic_result.metadatas)
    }
    text_by_id = {chunk_id: text for chunk_id, text in zip(semantic_result.ids, semantic_result.documents)}

    lexical_index = _get_lexical_index(document_id, chroma_store)
    lexical_hits = lexical_index.search(query, top_k=fetch_k)
    lexical_raw = {hit["chunk_id"]: hit["score"] for hit in lexical_hits}

    all_chunks_by_id = {c["chunk_id"]: c for c in chroma_store.get_document_chunks(document_id)}
    for chunk_id, chunk in all_chunks_by_id.items():
        text_by_id.setdefault(chunk_id, chunk["text"])
        metadata_by_id.setdefault(
            chunk_id,
            {
                "section_title": chunk["section_title"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
            },
        )

    semantic_norm = _normalize(semantic_raw)
    lexical_norm = _normalize(lexical_raw)

    all_ids = set(semantic_norm) | set(lexical_norm)
    results: list[HybridSearchResult] = []
    for chunk_id in all_ids:
        meta = metadata_by_id.get(chunk_id)
        text = text_by_id.get(chunk_id)
        if meta is None or text is None:
            continue
        semantic_score = semantic_norm.get(chunk_id, 0.0)
        lexical_score = lexical_norm.get(chunk_id, 0.0)
        combined_score = alpha * semantic_score + (1 - alpha) * lexical_score
        results.append(
            HybridSearchResult(
                chunk_id=chunk_id,
                text=text,
                section_title=meta["section_title"],
                page_start=meta["page_start"],
                page_end=meta["page_end"],
                semantic_score=semantic_score,
                lexical_score=lexical_score,
                combined_score=combined_score,
            )
        )

    results.sort(key=lambda r: r.combined_score, reverse=True)
    return results[:top_k]
