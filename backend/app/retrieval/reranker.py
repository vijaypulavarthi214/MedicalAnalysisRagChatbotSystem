from app.errors import RerankError
from app.models.schemas import HybridSearchResult, RerankedChunk


def rerank(query: str, candidates: list[HybridSearchResult], cohere_client, model: str, top_n: int = 3) -> list[RerankedChunk]:
    """Narrow hybrid-search candidates down to `top_n` via Cohere rerank.
    Raises RerankError on any Cohere failure — never silently falls back
    to un-reranked order.
    """
    if not candidates:
        return []
    try:
        response = cohere_client.rerank(
            model=model,
            query=query,
            documents=[c.text for c in candidates],
            top_n=min(top_n, len(candidates)),
        )
    except Exception as exc:
        raise RerankError(f"Cohere rerank failed: {type(exc).__name__}: {exc}") from exc

    reranked: list[RerankedChunk] = []
    for result in response.results:
        candidate = candidates[result.index]
        reranked.append(
            RerankedChunk(
                chunk_id=candidate.chunk_id,
                text=candidate.text,
                section_title=candidate.section_title,
                page_start=candidate.page_start,
                page_end=candidate.page_end,
                relevance_score=result.relevance_score,
            )
        )
    return reranked
