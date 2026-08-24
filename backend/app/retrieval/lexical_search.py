import re

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class LexicalSearchIndex:
    """BM25 index over a document's chunks. Chunk dicts need at minimum
    {"chunk_id", "text"} keys."""

    def __init__(self, chunks: list[dict]):
        self._chunk_ids = [c["chunk_id"] for c in chunks]
        corpus = [_tokenize(c["text"]) for c in chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, top_k: int) -> list[dict]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self._chunk_ids, scores), key=lambda pair: pair[1], reverse=True)
        return [{"chunk_id": chunk_id, "score": float(score)} for chunk_id, score in ranked[:top_k]]
