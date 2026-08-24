from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import get_settings


class EmbeddingService:
    """Wraps a single SentenceTransformer instance so indexing and querying
    always use the exact same model — required for embedding parity.
    """

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, batch_size=32, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode([text], show_progress_bar=False)[0]
        return vector.tolist()


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(get_settings().embedding_model)
