from app.embeddings.embedding_service import EmbeddingService


def test_embed_texts_and_embed_query_same_dimension():
    service = EmbeddingService("sentence-transformers/all-MiniLM-L6-v2")
    vectors = service.embed_texts(["hello world", "patient has diabetes"])
    query_vector = service.embed_query("hello world")
    assert len(vectors) == 2
    assert len(vectors[0]) == len(query_vector)
    assert len(query_vector) == 384


def test_embed_query_is_consistent_for_same_text():
    service = EmbeddingService("sentence-transformers/all-MiniLM-L6-v2")
    v1 = service.embed_query("blood pressure 120/80")
    v2 = service.embed_query("blood pressure 120/80")
    assert v1 == v2
