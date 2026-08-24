from app.retrieval.lexical_search import LexicalSearchIndex


def test_ranks_matching_chunk_above_non_matching():
    chunks = [
        {"chunk_id": "a", "text": "patient takes lisinopril for hypertension daily"},
        {"chunk_id": "b", "text": "patient reports no known drug allergies today"},
        {"chunk_id": "c", "text": "vitals were stable during the visit today"},
        {"chunk_id": "d", "text": "follow up appointment scheduled next month"},
    ]
    index = LexicalSearchIndex(chunks)

    results = index.search("lisinopril hypertension", top_k=4)

    assert results[0]["chunk_id"] == "a"
    assert results[0]["score"] > results[1]["score"]


def test_search_respects_top_k():
    chunks = [{"chunk_id": str(i), "text": f"chunk number {i} about diabetes"} for i in range(5)]
    index = LexicalSearchIndex(chunks)

    results = index.search("diabetes", top_k=3)

    assert len(results) == 3


def test_empty_corpus_returns_no_results():
    index = LexicalSearchIndex([])
    assert index.search("anything", top_k=5) == []
