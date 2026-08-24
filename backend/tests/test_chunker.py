import uuid

from app.ingestion.chunker import chunk_sections
from app.models.schemas import SectionBlock


def _section(text, title="MEDICATIONS", page_start=1, page_end=1, order_index=0):
    return SectionBlock(
        section_id=str(uuid.uuid4()),
        section_title=title,
        page_start=page_start,
        page_end=page_end,
        text=text,
        order_index=order_index,
    )


def test_small_section_stays_as_one_chunk():
    section = _section("word " * 50)
    chunks = chunk_sections([section], document_id="doc-1")
    assert len(chunks) == 1
    assert chunks[0].token_count == 50
    assert chunks[0].section_title == "MEDICATIONS"
    assert chunks[0].document_id == "doc-1"
    assert chunks[0].text.startswith("[Section: MEDICATIONS | Page: 1]")


def test_large_section_splits_into_target_sized_chunks_with_overlap():
    section = _section("word " * 1200)
    chunks = chunk_sections([section], document_id="doc-1")
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= 500 + 20
    # overlap: consecutive chunk bodies share trailing/leading words
    first_words = chunks[0].text.split("\n", 1)[1].split()
    second_words = chunks[1].text.split("\n", 1)[1].split()
    assert first_words[-10:] == second_words[:10]


def test_chunks_never_cross_section_boundaries():
    section_a = _section("alpha " * 30, title="HISTORY", page_start=1, page_end=1, order_index=0)
    section_b = _section("beta " * 30, title="ASSESSMENT", page_start=2, page_end=2, order_index=1)
    chunks = chunk_sections([section_a, section_b], document_id="doc-1")
    assert len(chunks) == 2
    assert "beta" not in chunks[0].text
    assert "alpha" not in chunks[1].text
    assert chunks[0].order_index == 0
    assert chunks[1].order_index == 1


def test_empty_section_produces_no_chunks():
    section = _section("   ")
    chunks = chunk_sections([section], document_id="doc-1")
    assert chunks == []


def test_page_range_label_for_multi_page_section():
    section = _section("word " * 10, page_start=1, page_end=3)
    chunks = chunk_sections([section], document_id="doc-1")
    assert chunks[0].text.startswith("[Section: MEDICATIONS | Page: 1-3]")
