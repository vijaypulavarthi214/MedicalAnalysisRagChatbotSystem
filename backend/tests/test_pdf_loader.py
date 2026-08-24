import fitz
import pytest

from app.errors import IngestionError
from app.ingestion.pdf_loader import load_pdf


def _build_pdf(pages_text: list[str]) -> bytes:
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=12)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_load_pdf_extracts_page_count_and_text():
    pdf_bytes = _build_pdf(["Page one content", "Page two content"])
    pages = load_pdf(pdf_bytes)
    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert "Page one content" in pages[0].text
    assert pages[1].page_number == 2
    assert "Page two content" in pages[1].text


def test_load_pdf_extracts_blocks_with_font_info():
    pdf_bytes = _build_pdf(["Hello World"])
    pages = load_pdf(pdf_bytes)
    assert len(pages[0].blocks) > 0
    block = pages[0].blocks[0]
    assert "text" in block and "font_size" in block and "is_bold" in block and "bbox" in block
    assert block["font_size"] > 0


def test_load_pdf_raises_ingestion_error_on_invalid_bytes():
    with pytest.raises(IngestionError):
        load_pdf(b"not a real pdf")


