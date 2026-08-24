import fitz

from app.errors import IngestionError
from app.models.schemas import PageContent


def load_pdf(pdf_bytes: bytes) -> list[PageContent]:
    """Extract per-page text and font-annotated blocks from a PDF.

    Each block dict: {"text", "bbox", "font_size", "is_bold", "block_no"} —
    used downstream by structure_parser to detect section headers.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise IngestionError(f"Failed to open PDF: {type(exc).__name__}") from exc

    if doc.page_count == 0:
        doc.close()
        raise IngestionError("PDF has no pages")

    pages: list[PageContent] = []
    try:
        for page_index in range(doc.page_count):
            page = doc[page_index]
            page_text = page.get_text("text")
            raw = page.get_text("dict")
            blocks: list[dict] = []
            for block_no, block in enumerate(raw.get("blocks", [])):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        flags = span.get("flags", 0)
                        font_name = span.get("font", "")
                        blocks.append(
                            {
                                "text": text,
                                "bbox": tuple(span.get("bbox", (0, 0, 0, 0))),
                                "font_size": float(span.get("size", 0.0)),
                                "is_bold": bool(flags & (1 << 4)) or "bold" in font_name.lower(),
                                "block_no": block_no,
                            }
                        )
            pages.append(
                PageContent(page_number=page_index + 1, text=page_text, blocks=blocks)
            )
    finally:
        doc.close()

    return pages
