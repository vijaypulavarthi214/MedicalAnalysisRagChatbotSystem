import uuid

from app.models.schemas import Chunk, SectionBlock

TARGET_WORDS = 500
OVERLAP_RATIO = 0.1


def _split_words(words: list[str], target: int, overlap_ratio: float) -> list[list[str]]:
    if len(words) <= target:
        return [words]
    overlap = max(1, int(target * overlap_ratio))
    pieces: list[list[str]] = []
    start = 0
    while start < len(words):
        end = min(start + target, len(words))
        pieces.append(words[start:end])
        if end == len(words):
            break
        start = end - overlap
    return pieces


def chunk_sections(sections: list[SectionBlock], document_id: str) -> list[Chunk]:
    """Split each SectionBlock into ~TARGET_WORDS-word chunks with overlap,
    never crossing section boundaries. Chunk text is prefixed with a
    [Section: X | Page: N] context header.
    """
    chunks: list[Chunk] = []
    order_index = 0
    for section in sections:
        words = section.text.split()
        if not words:
            continue
        pieces = _split_words(words, TARGET_WORDS, OVERLAP_RATIO)
        for piece in pieces:
            body = " ".join(piece)
            page_label = (
                str(section.page_start)
                if section.page_start == section.page_end
                else f"{section.page_start}-{section.page_end}"
            )
            prefixed_text = f"[Section: {section.section_title} | Page: {page_label}]\n{body}"
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    section_id=section.section_id,
                    section_title=section.section_title,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    order_index=order_index,
                    text=prefixed_text,
                    token_count=len(piece),
                )
            )
            order_index += 1
    return chunks
