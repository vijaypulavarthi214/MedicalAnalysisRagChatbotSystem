import re
import statistics
import uuid

from app.models.schemas import PageContent, SectionBlock

_HEADER_KEYWORDS = re.compile(
    r"^(HISTORY OF PRESENT ILLNESS|CHIEF COMPLAINT|MEDICATIONS?|ALLERGIES|"
    r"ASSESSMENT( AND PLAN)?|PLAN|LABS?( AND STUDIES)?|VITALS?|VITAL SIGNS|"
    r"DIAGNOSIS|DIAGNOSES|PAST MEDICAL HISTORY|SOCIAL HISTORY|FAMILY HISTORY|"
    r"PHYSICAL EXAM(INATION)?|REVIEW OF SYSTEMS|IMPRESSION|SUMMARY|"
    r"DISCHARGE (SUMMARY|INSTRUCTIONS)|PROCEDURE(S)?|IMAGING|RESULTS)$",
    re.IGNORECASE,
)

_MAX_HEADER_LEN = 60


def _is_probable_header(block: dict, median_font_size: float) -> bool:
    text = block["text"].strip()
    if not text or len(text) > _MAX_HEADER_LEN:
        return False
    if _HEADER_KEYWORDS.match(text):
        return True
    is_larger = median_font_size > 0 and block["font_size"] >= median_font_size * 1.15
    is_shouty = text.isupper() and len(text.split()) <= 6
    if (is_larger or block["is_bold"]) and is_shouty:
        return True
    return False


def parse_sections(pages: list[PageContent]) -> list[SectionBlock]:
    """Split pages into sections by detecting header blocks (larger/bold font,
    short all-caps lines, or known medical-report header keywords). Falls back
    to one section per page when no headers are detected anywhere in the doc.
    """
    all_font_sizes = [b["font_size"] for page in pages for b in page.blocks if b["font_size"] > 0]
    median_font_size = statistics.median(all_font_sizes) if all_font_sizes else 0.0

    headers: list[tuple[int, str]] = []  # (page_number, header_text) in reading order
    for page in pages:
        for block in page.blocks:
            if _is_probable_header(block, median_font_size):
                headers.append((page.page_number, block["text"].strip()))

    if not headers:
        return [
            SectionBlock(
                section_id=str(uuid.uuid4()),
                section_title=f"Page {page.page_number}",
                page_start=page.page_number,
                page_end=page.page_number,
                text=page.text.strip(),
                order_index=i,
            )
            for i, page in enumerate(pages)
            if page.text.strip()
        ]

    pages_by_number = {p.page_number: p for p in pages}
    sections: list[SectionBlock] = []
    for idx, (page_number, title) in enumerate(headers):
        start_page = page_number
        if idx + 1 < len(headers):
            next_header_page = headers[idx + 1][0]
            end_page = next_header_page - 1 if next_header_page > start_page else start_page
        else:
            end_page = pages[-1].page_number
        text_parts = []
        for pn in range(start_page, end_page + 1):
            page = pages_by_number.get(pn)
            if page:
                text_parts.append(page.text)
        section_text = "\n".join(text_parts).strip()
        if not section_text:
            continue
        sections.append(
            SectionBlock(
                section_id=str(uuid.uuid4()),
                section_title=title,
                page_start=start_page,
                page_end=end_page,
                text=section_text,
                order_index=len(sections),
            )
        )

    return sections
