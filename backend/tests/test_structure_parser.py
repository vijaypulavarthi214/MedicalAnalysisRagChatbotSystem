from app.ingestion.structure_parser import parse_sections
from app.models.schemas import PageContent


def _block(text, font_size=10.0, is_bold=False, block_no=0):
    return {"text": text, "bbox": (0, 0, 0, 0), "font_size": font_size, "is_bold": is_bold, "block_no": block_no}


def test_parse_sections_detects_keyword_headers():
    pages = [
        PageContent(
            page_number=1,
            text="MEDICATIONS\nLisinopril 10mg daily\nMetformin 500mg twice daily",
            blocks=[
                _block("MEDICATIONS", font_size=14.0, is_bold=True, block_no=0),
                _block("Lisinopril 10mg daily", block_no=1),
                _block("Metformin 500mg twice daily", block_no=2),
            ],
        ),
        PageContent(
            page_number=2,
            text="ASSESSMENT\nPatient stable, continue current regimen.",
            blocks=[
                _block("ASSESSMENT", font_size=14.0, is_bold=True, block_no=0),
                _block("Patient stable, continue current regimen.", block_no=1),
            ],
        ),
    ]

    sections = parse_sections(pages)

    assert len(sections) == 2
    assert sections[0].section_title == "MEDICATIONS"
    assert sections[0].page_start == 1
    assert sections[0].page_end == 1
    assert "Lisinopril" in sections[0].text
    assert sections[1].section_title == "ASSESSMENT"
    assert sections[1].page_start == 2
    assert sections[1].order_index == 1


def test_parse_sections_falls_back_to_one_section_per_page_when_no_headers():
    pages = [
        PageContent(page_number=1, text="Just some plain body text.", blocks=[_block("Just some plain body text.")]),
        PageContent(page_number=2, text="More plain body text.", blocks=[_block("More plain body text.")]),
    ]

    sections = parse_sections(pages)

    assert len(sections) == 2
    assert sections[0].section_title == "Page 1"
    assert sections[1].section_title == "Page 2"


def test_parse_sections_ignores_long_lines_as_headers():
    long_line = "This is a very long line of body text that should never be mistaken for a section header no matter the font"
    pages = [
        PageContent(
            page_number=1,
            text=long_line,
            blocks=[_block(long_line, font_size=20.0, is_bold=True)],
        )
    ]

    sections = parse_sections(pages)

    assert len(sections) == 1
    assert sections[0].section_title == "Page 1"
