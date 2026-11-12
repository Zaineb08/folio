"""PDF parsing, on synthetic PDFs built to contain the problems real papers have."""

import pymupdf

from folio.pdf import parse_pdf

SENTENCE = "The participants in the study reported substantially higher engagement with the new method. "


def _write(page, x, y, text, size=10, width=250, bold=False):
    rect = pymupdf.Rect(x, y, x + width, y + 700)
    page.insert_textbox(rect, text, fontsize=size, fontname="hebo" if bold else "helv")


def _save(doc, tmp_path, name="paper.pdf"):
    path = tmp_path / name
    doc.save(path)
    doc.close()
    return str(path)


def test_two_columns_are_read_in_order(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    left = " ".join(f"Leftcolumn{i} sentence goes here." for i in range(40))
    right = " ".join(f"Rightcolumn{i} sentence goes here." for i in range(40))
    _write(page, 50, 60, left)
    _write(page, 320, 60, right)
    result = parse_pdf(_save(doc, tmp_path))
    text = " ".join(chunk["text"] for chunk in result["chunks"])
    assert result["status"] == "ok"
    assert text.index("Leftcolumn39") < text.index("Rightcolumn0")


def test_running_headers_and_page_numbers_are_removed(tmp_path):
    doc = pymupdf.open()
    for number in range(6):
        page = doc.new_page()
        page.insert_text((60, 30), "Journal of Repeated Headers, Volume 12", fontsize=9)
        page.insert_text((300, 820), str(number + 1), fontsize=9)
        _write(page, 60, 80, SENTENCE * 20, width=480)
    result = parse_pdf(_save(doc, tmp_path))
    text = " ".join(chunk["text"] for chunk in result["chunks"])
    assert "Repeated Headers" not in text
    assert "engagement" in text


def test_references_are_excluded_but_appendix_is_kept(tmp_path):
    doc = pymupdf.open()
    for _ in range(3):
        _write(doc.new_page(), 60, 60, SENTENCE * 25, width=480)
    page = doc.new_page()
    page.insert_text((60, 60), "References", fontsize=14, fontname="hebo")
    _write(page, 60, 90, "Fakename, A. (1999). A cited work that must not be indexed. Journal of Things.\n" * 8, width=480)
    page = doc.new_page()
    page.insert_text((60, 60), "Appendix A", fontsize=14, fontname="hebo")
    _write(page, 60, 90, "The questionnaire included detailed items on engagement across the sample. " * 12, width=480)
    result = parse_pdf(_save(doc, tmp_path))
    text = " ".join(chunk["text"] for chunk in result["chunks"])
    assert "Fakename" not in text
    assert "questionnaire" in text


def test_sections_are_tracked_and_acknowledgements_skipped(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((60, 60), "2. Methods", fontsize=14, fontname="hebo")
    _write(page, 60, 80, "We interviewed many teachers about their experience in rural schools. " * 12, width=480)
    page = doc.new_page()
    page.insert_text((60, 60), "Acknowledgements", fontsize=14, fontname="hebo")
    _write(page, 60, 80, "We thank Generous Foundation for its kind support of this research project. " * 6, width=480)
    result = parse_pdf(_save(doc, tmp_path))
    assert result["chunks"][0]["section"] == "Methods"
    assert not any("Generous" in chunk["text"] for chunk in result["chunks"])


def _filler_pages(doc, count):
    for _ in range(count):
        _write(doc.new_page(), 60, 60, SENTENCE * 25, width=480)


def test_references_before_long_lettered_appendices_are_excluded(tmp_path):
    """ACL style: references on page 9 of 24, then appendices headed "A ...", "B ..." without the word Appendix."""
    doc = pymupdf.open()
    _filler_pages(doc, 8)
    page = doc.new_page()
    page.insert_text((60, 60), "References", fontsize=12, fontname="hebo")
    _write(page, 60, 90, "Fakename, A. 2024. A cited work that must not be indexed. In Proceedings of Things.\n" * 8, width=480)
    page = doc.new_page()
    page.insert_text((60, 60), "A Unsupervised Uncertainty Methods", fontsize=12, fontname="hebo")
    _write(page, 60, 90, "The questionnaire included detailed items on engagement across the sample. " * 12, width=480)
    _filler_pages(doc, 14)
    result = parse_pdf(_save(doc, tmp_path))
    text = " ".join(chunk["text"] for chunk in result["chunks"])
    assert "Fakename" not in text
    assert "questionnaire" in text
    assert "A Unsupervised Uncertainty Methods" in {chunk["section"] for chunk in result["chunks"]}


def test_an_appendix_letter_on_its_own_line_still_ends_the_references(tmp_path):
    doc = pymupdf.open()
    _filler_pages(doc, 8)
    page = doc.new_page()
    page.insert_text((60, 60), "References", fontsize=12, fontname="hebo")
    _write(page, 60, 90, "Fakename, A. 2024. A cited work that must not be indexed. In Proceedings of Things.\n" * 8, width=480)
    page = doc.new_page()
    page.insert_text((60, 60), "C", fontsize=12, fontname="hebo")
    page.insert_text((60, 75), "Quality of Dataset Annotation", fontsize=12, fontname="hebo")
    _write(page, 60, 100, "The questionnaire included detailed items on engagement across the sample. " * 12, width=480)
    _filler_pages(doc, 14)
    text = " ".join(chunk["text"] for chunk in parse_pdf(_save(doc, tmp_path))["chunks"])
    assert "Fakename" not in text and "questionnaire" in text


def test_a_reference_entry_starting_with_an_initial_doesnt_end_the_references(tmp_path):
    doc = pymupdf.open()
    _filler_pages(doc, 3)
    page = doc.new_page()
    page.insert_text((60, 60), "References", fontsize=14, fontname="hebo")
    _write(page, 60, 90, "A. Fakename and B. Othername. 2024. A cited work that must not be indexed. Journal of Things.\n" * 8, width=480)
    result = parse_pdf(_save(doc, tmp_path))
    assert "Fakename" not in " ".join(chunk["text"] for chunk in result["chunks"])


def test_a_references_line_in_an_early_contents_list_doesnt_end_the_document(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    _write(page, 60, 60, SENTENCE * 5, width=480)
    page = doc.new_page()
    page.insert_text((60, 60), "References", fontsize=12, fontname="hebo")
    _write(page, 60, 90, SENTENCE * 5, width=480)
    _filler_pages(doc, 6)
    page = doc.new_page()
    _write(page, 60, 60, "The questionnaire included detailed items on engagement across the sample. " * 12, width=480)
    _filler_pages(doc, 3)
    result = parse_pdf(_save(doc, tmp_path))
    assert "questionnaire" in " ".join(chunk["text"] for chunk in result["chunks"])


def test_references_are_found_when_number_and_word_are_on_separate_lines(tmp_path):
    doc = pymupdf.open()
    _filler_pages(doc, 3)
    page = doc.new_page()
    page.insert_text((60, 60), "7", fontsize=12, fontname="hebo")
    page.insert_text((60, 74), "REFERENCES", fontsize=12, fontname="hebo")
    _write(page, 60, 100, "[1] Fakename, A. A cited work that must not be indexed in any way.\n" * 8, width=480)
    result = parse_pdf(_save(doc, tmp_path))
    assert "Fakename" not in " ".join(chunk["text"] for chunk in result["chunks"])


def test_ieee_style_body_size_capital_heading_is_recognised(tmp_path):
    doc = pymupdf.open()
    _filler_pages(doc, 1)
    page = doc.new_page()
    page.insert_text((60, 60), "II. OVERVIEW OF RAG", fontsize=10)
    _write(page, 60, 80, SENTENCE * 15, width=480)
    result = parse_pdf(_save(doc, tmp_path))
    assert "OVERVIEW OF RAG" in {chunk["section"] for chunk in result["chunks"]}


def test_captions_and_first_page_furniture_are_not_sections(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((60, 40), "Research article", fontsize=11, fontname="hebo")
    page.insert_text((60, 60), "Jane Author1*, John Writer2", fontsize=12, fontname="hebo")
    _write(page, 60, 90, SENTENCE * 25, width=480)
    page = doc.new_page()
    page.insert_text((60, 60), "Figure 1", fontsize=11, fontname="hebo")
    _write(page, 60, 80, SENTENCE * 25, width=480)
    result = parse_pdf(_save(doc, tmp_path))
    sections = {chunk["section"] for chunk in result["chunks"]}
    assert sections == {""}


def test_bullet_is_stripped_from_section_names(tmp_path):
    doc = pymupdf.open()
    _filler_pages(doc, 1)
    page = doc.new_page()
    page.insert_text((60, 60), "• Data extraction", fontsize=11, fontname="hebo")
    _write(page, 60, 80, SENTENCE * 25, width=480)
    result = parse_pdf(_save(doc, tmp_path))
    assert "Data extraction" in {chunk["section"] for chunk in result["chunks"]}


def test_line_end_hyphens_are_rejoined(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    y = 60
    for line in ["The researchers discussed the method-", "ological choices in the first study today."] * 8:
        page.insert_text((60, y), line, fontsize=10)
        y += 13
    result = parse_pdf(_save(doc, tmp_path))
    assert "methodological" in result["chunks"][0]["text"]


def test_chunks_have_page_and_rects_for_highlighting(tmp_path):
    doc = pymupdf.open()
    for _ in range(3):
        _write(doc.new_page(), 60, 60, SENTENCE * 40, width=480)
    result = parse_pdf(_save(doc, tmp_path))
    assert len(result["chunks"]) > 1
    for chunk in result["chunks"]:
        assert 1 <= chunk["page"] <= chunk["page_end"] <= 3
        assert chunk["rects"]
        page, x0, y0, x1, y1 = chunk["rects"][0]
        assert page == chunk["page"] and x1 > x0 and y1 > y0
        assert len(chunk["text"].split()) <= 400


def test_scan_without_text_is_reported_not_silently_empty(tmp_path):
    doc = pymupdf.open()
    doc.new_page()
    doc.new_page()
    result = parse_pdf(_save(doc, tmp_path))
    assert result["status"] == "scanned"
    assert result["chunks"] == []
    assert result["error"]


def test_damaged_file_is_reported(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"this is not a pdf at all")
    result = parse_pdf(str(path))
    assert result["status"] == "error"
    assert result["chunks"] == []


def test_password_protected_pdf_is_reported(tmp_path):
    doc = pymupdf.open()
    _write(doc.new_page(), 60, 60, SENTENCE * 10, width=480)
    path = tmp_path / "locked.pdf"
    doc.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="secret")
    doc.close()
    result = parse_pdf(str(path))
    assert result["status"] == "encrypted"


def test_rotated_margin_stamp_is_ignored(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((20, 500), "arXiv:2005.11401v4 [cs.CL] 12 Apr 2021", fontsize=12, rotate=90)
    _write(page, 60, 60, SENTENCE * 20, width=480)
    result = parse_pdf(_save(doc, tmp_path))
    assert "arXiv" not in " ".join(chunk["text"] for chunk in result["chunks"])
