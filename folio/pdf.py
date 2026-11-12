"""Turn an academic PDF into clean, page-anchored passages.

Handles what generic text extraction gets wrong on papers: two-column reading
order, running headers and footers, margin stamps, hyphenated line breaks, and
bibliographies (which would otherwise pollute every search). Each passage keeps
the rectangles it came from, so the viewer can highlight it on the real page.

This module must stay light to import: it runs in worker processes.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import pymupdf

pymupdf.TOOLS.mupdf_display_errors(False)
pymupdf.TOOLS.mupdf_display_warnings(False)

PARSER_VERSION = 2  # Raise when parsing changes, so existing articles get re-indexed. 2: lettered appendices.

TARGET_WORDS = 260
MAX_WORDS = 360
OVERLAP_WORDS = 40
MIN_CHUNK_WORDS = 25
MIN_CHARS_PER_PAGE = 60  # Below this, a PDF is almost certainly a scan with no text layer.

_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st", "­": ""}
_LIGATURE_TABLE = str.maketrans(_LIGATURES)
_CID = re.compile(r"\(cid:\d+\)")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACES = re.compile(r"\s+")

_NUMBERING = r"(?:\d+(?:\.\d+)*\.?|[IVXLC]+\.|[A-Z]\.)\s+"
_NUMBERED_HEADING = re.compile(rf"^{_NUMBERING}\S")
_STRIP_NUMBERING = re.compile(rf"^{_NUMBERING}")
_REFERENCES = re.compile(
    rf"^(?:{_NUMBERING})?(references|bibliography|works cited|reference list|literature cited|"
    r"literatur|literaturverzeichnis|références|références bibliographiques|bibliographie|"
    r"referencias|bibliografía|riferimenti|riferimenti bibliografici)\s*:?$",
    re.IGNORECASE,
)
_RESUME_AFTER_REFERENCES = re.compile(rf"^(?:{_NUMBERING})?(appendix|appendices|supplement|annex)", re.IGNORECASE)
# ACL and similar styles letter their appendices ("A Unsupervised Methods") without the word "Appendix".
_LETTERED_APPENDIX = re.compile(r"^[A-Z](?:\.\d+)*\.?\s+[A-Z][a-z]")
# A "References" heading only counts after this share of the pages, so a contents list can't end a paper early.
# Long appendices can put the references well before the middle: page 9 of 24 in one ACL paper.
_REFERENCES_FROM = 0.25
_SKIPPED_SECTIONS = re.compile(
    rf"^(?:{_NUMBERING})?(acknowledge?ments?|funding|competing interests?|conflicts? of interest|"
    r"declaration of (competing )?interests?|author contributions?|credit authorship.*)\s*:?$",
    re.IGNORECASE,
)
_COMMON_HEADINGS = {
    "abstract", "summary", "introduction", "background", "related work", "literature review",
    "theoretical framework", "theory", "method", "methods", "methodology", "materials and methods",
    "research design", "data", "data and methods", "results", "findings", "analysis", "evaluation",
    "experiments", "discussion", "results and discussion", "implications", "limitations",
    "conclusion", "conclusions", "future work", "conclusion and future work", "appendix",
}
ABBREVIATIONS = {
    "al.", "fig.", "figs.", "eq.", "eqs.", "e.g.", "i.e.", "cf.", "vs.", "dr.", "prof.", "no.", "nos.",
    "vol.", "pp.", "p.", "ref.", "refs.", "sec.", "tab.", "approx.", "ca.", "st.", "mr.", "ms.", "mrs.",
    "jr.", "sr.", "inc.", "ltd.", "ed.", "eds.", "ch.", "resp.",
}
_BULLET = re.compile(r"^[•·▪‣◦∙●]\s*")
_CAPTION = re.compile(r"^(figure|fig\.|table|tab\.|box|scheme|algorithm|listing|supplementary figure)\s*[\dA-Z]?", re.IGNORECASE)
_INITIAL = re.compile(r"^[A-Z]\.$")
_TERMINAL = re.compile(r"[.?!][)\]\"'’”]*$")


@dataclass
class _Line:
    page: int
    bbox: tuple
    text: str


@dataclass
class _Block:
    page: int
    bbox: tuple
    lines: list[_Line]
    size: float
    bold: bool
    italic: bool = False

    @property
    def text(self) -> str:
        return " ".join(line.text for line in self.lines)


@dataclass
class _Chunk:
    section: str
    words: list = field(default_factory=list)


def _clean(text: str) -> str:
    text = _CID.sub("", text).translate(_LIGATURE_TABLE)
    text = _CONTROL.sub("", text)
    return _SPACES.sub(" ", text).strip()


def _page_blocks(page: pymupdf.Page, number: int, size_counts: Counter) -> list[_Block]:
    """Extract text blocks, splitting any block where the font changes (a heading run into a paragraph)."""
    blocks: list[_Block] = []
    data = page.get_text("dict", flags=pymupdf.TEXT_MEDIABOX_CLIP)
    for raw in data.get("blocks", []):
        if raw.get("type") != 0:
            continue
        current: list[tuple[_Line, float, bool, bool]] = []
        for line in raw.get("lines", []):
            # Rotated text is margin furniture, such as arXiv's identifier stamp.
            if abs(line["dir"][1]) > 0.1:
                continue
            text_parts, sizes, chars, bold_chars, italic_chars = [], Counter(), 0, 0, 0
            for span in line.get("spans", []):
                text = span.get("text", "")
                text_parts.append(text)
                length = len(text.strip())
                if not length:
                    continue
                sizes[round(span["size"], 1)] += length
                chars += length
                font = span.get("font", "").lower()
                flags = span.get("flags", 0)
                if flags & 16 or "bold" in font or "black" in font:
                    bold_chars += length
                if flags & 2 or "italic" in font or "oblique" in font:
                    italic_chars += length
            text = _clean("".join(text_parts))
            if not text or not chars:
                continue
            size = sizes.most_common(1)[0][0]
            bold = bold_chars / chars > 0.6
            italic = italic_chars / chars > 0.6
            size_counts.update({size: chars})
            info = (_Line(number, tuple(line["bbox"]), text), size, bold, italic)
            if current and (abs(current[-1][1] - size) > 0.6 or current[-1][2] != bold or current[-1][3] != italic):
                blocks.append(_make_block(number, current))
                current = []
            current.append(info)
        if current:
            blocks.append(_make_block(number, current))
    return blocks


def _make_block(number: int, parts: list[tuple[_Line, float, bool, bool]]) -> _Block:
    lines = [part[0] for part in parts]
    x0 = min(line.bbox[0] for line in lines)
    y0 = min(line.bbox[1] for line in lines)
    x1 = max(line.bbox[2] for line in lines)
    y1 = max(line.bbox[3] for line in lines)
    return _Block(number, (x0, y0, x1, y1), lines, parts[0][1], parts[0][2], parts[0][3])


def _reading_order(blocks: list[_Block], width: float) -> list[_Block]:
    """Order blocks column by column; a block spanning the gutter ends the columns above it."""
    middle = width / 2
    tolerance = width * 0.04
    ordered: list[_Block] = []
    left: list[_Block] = []
    right: list[_Block] = []

    def flush() -> None:
        ordered.extend(sorted(left, key=lambda b: b.bbox[1]))
        ordered.extend(sorted(right, key=lambda b: b.bbox[1]))
        left.clear()
        right.clear()

    for block in sorted(blocks, key=lambda b: (round(b.bbox[1]), b.bbox[0])):
        x0, _, x1, _ = block.bbox
        if x0 < middle - tolerance and x1 > middle + tolerance:
            flush()
            ordered.append(block)
        elif x1 <= middle + tolerance:
            left.append(block)
        else:
            right.append(block)
    flush()
    return ordered


def _strip_furniture(pages: list[list[_Block]], heights: list[float]) -> list[list[_Block]]:
    """Remove page numbers and running headers and footers."""
    margin = 0.1
    counts: Counter = Counter()

    def in_margin(block: _Block, height: float) -> bool:
        return block.bbox[3] < height * margin or block.bbox[1] > height * (1 - margin)

    def signature(block: _Block) -> str:
        return re.sub(r"\d+", "#", block.text.lower()).strip()

    for blocks, height in zip(pages, heights):
        seen = {signature(block) for block in blocks if in_margin(block, height)}
        counts.update(seen)

    threshold = max(3, int(0.25 * len(pages)))
    cleaned = []
    for blocks, height in zip(pages, heights):
        kept = []
        for block in blocks:
            if in_margin(block, height):
                sig = signature(block)
                if counts[sig] >= threshold or re.fullmatch(r"(page\s*)?[#\s\-–/of]*", sig):
                    continue
            kept.append(block)
        cleaned.append(kept)
    return cleaned


def _is_heading(block: _Block, body_size: float, first_page: bool = False) -> bool:
    text = _BULLET.sub("", block.text.strip())
    words = text.split()
    if not 1 <= len(words) <= 14 or len(block.lines) > 3:
        return False
    if sum(c.isalpha() for c in text) < 3:
        return False
    plain = _STRIP_NUMBERING.sub("", text).rstrip(":. ").lower()
    if plain in _COMMON_HEADINGS:
        return True
    numbered = bool(_NUMBERED_HEADING.match(text))
    # The rest of a first page is title, authors and journal furniture, not sections.
    if (first_page and not numbered) or _CAPTION.match(text):
        return False
    if text[-1] in ".,;" or (words[0][0].islower() and not numbered):
        return False
    styled = block.size >= body_size * 1.12 or (block.bold and block.size >= body_size * 0.95)
    if styled:
        return True
    # IEEE and similar styles set headings at body size, in capitals or italics.
    letters = [c for c in _STRIP_NUMBERING.sub("", text) if c.isalpha()]
    return numbered and len(words) <= 10 and (block.italic or all(c.isupper() for c in letters))


def _is_junk(block: _Block) -> bool:
    """Equations, axis labels and number grids are noise for search."""
    text = block.text
    letters = sum(c.isalpha() for c in text)
    return len(text) < 200 and letters / max(len(text), 1) < 0.3


def _block_words(block: _Block, lines: list[_Line]) -> list[tuple[str, int]]:
    words: list[tuple[str, int]] = []
    for line in block.lines:
        lines.append(line)
        index = len(lines) - 1
        tokens = line.text.split()
        if words and tokens and len(words[-1][0]) > 2 and words[-1][0].endswith("-") and tokens[0][0].islower():
            tokens[0] = words.pop()[0][:-1] + tokens[0]
        words.extend((token, index) for token in tokens)
    return words


def _is_sentence_end(word: str, following: str | None) -> bool:
    if not _TERMINAL.search(word):
        return False
    bare = word.rstrip(")]\"'’”")
    if bare.lower() in ABBREVIATIONS or _INITIAL.match(bare):
        return False
    return following is None or not following[0].islower()


def _merge_rects(indices: list[int], lines: list[_Line]) -> list[list[float]]:
    """Merge a passage's lines into as few page rectangles as possible."""
    rects: list[list[float]] = []
    for index in sorted(set(indices)):
        line = lines[index]
        x0, y0, x1, y1 = line.bbox
        if rects:
            last = rects[-1]
            overlaps = min(x1, last[3]) - max(x0, last[1]) > 0
            close = y0 - last[4] < (y1 - y0) * 1.2
            if last[0] == line.page and overlaps and close and y0 >= last[2] - 2:
                last[1], last[2] = min(last[1], x0), min(last[2], y0)
                last[3], last[4] = max(last[3], x1), max(last[4], y1)
                continue
        rects.append([line.page, x0, y0, x1, y1])
    return [[r[0], round(r[1], 1), round(r[2], 1), round(r[3], 1), round(r[4], 1)] for r in rects]


class _Chunker:
    def __init__(self, lines: list[_Line]):
        self.lines = lines
        self.chunks: list[dict] = []
        self.section = ""
        self.sentences: list[list[tuple[str, int]]] = []
        self.count = 0

    def add(self, sentence: list[tuple[str, int]]) -> None:
        while len(sentence) > MAX_WORDS:
            self.add(sentence[:MAX_WORDS])
            sentence = sentence[MAX_WORDS:]
        if self.count + len(sentence) > TARGET_WORDS and self.count >= MIN_CHUNK_WORDS:
            self._emit(keep_overlap=True)
        self.sentences.append(sentence)
        self.count += len(sentence)

    def new_section(self, name: str) -> None:
        if self.count >= MIN_CHUNK_WORDS:
            self._emit(keep_overlap=False)
        self.section = name

    def finish(self) -> list[dict]:
        if self.count >= MIN_CHUNK_WORDS or (self.count and not self.chunks):
            self._emit(keep_overlap=False)
        return self.chunks

    def _emit(self, keep_overlap: bool) -> None:
        words = [word for sentence in self.sentences for word in sentence]
        indices = [word[1] for word in words]
        pages = [self.lines[index].page for index in indices]
        self.chunks.append(
            {
                "page": pages[0],
                "page_end": pages[-1],
                "section": self.section,
                "text": " ".join(word[0] for word in words),
                "rects": _merge_rects(indices, self.lines),
            }
        )
        kept: list[list[tuple[str, int]]] = []
        if keep_overlap:
            total = 0
            for sentence in reversed(self.sentences):
                if total + len(sentence) > OVERLAP_WORDS:
                    break
                kept.insert(0, sentence)
                total += len(sentence)
        self.sentences = kept
        self.count = sum(len(sentence) for sentence in kept)


def parse_pdf(path: str) -> dict:
    """Parse one PDF. Returns plain data, so it can cross a process boundary.

    status is "ok", "scanned" (no text layer), "encrypted" or "error".
    """
    try:
        document = pymupdf.open(path)
    except Exception as exc:  # MuPDF raises many types on damaged files.
        return {"status": "error", "pages": 0, "chunks": [], "error": f"Couldn't open the PDF: {exc}"}
    try:
        if document.needs_pass:
            return {"status": "encrypted", "pages": document.page_count, "chunks": [], "error": "The PDF is password-protected."}
        return _parse_document(document)
    except Exception as exc:
        return {"status": "error", "pages": document.page_count, "chunks": [], "error": f"Couldn't read the PDF: {exc}"}
    finally:
        document.close()


def _parse_document(document: pymupdf.Document) -> dict:
    size_counts: Counter = Counter()
    pages: list[list[_Block]] = []
    heights: list[float] = []
    total_chars = 0
    for index, page in enumerate(document):
        blocks = _page_blocks(page, index + 1, size_counts)
        total_chars += sum(len(block.text) for block in blocks)
        pages.append(_reading_order(blocks, page.rect.width))
        heights.append(page.rect.height)

    page_count = len(pages)
    if page_count == 0 or total_chars / page_count < MIN_CHARS_PER_PAGE:
        return {"status": "scanned", "pages": page_count, "chunks": [], "error": "The PDF has no text layer. It's probably a scan and needs OCR."}

    pages = _strip_furniture(pages, heights)
    body_size = size_counts.most_common(1)[0][0] if size_counts else 10.0

    lines: list[_Line] = []
    chunker = _Chunker(lines)
    pending: list[tuple[str, int]] = []
    skipping = False
    in_references = False
    flat = [(index, block) for index, blocks in enumerate(pages) for block in blocks]

    def flush_pending() -> None:
        nonlocal pending
        if pending:
            chunker.add(pending)
            pending = []

    for position, (page_index, block) in enumerate(flat):
        first_line = block.lines[0].text.strip()
        whole = block.text.strip()
        reference_zone = page_index >= max(1, int(_REFERENCES_FROM * page_count))
        # The number and the word can sit on separate lines, and the first entries can share the block.
        if reference_zone and (_REFERENCES.match(first_line) or _REFERENCES.match(whole)):
            flush_pending()
            in_references = True
            continue
        if in_references:
            resumes = any(pattern.match(text) for pattern in (_RESUME_AFTER_REFERENCES, _LETTERED_APPENDIX) for text in (first_line, whole))
            if resumes and _is_heading(block, body_size):
                in_references = False
            else:
                continue
        if _is_heading(block, body_size, first_page=page_index == 0):
            flush_pending()
            name = _BULLET.sub("", whole)
            skipping = bool(_SKIPPED_SECTIONS.match(name))
            if not skipping:
                chunker.new_section(_STRIP_NUMBERING.sub("", name).rstrip(":. "))
            continue
        if skipping or _is_junk(block):
            continue

        pending.extend(_block_words(block, lines))

        # Split finished sentences off. An unfinished one may continue in the next column or page.
        start = 0
        for index, (word, _) in enumerate(pending):
            following = pending[index + 1][0] if index + 1 < len(pending) else None
            if following is not None and _is_sentence_end(word, following):
                chunker.add(pending[start : index + 1])
                start = index + 1
        pending = pending[start:]

        if pending:
            next_word = None
            if position + 1 < len(flat):
                next_text = flat[position + 1][1].text.split()
                next_word = next_text[0] if next_text else None
            continues = next_word is not None and next_word[0].islower() and not _TERMINAL.search(pending[-1][0])
            if not continues:
                flush_pending()

    flush_pending()
    chunks = chunker.finish()
    if not chunks:
        return {"status": "scanned", "pages": page_count, "chunks": [], "error": "No readable body text was found in the PDF."}
    return {"status": "ok", "pages": page_count, "chunks": chunks, "error": None}
