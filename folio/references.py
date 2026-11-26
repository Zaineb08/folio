"""Which works do several papers cite? Computed from their reference lists, not by a model.

Indexing leaves reference lists out of search, because a list of titles matches almost any
question. So this reads them straight from each PDF when asked. A cited work counts as
shared when its title appears word for word in another paper's reference list, ignoring
case, punctuation and words split across lines. Titles are copied faithfully between
papers far more often than authors or venues are, which makes them the safest key.
"""

from __future__ import annotations

import functools
import logging
import re

import pymupdf

from . import cite, pdf, store

log = logging.getLogger("folio.references")

# A title in quotes, as IEEE and ACM styles print it.
_QUOTED = re.compile(r"[“\"]([^”\"]{15,250}?)[,.]?[”\"]")
# A title after the year, as ACL and AAAI styles print it: "... 2021. Evaluating large language models ..."
_AFTER_YEAR = re.compile(r"\b(?:19|20)\d{2}[a-z]?\.\s+([A-Z][^.?!]{15,250}[.?!])")
# Venue names repeat across reference lists without being shared references.
_NOT_A_TITLE = re.compile(r"\b(proceedings|conference|journal|workshop|transactions|symposium|arxiv|preprint|in advances)\b", re.IGNORECASE)
_MIN_WORDS = 4
MAX_SHOWN = 25


def _joined(text: str) -> str:
    """One line of text, with words that a line break split rejoined."""
    text = text.translate(pdf._LIGATURE_TABLE)
    text = re.sub(r"-\s*\n\s*(?=[a-z])", "", text)
    return " ".join(text.split())


def _key(text: str) -> str:
    """Text to compare: lower case, no punctuation, and "fact-conflicting" the same as "factconflicting"."""
    return " ".join(re.findall(r"[a-z0-9]+", text.replace("-", "").lower()))


def _main_title(title: str) -> str | None:
    """The part before a subtitle, when it's distinctive enough to match on its own."""
    main = _key(title.split(":")[0]) if ":" in title else ""
    return main if len(main.split()) >= 5 else None


@functools.lru_cache(maxsize=256)
def _reference_pages(path: str, fingerprint: str) -> tuple[tuple[int, str], ...]:
    """The text of each page from the reference list onwards, as (page number, text)."""
    with pymupdf.open(path) as document:
        texts = [page.get_text() for page in document]
    first = max(1, int(pdf._REFERENCES_FROM * len(texts)))  # As in parsing, a contents list can't count.
    for index in range(first, len(texts)):
        lines = texts[index].splitlines()
        for number, line in enumerate(lines):
            if pdf._REFERENCES.match(line.strip()):
                pages = [(index + 1, "\n".join(lines[number + 1 :]))]
                pages += [(later + 1, texts[later]) for later in range(index + 1, len(texts))]
                return tuple(pages)
    return ()


def reference_lists(doc_keys: list[str]) -> dict[str, tuple[tuple[int, str], ...]]:
    lists = {}
    for key in doc_keys:
        doc = store.get_doc(key)
        if not doc or not doc.get("pdf_path"):
            continue
        try:
            lists[key] = _reference_pages(doc["pdf_path"], doc.get("fingerprint") or "")
        except Exception as exc:  # MuPDF raises many types on damaged files.
            log.warning("Couldn't read the references of %s: %s", key, exc)
    return lists


def _titles(raw: str) -> list[str]:
    titles = [match.group(1).strip() for pattern in (_QUOTED, _AFTER_YEAR) for match in pattern.finditer(raw)]
    return [title.rstrip(".?! ") for title in titles if not _NOT_A_TITLE.search(title)]


def shared(doc_keys: list[str]) -> dict:
    """The works cited by at least two of these papers, most widely cited first."""
    lists = reference_lists(doc_keys)
    keyed_pages = {key: [(page, f" {_key(_joined(text))} ") for page, text in pages] for key, pages in lists.items()}
    keyed_whole = {key: " ".join(text for _, text in pages) for key, pages in keyed_pages.items()}

    candidates: dict[str, str] = {}  # comparable title -> title as printed
    for pages in lists.values():
        for title in _titles(_joined("\n".join(text for _, text in pages))):
            key = _key(title)
            if len(key.split()) >= _MIN_WORDS:
                candidates.setdefault(key, title)

    def cites(text: str, key: str, main: str | None) -> bool:
        # Papers print subtitles differently, so a long main title is enough on its own.
        return f" {key} " in text or bool(main and f" {main} " in text)

    works = []
    for key, title in candidates.items():
        main = _main_title(title)
        citing = [doc_key for doc_key, text in keyed_whole.items() if cites(text, key, main)]
        if len(citing) >= 2:
            works.append((key, main, title, citing))
    # The same work can be picked up with a word more or less from different lists. Keep the widest-cited version.
    works.sort(key=lambda work: (-len(work[3]), len(work[0])))
    kept: list[tuple[str, str | None, str, list[str]]] = []
    for key, main, title, citing in works:
        same = lambda other, other_main: key in other or other in key or (main and main == other_main)  # noqa: E731
        if not any(same(other, other_main) for other, other_main, _, _ in kept):
            kept.append((key, main, title, citing))

    docs = {key: store.get_doc(key) for key in doc_keys}
    result = []
    for key, main, title, citing in kept[:MAX_SHOWN]:
        cited_by = []
        for doc_key in doc_keys:
            if doc_key in citing:
                page = next((page for page, text in keyed_pages[doc_key] if cites(text, key, main)), keyed_pages[doc_key][0][0])
                cited_by.append({"doc_key": doc_key, "cite": cite.short_cite(docs[doc_key]), "page": page})
        result.append({"title": title[0].upper() + title[1:], "cited_by": cited_by, "count": len(cited_by)})
    return {
        "papers": len(doc_keys),
        "works": result,
        "missing": [cite.short_cite(docs[key]) for key in doc_keys if docs.get(key) and not lists.get(key)],
    }
