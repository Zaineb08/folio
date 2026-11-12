"""Citations built from Zotero's metadata, never from model output, so they can't be invented."""

from __future__ import annotations

import re
import unicodedata

_BIBTEX_TYPES = {
    "journalArticle": "article",
    "conferencePaper": "inproceedings",
    "book": "book",
    "bookSection": "incollection",
    "thesis": "phdthesis",
    "report": "techreport",
    "preprint": "misc",
}
_BIBTEX_ESCAPES = str.maketrans({"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "$": r"\$"})


def _authors(doc: dict) -> list[dict]:
    creators = doc.get("creators") or []
    authors = [c for c in creators if c.get("type") == "author"]
    return authors or [c for c in creators if c.get("type") == "editor"] or creators


def short_cite(doc: dict) -> str:
    """For example: Lewis et al., 2020."""
    authors = _authors(doc)
    year = doc.get("year") or "n.d."
    if not authors:
        title = doc.get("title") or "Untitled"
        return f"{title[:40]}{'…' if len(title) > 40 else ''}, {year}"
    if len(authors) == 1:
        names = authors[0]["last"]
    elif len(authors) == 2:
        names = f"{authors[0]['last']} & {authors[1]['last']}"
    else:
        names = f"{authors[0]['last']} et al."
    return f"{names}, {year}"


def _apa_name(author: dict) -> str:
    # Every given name counts, however it's written: "W.L.J.M." and "Jean-Paul" both work.
    initials = " ".join(f"{part[0].upper()}." for part in re.findall(r"[^\W\d_]+", author.get("first", "")))
    return f"{author['last']}, {initials}" if initials else author["last"]


def apa(doc: dict) -> str:
    """A reference in APA style. Check it against your style guide before submitting."""
    authors = _authors(doc)
    names = [_apa_name(author) for author in authors]
    if len(names) > 20:
        who = ", ".join(names[:19]) + ", … " + names[-1]
    elif len(names) > 1:
        who = ", ".join(names[:-1]) + ", & " + names[-1]
    else:
        who = names[0] if names else ""
    year = doc.get("year") or "n.d."
    title = doc.get("title", "Untitled").rstrip(". ")
    parts = [f"{who} ({year})." if who else f"({year}).", title if title.endswith(("?", "!")) else f"{title}."]
    if doc.get("venue"):
        parts.append(f"{doc['venue']}.")
    if doc.get("doi"):
        doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doc["doi"], flags=re.IGNORECASE)
        parts.append(f"https://doi.org/{doi}")
    elif doc.get("url"):
        parts.append(doc["url"])
    return " ".join(parts)


def _ascii(text: str) -> str:
    """Devillé becomes deville, because BibTeX keys should be plain letters."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def bibtex_key(doc: dict) -> str:
    authors = _authors(doc)
    last = re.sub(r"[^a-z]", "", _ascii(authors[0]["last"])) if authors else "anon"
    words = re.findall(r"[a-z]+", _ascii(doc.get("title") or ""))
    word = next((w for w in words if len(w) > 3), "untitled")
    return f"{last or 'anon'}{doc.get('year') or ''}{word}"


def _braced(value) -> str:
    return "{" + str(value).translate(_BIBTEX_ESCAPES) + "}"


def bibtex(doc: dict) -> str:
    entry = _BIBTEX_TYPES.get(doc.get("item_type"), "misc")
    fields: list[tuple[str, object]] = [("title", doc.get("title") or "Untitled")]
    authors = _authors(doc)
    if authors:
        fields.append(("author", " and ".join(f"{a['last']}, {a['first']}" if a.get("first") else a["last"] for a in authors)))
    if doc.get("year"):
        fields.append(("year", doc["year"]))
    if doc.get("venue"):
        venue_field = {"article": "journal", "inproceedings": "booktitle", "incollection": "booktitle", "phdthesis": "school", "techreport": "institution", "book": "publisher"}.get(entry, "howpublished")
        fields.append((venue_field, doc["venue"]))
    if doc.get("doi"):
        fields.append(("doi", doc["doi"]))
    if doc.get("url"):
        fields.append(("url", doc["url"]))
    body = ",\n".join(f"  {name} = {_braced(value)}" for name, value in fields)
    return f"@{entry}{{{bibtex_key(doc)},\n{body}\n}}"
