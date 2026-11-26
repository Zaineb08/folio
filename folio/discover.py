"""Find new articles through OpenAlex or Crossref (free, no account) and add them to Zotero.

Only what the student types here leaves the computer. Nothing from their library or
their questions is sent. PDFs are fetched only for open-access articles they choose.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
import time
from urllib.parse import urljoin, urlparse

import httpx

from . import __version__, store, zotero

log = logging.getLogger("folio.discover")

API = "https://api.openalex.org/works"
CROSSREF_API = "https://api.crossref.org/works"
_OPENALEX_RETRIES = (0, 2, 5)  # Seconds to wait before each attempt.
_USER_AGENT = f"Folio/{__version__} (local research assistant)"
_MAX_PDF_BYTES = 150 * 1024 * 1024
_FIELDS = (
    "id,doi,display_name,publication_year,authorships,primary_location,best_oa_location,"
    "open_access,cited_by_count,abstract_inverted_index,type"
)
_ITEM_TYPES = {
    "article": "journalArticle",
    "review": "journalArticle",
    "preprint": "preprint",
    "book-chapter": "bookSection",
    "book": "book",
    "dissertation": "thesis",
    "report": "report",
}
_VENUE_FIELD = {"journalArticle": "publicationTitle", "preprint": "repository", "bookSection": "bookTitle", "report": "institution", "thesis": "university"}
_TAG = re.compile(r"<[^>]+>")
_SURNAME_PARTICLES = {"de", "del", "della", "der", "den", "di", "da", "dos", "du", "la", "le", "van", "von", "ter", "ten", "el", "al", "ben", "bin", "ibn"}


class DiscoverError(RuntimeError):
    pass


def _plain(text: str | None) -> str:
    return re.sub(r"\s+", " ", _TAG.sub("", text or "")).strip()


def _abstract(inverted: dict | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]}."""
    if not inverted:
        return ""
    positions = [(position, word) for word, places in inverted.items() for position in places]
    return " ".join(word for _, word in sorted(positions))


def _split_name(name: str) -> dict:
    """Split a full name. Particles belong to the surname: "H. de Vet" is de Vet, H., and "R. De Luca" is De Luca, R."""
    parts = (name or "").strip().split()
    if not parts:
        return {"first": "", "last": "Unknown"}
    start = len(parts) - 1
    while start > 1 and (parts[start - 1][0].islower() or parts[start - 1].lower() in _SURNAME_PARTICLES):
        start -= 1
    return {"first": " ".join(parts[:start]), "last": " ".join(parts[start:])}


def _normalise_doi(doi: str | None) -> str:
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", doi or "", flags=re.IGNORECASE).strip().lower()


def _normalise(work: dict, known_dois: set[str], known_titles: set[str]) -> dict:
    location = work.get("primary_location") or {}
    source = location.get("source") or {}
    best = work.get("best_oa_location") or {}
    open_access = work.get("open_access") or {}
    title = _plain(work.get("display_name")) or "(untitled)"
    doi = _normalise_doi(work.get("doi"))
    return {
        "id": work.get("id", ""),
        "title": title,
        "authors": [_split_name((a.get("author") or {}).get("display_name", "")) for a in work.get("authorships", [])[:30]],
        "year": work.get("publication_year"),
        "venue": _plain(source.get("display_name")),
        "doi": doi,
        "url": best.get("landing_page_url") or location.get("landing_page_url") or (f"https://doi.org/{doi}" if doi else ""),
        "pdf_url": best.get("pdf_url") or "",
        "is_oa": bool(open_access.get("is_oa")),
        "cited_by": work.get("cited_by_count") or 0,
        "abstract": _plain(_abstract(work.get("abstract_inverted_index"))),
        "type": work.get("type") or "article",
        "in_library": bool(doi and doi in known_dois) or title.lower() in known_titles,
    }


def _search_openalex(query: str, open_access_only: bool, year_min: int | None, sort: str, limit: int) -> list[dict]:
    filters = ["type:article|preprint|review|book-chapter|dissertation"]
    if open_access_only:
        filters.append("is_oa:true")
    if year_min:
        filters.append(f"from_publication_date:{year_min}-01-01")
    params = {"search": query, "filter": ",".join(filters), "per_page": min(limit, 50), "select": _FIELDS}
    if sort == "cited":
        params["sort"] = "cited_by_count:desc"
    elif sort == "newest":
        params["sort"] = "publication_date:desc"

    # OpenAlex pauses anonymous search under heavy load, usually only briefly.
    for delay in _OPENALEX_RETRIES:
        time.sleep(delay)
        response = httpx.get(API, params=params, headers={"User-Agent": _USER_AGENT}, timeout=20)
        if response.status_code == 200:
            return response.json().get("results", [])
        if response.status_code not in (429, 500, 502, 503, 504):
            break
    raise httpx.HTTPError(f"OpenAlex returned HTTP {response.status_code}")


def _normalise_crossref(work: dict) -> dict:
    authors = [
        {"first": author.get("given", ""), "last": author.get("family") or author.get("name", "Unknown")}
        for author in work.get("author", [])[:30]
    ]
    date_parts = (work.get("issued") or {}).get("date-parts") or [[None]]
    licences = [licence.get("URL", "") for licence in work.get("license", [])]
    pdf = next((link["URL"] for link in work.get("link", []) if link.get("content-type") == "application/pdf"), "")
    is_oa = any("creativecommons.org" in licence for licence in licences)
    types = {"journal-article": "article", "posted-content": "preprint", "book-chapter": "book-chapter", "dissertation": "dissertation"}
    return {
        "id": work.get("DOI", ""),
        "title": _plain((work.get("title") or [""])[0]) or "(untitled)",
        "authors": authors,
        "year": date_parts[0][0],
        "venue": _plain((work.get("container-title") or [""])[0]),
        "doi": _normalise_doi(work.get("DOI")),
        "url": work.get("URL", ""),
        "pdf_url": pdf if is_oa else "",
        "is_oa": is_oa,
        "cited_by": work.get("is-referenced-by-count") or 0,
        "abstract": _plain(work.get("abstract")),
        "type": types.get(work.get("type"), "article"),
    }


def _search_crossref(query: str, open_access_only: bool, year_min: int | None, sort: str, limit: int) -> list[dict]:
    filters = ["type:journal-article,type:posted-content,type:book-chapter,type:dissertation"]
    if year_min:
        filters.append(f"from-pub-date:{year_min}")
    params: dict = {
        "query": query,
        "rows": 100 if open_access_only else min(limit, 50),
        "filter": ",".join(filters),
        "select": "DOI,title,author,issued,container-title,is-referenced-by-count,abstract,type,link,URL,license",
    }
    if sort == "cited":
        params.update(sort="is-referenced-by-count", order="desc")
    elif sort == "newest":
        params.update(sort="published", order="desc")
    response = httpx.get(CROSSREF_API, params=params, headers={"User-Agent": _USER_AGENT}, timeout=30)
    response.raise_for_status()
    works = [_normalise_crossref(work) for work in response.json().get("message", {}).get("items", [])]
    if open_access_only:
        works = [work for work in works if work["is_oa"]]
    return works[:limit]


def search(query: str, open_access_only: bool = True, year_min: int | None = None, sort: str = "relevance", limit: int = 25) -> dict:
    """Search OpenAlex, falling back to Crossref if it's unavailable."""
    docs = store.list_docs()
    known_dois = {_normalise_doi(doc.get("doi")) for doc in docs if doc.get("doi")}
    known_titles = {(doc.get("title") or "").lower() for doc in docs}

    try:
        works = _search_openalex(query, open_access_only, year_min, sort, limit)
        results = [_normalise(work, known_dois, known_titles) for work in works]
        return {"results": results, "source": "OpenAlex"}
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("OpenAlex unavailable, using Crossref: %s", exc)
    try:
        results = _search_crossref(query, open_access_only, year_min, sort, limit)
    except (httpx.HTTPError, ValueError) as exc:
        raise DiscoverError("Neither OpenAlex nor Crossref could be reached. Check your connection and try again.") from exc
    for work in results:
        work["in_library"] = bool(work["doi"] and work["doi"] in known_dois) or work["title"].lower() in known_titles
    return {"results": results, "source": "Crossref"}


def _is_public_host(host: str) -> bool:
    """Refuse to fetch from the computer itself or its local network."""
    try:
        addresses = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


def download_pdf(url: str) -> bytes | None:
    """Fetch a PDF from a URL. Returns None if what's there isn't a PDF (often a publisher page)."""
    headers = {"User-Agent": _USER_AGENT, "Accept": "application/pdf,*/*;q=0.8"}
    try:
        for _ in range(6):
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname or not _is_public_host(parsed.hostname):
                return None
            with httpx.stream("GET", url, headers=headers, timeout=60) as response:
                if response.is_redirect:
                    url = urljoin(url, response.headers.get("location", ""))
                    continue
                if response.status_code != 200:
                    return None
                data = bytearray()
                for block in response.iter_bytes():
                    data.extend(block)
                    if len(data) > _MAX_PDF_BYTES:
                        return None
                    if len(data) >= 1024 and b"%PDF-" not in bytes(data[:1024]):
                        return None
                return bytes(data) if b"%PDF-" in bytes(data[:1024]) else None
    except httpx.HTTPError:
        return None
    return None


def add_to_zotero(work: dict, tags: list[str] | None = None) -> dict:
    """Save an article's metadata to Zotero, with its PDF when an open-access copy can be fetched."""
    item_type = _ITEM_TYPES.get(work.get("type"), "journalArticle")
    item = {
        "itemType": item_type,
        "title": work.get("title") or "(untitled)",
        "creators": [
            {"firstName": author.get("first", ""), "lastName": author.get("last", ""), "creatorType": "author"}
            for author in work.get("authors", [])
        ],
        "date": str(work.get("year") or ""),
        "url": work.get("url") or "",
        "abstractNote": work.get("abstract") or "",
        "tags": [{"tag": tag} for tag in (tags or [])],
    }
    if work.get("doi") and item_type in ("journalArticle", "preprint", "bookSection"):
        item["DOI"] = work["doi"]
    venue_field = _VENUE_FIELD.get(item_type)
    if venue_field and work.get("venue"):
        item[venue_field] = work["venue"]

    pdf = download_pdf(work["pdf_url"]) if work.get("pdf_url") else None
    try:
        return zotero.save_item(item, pdf=pdf, pdf_url=work.get("pdf_url") or "")
    except zotero.ZoteroError as exc:
        raise DiscoverError(str(exc)) from exc
