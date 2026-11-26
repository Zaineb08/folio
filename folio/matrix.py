"""Literature matrix: one row per paper, one column per question, each cell written from that paper's pages.

Cells are filled one at a time, each from the passages of that paper that best answer its
column's question, plus the paper's opening. That keeps every cell tied to pages the student
can check, and lets a question asked meanwhile slip in between two cells instead of waiting
for the whole matrix. Only the local model fills cells: a matrix takes dozens of calls,
which Claude would charge for.
"""

from __future__ import annotations

import io
import logging
import re
import threading
import time

from . import cite, config, llm, search, store
from .answer import _CITATION, _INNER_REFERENCE, _excerpt_block, _numbers_in, generation_lock
from .models import engine, gpu_lock
from .search import _hit, best_in_paper

log = logging.getLogger("folio.matrix")

_BEST_PASSAGES = 3
_REPLY_TOKENS = 120
# The words authors use for each usual column find the right pages far better than a question
# like "What did the authors do?" on its own, which left Method empty for papers that clearly have one.
_SEARCH_HINTS = {
    "Research question": "research question aim objective goal motivation this paper investigates",
    "Method": "methodology approach study design experimental setup procedure implementation evaluation",
    "Data and participants": "dataset benchmark participants subjects recruited samples models evaluated",
    "Main findings": "results show found findings key contributions conclusion",
    "Limitations": "limitations threats to validity shortcomings future work",
}
# Sections about other people's work, which the model would otherwise report as this paper's own.
_OTHER_WORK = re.compile(r"related work|background|prior work|previous work|literature review|state of the art", re.I)
# Section names are often subheadings like "Evaluations of Copilot", so a passage crowded with
# citations counts as other people's work too. Not in a survey, where that's the paper's own content.
_CITATION_MARK = re.compile(r"et al\.|\[\d+(?:\s*[,–-]\s*\d+)*\]|\([A-Z][^()]{0,60}?,? (?:19|20)\d{2}[a-z]?\)")
_SURVEY = re.compile(r"survey|review|\bSoK\b|mapping study", re.I)
# Headings that name the paper's own parts. Threats to validity often cite others, but they're still its own.
_OWN_SECTION = re.compile(r"limitation|threat|validity|conclu|discussion|result|finding|method|approach|experiment|setup|implementation|participant|recruit", re.I)
_CROWDED = 3


def _about_other_work(doc: dict, chunk: dict) -> bool:
    section = chunk.get("section") or ""
    if _OTHER_WORK.search(section):
        return True
    if _OWN_SECTION.search(section) or _SURVEY.search(doc.get("title") or ""):
        return False
    return len(_CITATION_MARK.findall(chunk["text"])) >= _CROWDED
_MAX_CELL = 420  # characters

_PROMPT = """{excerpts}

You are filling in one cell of a literature matrix, a table that compares research papers.
Paper: {cite}, "{title}"
Column: {label}
What goes in this column: {question}

Using only the excerpts above, write at most 30 words in note form about this paper's own work, not about work it cites. Cite the excerpts you used, like [2].
If the excerpts don't say, reply only: Not stated"""


def _local_model() -> str:
    model = config.load_settings()["local_model"]
    if model not in llm.list_models():
        raise llm.LLMError("The matrix is filled by the local model, and Folio can't reach it. Start Ollama, then try again.")
    return model


def _passages(doc: dict, column: dict) -> list[dict]:
    """The paper's opening, which usually summarises it, plus its passages that best answer the column."""
    hits = best_in_paper(doc["key"], column["question"], _SEARCH_HINTS.get(column["label"], ""), k=_BEST_PASSAGES, skip=_about_other_work)
    rows = store.chunks_of(doc["key"])
    if rows and rows[0]["id"] not in {hit["chunk_id"] for hit in hits}:
        opening = store.get_chunk(rows[0]["id"])
        if opening:
            hits = [_hit(opening, doc, 0.0), *hits]
    return hits


def _pages(sources: list[dict], numbers: list[int]) -> str:
    pages = list(dict.fromkeys(source["page"] for source in sources if source["n"] in numbers))
    if not pages:
        return ""
    if pages == [0]:
        return "(abstract)"
    pages = [str(page) for page in pages if page]
    return f"(p. {pages[0]})" if len(pages) == 1 else f"(pp. {', '.join(pages)})"


def plain_text(text: str, sources: list[dict]) -> str:
    """A cell's text with its [n] citations turned into page numbers, for a spreadsheet."""
    numbers = lambda group: [int(n) for n in re.split(r"\s*[,;]\s*", group)]  # noqa: E731
    return _CITATION.sub(lambda match: " " + _pages(sources, numbers(match.group(1))), text).replace("  ", " ").strip()


def cell_from_reply(reply: str, hits: list[dict], year: int | None = None) -> tuple[str, list[dict], list[dict]]:
    """Turn the model's reply into a cell's text, the passages it came from, and anything to check."""
    text = " ".join(reply.split()).strip("\"' ")
    sources = [
        {"n": n, "chunk_id": hit["chunk_id"], "page": hit["page"], "page_end": hit["page_end"], "section": hit["section"], "text": hit["text"]}
        for n, hit in enumerate(hits, 1)
    ]
    if not text or text.lower().startswith("not stated"):
        return "", sources, []  # Keep the pages searched, so the student can look for themselves.
    valid = {source["n"] for source in sources}

    def keep_valid(match: re.Match) -> str:
        kept = [n for n in re.split(r"\s*[,;]\s*", match.group(1)) if int(n) in valid]
        return f"[{', '.join(kept)}]" if kept else ""

    text = _CITATION.sub(keep_valid, text)
    if len(text) > _MAX_CELL:
        text = text[:_MAX_CELL].rsplit(" ", 1)[0] + "…"
    cited = {int(n) for group in _CITATION.findall(text) for n in re.split(r"\s*[,;]\s*", group)}
    used = [source for source in sources if source["n"] in cited] or sources
    allowed = _numbers_in(" ".join(_INNER_REFERENCE.sub("", source["text"]) for source in used)) | {str(year)}
    bare = _CITATION.sub("", text)
    missing = sorted(n for n in _numbers_in(bare) if (len(n) >= 2 or "." in n) and n not in allowed)
    flags = [{"kind": "number", "detail": f"{', '.join(missing)} isn't in the passages this cell is based on. Check it."}] if missing else []
    return text, sources, flags


def fill_cell(model: str, doc: dict, column: dict) -> None:
    hits = _passages(doc, column)
    if hits:
        prompt = _PROMPT.format(
            excerpts=_excerpt_block(hits), cite=cite.short_cite(doc), title=doc["title"], label=column["label"], question=column["question"]
        )
        with generation_lock, gpu_lock:
            engine.release_gpu()
            reply = llm.chat_once(model, [{"role": "user", "content": prompt}], num_predict=_REPLY_TOKENS)
    else:
        reply = ""
    text, sources, flags = cell_from_reply(reply, hits, doc.get("year"))
    store.save_matrix_cell(doc["key"], column["id"], doc.get("fingerprint"), text, sources, flags, model)


def pending_cells(doc_keys: list[str]) -> list[tuple[dict, dict]]:
    """Cells that are empty, or were written before their paper was last indexed."""
    docs = {doc["key"]: doc for doc in store.list_docs() if doc.get("chunks")}
    columns = store.matrix_columns()
    done = {(cell["doc_key"], cell["column_id"]): cell["fingerprint"] for cell in store.matrix_cells()}
    return [
        (docs[key], column)
        for key in dict.fromkeys(doc_keys)
        if key in docs
        for column in columns
        if done.get((key, column["id"]), object()) != docs[key].get("fingerprint")
    ]


class Filler:
    """Fills cells in the background, one at a time, so it can be stopped between any two."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.state = {"running": False, "done": 0, "total": 0, "current": "", "error": None, "stopped": False, "seconds_per_cell": None}

    def _begin(self, doc_keys: list[str]) -> tuple[str, list] | None:
        with self._lock:
            if self.state["running"]:
                return None
            model = _local_model()
            tasks = pending_cells(doc_keys)
            self._stop.clear()
            self.state = {"running": True, "done": 0, "total": len(tasks), "current": "", "error": None, "stopped": False, "seconds_per_cell": None}
            return model, tasks

    def start(self, doc_keys: list[str]) -> bool:
        begun = self._begin(doc_keys)
        if begun is None:
            return False
        threading.Thread(target=self._run, args=begun, daemon=True, name="folio-matrix").start()
        return True

    def run_blocking(self, doc_keys: list[str]) -> None:
        begun = self._begin(doc_keys)
        if begun is not None:
            self._run(*begun)

    def stop(self) -> None:
        self._stop.set()

    def _run(self, model: str, tasks: list[tuple[dict, dict]]) -> None:
        started = time.time()
        try:
            for doc, column in tasks:
                if self._stop.is_set():
                    self.state["stopped"] = True
                    break
                self.state["current"] = f"{cite.short_cite(doc)}: {column['label']}"
                fill_cell(model, doc, column)
                self.state["done"] += 1
                self.state["seconds_per_cell"] = round((time.time() - started) / self.state["done"], 1)
        except llm.LLMError as exc:
            self.state["error"] = str(exc)
        except Exception as exc:
            log.exception("Filling the matrix failed")
            self.state["error"] = f"Something went wrong: {exc}"
        finally:
            self.state["running"] = False
            self.state["current"] = ""


filler = Filler()


def _one_per_paper(docs: list[dict]) -> list[dict]:
    return search.one_per_paper(docs)


def overview() -> dict:
    indexed = [doc for doc in store.list_docs() if doc.get("chunks")]
    docs = sorted(_one_per_paper(indexed), key=lambda doc: (cite.short_cite(doc).lower(), doc.get("year") or 0))
    fingerprints = {doc["key"]: doc.get("fingerprint") for doc in docs}
    cells = [
        {
            "doc_key": cell["doc_key"],
            "column_id": cell["column_id"],
            "text": cell["text"],
            "sources": cell["sources"],
            "flags": cell["flags"],
            "stale": cell["fingerprint"] != fingerprints[cell["doc_key"]],
        }
        for cell in store.matrix_cells()
        if cell["doc_key"] in fingerprints
    ]
    rows = [
        {
            "key": doc["key"],
            "cite": cite.short_cite(doc),
            "title": doc["title"],
            "year": doc.get("year"),
            "apa": cite.apa(doc),
            "tags": doc["tags"],
            "collections": doc["collections"],
            "has_pdf": bool(doc.get("pdf_path")),
            "pages": doc.get("pages") or 0,
        }
        for doc in docs
    ]
    return {"columns": store.matrix_columns(), "rows": rows, "cells": cells, "job": dict(filler.state), "duplicates": len(indexed) - len(docs)}


def export_xlsx(doc_keys: list[str] | None = None) -> bytes:
    """The matrix as an Excel workbook, with page numbers in place of citation numbers."""
    from openpyxl import Workbook
    from openpyxl.comments import Comment
    from openpyxl.styles import Alignment, Font, PatternFill

    data = overview()
    wanted = set(doc_keys) if doc_keys else None
    rows = [row for row in data["rows"] if wanted is None or row["key"] in wanted]
    columns = data["columns"]
    cells = {(cell["doc_key"], cell["column_id"]): cell for cell in data["cells"]}

    book = Workbook()
    sheet = book.active
    sheet.title = "Matrix"

    def put(row: int, column: int, value) -> object:
        target = sheet.cell(row=row, column=column, value=value)
        if isinstance(value, str) and value.startswith("="):
            target.data_type = "s"  # Text from a paper must never run as a formula.
        target.alignment = Alignment(wrap_text=True, vertical="top")
        return target

    header = ["Paper", "Year", *(column["label"] for column in columns), "Reference"]
    for index, label in enumerate(header, 1):
        target = put(1, index, label)
        target.font = Font(bold=True)
        target.fill = PatternFill("solid", fgColor="EDE6D6")
    for row_number, row in enumerate(rows, 2):
        put(row_number, 1, row["cite"])
        put(row_number, 2, row["year"])
        for index, column in enumerate(columns, 3):
            cell = cells.get((row["key"], column["id"]))
            if cell is None:
                continue
            if cell["text"]:
                text = plain_text(cell["text"], cell["sources"])
            else:
                searched = _pages(cell["sources"], [source["n"] for source in cell["sources"]])
                text = f"Not stated in the passages found {searched.replace('(', '(searched ')}".strip()
            target = put(row_number, index, text)
            notes = [flag["detail"] for flag in cell["flags"]]
            if cell["stale"]:
                notes.append("The paper changed after this cell was filled. Fill it again in Folio.")
            if notes:
                target.comment = Comment(" ".join(notes), "Folio")
        put(row_number, len(columns) + 3, row["apa"])

    sheet.freeze_panes = "C2"
    sheet.column_dimensions["A"].width = 24
    sheet.column_dimensions["B"].width = 7
    for index in range(3, len(columns) + 3):
        sheet.column_dimensions[sheet.cell(row=1, column=index).column_letter].width = 42
    sheet.column_dimensions[sheet.cell(row=1, column=len(columns) + 3).column_letter].width = 60

    about = book.create_sheet("About")
    for line in (
        "Made by Folio from your Zotero library.",
        "Each cell was written by the model on your PC from up to 4 passages of that paper: its opening and the 3 that best match the column.",
        "Page numbers show where each note came from. Read those pages before you rely on a cell or cite it.",
        "A comment on a cell means Folio found something to check, such as a figure that isn't in the passages.",
        "\"Not stated in the passages found\" means the passages searched didn't say. The paper may still say it elsewhere.",
    ):
        about.append([line])
    about.column_dimensions["A"].width = 120

    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()
