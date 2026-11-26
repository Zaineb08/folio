"""The local web server behind Folio's interface. It only ever listens on the local computer."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import shutil
import subprocess
import threading
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

import pymupdf
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import answer, claims, cite, claude, config, discover, llm, matrix, store, zotero
from .indexer import indexer
from .models import engine, models_ready
from .search import Scope

log = logging.getLogger("folio.app")
STATIC_DIR = Path(__file__).parent / "static"
_LOCAL_HOSTS = {"127.0.0.1", "localhost"}
_CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "font-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)


class LocalOnly:
    """Reject requests from other sites and apply a strict content policy.

    Article text is untrusted, and any web page can try to call a localhost server,
    so this checks the Host and Origin headers on every request.
    """

    def __init__(self, inner) -> None:
        self.inner = inner

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.inner(scope, receive, send)
            return
        headers = {key.decode("latin-1"): value.decode("latin-1") for key, value in scope["headers"]}
        host = urlparse(f"//{headers.get('host', '')}").hostname
        origin = headers.get("origin")
        origin_host = urlparse(origin).hostname if origin else None
        if host not in _LOCAL_HOSTS or (origin and origin_host not in _LOCAL_HOSTS):
            await Response("Folio only accepts requests from the computer it runs on.", status_code=403)(scope, receive, send)
            return

        async def send_with_headers(message) -> None:
            if message["type"] == "http.response.start":
                extra = [
                    (b"content-security-policy", _CSP.encode()),
                    (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"no-referrer"),
                ]
                message = {**message, "headers": list(message["headers"]) + extra}
            await send(message)

        await self.inner(scope, receive, send_with_headers)


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_warm_up, daemon=True, name="folio-warmup").start()
    indexer.watch()
    yield
    indexer.stop()


def _warm_up() -> None:
    """Get everything ready so the first question isn't slow.

    The reranker loads first, because Ollama sizes the answer model around what's free.
    """
    llm.ensure_running()
    settle_answer_model(llm.list_models())
    engine.preload()
    indexer.run_blocking()
    model = config.load_settings()["local_model"]  # It also checks Claude's claims when Claude is selected.
    if model in llm.list_models():
        llm.preload(model)


def settle_answer_model(installed: list[str]) -> None:
    """Until the student picks a model, use the one that suits the computer, or failing that, the other one installed.

    So someone who downloaded the larger model on a smaller PC still gets answers, just slower.
    """
    if "llm_model" in config.stored_settings() or not installed or config.recommended_model() in installed:
        return
    for model in (config.SMALL_LLM, config.LARGE_LLM):
        if model in installed:
            config.save_settings(llm_model=model, local_model=model)
            return


app = FastAPI(title="Folio", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(LocalOnly)


# ---------------------------------------------------------------------- status

_gpu_name: str | None = None
_gpu_checked = False


def _gpu() -> str | None:
    """The graphics card's name, shown in the status panel."""
    global _gpu_name, _gpu_checked
    if not _gpu_checked:
        _gpu_checked = True
        exe = shutil.which("nvidia-smi")
        if exe:
            try:
                output = subprocess.run(
                    [exe, "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                _gpu_name = output.stdout.strip().splitlines()[0] if output.stdout.strip() else None
            except (OSError, subprocess.SubprocessError):
                _gpu_name = None
    return _gpu_name


@app.get("/api/status")
def status() -> dict:
    settings = config.load_settings()
    ollama_up = llm.is_up()
    models = llm.list_models() if ollama_up else []
    data_dir = zotero.find_data_dir()
    selected = settings["llm_model"]
    has_key = claude.has_key()
    uses_claude = claude.is_claude(selected)
    return {
        "zotero": {
            "installed": bool(zotero.executable()),
            "found": bool(data_dir),
            "running": zotero.is_running(),
            "data_dir": str(data_dir) if data_dir else None,
        },
        "llm": {
            "up": has_key if uses_claude else ollama_up,
            "models": models,
            "selected": selected,
            "installed": has_key if uses_claude else selected in models,
        },
        # Folio's own models, offered for download when they're missing. "suits" marks the one for this computer.
        "suggested": [
            {"id": model, "download_gb": size, "suits": model == config.recommended_model()}
            for model, size in config.LLM_DOWNLOAD_GB.items()
        ],
        "download": dict(_download),
        "claude": {"has_key": has_key, "models": claude.public_models()},  # Whether a key exists, never the key.
        "search_models_ready": models_ready(),
        "gpu": _gpu(),
        "library": store.counts(),
        "sync": indexer.state,
    }


class SettingsBody(BaseModel):
    llm_model: str | None = None


@app.post("/api/settings")
def update_settings(body: SettingsBody) -> dict:
    if body.llm_model:
        if claude.is_claude(body.llm_model):
            if not claude.has_key():
                raise HTTPException(400, "Add your Anthropic API key first.")
            config.save_settings(llm_model=body.llm_model)
        elif body.llm_model in llm.list_models():
            config.save_settings(llm_model=body.llm_model, local_model=body.llm_model)
        else:
            raise HTTPException(400, "That model isn't installed.")
    return config.load_settings()


_download = {"model": None, "done": 0, "total": 0, "error": None, "active": False}
_download_lock = threading.Lock()


class DownloadBody(BaseModel):
    model: str
    select: bool = True


@app.post("/api/models/download")
def download_model(body: DownloadBody) -> dict:
    """Download one of Folio's answer models through Ollama, then use it if asked."""
    if body.model not in config.LLM_DOWNLOAD_GB:
        raise HTTPException(400, "Folio can only download its own answer models.")
    if not llm.ensure_running():
        raise HTTPException(400, "Ollama isn't installed or won't start. Install it with: winget install --id Ollama.Ollama")
    with _download_lock:
        if _download["active"]:
            raise HTTPException(409, f"{_download['model']} is already downloading.")
        _download.update(model=body.model, done=0, total=0, error=None, active=True)
    threading.Thread(target=_run_download, args=(body.model, body.select), daemon=True, name="folio-download").start()
    return dict(_download)


def _run_download(model: str, select: bool) -> None:
    def progress(done: int, total: int) -> None:
        _download.update(done=done, total=total)

    try:
        llm.pull(model, progress)
        if select:
            config.save_settings(llm_model=model, local_model=model)
    except llm.LLMError as exc:
        _download["error"] = str(exc)
    finally:
        _download["active"] = False


class KeyBody(BaseModel):
    key: str = Field(min_length=10, max_length=500)


@app.post("/api/claude/key")
def save_claude_key(body: KeyBody) -> dict:
    """Check the key with Anthropic, then keep it in Windows Credential Manager. It's never sent back."""
    try:
        claude.save_key(body.key)
    except llm.LLMError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"saved": True}


@app.delete("/api/claude/key")
def delete_claude_key() -> dict:
    claude.delete_key()
    settings = config.load_settings()
    if claude.is_claude(settings["llm_model"]) and not claude.has_key():
        config.save_settings(llm_model=settings["local_model"])
    return {"deleted": True}


# --------------------------------------------------------------------- library


@app.get("/api/library")
def library() -> dict:
    docs = store.list_docs()
    tags = Counter(tag for doc in docs for tag in doc["tags"])
    collections = Counter(name for doc in docs for name in doc["collections"])
    years = [doc["year"] for doc in docs if doc.get("year")]
    public = [
        {
            "key": doc["key"],
            "title": doc["title"],
            "cite": cite.short_cite(doc),
            "year": doc.get("year"),
            "venue": doc.get("venue"),
            "status": doc["status"],
            "error": doc.get("error"),
            "pages": doc.get("pages") or 0,
            "chunks": doc.get("chunks") or 0,
            "has_pdf": bool(doc.get("pdf_path")),
            "tags": doc["tags"],
            "collections": doc["collections"],
            "date_added": doc.get("date_added"),
        }
        for doc in docs
    ]
    return {
        "docs": public,
        "tags": [{"name": name, "count": count} for name, count in sorted(tags.items())],
        "collections": [{"name": name, "count": count} for name, count in sorted(collections.items())],
        "years": [min(years), max(years)] if years else None,
    }


@app.post("/api/sync")
def sync(force: bool = False) -> dict:
    return {"started": indexer.start(force=force)}


def _require_doc(doc_key: str) -> dict:
    doc = store.get_doc(doc_key)
    if not doc:
        raise HTTPException(404, "That article isn't in your library.")
    return doc


def _require_pdf(doc: dict) -> str:
    path = doc.get("pdf_path")
    if not path or not os.path.isfile(path):
        raise HTTPException(404, "This article has no PDF in Zotero.")
    return path


@app.get("/api/pdf/{doc_key}")
def pdf_file(doc_key: str) -> FileResponse:
    path = _require_pdf(_require_doc(doc_key))
    return FileResponse(path, media_type="application/pdf", content_disposition_type="inline", filename=os.path.basename(path))


@app.get("/api/page/{doc_key}/{page}")
def page_image(doc_key: str, page: int, chunk: int | None = None, scale: float = 2.0) -> Response:
    """Draw a page as an image, with the cited passage highlighted on it."""
    path = _require_pdf(_require_doc(doc_key))
    scale = min(max(scale, 0.5), 4.0)
    rects: list[list[float]] = []
    if chunk is not None:
        row = store.get_chunk(chunk)
        if row and row["doc_key"] == doc_key:
            rects = [rect for rect in json.loads(row["rects"]) if rect[0] == page]
    try:
        with pymupdf.open(path) as document:
            if not 1 <= page <= document.page_count:
                raise HTTPException(404, "That page doesn't exist.")
            sheet = document[page - 1]
            for rect in rects:
                area = pymupdf.Rect(rect[1:]) * sheet.derotation_matrix
                sheet.draw_rect(area, color=None, fill=(1.0, 0.82, 0.25), fill_opacity=0.32, overlay=True)
            image = sheet.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).tobytes("png")
    except HTTPException:
        raise
    except Exception as exc:
        log.warning("Couldn't render %s page %s: %s", doc_key, page, exc)
        raise HTTPException(500, "Couldn't draw that page.") from exc
    return Response(image, media_type="image/png", headers={"Cache-Control": "private, max-age=600"})


@app.get("/api/chunk/{chunk_id}")
def chunk_details(chunk_id: int) -> dict:
    row = store.get_chunk(chunk_id)
    if not row:
        raise HTTPException(404, "That passage no longer exists.")
    return {
        **{key: row[key] for key in ("id", "doc_key", "page", "page_end", "section", "text")},
        "rects": json.loads(row["rects"]),
    }


@app.post("/api/zotero/launch")
def launch_zotero() -> dict:
    if not zotero.ensure_running():
        raise HTTPException(503, "Zotero couldn't be started. Open it from the Start menu.")
    return {"running": True}


class OpenBody(BaseModel):
    doc_key: str
    page: int | None = None


@app.post("/api/zotero/open")
def open_in_zotero(body: OpenBody) -> dict:
    """Show an article in Zotero, at the cited page if it has a PDF."""
    doc = _require_doc(body.doc_key)
    if not zotero.ensure_running():
        raise HTTPException(503, "Zotero couldn't be started.")
    if doc.get("attachment_key") and body.page:
        uri = zotero.open_pdf_uri(doc.get("group_id"), doc["attachment_key"], body.page)
    else:
        uri = zotero.select_uri(doc.get("group_id"), doc["item_key"])
    os.startfile(uri)
    return {"opened": uri}


class CiteBody(BaseModel):
    doc_keys: list[str] = Field(max_length=200)


@app.post("/api/cite/bibtex", response_class=PlainTextResponse)
def bibtex(body: CiteBody) -> str:
    docs = [store.get_doc(key) for key in dict.fromkeys(body.doc_keys)]
    return "\n\n".join(cite.bibtex(doc) for doc in docs if doc)


# ------------------------------------------------------------------- answering


class ScopeBody(BaseModel):
    collections: list[str] = []
    tags: list[str] = []
    year_min: int | None = None
    year_max: int | None = None
    doc_keys: list[str] = []


class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    conversation_id: int | None = None
    mode: str = "answer"
    scope: ScopeBody = ScopeBody()
    model: str | None = None  # Overrides the selected model for this one question.


@app.post("/api/ask")
async def ask(body: AskBody) -> StreamingResponse:
    if body.mode not in answer.MODES:
        raise HTTPException(400, "Unknown mode.")
    if body.model and not claude.is_claude(body.model) and body.model not in llm.list_models():
        raise HTTPException(400, "That model isn't available.")
    if body.mode == "explain" and len(body.scope.doc_keys) != 1:
        raise HTTPException(400, "Explaining works on one paper at a time.")
    job = answer.start(
        answer.AskRequest(
            question=body.question,
            conversation_id=body.conversation_id,
            mode=body.mode,
            scope=Scope(**body.scope.model_dump()),
            model=body.model,
        )
    )

    async def events():
        try:
            while True:
                try:
                    event = job.events.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.02)
                    continue
                if event is None:
                    return
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            job.cancelled.set()

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})


class ClaimBody(BaseModel):
    text: str = Field(min_length=10, max_length=1500)
    scope: ScopeBody = ScopeBody()


@app.post("/api/claim")
def find_a_source(body: ClaimBody) -> dict:
    try:
        return claims.find_sources(body.text, Scope(**body.scope.model_dump()))
    except llm.LLMError as exc:
        raise HTTPException(503, str(exc)) from exc


# ----------------------------------------------------------- literature matrix


@app.get("/api/matrix")
def matrix_overview() -> dict:
    return matrix.overview()


class FillBody(BaseModel):
    doc_keys: list[str] = Field(min_length=1, max_length=5000)


@app.post("/api/matrix/fill")
def matrix_fill(body: FillBody) -> dict:
    try:
        return {"started": matrix.filler.start(body.doc_keys)}
    except llm.LLMError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post("/api/matrix/stop")
def matrix_stop() -> dict:
    matrix.filler.stop()
    return {"stopping": True}


class ColumnBody(BaseModel):
    label: str = Field(min_length=1, max_length=60)
    question: str = Field(min_length=5, max_length=300)


@app.post("/api/matrix/columns")
def matrix_add_column(body: ColumnBody) -> dict:
    label, question = body.label.strip(), body.question.strip()
    if not label or len(question) < 5:
        raise HTTPException(400, "Give the column a name and say what to look for.")
    return store.add_matrix_column(label, question)


@app.delete("/api/matrix/columns/{column_id}")
def matrix_delete_column(column_id: int) -> dict:
    store.delete_matrix_column(column_id)
    return {"deleted": True}


@app.get("/api/matrix/export")
def matrix_export(keys: str = "") -> Response:
    workbook = matrix.export_xlsx([key for key in keys.split(",") if key] or None)
    return Response(
        workbook,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="literature-matrix.xlsx"', "Cache-Control": "no-store"},
    )


@app.post("/api/ask/stop")
def stop_answer() -> dict:
    answer.stop_active()
    return {"stopped": True}


@app.get("/api/conversations")
def conversations() -> list[dict]:
    return store.list_conversations()


@app.get("/api/conversations/{conversation_id}")
def conversation(conversation_id: int) -> dict:
    return {"messages": store.get_messages(conversation_id)}


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: int) -> dict:
    store.delete_conversation(conversation_id)
    return {"deleted": True}


# -------------------------------------------------------------------- discover


@app.get("/api/discover")
def discover_search(q: str, oa: bool = True, year_min: int | None = None, sort: str = "relevance") -> dict:
    if not q.strip():
        return {"results": [], "source": "OpenAlex"}
    try:
        return discover.search(q.strip(), open_access_only=oa, year_min=year_min, sort=sort)
    except discover.DiscoverError as exc:
        raise HTTPException(502, str(exc)) from exc


class AddBody(BaseModel):
    work: dict


@app.post("/api/discover/add")
def discover_add(body: AddBody) -> dict:
    try:
        result = discover.add_to_zotero(body.work)
    except discover.DiscoverError as exc:
        raise HTTPException(502, str(exc)) from exc
    indexer.start()
    return result


# --------------------------------------------------------------- the interface


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


class _FreshStaticFiles(StaticFiles):
    """The interface's own files. Browsers check for a newer copy on every load, which costs
    nothing on this computer, so a new version of Folio shows without a forced refresh."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


app.mount("/static", _FreshStaticFiles(directory=STATIC_DIR), name="static")
