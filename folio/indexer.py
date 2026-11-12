"""Keep Folio's index in step with Zotero, incrementally and in the background."""

from __future__ import annotations

import logging
import multiprocessing
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from typing import Iterator

from . import config, llm, store, zotero
from .models import download_models, engine, gpu_lock, models_ready
from .pdf import parse_pdf

log = logging.getLogger("folio.indexer")

_WATCH_INTERVAL = 15  # seconds
_PARSE_WORKERS = min(8, max(2, (os.cpu_count() or 4) // 3))


def _embedding_text(title: str, section: str, text: str) -> str:
    """Title and section give a passage context that its words alone may lack."""
    return "\n".join(part for part in (title[:200], section[:120], text) if part)


def _abstract_chunk(abstract: str) -> dict:
    return {"page": 0, "page_end": 0, "section": "Abstract", "text": abstract.strip(), "rects": []}


def _parse_all(paths: dict[str, str]) -> Iterator[tuple[str, dict]]:
    """Parse PDFs across CPU cores. A PDF that crashes the parser fails alone, not the whole batch."""
    if not paths:
        return
    if len(paths) <= 2:
        for key, path in paths.items():
            yield key, _parse_isolated(path)
        return
    remaining = dict(paths)
    context = multiprocessing.get_context("spawn")
    try:
        with ProcessPoolExecutor(max_workers=_PARSE_WORKERS, mp_context=context) as pool:
            futures = {pool.submit(parse_pdf, path): key for key, path in paths.items()}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    result = future.result()
                except BrokenProcessPool:
                    continue
                remaining.pop(key, None)
                yield key, result
    except BrokenProcessPool:
        pass
    for key, path in remaining.items():
        yield key, _parse_isolated(path)


def _parse_isolated(path: str) -> dict:
    context = multiprocessing.get_context("spawn")
    try:
        with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
            return pool.submit(parse_pdf, path).result(timeout=300)
    except Exception as exc:
        return {"status": "error", "pages": 0, "chunks": [], "error": f"The parser failed on this PDF: {exc}"}


class Indexer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_stamp: tuple | None = None
        self._stop = threading.Event()
        self.state: dict = {
            "phase": "idle", "done": 0, "total": 0, "current": "", "error": None, "last_sync": None,
        }

    def start(self, force: bool = False) -> bool:
        """Start a sync in the background. Returns False if one is already running."""
        if self._lock.locked():
            return False
        threading.Thread(target=self._run_guarded, args=(force,), daemon=True, name="folio-sync").start()
        return True

    def run_blocking(self, force: bool = False) -> None:
        self._run_guarded(force)

    def _update(self, **changes) -> None:
        self.state = {**self.state, **changes}

    def _run_guarded(self, force: bool) -> None:
        if not self._lock.acquire(blocking=False):
            return
        try:
            self._update(phase="reading", done=0, total=0, current="", error=None)
            self._sync(force)
            self._update(phase="idle", current="", last_sync=time.time())
        except zotero.ZoteroError as exc:
            self._update(phase="idle", current="", error=str(exc))
        except Exception as exc:
            log.exception("Sync failed")
            self._update(phase="idle", current="", error=f"Sync failed: {exc}")
        finally:
            self._lock.release()

    def _sync(self, force: bool) -> None:
        self._last_stamp = zotero.database_stamp()
        library = zotero.read_library()
        current = {item.doc_key: item for item in library.items}
        known = store.fingerprints()

        for key in known.keys() - current.keys():
            store.delete_doc(key)
        store.upsert_docs_metadata(library.items)

        todo = [item for key, item in current.items() if force or known.get(key) != item.fingerprint]
        if not todo:
            return
        if not models_ready():
            self._update(phase="downloading", current="Search models")
            download_models(lambda text: self._update(current=text))

        self._update(phase="indexing", total=len(todo), done=0, current="")
        by_key = {item.doc_key: item for item in todo}

        # Items without a PDF can still be searched through their abstracts.
        for item in todo:
            if not item.pdf_path:
                if item.abstract.strip():
                    self._store(item, "abstract_only", None, 0, [_abstract_chunk(item.abstract)])
                else:
                    self._store(item, "metadata_only", None, 0, [])

        paths = {item.doc_key: item.pdf_path for item in todo if item.pdf_path}
        try:
            for key, parsed in _parse_all(paths):
                item = by_key[key]
                chunks = parsed["chunks"]
                if not chunks and item.abstract.strip():
                    chunks = [_abstract_chunk(item.abstract)]
                self._store(item, parsed["status"], parsed["error"], parsed["pages"], chunks)
        finally:
            with gpu_lock:
                engine.release_gpu()

    def _store(self, item, status: str, error: str | None, pages: int, chunks: list[dict]) -> None:
        self._update(current=item.title)
        texts = [_embedding_text(item.title, chunk["section"], chunk["text"]) for chunk in chunks]
        vectors = None
        if texts:
            with gpu_lock:
                # If the answer model is loaded, it has to make room.
                if not engine.gpu_in_use:
                    llm.unload_all()
                vectors = engine.embed_documents(texts)
        store.save_indexed(item.doc_key, item.fingerprint, status, error, pages, chunks, vectors)
        self._update(done=self.state["done"] + 1)

    # ---------------------------------------------------------------- auto-sync

    def watch(self) -> None:
        """Notice new articles in Zotero without being asked."""
        threading.Thread(target=self._watch_loop, daemon=True, name="folio-watch").start()

    def stop(self) -> None:
        self._stop.set()

    def _watch_loop(self) -> None:
        while not self._stop.wait(_WATCH_INTERVAL):
            stamp = zotero.database_stamp()
            if stamp is not None and stamp != self._last_stamp:
                self.start()


indexer = Indexer()
