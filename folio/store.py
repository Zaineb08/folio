"""Folio's own database: documents, passages, vectors, keyword index and conversations.

Vectors live in SQLite beside their passages, so a document is always written
atomically: a crash can never leave search and text disagreeing. For search,
they're held in one NumPy matrix, which is fast at any realistic thesis library size.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Iterable, Iterator

import numpy as np

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    key TEXT PRIMARY KEY,
    item_key TEXT NOT NULL,
    library_id INTEGER NOT NULL,
    group_id INTEGER,
    attachment_key TEXT,
    item_type TEXT,
    title TEXT NOT NULL,
    creators TEXT NOT NULL DEFAULT '[]',
    year INTEGER,
    venue TEXT DEFAULT '',
    doi TEXT DEFAULT '',
    url TEXT DEFAULT '',
    abstract TEXT DEFAULT '',
    tags TEXT NOT NULL DEFAULT '[]',
    collections TEXT NOT NULL DEFAULT '[]',
    pdf_path TEXT,
    date_added TEXT,
    fingerprint TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    error TEXT,
    pages INTEGER DEFAULT 0,
    chunks INTEGER DEFAULT 0,
    indexed_at REAL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    doc_key TEXT NOT NULL REFERENCES docs(key) ON DELETE CASCADE,
    ord INTEGER NOT NULL,
    page INTEGER NOT NULL,
    page_end INTEGER NOT NULL,
    section TEXT DEFAULT '',
    text TEXT NOT NULL,
    rects TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_key);
CREATE TABLE IF NOT EXISTS vectors (
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    vec BLOB NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, title, section, tokenize = "unicode61 remove_diacritics 2"
);
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    mode TEXT,
    query TEXT,
    sources TEXT,
    checks TEXT,
    meta TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_conv ON messages(conversation_id, id);
CREATE TABLE IF NOT EXISTS matrix_columns (
    id INTEGER PRIMARY KEY,
    label TEXT NOT NULL,
    question TEXT NOT NULL,
    position INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS matrix_cells (
    doc_key TEXT NOT NULL REFERENCES docs(key) ON DELETE CASCADE,
    column_id INTEGER NOT NULL REFERENCES matrix_columns(id) ON DELETE CASCADE,
    fingerprint TEXT,
    text TEXT NOT NULL,
    sources TEXT NOT NULL,
    flags TEXT NOT NULL,
    model TEXT,
    updated_at REAL NOT NULL,
    PRIMARY KEY (doc_key, column_id)
);
"""

# What most literature matrices compare. The student can remove these and add their own.
MATRIX_DEFAULT_COLUMNS = (
    ("Research question", "What question or aim does this paper address?"),
    ("Method", "What did the authors do: the study design, approach or tool?"),
    ("Data and participants", "What data, benchmarks, models or participants did they use?"),
    ("Main findings", "What are the main results or findings?"),
    ("Limitations", "What limitations or threats to validity do the authors state?"),
)

_init_lock = threading.Lock()
_initialised = False
_JSON_DOC_FIELDS = ("creators", "tags", "collections")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    global _initialised
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    with _init_lock:
        if not _initialised:
            had_matrix = conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'matrix_columns'").fetchone()
            conn.executescript(_SCHEMA)
            if not had_matrix:  # Only once, so columns the student removed don't come back.
                conn.executemany(
                    "INSERT INTO matrix_columns (label, question, position) VALUES (?,?,?)",
                    [(label, question, position) for position, (label, question) in enumerate(MATRIX_DEFAULT_COLUMNS)],
                )
                conn.commit()
            # Databases made before answers recorded which model wrote them.
            if "meta" not in {row[1] for row in conn.execute("PRAGMA table_info(messages)")}:
                conn.execute("ALTER TABLE messages ADD COLUMN meta TEXT")
            _initialised = True
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def reset_for_tests() -> None:
    global _initialised
    _initialised = False
    vector_index.mark_stale()


def _doc_from_row(row: sqlite3.Row) -> dict:
    doc = dict(row)
    for name in _JSON_DOC_FIELDS:
        doc[name] = json.loads(doc[name] or "[]")
    return doc


# ------------------------------------------------------------------- documents


def upsert_docs_metadata(items: list) -> None:
    """Save items' metadata without touching their indexing state."""
    with connect() as conn:
        conn.executemany(
            """INSERT INTO docs (key, item_key, library_id, group_id, attachment_key, item_type, title, creators,
                                 year, venue, doi, url, abstract, tags, collections, pdf_path, date_added)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(key) DO UPDATE SET
                 group_id=excluded.group_id, attachment_key=excluded.attachment_key, item_type=excluded.item_type,
                 title=excluded.title, creators=excluded.creators, year=excluded.year, venue=excluded.venue,
                 doi=excluded.doi, url=excluded.url, abstract=excluded.abstract, tags=excluded.tags,
                 collections=excluded.collections, pdf_path=excluded.pdf_path, date_added=excluded.date_added""",
            [
                (
                    item.doc_key, item.key, item.library_id, item.group_id, item.attachment_key, item.item_type,
                    item.title, json.dumps(item.creators), item.year, item.venue, item.doi, item.url, item.abstract,
                    json.dumps(item.tags), json.dumps(item.collections), item.pdf_path, item.date_added,
                )
                for item in items
            ],
        )


def list_docs() -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM docs ORDER BY date_added DESC, title").fetchall()
    return [_doc_from_row(row) for row in rows]


def get_doc(key: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM docs WHERE key = ?", (key,)).fetchone()
    return _doc_from_row(row) if row else None


def fingerprints() -> dict[str, str | None]:
    with connect() as conn:
        return {row["key"]: row["fingerprint"] for row in conn.execute("SELECT key, fingerprint FROM docs")}


def delete_doc(key: str) -> None:
    with connect() as conn:
        _delete_chunks(conn, key)
        conn.execute("DELETE FROM docs WHERE key = ?", (key,))
    vector_index.mark_stale()


def _delete_chunks(conn: sqlite3.Connection, key: str) -> None:
    conn.execute("DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE doc_key = ?)", (key,))
    conn.execute("DELETE FROM chunks WHERE doc_key = ?", (key,))


def save_indexed(key: str, fingerprint: str, status: str, error: str | None, pages: int, chunks: list[dict], vectors: np.ndarray | None) -> None:
    """Replace one document's passages and vectors in a single transaction."""
    with connect() as conn:
        _delete_chunks(conn, key)
        title = conn.execute("SELECT title FROM docs WHERE key = ?", (key,)).fetchone()["title"]
        for ord_, chunk in enumerate(chunks):
            chunk_id = conn.execute(
                "INSERT INTO chunks (doc_key, ord, page, page_end, section, text, rects) VALUES (?,?,?,?,?,?,?)",
                (key, ord_, chunk["page"], chunk["page_end"], chunk["section"], chunk["text"], json.dumps(chunk["rects"], separators=(",", ":"))),
            ).lastrowid
            conn.execute(
                "INSERT INTO chunks_fts (rowid, text, title, section) VALUES (?,?,?,?)",
                (chunk_id, chunk["text"], title, chunk["section"]),
            )
            conn.execute("INSERT INTO vectors (chunk_id, vec) VALUES (?,?)", (chunk_id, vectors[ord_].astype(np.float32).tobytes()))
        conn.execute(
            "UPDATE docs SET fingerprint=?, status=?, error=?, pages=?, chunks=?, indexed_at=? WHERE key=?",
            (fingerprint, status, error, pages, len(chunks), time.time(), key),
        )
    vector_index.mark_stale()


def counts() -> dict:
    with connect() as conn:
        docs = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        statuses = {row[0]: row[1] for row in conn.execute("SELECT status, COUNT(*) FROM docs GROUP BY status")}
    return {"docs": docs, "chunks": chunks, "statuses": statuses}


# --------------------------------------------------------------------- passages


def get_chunks(ids: Iterable[int]) -> dict[int, dict]:
    ids = list(ids)
    if not ids:
        return {}
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM chunks WHERE id IN (SELECT value FROM json_each(?))", (json.dumps(ids),)
        ).fetchall()
    return {row["id"]: dict(row) for row in rows}


def get_chunk(chunk_id: int) -> dict | None:
    return get_chunks([chunk_id]).get(chunk_id)


def chunks_of(doc_key: str) -> list[dict]:
    """A document's passages in reading order, without their text."""
    with connect() as conn:
        rows = conn.execute("SELECT id, section FROM chunks WHERE doc_key = ? ORDER BY ord", (doc_key,)).fetchall()
    return [dict(row) for row in rows]


def keyword_search(query: str, doc_keys: set[str] | None, limit: int) -> list[int]:
    """Exact-term search, which catches names, acronyms and jargon that embeddings blur."""
    terms = list(dict.fromkeys(re.findall(r"[^\W_]+", query.lower())))[:32]
    if not terms:
        return []
    match = " OR ".join(f'"{term}"' for term in terms)
    with connect() as conn:
        if doc_keys is None:
            rows = conn.execute(
                "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts, 1.0, 2.0, 1.0) LIMIT ?",
                (match, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT f.rowid FROM chunks_fts f JOIN chunks c ON c.id = f.rowid
                   WHERE chunks_fts MATCH ? AND c.doc_key IN (SELECT value FROM json_each(?))
                   ORDER BY bm25(chunks_fts, 1.0, 2.0, 1.0) LIMIT ?""",
                (match, json.dumps(sorted(doc_keys)), limit),
            ).fetchall()
    return [row[0] for row in rows]


class VectorIndex:
    """All passage vectors in one matrix, reloaded lazily after indexing changes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stale = True
        self._ids = np.empty(0, dtype=np.int64)
        self._doc_codes = np.empty(0, dtype=np.int32)
        self._codes: dict[str, int] = {}
        self._matrix = np.empty((0, config.EMBED_DIM), dtype=np.float32)

    def mark_stale(self) -> None:
        self._stale = True

    def _load(self) -> None:
        with connect() as conn:
            rows = conn.execute(
                "SELECT v.chunk_id, c.doc_key, v.vec FROM vectors v JOIN chunks c ON c.id = v.chunk_id ORDER BY v.chunk_id"
            ).fetchall()
        codes: dict[str, int] = {}
        ids = np.empty(len(rows), dtype=np.int64)
        doc_codes = np.empty(len(rows), dtype=np.int32)
        matrix = np.empty((len(rows), config.EMBED_DIM), dtype=np.float32)
        for index, row in enumerate(rows):
            ids[index] = row[0]
            doc_codes[index] = codes.setdefault(row[1], len(codes))
            matrix[index] = np.frombuffer(row[2], dtype=np.float32)
        self._ids, self._doc_codes, self._matrix, self._codes = ids, doc_codes, matrix, codes
        self._stale = False

    def search(self, query_vector: np.ndarray, doc_keys: set[str] | None, limit: int) -> list[int]:
        with self._lock:
            if self._stale:
                self._load()
            if not len(self._ids):
                return []
            scores = self._matrix @ query_vector
            if doc_keys is not None:
                allowed = [self._codes[key] for key in doc_keys if key in self._codes]
                scores = np.where(np.isin(self._doc_codes, allowed), scores, -np.inf)
            limit = min(limit, len(scores))
            top = np.argpartition(-scores, limit - 1)[:limit]
            top = top[np.argsort(-scores[top])]
            return [int(self._ids[index]) for index in top if np.isfinite(scores[index])]


vector_index = VectorIndex()


# ---------------------------------------------------------------- conversations


def create_conversation(title: str) -> int:
    now = time.time()
    with connect() as conn:
        return conn.execute(
            "INSERT INTO conversations (title, created_at, updated_at) VALUES (?,?,?)", (title[:120], now, now)
        ).lastrowid


def list_conversations(limit: int = 100) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, title, updated_at FROM conversations ORDER BY updated_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


def get_messages(conversation_id: int) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id", (conversation_id,)
        ).fetchall()
    messages = []
    for row in rows:
        message = dict(row)
        message["sources"] = json.loads(message["sources"]) if message["sources"] else []
        message["checks"] = json.loads(message["checks"]) if message["checks"] else None
        message["meta"] = json.loads(message["meta"]) if message["meta"] else None
        messages.append(message)
    return messages


def add_message(conversation_id: int, role: str, content: str, mode: str | None = None, query: str | None = None, sources: list | None = None, checks: dict | None = None, meta: dict | None = None) -> int:
    now = time.time()
    with connect() as conn:
        message_id = conn.execute(
            "INSERT INTO messages (conversation_id, role, content, mode, query, sources, checks, meta, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (conversation_id, role, content, mode, query, json.dumps(sources) if sources else None, json.dumps(checks) if checks else None, json.dumps(meta) if meta else None, now),
        ).lastrowid
        conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
    return message_id


def delete_conversation(conversation_id: int) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))


# ------------------------------------------------------------- literature matrix


def matrix_columns() -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT id, label, question FROM matrix_columns ORDER BY position, id").fetchall()
    return [dict(row) for row in rows]


def add_matrix_column(label: str, question: str) -> dict:
    with connect() as conn:
        position = conn.execute("SELECT COALESCE(MAX(position) + 1, 0) FROM matrix_columns").fetchone()[0]
        column_id = conn.execute(
            "INSERT INTO matrix_columns (label, question, position) VALUES (?,?,?)", (label, question, position)
        ).lastrowid
    return {"id": column_id, "label": label, "question": question}


def delete_matrix_column(column_id: int) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM matrix_columns WHERE id = ?", (column_id,))


def matrix_cells() -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM matrix_cells").fetchall()
    cells = []
    for row in rows:
        cell = dict(row)
        cell["sources"] = json.loads(cell["sources"])
        cell["flags"] = json.loads(cell["flags"])
        cells.append(cell)
    return cells


def save_matrix_cell(doc_key: str, column_id: int, fingerprint: str | None, text: str, sources: list, flags: list, model: str) -> None:
    with connect() as conn:
        # The column, or the paper, may have been removed while its cell was being written.
        column_exists = conn.execute("SELECT 1 FROM matrix_columns WHERE id = ?", (column_id,)).fetchone()
        doc_exists = conn.execute("SELECT 1 FROM docs WHERE key = ?", (doc_key,)).fetchone()
        if column_exists and doc_exists:
            conn.execute(
                "INSERT OR REPLACE INTO matrix_cells (doc_key, column_id, fingerprint, text, sources, flags, model, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (doc_key, column_id, fingerprint, text, json.dumps(sources), json.dumps(flags), model, time.time()),
            )
