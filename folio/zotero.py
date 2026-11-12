"""Read the Zotero library and add items to it.

Reading works from a private copy of zotero.sqlite, so Zotero's own files are
never touched and it doesn't matter whether Zotero is open. Writing goes only
through Zotero's own connector API, the same one its browser extension uses.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from . import config
from .pdf import PARSER_VERSION

_PREF_RE = re.compile(r'^user_pref\("([^"]+)",\s*(.+)\);\s*$')
_VENUE_FIELDS = (
    "publicationTitle",
    "proceedingsTitle",
    "conferenceName",
    "bookTitle",
    "university",
    "repository",
    "publisher",
)


@dataclass
class Item:
    library_id: int
    group_id: int | None
    key: str
    item_type: str
    title: str
    creators: list[dict] = field(default_factory=list)
    year: int | None = None
    venue: str = ""
    doi: str = ""
    url: str = ""
    abstract: str = ""
    tags: list[str] = field(default_factory=list)
    collections: list[str] = field(default_factory=list)
    attachment_key: str | None = None
    pdf_path: str | None = None
    date_added: str = ""
    fields: dict = field(default_factory=dict)

    @property
    def doc_key(self) -> str:
        return f"{self.library_id}:{self.key}"

    @property
    def fingerprint(self) -> str:
        """Changes whenever the item needs indexing again."""
        if self.pdf_path:
            try:
                stat = os.stat(self.pdf_path)
                return f"v{PARSER_VERSION}:pdf:{stat.st_size}:{int(stat.st_mtime)}"
            except OSError:
                pass
        digest = hashlib.md5(self.abstract.encode("utf-8")).hexdigest()[:12]
        return f"v{PARSER_VERSION}:abs:{digest}"


@dataclass
class Library:
    data_dir: Path
    items: list[Item]
    collections: list[str]


class ZoteroError(RuntimeError):
    pass


# ---------------------------------------------------------------- locating data


def _profile_prefs() -> dict:
    """Read Zotero's preferences from its most recently used profile."""
    root = Path(os.environ.get("APPDATA") or Path.home()) / "Zotero" / "Zotero" / "Profiles"
    candidates = sorted(root.glob("*/prefs.js"), key=lambda p: p.stat().st_mtime, reverse=True) if root.exists() else []
    prefs: dict = {}
    if not candidates:
        return prefs
    try:
        text = candidates[0].read_text(encoding="utf-8", errors="replace")
    except OSError:
        return prefs
    for line in text.splitlines():
        match = _PREF_RE.match(line.strip())
        if match:
            try:
                prefs[match.group(1)] = json.loads(match.group(2))
            except ValueError:
                pass
    return prefs


def find_data_dir() -> Path | None:
    override = config.load_settings().get("zotero_dir")
    candidates = [Path(override)] if override else []
    prefs = _profile_prefs()
    if prefs.get("extensions.zotero.useDataDir") and prefs.get("extensions.zotero.dataDir"):
        candidates.append(Path(prefs["extensions.zotero.dataDir"]))
    candidates.append(Path.home() / "Zotero")
    for candidate in candidates:
        if (candidate / "zotero.sqlite").exists():
            return candidate
    return None


def _stamp(data_dir: Path) -> tuple:
    """Size and modification time of the database and its write-ahead log."""
    stamp = []
    for name in ("zotero.sqlite", "zotero.sqlite-wal"):
        try:
            stat = (data_dir / name).stat()
            stamp.extend((stat.st_size, stat.st_mtime_ns))
        except OSError:
            stamp.extend((None, None))
    return tuple(stamp)


def database_stamp() -> tuple | None:
    """Changes whenever Zotero writes to its database."""
    data_dir = find_data_dir()
    return _stamp(data_dir) if data_dir else None


def _snapshot(data_dir: Path, destination: Path) -> None:
    """Copy the database and its log, again if Zotero wrote while they were being copied.

    Zotero keeps recent changes in zotero.sqlite-wal, so the main file alone is out of date.
    """
    for _ in range(5):
        before = _stamp(data_dir)
        shutil.copy2(data_dir / "zotero.sqlite", destination)
        wal = data_dir / "zotero.sqlite-wal"
        wal_copy = destination.with_name(destination.name + "-wal")
        wal_copy.unlink(missing_ok=True)
        if wal.exists():
            shutil.copy2(wal, wal_copy)
        if _stamp(data_dir) == before:
            return
        time.sleep(0.5)


# ---------------------------------------------------------------------- reading


def _year_of(date: str) -> int | None:
    match = re.search(r"\b(1[5-9]\d\d|20\d\d|21\d\d)\b", date or "")
    return int(match.group(1)) if match else None


def _resolve_attachment(data_dir: Path, base_path: str | None, att_key: str, link_mode: int, path: str | None) -> str | None:
    """Turn Zotero's stored attachment path into a real path, if the file exists."""
    if not path:
        return None
    if path.startswith("storage:"):
        resolved = data_dir / "storage" / att_key / path[len("storage:"):]
    elif path.startswith("attachments:"):
        if not base_path:
            return None
        resolved = Path(base_path) / path[len("attachments:"):]
    else:
        resolved = Path(path)
    return str(resolved) if resolved.is_file() else None


def read_library(data_dir: Path | None = None) -> Library:
    data_dir = data_dir or find_data_dir()
    if not data_dir:
        raise ZoteroError("Zotero's database wasn't found. Open Zotero once so it can create it.")
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=config.DATA_DIR, prefix="zotero-") as tmp:
        copy = Path(tmp) / "zotero.sqlite"
        _snapshot(data_dir, copy)
        conn = sqlite3.connect(copy)
        try:
            return _read_tables(conn, data_dir)
        finally:
            conn.close()


def _read_tables(conn: sqlite3.Connection, data_dir: Path) -> Library:
    conn.row_factory = sqlite3.Row
    base_path = _profile_prefs().get("extensions.zotero.baseAttachmentPath")

    groups = {r["libraryID"]: r["groupID"] for r in conn.execute("SELECT groupID, libraryID FROM groups")}
    libraries = {r["libraryID"] for r in conn.execute("SELECT libraryID FROM libraries WHERE type IN ('user','group')")}
    deleted = {r["itemID"] for r in conn.execute("SELECT itemID FROM deletedItems")}

    fields: dict[int, dict] = {}
    for row in conn.execute(
        """SELECT d.itemID, f.fieldName, v.value
           FROM itemData d
           JOIN fields f ON f.fieldID = d.fieldID
           JOIN itemDataValues v ON v.valueID = d.valueID"""
    ):
        fields.setdefault(row["itemID"], {})[row["fieldName"]] = row["value"]

    creators: dict[int, list[dict]] = {}
    for row in conn.execute(
        """SELECT ic.itemID, c.firstName, c.lastName, ct.creatorType
           FROM itemCreators ic
           JOIN creators c ON c.creatorID = ic.creatorID
           JOIN creatorTypes ct ON ct.creatorTypeID = ic.creatorTypeID
           ORDER BY ic.itemID, ic.orderIndex"""
    ):
        creators.setdefault(row["itemID"], []).append(
            {"first": row["firstName"] or "", "last": row["lastName"] or "", "type": row["creatorType"]}
        )

    tags: dict[int, list[str]] = {}
    for row in conn.execute("SELECT it.itemID, t.name FROM itemTags it JOIN tags t ON t.tagID = it.tagID"):
        tags.setdefault(row["itemID"], []).append(row["name"])

    # Collections, shown as "Parent / Child".
    rows = list(conn.execute("SELECT collectionID, collectionName, parentCollectionID FROM collections"))
    by_id = {r["collectionID"]: r for r in rows}

    def collection_path(cid: int) -> str:
        parts, seen = [], set()
        while cid in by_id and cid not in seen:
            seen.add(cid)
            parts.append(by_id[cid]["collectionName"])
            cid = by_id[cid]["parentCollectionID"]
        return " / ".join(reversed(parts))

    # A collection in Zotero's trash, or inside one, isn't offered as a filter.
    trashed: set[int] = set()
    if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='deletedCollections'").fetchone():
        trashed = {r["collectionID"] for r in conn.execute("SELECT collectionID FROM deletedCollections")}

    def in_trash(cid: int) -> bool:
        seen = set()
        while cid in by_id and cid not in seen:
            if cid in trashed:
                return True
            seen.add(cid)
            cid = by_id[cid]["parentCollectionID"]
        return False

    paths = {cid: collection_path(cid) for cid in by_id if not in_trash(cid)}
    item_collections: dict[int, list[str]] = {}
    for row in conn.execute("SELECT collectionID, itemID FROM collectionItems"):
        if row["collectionID"] in paths:
            item_collections.setdefault(row["itemID"], []).append(paths[row["collectionID"]])

    # The oldest PDF present on disk is the item's main file, which is also Zotero's default.
    attachments: dict[int, tuple[str, str]] = {}
    standalone = []
    for row in conn.execute(
        """SELECT a.itemID, a.parentItemID, a.linkMode, a.contentType, a.path, i.key, i.libraryID, i.dateAdded
           FROM itemAttachments a JOIN items i ON i.itemID = a.itemID
           ORDER BY i.dateAdded"""
    ):
        if row["itemID"] in deleted or row["contentType"] != "application/pdf":
            continue
        resolved = _resolve_attachment(data_dir, base_path, row["key"], row["linkMode"], row["path"])
        if row["parentItemID"] is None:
            standalone.append((row, resolved))
        elif resolved and row["parentItemID"] not in attachments:
            attachments[row["parentItemID"]] = (row["key"], resolved)

    items: list[Item] = []
    for row in conn.execute(
        """SELECT i.itemID, i.key, i.libraryID, i.dateAdded, t.typeName
           FROM items i JOIN itemTypes t ON t.itemTypeID = i.itemTypeID
           WHERE t.typeName NOT IN ('attachment', 'note', 'annotation')"""
    ):
        item_id = row["itemID"]
        if item_id in deleted or row["libraryID"] not in libraries:
            continue
        data = fields.get(item_id, {})
        att_key, pdf_path = attachments.get(item_id, (None, None))
        items.append(
            Item(
                library_id=row["libraryID"],
                group_id=groups.get(row["libraryID"]),
                key=row["key"],
                item_type=row["typeName"],
                title=data.get("title", "").strip() or "(untitled)",
                creators=creators.get(item_id, []),
                year=_year_of(data.get("date", "")),
                venue=next((data[f] for f in _VENUE_FIELDS if data.get(f)), ""),
                doi=data.get("DOI", ""),
                url=data.get("url", ""),
                abstract=data.get("abstractNote", ""),
                tags=sorted(tags.get(item_id, [])),
                collections=sorted(item_collections.get(item_id, [])),
                attachment_key=att_key,
                pdf_path=pdf_path,
                date_added=row["dateAdded"],
                fields=data,
            )
        )

    # A PDF dropped into Zotero with no parent item is still something to search.
    for row, resolved in standalone:
        if row["libraryID"] not in libraries or not resolved:
            continue
        data = fields.get(row["itemID"], {})
        items.append(
            Item(
                library_id=row["libraryID"],
                group_id=groups.get(row["libraryID"]),
                key=row["key"],
                item_type="attachment",
                title=data.get("title", "").strip() or Path(resolved).stem,
                tags=sorted(tags.get(row["itemID"], [])),
                collections=sorted(item_collections.get(row["itemID"], [])),
                attachment_key=row["key"],
                pdf_path=resolved,
                date_added=row["dateAdded"],
                fields=data,
            )
        )

    return Library(data_dir=data_dir, items=items, collections=sorted(set(paths.values())))


# ----------------------------------------------------------------- deep links


def _library_prefix(group_id: int | None) -> str:
    return f"groups/{group_id}" if group_id else "library"


def select_uri(group_id: int | None, key: str) -> str:
    return f"zotero://select/{_library_prefix(group_id)}/items/{key}"


def open_pdf_uri(group_id: int | None, attachment_key: str, page: int | None = None) -> str:
    uri = f"zotero://open-pdf/{_library_prefix(group_id)}/items/{attachment_key}"
    return f"{uri}?page={page}" if page else uri


# ------------------------------------------------------------ connector (writes)


def executable() -> str | None:
    for root in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)"), os.environ.get("LOCALAPPDATA")):
        if root:
            path = Path(root) / "Zotero" / "zotero.exe"
            if path.exists():
                return str(path)
    return shutil.which("zotero")


def is_running() -> bool:
    try:
        return httpx.get(f"{config.ZOTERO_URL}/connector/ping", timeout=1.5).status_code == 200
    except httpx.HTTPError:
        return False


def ensure_running(timeout: float = 30.0) -> bool:
    """Start Zotero if it's closed, and wait until it answers."""
    if is_running():
        return True
    exe = executable()
    if not exe:
        return False
    subprocess.Popen([exe], close_fds=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_running():
            return True
        time.sleep(0.5)
    return False


def save_item(item: dict, pdf: bytes | None = None, pdf_url: str = "") -> dict:
    """Create one item in Zotero (in whatever library or collection is selected there)."""
    if not ensure_running():
        raise ZoteroError("Zotero isn't running and couldn't be started.")
    session = uuid.uuid4().hex
    connector_id = uuid.uuid4().hex[:8]
    payload = {
        "sessionID": session,
        "uri": item.get("url") or pdf_url or "https://folio.invalid/",
        "items": [{**item, "id": connector_id, "attachments": []}],
    }
    headers = {"X-Zotero-Connector-API-Version": "3"}
    try:
        response = httpx.post(f"{config.ZOTERO_URL}/connector/saveItems", json=payload, headers=headers, timeout=60)
    except httpx.HTTPError as exc:
        raise ZoteroError(f"Couldn't reach Zotero: {exc}") from exc
    if response.status_code != 201:
        raise ZoteroError(f"Zotero refused the item (HTTP {response.status_code}): {response.text[:200]}")

    attached = False
    if pdf:
        metadata = {
            "sessionID": session,
            "parentItemID": connector_id,
            "title": "Full Text PDF",
            "url": pdf_url or item.get("url") or "",
        }
        try:
            response = httpx.post(
                f"{config.ZOTERO_URL}/connector/saveAttachment",
                content=pdf,
                headers={"Content-Type": "application/pdf", "X-Metadata": json.dumps(metadata)},
                timeout=180,
            )
        except httpx.HTTPError as exc:
            raise ZoteroError(f"Zotero saved the item but not its PDF: {exc}") from exc
        attached = response.status_code == 201
        if not attached:
            raise ZoteroError(f"Zotero saved the item but not its PDF (HTTP {response.status_code}).")
    return {"saved": True, "pdf_attached": attached}
