"""Reading Zotero's schema, from a miniature database of the same shape."""

import sqlite3

import pytest

from folio import zotero

SCHEMA = """
CREATE TABLE libraries (libraryID INTEGER PRIMARY KEY, type TEXT);
CREATE TABLE groups (groupID INTEGER PRIMARY KEY, libraryID INT);
CREATE TABLE itemTypes (itemTypeID INTEGER PRIMARY KEY, typeName TEXT);
CREATE TABLE items (itemID INTEGER PRIMARY KEY, itemTypeID INT, libraryID INT, key TEXT, dateAdded TEXT);
CREATE TABLE deletedItems (itemID INTEGER PRIMARY KEY);
CREATE TABLE fields (fieldID INTEGER PRIMARY KEY, fieldName TEXT);
CREATE TABLE itemDataValues (valueID INTEGER PRIMARY KEY, value TEXT);
CREATE TABLE itemData (itemID INT, fieldID INT, valueID INT);
CREATE TABLE creators (creatorID INTEGER PRIMARY KEY, firstName TEXT, lastName TEXT);
CREATE TABLE creatorTypes (creatorTypeID INTEGER PRIMARY KEY, creatorType TEXT);
CREATE TABLE itemCreators (itemID INT, creatorID INT, creatorTypeID INT, orderIndex INT);
CREATE TABLE tags (tagID INTEGER PRIMARY KEY, name TEXT);
CREATE TABLE itemTags (itemID INT, tagID INT);
CREATE TABLE collections (collectionID INTEGER PRIMARY KEY, collectionName TEXT, parentCollectionID INT);
CREATE TABLE collectionItems (collectionID INT, itemID INT);
CREATE TABLE itemAttachments (itemID INTEGER PRIMARY KEY, parentItemID INT, linkMode INT, contentType TEXT, path TEXT);
"""


@pytest.fixture
def library(tmp_path, monkeypatch):
    monkeypatch.setattr(zotero, "_profile_prefs", lambda: {})
    conn = sqlite3.connect(tmp_path / "zotero.sqlite")
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO libraries VALUES (?,?)", [(1, "user"), (2, "group"), (3, "feed")])
    conn.execute("INSERT INTO groups VALUES (555, 2)")
    conn.executemany("INSERT INTO itemTypes VALUES (?,?)", [(1, "journalArticle"), (2, "attachment"), (3, "note"), (4, "annotation")])
    conn.executemany(
        "INSERT INTO items VALUES (?,?,?,?,?)",
        [
            (1, 1, 1, "PARENT01", "2026-01-01"),
            (2, 2, 1, "ATTACH01", "2026-01-02"),
            (3, 2, 1, "ATTACH02", "2026-01-03"),  # A second PDF on the same parent.
            (4, 3, 1, "NOTE0001", "2026-01-04"),
            (5, 1, 1, "TRASHED1", "2026-01-05"),
            (6, 1, 2, "GROUPIT1", "2026-01-06"),
            (7, 1, 3, "FEEDITEM", "2026-01-07"),
            (8, 2, 1, "STANDALN", "2026-01-08"),
            (9, 1, 1, "NOPDF001", "2026-01-09"),
        ],
    )
    conn.execute("INSERT INTO deletedItems VALUES (5)")
    conn.executemany("INSERT INTO fields VALUES (?,?)", [(1, "title"), (2, "date"), (3, "DOI"), (4, "publicationTitle"), (5, "abstractNote")])
    values = ["My Article", "2020-05-00 May 2020", "10.1/abc", "A Journal", "An abstract.", "Group Article", "No PDF here", "Loose file"]
    conn.executemany("INSERT INTO itemDataValues VALUES (?,?)", list(enumerate(values, 1)))
    conn.executemany(
        "INSERT INTO itemData VALUES (?,?,?)",
        [(1, 1, 1), (1, 2, 2), (1, 3, 3), (1, 4, 4), (1, 5, 5), (6, 1, 6), (9, 1, 7), (9, 5, 5), (8, 1, 8)],
    )
    conn.executemany("INSERT INTO creators VALUES (?,?,?)", [(1, "Ada", "Lovelace"), (2, "Alan", "Turing")])
    conn.execute("INSERT INTO creatorTypes VALUES (1, 'author')")
    conn.executemany("INSERT INTO itemCreators VALUES (?,?,?,?)", [(1, 2, 1, 1), (1, 1, 1, 0)])
    conn.execute("INSERT INTO tags VALUES (1, 'rag')")
    conn.execute("INSERT INTO itemTags VALUES (1, 1)")
    conn.executemany("INSERT INTO collections VALUES (?,?,?)", [(1, "Thesis", None), (2, "Chapter 1", 1)])
    conn.execute("INSERT INTO collectionItems VALUES (2, 1)")
    conn.executemany(
        "INSERT INTO itemAttachments VALUES (?,?,?,?,?)",
        [
            (2, 1, 1, "application/pdf", "storage:first.pdf"),
            (3, 1, 1, "application/pdf", "storage:second.pdf"),
            (8, None, 1, "application/pdf", "storage:loose.pdf"),
        ],
    )
    conn.commit()
    conn.close()
    for key, name in (("ATTACH01", "first.pdf"), ("ATTACH02", "second.pdf"), ("STANDALN", "loose.pdf")):
        folder = tmp_path / "storage" / key
        folder.mkdir(parents=True)
        (folder / name).write_bytes(b"%PDF-1.7 test")
    return zotero.read_library(tmp_path)


def by_key(library):
    return {item.key: item for item in library.items}


def test_changes_still_in_the_write_ahead_log_are_read(tmp_path, monkeypatch):
    """Zotero 10 uses WAL mode, so its newest items aren't in the main file yet."""
    monkeypatch.setattr(zotero, "_profile_prefs", lambda: {})
    live = sqlite3.connect(tmp_path / "zotero.sqlite")
    live.execute("PRAGMA journal_mode=WAL")
    live.execute("PRAGMA wal_autocheckpoint=0")
    live.executescript(SCHEMA)
    live.execute("INSERT INTO libraries VALUES (1, 'user')")
    live.execute("INSERT INTO itemTypes VALUES (1, 'journalArticle')")
    live.execute("INSERT INTO items VALUES (1, 1, 1, 'INWALONL', '2026-01-01')")
    live.commit()
    try:
        assert (tmp_path / "zotero.sqlite-wal").stat().st_size > 0
        assert "INWALONL" in by_key(zotero.read_library(tmp_path))
        stamp = zotero._stamp(tmp_path)
        live.execute("INSERT INTO items VALUES (2, 1, 1, 'NEWWRITE', '2026-01-02')")
        live.commit()
        assert zotero._stamp(tmp_path) != stamp, "the watcher must notice writes that only reach the log"
        assert "NEWWRITE" in by_key(zotero.read_library(tmp_path))
    finally:
        live.close()


def test_only_real_items_are_read(library):
    assert set(by_key(library)) == {"PARENT01", "GROUPIT1", "STANDALN", "NOPDF001"}


def test_metadata_is_complete(library):
    item = by_key(library)["PARENT01"]
    assert item.title == "My Article" and item.year == 2020 and item.venue == "A Journal"
    assert item.doi == "10.1/abc" and item.abstract == "An abstract."
    assert [c["last"] for c in item.creators] == ["Lovelace", "Turing"]
    assert item.tags == ["rag"]
    assert item.collections == ["Thesis / Chapter 1"]
    assert library.collections == ["Thesis", "Thesis / Chapter 1"]


def test_collections_in_the_trash_are_left_out(tmp_path, monkeypatch):
    """Deleting a collection in Zotero only moves it to the trash; its items stay in the library."""
    monkeypatch.setattr(zotero, "_profile_prefs", lambda: {})
    conn = sqlite3.connect(tmp_path / "zotero.sqlite")
    conn.executescript(SCHEMA + "CREATE TABLE deletedCollections (collectionID INTEGER PRIMARY KEY);")
    conn.execute("INSERT INTO libraries VALUES (1, 'user')")
    conn.execute("INSERT INTO itemTypes VALUES (1, 'journalArticle')")
    conn.execute("INSERT INTO items VALUES (1, 1, 1, 'KEPTITEM', '2026-01-01')")
    conn.executemany(
        "INSERT INTO collections VALUES (?,?,?)", [(1, "Mine", None), (2, "Old import", None), (3, "Inside old", 2)]
    )
    conn.executemany("INSERT INTO collectionItems VALUES (?,?)", [(1, 1), (2, 1), (3, 1)])
    conn.execute("INSERT INTO deletedCollections VALUES (2)")
    conn.commit()
    conn.close()
    library = zotero.read_library(tmp_path)
    assert library.collections == ["Mine"]
    assert by_key(library)["KEPTITEM"].collections == ["Mine"]


def test_oldest_pdf_is_the_main_file(library):
    item = by_key(library)["PARENT01"]
    assert item.attachment_key == "ATTACH01" and item.pdf_path.endswith("first.pdf")


def test_group_items_know_their_group(library):
    item = by_key(library)["GROUPIT1"]
    assert item.group_id == 555 and item.doc_key == "2:GROUPIT1"


def test_item_without_pdf_is_kept(library):
    item = by_key(library)["NOPDF001"]
    assert item.pdf_path is None and item.abstract == "An abstract."


def test_standalone_pdf_becomes_an_item(library):
    item = by_key(library)["STANDALN"]
    assert item.title == "Loose file" and item.attachment_key == "STANDALN"


def test_fingerprint_changes_with_the_file(library, tmp_path):
    item = by_key(library)["PARENT01"]
    before = item.fingerprint
    (tmp_path / "storage" / "ATTACH01" / "first.pdf").write_bytes(b"%PDF-1.7 a longer test file")
    assert item.fingerprint != before


def test_deep_links():
    assert zotero.select_uri(None, "ABC") == "zotero://select/library/items/ABC"
    assert zotero.select_uri(555, "ABC") == "zotero://select/groups/555/items/ABC"
    assert zotero.open_pdf_uri(None, "ATT", 7) == "zotero://open-pdf/library/items/ATT?page=7"
