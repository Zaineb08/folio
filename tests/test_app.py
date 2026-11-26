"""The web server's safety checks and basic routes."""

import pymupdf
import pytest
from fastapi.testclient import TestClient

from folio import app as app_module
from folio import store, zotero


@pytest.fixture
def client():
    # No lifespan: tests shouldn't start the background sync or load models.
    return TestClient(app_module.app, base_url="http://127.0.0.1:8799")


def test_index_is_served_with_a_strict_content_policy(client):
    response = client.get("/")
    assert response.status_code == 200
    policy = response.headers["content-security-policy"]
    assert "script-src 'self'" in policy and "unsafe-inline" not in policy


def test_browsers_check_for_new_versions_of_the_interface(client):
    """Otherwise a browser can keep showing the old look for a day after Folio is updated."""
    for path in ("/static/styles.css", "/static/app.js", "/static/icon.svg"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-cache"
    # A copy the browser already has is confirmed without being sent again.
    first = client.get("/static/styles.css")
    again = client.get("/static/styles.css", headers={"if-none-match": first.headers["etag"]})
    assert again.status_code == 304


def test_requests_for_other_hosts_are_refused(client):
    assert client.get("/api/library", headers={"host": "evil.example"}).status_code == 403


def test_other_sites_cannot_call_the_api(client):
    response = client.post("/api/sync", headers={"origin": "https://evil.example"})
    assert response.status_code == 403
    response = client.post("/api/ask/stop", headers={"origin": "http://127.0.0.1:8799"})
    assert response.status_code == 200


def test_unknown_mode_is_rejected(client):
    assert client.post("/api/ask", json={"question": "x", "mode": "nonsense"}).status_code == 400


def test_empty_question_is_rejected(client):
    assert client.post("/api/ask", json={"question": ""}).status_code == 422


def test_missing_doc_and_chunk_are_404(client):
    assert client.get("/api/page/1:NOPE/1").status_code == 404
    assert client.get("/api/pdf/1:NOPE").status_code == 404
    assert client.get("/api/chunk/999999").status_code == 404
    assert client.post("/api/zotero/open", json={"doc_key": "1:NOPE"}).status_code == 404


@pytest.fixture
def doc_with_pdf(tmp_path):
    path = tmp_path / "a.pdf"
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "Hello highlight", fontsize=12)
    document.save(path)
    document.close()
    item = zotero.Item(library_id=1, group_id=None, key="PAGE0001", item_type="journalArticle", title="T", pdf_path=str(path))
    store.upsert_docs_metadata([item])
    yield item.doc_key
    store.delete_doc(item.doc_key)


def test_page_image_renders_and_checks_bounds(client, doc_with_pdf):
    response = client.get(f"/api/page/{doc_with_pdf}/1")
    assert response.status_code == 200 and response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert client.get(f"/api/page/{doc_with_pdf}/99").status_code == 404
    assert client.get(f"/api/page/{doc_with_pdf}/0").status_code == 404


def test_pdf_is_served_only_through_the_library(client, doc_with_pdf):
    assert client.get(f"/api/pdf/{doc_with_pdf}").status_code == 200
    assert client.get("/api/pdf/..%2F..%2Fwindows").status_code == 404


def test_settings_only_accept_installed_models(client, monkeypatch):
    monkeypatch.setattr(app_module.llm, "list_models", lambda: ["good:1"])
    assert client.post("/api/settings", json={"llm_model": "bad:1"}).status_code == 400
    assert client.post("/api/settings", json={"llm_model": "good:1"}).json()["llm_model"] == "good:1"
