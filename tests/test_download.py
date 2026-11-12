"""Downloading an answer model from Folio's menu, so nobody needs the command line."""

import json
import time

import pytest
from fastapi.testclient import TestClient

from folio import app as app_module
from folio import config, llm


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(llm, "ensure_running", lambda: True)
    app_module._download.update(model=None, done=0, total=0, error=None, active=False)
    return TestClient(app_module.app, base_url="http://127.0.0.1:8799")


def wait_for_download():
    for _ in range(200):
        if not app_module._download["active"]:
            return dict(app_module._download)
        time.sleep(0.01)
    raise AssertionError("the download never finished")


def test_the_menu_offers_both_models_and_marks_the_one_for_this_pc(client, monkeypatch):
    monkeypatch.setattr(llm, "is_up", lambda: True)
    monkeypatch.setattr(llm, "list_models", lambda: [config.LARGE_LLM])
    suggested = client.get("/api/status").json()["suggested"]
    assert {m["id"] for m in suggested} == {config.LARGE_LLM, config.SMALL_LLM}
    assert [m["id"] for m in suggested if m["suits"]] == [config.recommended_model()]


def test_only_folios_own_models_can_be_downloaded(client):
    assert client.post("/api/models/download", json={"model": "some/huge-model:405b"}).status_code == 400


def test_a_finished_download_is_selected(client, monkeypatch):
    def fake_pull(model, on_progress):
        on_progress(1, 2)
        on_progress(2, 2)

    monkeypatch.setattr(llm, "pull", fake_pull)
    response = client.post("/api/models/download", json={"model": config.SMALL_LLM})
    assert response.status_code == 200 and response.json()["active"]
    state = wait_for_download()
    assert state["error"] is None and state["done"] == state["total"] == 2
    assert config.load_settings()["llm_model"] == config.SMALL_LLM


def test_a_failed_download_says_why_and_keeps_the_old_model(client, monkeypatch):
    config.save_settings(llm_model=config.LARGE_LLM, local_model=config.LARGE_LLM)

    def failing_pull(model, on_progress):
        raise llm.LLMError("Couldn't download it. Check the internet connection and that Ollama is running.")

    monkeypatch.setattr(llm, "pull", failing_pull)
    client.post("/api/models/download", json={"model": config.SMALL_LLM})
    state = wait_for_download()
    assert "internet connection" in state["error"]
    assert config.load_settings()["llm_model"] == config.LARGE_LLM


def test_without_ollama_the_download_explains_how_to_install_it(client, monkeypatch):
    monkeypatch.setattr(llm, "ensure_running", lambda: False)
    response = client.post("/api/models/download", json={"model": config.SMALL_LLM})
    assert response.status_code == 400 and "winget install" in response.json()["detail"]


class FakeStream:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_lines(self):
        yield from self.lines


def test_progress_adds_up_every_part_of_the_model(monkeypatch):
    """Ollama reports each part of a model separately; the bar should cover all of them."""
    lines = [json.dumps(m) for m in (
        {"status": "pulling manifest"},
        {"status": "pulling a", "digest": "a", "total": 300, "completed": 100},
        {"status": "pulling b", "digest": "b", "total": 100, "completed": 100},
        {"status": "pulling a", "digest": "a", "total": 300, "completed": 300},
        {"status": "success"},
    )]
    monkeypatch.setattr(llm.httpx, "stream", lambda *args, **kwargs: FakeStream(lines))
    monkeypatch.setattr(llm, "list_models", lambda: [config.SMALL_LLM])
    seen = []
    llm.pull(config.SMALL_LLM, lambda done, total: seen.append((done, total)))
    assert seen == [(100, 300), (200, 400), (400, 400)]


def test_an_error_from_ollama_is_reported(monkeypatch):
    lines = [json.dumps({"error": "pull model manifest: file does not exist"})]
    monkeypatch.setattr(llm.httpx, "stream", lambda *args, **kwargs: FakeStream(lines))
    with pytest.raises(llm.LLMError, match="does not exist"):
        llm.pull(config.SMALL_LLM, lambda done, total: None)
