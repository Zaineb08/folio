"""Claude as an answer model, against a fake Anthropic client: no key, network or cost."""

from types import SimpleNamespace as NS

import anthropic
import keyring
import numpy as np  # noqa: F401  Loads before the app, as in normal use.
import pytest
from fastapi.testclient import TestClient

from folio import answer, claude, config, llm, store
from folio import app as app_module

MODEL = "claude-opus-5-5"


def final_message(stop_reason="end_turn", model=MODEL, input_tokens=1000, output_tokens=500, iterations=None, category=None):
    usage = NS(input_tokens=input_tokens, output_tokens=output_tokens, iterations=iterations)
    return NS(stop_reason=stop_reason, model=model, usage=usage, stop_details=NS(category=category) if category else None)


class FakeClient:
    """Records the request, then streams a canned reply through the SDK's interface."""

    calls: list[dict] = []
    tokens = ["Hello ", "world."]
    final = final_message()
    error: Exception | None = None

    def __init__(self, api_key=None):
        self.api_key = api_key
        self.beta = NS(messages=NS(stream=self._stream))
        self.models = NS(list=self._list)

    def _list(self, **_):
        if FakeClient.error:
            raise FakeClient.error
        return []

    def _stream(self, **kwargs):
        FakeClient.calls.append({**kwargs, "api_key": self.api_key})
        if FakeClient.error:
            raise FakeClient.error
        client = self

        class Stream:
            text_stream = iter(client.tokens)

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def get_final_message(self):
                return client.final

        return Stream()


def api_error(cls, message="boom", status=None):
    """Make an SDK error without needing a real HTTP response."""
    error = cls.__new__(cls)
    error.message = message
    if status is not None:
        error.status_code = status
    return error


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    vault: dict = {}

    def delete(service, user):
        if (service, user) not in vault:
            raise keyring.errors.PasswordDeleteError("nothing saved")
        del vault[(service, user)]

    monkeypatch.setattr(keyring, "get_password", lambda s, u: vault.get((s, u)))
    monkeypatch.setattr(keyring, "set_password", lambda s, u, p: vault.__setitem__((s, u), p))
    monkeypatch.setattr(keyring, "delete_password", delete)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    FakeClient.calls, FakeClient.tokens, FakeClient.final, FakeClient.error = [], ["Hello ", "world."], final_message(), None
    config.save_settings(llm_model=config.DEFAULT_LLM, local_model=config.DEFAULT_LLM)
    yield vault


def stream(messages=None, **kwargs):
    return "".join(claude.chat_stream(MODEL, messages or [{"role": "user", "content": "hi"}], **kwargs))


# ------------------------------------------------------------------- the request


def test_request_follows_the_current_api(isolated):
    claude.save_key("sk-ant-test-key")
    messages = [{"role": "system", "content": "Be careful."}, {"role": "user", "content": "Hi"}]
    assert stream(messages) == "Hello world."
    call = FakeClient.calls[0]
    assert call["model"] == MODEL and call["system"] == "Be careful."
    assert call["messages"] == [{"role": "user", "content": "Hi"}]
    assert call["fallbacks"] == "default" and call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["output_config"] == {"effort": "high"}
    for removed in ("temperature", "top_p", "top_k", "thinking"):
        assert removed not in call, f"{removed} is rejected or unnecessary on this model"


def test_no_key_gives_a_clear_error():
    with pytest.raises(llm.LLMError, match="No Anthropic API key"):
        stream()
    assert not FakeClient.calls


def test_a_key_in_the_environment_is_used(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
    stream()
    assert FakeClient.calls[0]["api_key"] == "sk-ant-from-env"


# ----------------------------------------------------------------- usage and cost


def test_cost_is_calculated_from_usage():
    seen = {}
    claude.save_key("sk-ant-test-key")
    FakeClient.final = final_message(input_tokens=10_000, output_tokens=2_000)
    stream(on_finish=seen.update)
    assert seen["cost_usd"] == pytest.approx((10_000 * 4 + 2_000 * 20) / 1e6)
    assert seen["fell_back"] is False and seen["served_label"] == "Claude Opus 5.5" and seen["truncated"] is False


def test_a_fallback_is_reported_and_all_attempts_are_counted():
    seen = {}
    claude.save_key("sk-ant-test-key")
    iterations = [NS(type="message", input_tokens=1000, output_tokens=0), NS(type="fallback_message", input_tokens=1000, output_tokens=1000)]
    FakeClient.final = final_message(model="claude-opus-4-8", iterations=iterations)
    stream(on_finish=seen.update)
    assert seen["fell_back"] is True and seen["served_label"] == "Claude Opus 4.8" and seen["label"] == "Claude Opus 5.5"
    assert seen["cost_usd"] == pytest.approx((1000 * 4 + 1000 * 5 + 1000 * 25) / 1e6)


def test_a_cut_off_answer_is_reported():
    seen = {}
    claude.save_key("sk-ant-test-key")
    FakeClient.final = final_message(stop_reason="max_tokens")
    stream(on_finish=seen.update)
    assert seen["truncated"] is True


def test_an_unknown_model_has_no_invented_price():
    assert claude._summarise(final_message(model="claude-future-9"), "claude-future-9")["cost_usd"] is None


# ----------------------------------------------------------------- when things fail


def test_a_refusal_raises_declined_with_its_category():
    claude.save_key("sk-ant-test-key")
    FakeClient.final = final_message(stop_reason="refusal", category="cyber")
    with pytest.raises(llm.Declined, match="cyber") as caught:
        stream()
    assert caught.value.category == "cyber"


def test_a_refusal_without_details_is_still_handled():
    claude.save_key("sk-ant-test-key")
    FakeClient.final = final_message(stop_reason="refusal")
    with pytest.raises(llm.Declined):
        stream()


@pytest.mark.parametrize(
    "cls, message, status, expected",
    [
        (anthropic.AuthenticationError, "invalid x-api-key", 401, "didn't accept"),
        (anthropic.PermissionDeniedError, "no", 403, "permission"),
        (anthropic.NotFoundError, "no model", 404, "isn't available"),
        (anthropic.RateLimitError, "slow down", 429, "rate limit"),
        (anthropic.BadRequestError, "Your credit balance is too low", 400, "out of credit"),
        (anthropic.BadRequestError, "bad field", 400, "bad field"),
        (anthropic.InternalServerError, "oops", 500, "having problems"),
        (anthropic.APIConnectionError, "offline", None, "Couldn't reach"),
        (anthropic.APITimeoutError, "slow", None, "too long"),
    ],
)
def test_api_errors_become_readable(cls, message, status, expected):
    claude.save_key("sk-ant-test-key")
    FakeClient.error = api_error(cls, message, status)
    with pytest.raises(llm.LLMError, match=expected):
        stream()


# ---------------------------------------------------------------------- the key


def test_a_key_is_checked_before_being_saved(isolated):
    FakeClient.error = api_error(anthropic.AuthenticationError, "invalid", 401)
    with pytest.raises(llm.LLMError, match="didn't accept"):
        claude.save_key("sk-ant-bad-key")
    assert not claude.has_key() and not isolated


def test_saving_trims_and_deleting_removes(isolated):
    claude.save_key("  sk-ant-good-key \n")
    assert claude.get_key() == "sk-ant-good-key"
    claude.delete_key()
    assert not claude.has_key()
    claude.delete_key()  # Deleting twice is harmless.


# ------------------------------------------------------------------- the web API


@pytest.fixture
def client():
    return TestClient(app_module.app, base_url="http://127.0.0.1:8799")


SECRET = "sk-ant-very-secret-key"


def test_the_key_is_never_sent_back(client):
    assert client.post("/api/claude/key", json={"key": SECRET}).json() == {"saved": True}
    status = client.get("/api/status")
    assert status.json()["claude"]["has_key"] is True and SECRET not in status.text
    assert SECRET not in client.post("/api/settings", json={"llm_model": MODEL}).text
    assert SECRET not in config.SETTINGS_PATH.read_text(encoding="utf-8")


def test_a_rejected_key_gives_a_readable_400(client):
    FakeClient.error = api_error(anthropic.AuthenticationError, "invalid", 401)
    response = client.post("/api/claude/key", json={"key": SECRET})
    assert response.status_code == 400 and "didn't accept" in response.json()["detail"]


def test_other_sites_cannot_set_or_remove_the_key(client):
    assert client.post("/api/claude/key", json={"key": SECRET}, headers={"origin": "https://evil.example"}).status_code == 403
    assert client.delete("/api/claude/key", headers={"origin": "https://evil.example"}).status_code == 403


def test_claude_cannot_be_selected_without_a_key(client):
    assert client.post("/api/settings", json={"llm_model": MODEL}).status_code == 400
    assert config.load_settings()["llm_model"] == config.DEFAULT_LLM


def test_removing_the_key_returns_to_the_local_model(client):
    client.post("/api/claude/key", json={"key": SECRET})
    client.post("/api/settings", json={"llm_model": MODEL})
    assert client.delete("/api/claude/key").json() == {"deleted": True}
    assert config.load_settings()["llm_model"] == config.DEFAULT_LLM


def test_status_describes_the_selected_model(client, monkeypatch):
    monkeypatch.setattr(llm, "is_up", lambda: False)
    client.post("/api/claude/key", json={"key": SECRET})
    client.post("/api/settings", json={"llm_model": MODEL})
    state = client.get("/api/status").json()["llm"]
    assert state["selected"] == MODEL and state["up"] is True and state["installed"] is True


def test_asking_with_an_unknown_model_is_refused(client, monkeypatch):
    monkeypatch.setattr(llm, "list_models", lambda: [config.DEFAULT_LLM])
    assert client.post("/api/ask", json={"question": "x", "model": "gpt-99"}).status_code == 400


# ---------------------------------------------------------------- answering flow

SOURCE = {"chunk_id": 1, "doc_key": "1:A", "text": "Teachers value autonomy above pay.", "page": 1, "page_end": 1, "section": "", "score": 0.9, "title": "T", "cite": "A, 2020", "apa": "A", "year": 2020, "venue": "", "has_pdf": False, "pages": 1, "overview": False}


def run(monkeypatch, tokens=("Teachers value autonomy above pay [1].",), final=None, local=(config.DEFAULT_LLM,), history=None, **kwargs):
    """Answer one question with a fake Claude. Returns the events and which model did each side job."""
    claude.save_key("sk-ant-test-key")
    FakeClient.tokens = list(tokens)
    FakeClient.final = final or final_message()
    helpers = []

    def local_once(model, messages, num_predict=128):
        helpers.append(model)
        return "YES"

    monkeypatch.setattr(answer, "retrieve", lambda *a, **k: [dict(SOURCE)])
    monkeypatch.setattr(llm, "list_models", lambda: list(local))
    monkeypatch.setattr(llm, "chat_once", local_once)
    job = answer.start(answer.AskRequest(question="What do teachers value?", model=MODEL, conversation_id=history, **kwargs))
    events = []
    while (event := job.events.get(timeout=20)) is not None:
        events.append(event)
    return events, helpers


def of_type(events, kind):
    return [event for event in events if event["type"] == kind]


def saved_answers(events):
    conversation = next(e for e in events if e["type"] == "conversation")["id"]
    return [m for m in store.get_messages(conversation) if m["role"] == "assistant"]


def test_claude_writes_and_the_local_model_checks(monkeypatch):
    events, helpers = run(monkeypatch)
    assert helpers == [config.DEFAULT_LLM], "checking should be free and independent"
    assert len(FakeClient.calls) == 1, "only the answer itself should cost money"
    meta = of_type(events, "meta")[0]
    assert meta["provider"] == "claude" and meta["checked_by"] == config.DEFAULT_LLM and meta["cost_usd"] > 0
    assert of_type(events, "check")[0]["flagged"] == 0
    assert saved_answers(events)[0]["meta"]["served_by"] == MODEL


def test_without_a_local_model_claude_checks_itself(monkeypatch):
    events, helpers = run(monkeypatch, local=())
    assert helpers == [MODEL]
    assert of_type(events, "meta")[0]["checked_by"] == MODEL


def test_a_decline_discards_partial_text_and_saves_no_answer(monkeypatch):
    events, _ = run(monkeypatch, tokens=("Partial ", "text"), final=final_message(stop_reason="refusal", category="cyber"))
    kinds = [event["type"] for event in events]
    assert kinds.index("token") < kinds.index("reset") < kinds.index("error")
    assert "safety filter" in of_type(events, "error")[0]["message"]
    assert saved_answers(events) == [] and not of_type(events, "check")


def test_history_sent_to_claude_starts_with_the_user(monkeypatch):
    conversation = store.create_conversation("x")
    for role, text in [("user", "q1"), ("assistant", "a1"), ("user", "q2"), ("assistant", "a2"), ("user", "q3 never answered")]:
        store.add_message(conversation, role, text)
    run(monkeypatch, history=conversation)
    assert FakeClient.calls[0]["messages"][0]["role"] == "user"
    store.delete_conversation(conversation)


def test_old_databases_gain_the_new_column():
    with store.connect() as conn:
        assert "meta" in {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
