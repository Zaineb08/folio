"""The weak-evidence warning, Explain this paper and Find a source, without real models."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from folio import answer, claims, config, llm, search, store, zotero
from folio import app as app_module

HIT = {"chunk_id": 1, "doc_key": "1:A", "text": "Teachers value autonomy above pay. They reported this in 12 interviews overall.", "page": 3, "page_end": 3, "section": "", "score": 0.9, "title": "T", "cite": "A et al., 2020", "apa": "A. (2020). T.", "year": 2020, "venue": "", "has_pdf": False, "pages": 9, "overview": False, "named": False}


def hit(score=0.9, **changes):
    return {**HIT, "score": score, **changes}


# -------------------------------------------------------------- weak evidence


@pytest.mark.parametrize(
    "hits, expected",
    [
        ([], None),
        ([hit(0.95), hit(0.4)], "strong"),
        ([hit(0.7)], "strong"),
        ([hit(0.69), hit(0.2)], "weak"),
        ([hit(0.05)], "weak"),
        ([hit(0.05, named=True), hit(0.01)], "strong"),  # Passages rarely contain their own authors' names.
        ([hit(0.01, overview=True)], "sample"),
    ],
)
def test_evidence_levels(hits, expected):
    assert search.assess_evidence(hits) == expected


def run(monkeypatch, hits, mode="answer", scope=None, question="What do teachers value?", reply="Teachers value autonomy above pay [1]."):
    """Answer with a fake local model. Returns the events and what the model was sent."""
    sent = []

    def fake_stream(model, messages, on_finish=None, **kwargs):
        sent.append(messages)
        yield reply

    monkeypatch.setattr(answer, "retrieve", lambda *a, **k: [dict(h) for h in hits])
    monkeypatch.setattr(llm, "chat_stream", fake_stream)
    monkeypatch.setattr(llm, "chat_once", lambda *a, **k: "YES")
    monkeypatch.setattr(llm, "loaded_models", lambda: [])
    monkeypatch.setattr(llm, "list_models", lambda: [config.DEFAULT_LLM])
    job = answer.start(answer.AskRequest(question=question, mode=mode, model=config.DEFAULT_LLM, scope=scope or search.Scope()))
    events = []
    while (event := job.events.get(timeout=20)) is not None:
        events.append(event)
    conversation = next(e["id"] for e in events if e["type"] == "conversation")
    saved = store.get_messages(conversation)
    store.delete_conversation(conversation)
    return events, sent, saved


def of_type(events, kind):
    return next(e for e in events if e["type"] == kind)


def test_weak_evidence_warns_the_reader_and_restrains_the_model(monkeypatch):
    events, sent, saved = run(monkeypatch, [hit(0.2)])
    assert of_type(events, "sources")["evidence"] == "weak"
    assert "match the question only weakly" in sent[0][-1]["content"]
    assert saved[-1]["meta"]["evidence"] == "weak", "the warning has to survive reopening the conversation"


def test_strong_evidence_adds_no_warning(monkeypatch):
    events, sent, _ = run(monkeypatch, [hit(0.95)])
    assert of_type(events, "sources")["evidence"] == "strong"
    assert "weakly" not in sent[0][-1]["content"]


# ----------------------------------------------------------- explain this paper


def test_explaining_uses_passages_from_across_the_paper(monkeypatch):
    monkeypatch.setattr(answer, "paper_passages", lambda key, k: ([hit(1.0), hit(1.0, chunk_id=2)], {"used": 2, "total": 40}))
    monkeypatch.setattr(answer, "retrieve", lambda *a, **k: pytest.fail("explaining shouldn't search by the question"))
    events, sent, saved = run_explain(monkeypatch)
    prompt = sent[0][-1]["content"]
    assert "new research student" in prompt and "Terms to know" in prompt and "Questions to ask as you read" in prompt
    assert of_type(events, "sources")["evidence"] is None, "these passages are chosen, not matched, so no score applies"
    assert saved[-1]["meta"]["coverage"] == {"used": 2, "total": 40}


def run_explain(monkeypatch):
    sent = []

    def fake_stream(model, messages, on_finish=None, **kwargs):
        sent.append(messages)
        yield "## In one sentence\nTeachers value autonomy above pay [1]."

    monkeypatch.setattr(llm, "chat_stream", fake_stream)
    monkeypatch.setattr(llm, "chat_once", lambda *a, **k: "YES")
    monkeypatch.setattr(llm, "loaded_models", lambda: [])
    monkeypatch.setattr(llm, "list_models", lambda: [config.DEFAULT_LLM])
    job = answer.start(answer.AskRequest(question="Explain this paper: T", mode="explain", model=config.DEFAULT_LLM, scope=search.Scope(doc_keys=["1:A"])))
    events = []
    while (event := job.events.get(timeout=20)) is not None:
        events.append(event)
    conversation = next(e["id"] for e in events if e["type"] == "conversation")
    saved = store.get_messages(conversation)
    store.delete_conversation(conversation)
    return events, sent, saved


class FakeEngine:
    def encode_query(self, text):
        vector = np.zeros(config.EMBED_DIM, dtype=np.float32)
        vector[hash(text) % 5] = 1.0
        return vector


def test_paper_passages_cover_the_paper_in_reading_order(monkeypatch):
    item = zotero.Item(library_id=1, group_id=None, key="WHOLE001", item_type="journalArticle", title="Whole paper")
    store.upsert_docs_metadata([item])
    sections = ["", "Introduction", "Methods", "Results", "Discussion", "Conclusion"] * 5
    chunks = [{"page": i + 1, "page_end": i + 1, "section": name, "text": f"Passage number {i} here.", "rects": []} for i, name in enumerate(sections)]
    vectors = np.zeros((len(chunks), config.EMBED_DIM), dtype=np.float32)
    vectors[np.arange(len(chunks)), np.arange(len(chunks)) % 5] = 1.0
    store.save_indexed(item.doc_key, "fp", "ok", None, len(chunks), chunks, vectors)
    monkeypatch.setattr(search, "engine", FakeEngine())
    hits, coverage = search.paper_passages(item.doc_key, k=12)
    store.delete_doc(item.doc_key)
    pages = [h["page"] for h in hits]
    assert pages == sorted(pages), "a paper should be read in its own order"
    assert pages[0] == 1 and len(hits) <= 12
    assert coverage == {"used": len(hits), "total": len(chunks)}
    assert {h["section"] for h in hits} & {"Conclusion"}, "the conclusion should always be included"


def test_paper_passages_for_an_unknown_paper_is_empty():
    assert search.paper_passages("1:NOPE") == ([], {"used": 0, "total": 0})


@pytest.fixture
def client():
    return TestClient(app_module.app, base_url="http://127.0.0.1:8799")


def test_explaining_needs_exactly_one_paper(client):
    assert client.post("/api/ask", json={"question": "Explain", "mode": "explain"}).status_code == 400
    two = {"question": "Explain", "mode": "explain", "scope": {"doc_keys": ["1:A", "1:B"]}}
    assert client.post("/api/ask", json=two).status_code == 400


def test_questions_for_the_reader_and_openly_general_definitions_are_not_uncited_claims():
    source = [{"n": 1, "text": "Teachers value autonomy above pay."}]
    text = (
        "Teachers value autonomy above pay [1].\n"
        "- How representative was the sample of teachers that these authors chose to interview for this study?\n"
        "- AUROC: a measure of how well a classifier separates two classes overall (general definition)\n"
        "- Most researchers now agree that this approach is clearly superior to all the others."
    )
    flags = answer.check_citations(text, source, lambda c, p: True)["flags"]
    assert [flag["text"][:4] for flag in flags] == ["Most"], "only the real uncited claim should be flagged"


def test_a_cited_question_for_the_reader_isnt_judged_as_a_claim():
    source = [{"n": 1, "text": "Teachers value autonomy above pay."}]
    text = "Teachers value autonomy above pay [1].\n- How subjective was the way they classified these answers [1]?"
    checks = answer.check_citations(text, source, lambda claim, passage: "subjective" not in claim)
    assert checks["flags"] == []


# ---------------------------------------------------------------- find a source


class SentenceEngine:
    def rerank(self, query, passages):
        words = set(query.lower().split())
        return np.array([len(words & set(p.lower().split())) for p in passages], dtype=np.float32)

    def release_gpu(self):
        pass


@pytest.fixture
def judged(monkeypatch):
    """Fake search and model: `replies` maps a passage's text to the model's verdict."""
    replies = {}
    monkeypatch.setattr(claims, "engine", SentenceEngine())
    monkeypatch.setattr(llm, "list_models", lambda: [config.DEFAULT_LLM])
    monkeypatch.setattr(llm, "chat_once", lambda model, messages, num_predict=128: next(v for k, v in replies.items() if k in messages[0]["content"]))
    config.save_settings(llm_model=config.DEFAULT_LLM, local_model=config.DEFAULT_LLM)

    def setup(hits, mapping):
        replies.clear()
        replies.update(mapping)
        monkeypatch.setattr(claims, "retrieve", lambda *a, **k: hits)

    return setup


def reply(verdict, quote="Teachers value autonomy above pay."):
    return f"QUOTE: {quote}\nVERDICT: {verdict}"


def test_sources_are_grouped_supporting_first_and_unrelated_ones_dropped(judged):
    hits = [hit(0.9, text="Teachers value related topic passages.", chunk_id=1), hit(0.8, text="Teachers value autonomy above pay.", chunk_id=2), hit(0.7, text="Teachers value pay above autonomy.", chunk_id=3), hit(0.6, text="Something else entirely is said here.", chunk_id=4)]
    judged(
        hits,
        {
            "Teachers value related": reply("RELATED", "Teachers value related topic passages."),
            "Teachers value pay": reply("**contradicts**", "Teachers value pay above autonomy."),
            "Something else": reply("UNRELATED", "Something else entirely is said here."),
            "Teachers value autonomy": reply("SUPPORTS."),  # Last, because every prompt contains the student's sentence.
        },
    )
    results = claims.find_sources("Teachers value autonomy above pay.")["results"]
    assert [r["verdict"] for r in results] == ["supports", "disagrees", "related"]
    assert results[0]["citation"] == "(A et al., 2020, p. 3)"


def test_a_passage_cant_support_a_figure_it_doesnt_contain(judged):
    judged([hit(0.9)], {"Teachers value": reply("SUPPORTS")})
    result = claims.find_sources("Teachers value autonomy above pay in 95% of the cases.")["results"][0]
    assert result["verdict"] == "related" and "95" in result["note"]
    judged([hit(0.9)], {"Teachers value": reply("SUPPORTS")})
    fine = claims.find_sources("Teachers value autonomy above pay, across 12 interviews (A et al., 2020).")["results"][0]
    assert fine["verdict"] == "supports" and fine["note"] is None


def test_a_quote_that_isnt_in_the_passage_cant_carry_a_verdict(judged):
    """Models sometimes just echo the student's sentence back."""
    sentence = "Teachers always value autonomy above pay."
    judged([hit(0.9)], {"Teachers value": reply("SUPPORTS", sentence)})
    result = claims.find_sources(sentence)["results"][0]
    assert result["verdict"] == "related"
    assert result["quote"] == "Teachers value autonomy above pay.", "show what the passage really says"


def test_a_question_the_paper_asks_isnt_a_finding(judged):
    judged([hit(0.9, text="Do teachers value autonomy above pay? We asked them directly in person.")], {"Do teachers": reply("SUPPORTS", "Do teachers value autonomy above pay?")})
    assert claims.find_sources("Teachers value autonomy above pay.")["results"][0]["verdict"] == "related"


def test_support_resting_on_a_quote_with_nothing_in_common_is_dropped(judged):
    judged([hit(0.9)], {"Teachers value": reply("SUPPORTS", "They reported this in 12 interviews overall.")})
    assert claims.find_sources("Schools should close during winter.")["results"][0]["verdict"] == "related"


def test_disagreement_needs_no_overlap_with_the_sentence(judged):
    judged([hit(0.9)], {"Teachers value": reply("CONTRADICTS", "They reported this in 12 interviews overall.")})
    assert claims.find_sources("Schools should close during winter.")["results"][0]["verdict"] == "disagrees"


@pytest.mark.parametrize(
    "quote",
    [
        "Teachers value autonomy [14].",
        "Smith et al. show that teachers value autonomy.",
        "Teachers value autonomy (Smith & Jones, 2019).",
        "They found that teachers value autonomy.",
    ],
)
def test_it_points_out_when_a_passage_reports_another_papers_work(judged, quote):
    judged([hit(0.9, text=f"Background comes first in this passage. {quote}")], {"Background": reply("SUPPORTS", quote)})
    note = claims.find_sources("Teachers value autonomy.")["results"][0]["note"]
    assert note and "original paper" in note


def test_a_papers_own_finding_gets_no_such_note(judged):
    judged([hit(0.9)], {"Teachers value": reply("SUPPORTS")})
    assert claims.find_sources("Teachers value autonomy above pay.")["results"][0]["note"] is None


def test_the_same_line_from_overlapping_passages_shows_once(judged):
    judged([hit(0.9, chunk_id=1), hit(0.8, chunk_id=2)], {"Teachers value": reply("SUPPORTS")})
    assert len(claims.find_sources("Teachers value autonomy above pay.")["results"]) == 1


def test_barely_matching_passages_arent_sent_to_the_model(judged, monkeypatch):
    judged([hit(0.05)], {})
    monkeypatch.setattr(llm, "chat_once", lambda *a, **k: pytest.fail("this passage shouldn't reach the model"))
    assert claims.find_sources("Teachers value autonomy above pay.")["results"] == []


@pytest.mark.parametrize("answer_text", ["Maybe", "QUOTE: Teachers value autonomy above pay.", ""])
def test_an_unclear_reply_is_treated_cautiously(judged, answer_text):
    judged([hit(0.9)], {"Teachers value": answer_text})
    assert claims.find_sources("Teachers value autonomy above pay.")["results"][0]["verdict"] == "related"


def test_a_safety_refusal_counts_as_related(judged, monkeypatch):
    def refuse(*a, **k):
        raise llm.Declined("cyber")

    judged([hit(0.9)], {})
    monkeypatch.setattr(llm, "chat_once", refuse)
    assert claims.find_sources("Teachers value autonomy above pay.")["results"][0]["verdict"] == "related"


def test_nothing_found_returns_no_results(judged):
    judged([], {})
    assert claims.find_sources("The sky over Paris is often grey.")["results"] == []


def test_the_api_checks_length_and_reports_model_problems(client, monkeypatch):
    assert client.post("/api/claim", json={"text": "short"}).status_code == 422

    def down(*a, **k):
        raise llm.LLMError("Ollama isn't running.")

    monkeypatch.setattr(claims, "find_sources", down)
    response = client.post("/api/claim", json={"text": "A sentence long enough to check."})
    assert response.status_code == 503 and "Ollama" in response.json()["detail"]
