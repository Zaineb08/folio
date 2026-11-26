"""Research mode: planning searches, searching again for gaps, and explaining in French, with fake models."""

import threading

import pytest

from folio import answer, config, llm, research, search, store

BASE = {"text": "Models invent package names in 19.7% of samples.", "page": 3, "page_end": 3, "section": "", "score": 0.9, "title": "T", "cite": "A et al., 2025", "apa": "A. (2025). T.", "year": 2025, "venue": "", "has_pdf": True, "pages": 9, "overview": False, "named": False}


def hit(chunk_id, doc_key="1:A", **changes):
    return {**BASE, "chunk_id": chunk_id, "doc_key": doc_key, **changes}


# ------------------------------------------------------------------- language


@pytest.mark.parametrize(
    "said, expected",
    [
        (["Quels outils détectent les paquets inventés ?"], research.FRENCH),
        (["What detects invented packages?"], research.FRENCH),  # French unless asked.
        (["Which tools exist?", "Answer in English please"], research.ENGLISH),
        (["Réponds en anglais", "et les limites ?"], research.ENGLISH),  # Stays switched for later questions.
        (["in english", "ok, repasse en français maintenant"], research.FRENCH),
        (["English please"], research.ENGLISH),
        (["Do models hallucinate more on prompts written in English?"], research.FRENCH),  # About English, not a request.
        (["Can you explain it in English?"], research.ENGLISH),
    ],
)
def test_the_reply_language(said, expected):
    history = [{"role": "user", "content": text} for text in said[:-1]]
    assert research.reply_language(history, said[-1]) == expected


def test_french_answers_keep_quotations_in_english():
    prompt = research.system_prompt(research.FRENCH)
    assert "Write in French" in prompt and "(connaissance générale)" in prompt
    assert "keep the quotation in English" in prompt
    assert "Write in English" in research.system_prompt(research.ENGLISH)
    assert "(general knowledge)" in research.system_prompt(research.ENGLISH)


# -------------------------------------------------------------------- planning


def test_queries_are_cleaned_from_the_models_reply():
    reply = 'Here are the queries:\n1. package hallucination rate\n- "slopsquatting attacks"\n* Package hallucination rate\n\nDONE'
    assert research.parse_queries(reply, 4) == ["package hallucination rate", "slopsquatting attacks"]
    assert research.parse_queries("a\nb\nc query\nd query\ne query", 2) == ["c query", "d query"]
    assert research.parse_queries("slopsquatting attacks\nnew one", 2, ["Slopsquatting attacks"]) == ["new one"]
    assert research.parse_queries("DONE", 2) == []
    assert research.parse_queries("The passages already cover every part of the question well, so nothing is missing.", 2) == []


def test_passages_are_taken_from_each_search_in_turn():
    results = {
        "first": [hit(1), hit(2), hit(3)],
        "second": [hit(1), hit(10, "1:B"), hit(11, "1:B")],
    }
    merged = research.merge(results)
    assert [h["chunk_id"] for h in merged] == [1, 2, 10, 3, 11], "a passage two searches found is used once"


def test_no_paper_takes_over_the_answer(monkeypatch):
    monkeypatch.setattr(research, "PER_PAPER", 2)
    merged = research.merge({"q": [hit(n) for n in range(1, 6)] + [hit(9, "1:B")]})
    assert [h["chunk_id"] for h in merged] == [1, 2, 9]


def fake_planner(plan, review):
    """chat_once that plans, then reviews, recording each prompt."""
    prompts = []

    def chat_once(model, messages, num_predict=128):
        prompt = messages[0]["content"]
        prompts.append(prompt)
        return plan if "plan searches" in prompt else review

    return chat_once, prompts


def test_the_question_in_english_is_searched_first():
    reply = "English: What risks do hallucinated packages pose?\npackage hallucination rate\nslopsquatting"
    assert research.plan_reply(reply, "Quels risques ?", 3) == [
        "What risks do hallucinated packages pose?", "package hallucination rate", "slopsquatting"
    ]
    # Without a translation, the question as asked still finds passages by meaning.
    assert research.plan_reply("package hallucination rate", "Quels risques ?", 3) == ["Quels risques ?", "package hallucination rate"]


def test_the_planner_sees_the_papers_titles(monkeypatch):
    monkeypatch.setattr(store, "list_docs", lambda: [
        {"key": "1:A", "title": "We Have a Package for You!", "collections": [], "tags": [], "year": 2025},
        {"key": "1:B", "title": "Lost at C", "collections": [], "tags": [], "year": 2023},
    ])
    monkeypatch.setattr(search.Scope, "resolve", lambda self, docs: {"1:A"})
    chat_once, prompts = fake_planner("English: q", "DONE")
    monkeypatch.setattr(llm, "chat_once", chat_once)
    monkeypatch.setattr(research, "retrieve", lambda *a, **k: [hit(1)])
    research.gather("Quels risques posent les paquets hallucinés ?", [], search.Scope(), "m", lambda e: None, threading.Event())
    assert "We Have a Package for You!" in prompts[0]
    assert "Lost at C" not in prompts[0], "only papers inside the chosen scope"


def test_it_searches_again_for_what_is_missing(monkeypatch):
    chat_once, prompts = fake_planner(
        "English: What risks do invented packages pose?\npackage hallucination rate\nslopsquatting",
        "detection tools for invented packages",
    )
    searched = []

    def fake_retrieve(query, scope, k, per_doc, allow_overview):
        searched.append(query)
        assert allow_overview is False, "each search should be an ordinary close search"
        return [hit(len(searched), f"1:{query[:3]}")]

    monkeypatch.setattr(llm, "chat_once", chat_once)
    monkeypatch.setattr(research, "retrieve", fake_retrieve)
    events = []
    hits, steps = research.gather("Quels risques posent les paquets inventés ?", [], search.Scope(), "m", events.append, threading.Event())

    assert searched == [
        "What risks do invented packages pose?", "package hallucination rate", "slopsquatting", "detection tools for invented packages"
    ]
    assert [s["round"] for s in steps] == [1, 1, 1, 2]
    assert len(hits) == 4
    assert "Models invent package names" in prompts[1], "the review has to see what was found"
    assert [e["query"] for e in events if e["type"] == "step"] == searched


def test_it_stops_when_the_first_searches_are_enough(monkeypatch):
    chat_once, _ = fake_planner("English: package hallucination rate", "DONE")
    monkeypatch.setattr(llm, "chat_once", chat_once)
    monkeypatch.setattr(research, "retrieve", lambda *a, **k: [hit(1)])
    _, steps = research.gather("q", [], search.Scope(), "m", lambda e: None, threading.Event())
    assert len(steps) == 1


def test_without_a_plan_it_searches_the_question_itself(monkeypatch):
    def broken(*args, **kwargs):
        raise llm.LLMError("Ollama isn't running")

    searched = []
    monkeypatch.setattr(llm, "chat_once", broken)
    monkeypatch.setattr(research, "retrieve", lambda query, *a, **k: searched.append(query) or [hit(1)])
    hits, _ = research.gather("Quels outils existent ?", [], search.Scope(), "m", lambda e: None, threading.Event())
    assert searched == ["Quels outils existent ?"] and hits


def test_nothing_found_means_no_second_round(monkeypatch):
    chat_once, prompts = fake_planner("English: a query\nanother query", "pytest.fail")
    monkeypatch.setattr(llm, "chat_once", chat_once)
    monkeypatch.setattr(research, "retrieve", lambda *a, **k: [])
    hits, steps = research.gather("q", [], search.Scope(), "m", lambda e: None, threading.Event())
    assert hits == [] and len(steps) == 2 and len(prompts) == 1


def test_stopping_ends_the_searches(monkeypatch):
    chat_once, _ = fake_planner("English: one query\ntwo query\nthree query", "more query")
    stop = threading.Event()
    searched = []

    def fake_retrieve(query, *args, **kwargs):
        searched.append(query)
        stop.set()
        return [hit(1)]

    monkeypatch.setattr(llm, "chat_once", chat_once)
    monkeypatch.setattr(research, "retrieve", fake_retrieve)
    research.gather("q", [], search.Scope(), "m", lambda e: None, stop)
    assert searched == ["one query"]


# ---------------------------------------------------------------- whole answer


def run_research(monkeypatch, reply, question="Quels risques posent les paquets inventés ?"):
    sent = []

    def fake_stream(model, messages, on_finish=None, **kwargs):
        sent.append(messages)
        yield reply

    def chat_once(model, messages, num_predict=128):
        prompt = messages[0]["content"]
        if "plan searches" in prompt:
            return "English: package hallucination rate"
        if "not cover yet" in prompt:
            return "DONE"
        return "YES"

    monkeypatch.setattr(research, "retrieve", lambda *a, **k: [hit(1), hit(2, "1:B", text="Attackers register invented names, which is called slopsquatting.")])
    monkeypatch.setattr(answer, "retrieve", lambda *a, **k: pytest.fail("research mode plans its own searches"))
    monkeypatch.setattr(llm, "chat_stream", fake_stream)
    monkeypatch.setattr(llm, "chat_once", chat_once)
    monkeypatch.setattr(llm, "loaded_models", lambda: [])
    monkeypatch.setattr(llm, "list_models", lambda: [config.DEFAULT_LLM])
    job = answer.start(answer.AskRequest(question=question, mode="research", model=config.DEFAULT_LLM))
    events = []
    while (event := job.events.get(timeout=20)) is not None:
        events.append(event)
    conversation = next(e["id"] for e in events if e["type"] == "conversation")
    saved = store.get_messages(conversation)
    store.delete_conversation(conversation)
    return events, sent, saved


FRENCH_REPLY = (
    "Les modèles inventent des noms de paquets dans 19,7 % des cas [1]. "
    "Un gestionnaire de paquets installe des bibliothèques écrites par d'autres développeurs (connaissance générale). "
    "Énoncé suivant : des attaquants enregistrent ces noms inventés, ce qu'on appelle le slopsquatting [2]."
)


def test_a_research_answer_is_written_in_french_and_checked(monkeypatch):
    events, sent, saved = run_research(monkeypatch, FRENCH_REPLY)
    assert "Write in French" in sent[0][0]["content"]
    assert research.INSTRUCTION in sent[0][-1]["content"]
    check = next(e for e in events if e["type"] == "check")
    assert check["flagged"] == 0, check["flags"]
    assert check["claims"] == 2, "the general-knowledge sentence isn't a claim about the papers"
    meta = saved[-1]["meta"]
    assert meta["language"] == "French"
    assert meta["steps"] == [{"query": "package hallucination rate", "round": 1, "passages": 2, "papers": 2}]
    assert saved[-1]["query"] == "package hallucination rate"


def test_a_wrong_figure_in_french_is_still_caught(monkeypatch):
    events, _, _ = run_research(monkeypatch, "Les modèles inventent des noms de paquets dans 21,5 % des cas [1].")
    check = next(e for e in events if e["type"] == "check")
    assert [f["kind"] for f in check["flags"]] == ["number"]


def test_research_mode_is_offered_by_the_server():
    assert "research" in answer.MODES


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Le taux est de 19,7 % [1]. Énoncé suivant ici [1].", 2),  # A French sentence can start with É.
        ("Rate is 1,000 samples [1]. Next one [1].", 2),
    ],
)
def test_french_sentences_split_cleanly(text, expected):
    assert len(answer.split_claims(text)) == expected


def test_french_decimals_match_the_papers_figures():
    assert answer._numbers_in("19,7 %") == {"19.7"}
    assert answer._numbers_in("1,000 samples") == {"1000"}
    assert answer._numbers_in("576 000 échantillons") == answer._numbers_in("576,000 samples") == {"576000"}
    assert answer._numbers_in("1 234 567") == {"1234567"}
    assert answer._numbers_in("18 des 32 configurations") == {"18", "32"}
    assert answer._numbers_in("in 2020 576 samples") == {"2020", "576"}
    assert answer._numbers_in("1 137 extraits") == answer._numbers_in("contains 1, 137 GPT-3.5 snippets") - {"3.5"} == {"1137"}
