"""Questions about several named papers: one copy each, every paper read, the list kept, shared references computed."""

import pytest

from folio import answer, config, llm, references, search, store


def doc(key, title, first, year, second=None, pdf=True, chunks=40):
    creators = [{"type": "author", "last": first}] + ([{"type": "author", "last": second}] if second else [])
    return {"key": key, "title": title, "creators": creators, "year": year, "pdf_path": f"{key}.pdf" if pdf else None,
            "chunks": chunks, "collections": [], "tags": []}


LIBRARY = [
    doc("1:L1", "Beyond Functional Correctness: Exploring Hallucinations in LLM-Generated Code", "Liu", 2026),
    doc("1:L2", "Beyond Functional Correctness: Exploring Hallucinations in LLM-Generated Code", "Liu", 2026),
    doc("1:L3", "Beyond Functional Correctness: Exploring Hallucinations in LLM-Generated Code", "Liu", 2026, pdf=False, chunks=1),
    doc("1:T", "CodeHalu: Investigating Code Hallucinations in LLMs via Execution-based Verification", "Tian", 2025),
    doc("1:R", "Code Hallucination", "Rahman", 2024, second="Kundu"),
    doc("1:S", "SPLASH: Contrastive Audit Pressure for Detecting Security Hallucinations", "Rahman", 2026, second="Wang"),
    doc("1:D", "Refusing the Impossible: A Taxonomy and Benchmark for Code Hallucination", "Dasu", 2026),
]


@pytest.fixture
def library(monkeypatch):
    monkeypatch.setattr(store, "list_docs", lambda: [dict(d) for d in LIBRARY])


# ------------------------------------------------------------- one copy each


def test_a_paper_in_zotero_three_times_is_read_once():
    docs = {d["key"]: d for d in LIBRARY}
    allowed = search._one_copy_each(None, docs)
    assert sum(key.startswith("1:L") for key in allowed) == 1
    assert "1:L3" not in allowed, "the copy without a PDF is never the one kept"
    assert search._one_copy_each({"1:L2", "1:L3"}, docs) == search._one_copy_each({"1:L1"}, docs)


# ------------------------------------------------------------- named papers


def test_papers_are_found_by_title_short_name_or_author_in_the_order_given(library):
    question = (
        "For each paper: Refusing the Impossible (Dasu et al., 2026), then CodeHalu, "
        "then Beyond Functional Correctness: Exploring Hallucinations in LLM-Generated Code, then Rahman et Kundu, 2024."
    )
    named = search.named_papers(question)
    assert named[0] == "1:D" and named[1] == "1:T" and named[3] == "1:R"
    assert named[2] in {"1:L1", "1:L2"}, "one copy of Liu, the one with the PDF"
    assert "1:S" not in named, "Kundu and 2024 point to Rahman's other paper"


def test_a_short_generic_title_is_not_taken_as_naming_a_paper(library):
    assert search.named_papers("How do papers define code hallucination?") == []


def test_short_names_must_match_their_capitals(library):
    assert search.named_papers("what does a codehalu-like splash mean") == []
    assert search.named_papers("Compare SPLASH and CodeHalu") == ["1:S", "1:T"]


def test_an_author_with_two_papers_and_no_other_clue_names_both(library):
    assert set(search.named_papers("What does Rahman argue?")) == {"1:R", "1:S"}


# ---------------------------------------------------------- passages per paper


def test_every_named_paper_gets_its_own_passages(monkeypatch):
    rows = {key: [{"id": key * 100 + n, "section": ""} for n in range(30)] for key in (1, 2)}
    monkeypatch.setattr(store, "get_doc", lambda key: {"key": key, "title": f"T{key}", "pdf_path": "x", "pages": 9, "creators": [], "year": 2024})
    monkeypatch.setattr(store, "chunks_of", lambda key: rows[key])
    monkeypatch.setattr(store, "get_chunks", lambda ids: {i: {"id": i, "doc_key": i // 100, "text": f"t{i}", "page": i % 100, "page_end": i % 100, "section": "", "ord": i % 100} for i in ids})

    def best(key, aspect, k):
        start = 10 if "definition" in aspect else 20
        return [{"chunk_id": key * 100 + start + n, "score": 0.9 - n / 10} for n in range(k)]

    monkeypatch.setattr(search, "best_in_paper", best)
    hits = search.passages_per_paper([1, 2], ["their definition", "the gap"], per_paper=5)
    for key in (1, 2):
        mine = [h["chunk_id"] % 100 for h in hits if h["doc_key"] == key]
        assert mine == [0, 10, 11, 20, 21], "the opening, then the best for each aspect in turn, in page order"
    assert all(h["named"] for h in hits)


# ------------------------------------------------------ the list carries over


def ask(question, history=(), mode="answer"):
    return answer.AskRequest(question=question, mode=mode)


def test_a_follow_up_stays_on_the_papers_already_discussed(monkeypatch):
    monkeypatch.setattr(answer, "named_papers", lambda question, scope: [])
    history = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a", "meta": {"papers": ["1:A", "1:B"]}}]
    assert answer.papers_in_question(ask("Could I write a review on one topic from these?"), history) == (["1:A", "1:B"], True)
    assert answer.papers_in_question(ask("Which other papers in my library cover this?"), history) == ([], False)
    assert answer.papers_in_question(ask("Quels autres articles en parlent ?"), history) == ([], False)
    assert answer.papers_in_question(ask("Explain it", mode="explain"), history) == ([], False)


def test_one_named_paper_keeps_the_ordinary_search(monkeypatch):
    monkeypatch.setattr(answer, "named_papers", lambda question, scope: ["1:A"])
    history = [{"role": "assistant", "content": "a", "meta": {"papers": ["1:A", "1:B"]}}]
    assert answer.papers_in_question(ask("And what does Liu say?"), history) == ([], False)


# ------------------------------------------------------------ shared references


def test_titles_are_found_in_each_citation_style():
    ieee = '[4] M. Chen et al., “Evaluating large language models trained on code,” arXiv preprint, 2021.'
    acl = "Mark Chen, Jerry Tworek. 2021. Evaluating large language models trained on code. arXiv preprint."
    assert references._titles(references._joined(ieee)) == ["Evaluating large language models trained on code"]
    assert "Evaluating large language models trained on code" in references._titles(references._joined(acl))
    venue = "Z. Ji. 2023. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics."
    assert references._titles(venue) == [], "a venue repeated across lists isn't a shared reference"


def test_shared_references_are_matched_by_title_across_styles(monkeypatch):
    lists = {
        "1:A": ((17, '[4] M. Chen, “Evaluating large lan-\nguage models trained on code,” 2021.\n[9] L. Huang, “A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions,” 2023.'),),
        "1:B": ((8, "Chen, M. 2021. Evaluating Large Language Models Trained on Code. arXiv.\nHuang, L. 2023. A survey on hallucination in large language models: principles and open questions. arXiv."),),
        "1:C": ((9, "Allamanis, M. 2024. Unsupervised evaluation of code LLMs. arXiv."),),
    }
    monkeypatch.setattr(references, "reference_lists", lambda keys: {k: lists[k] for k in keys})
    monkeypatch.setattr(store, "get_doc", lambda key: {"key": key, "title": key, "creators": [{"type": "author", "last": key[-1]}], "year": 2024})
    result = references.shared(["1:A", "1:B", "1:C"])
    titles = {work["title"]: work for work in result["works"]}
    assert titles["Evaluating large language models trained on code"]["count"] == 2, "a word split across lines still matches"
    huang = next(work for title, work in titles.items() if title.startswith("A survey on hallucination"))
    assert huang["count"] == 2, "a subtitle printed differently is still the same work"
    assert [c["page"] for c in huang["cited_by"]] == [17, 8]
    assert result["papers"] == 3 and len(result["works"]) == 2


def test_the_model_isnt_asked_for_what_folio_computes():
    question = "Pour chaque article :\nLeur définition, avec la page.\nLes références qu'ils citent tous."
    shown = answer.without_references_request(question)
    assert "Leur définition, avec la page." in shown and "citent tous" not in shown
    long_line = "Explain how each paper's definition differs from the definitions in the works it cites, and why that matters for my review."
    assert answer.without_references_request(long_line) == long_line, "a longer request is about more than the reference list"


# ------------------------------------------------------------------ whole answer


PAPER_TEXT = {
    "1:A": ["We define code hallucination as ungrounded code generation.", "Prior benchmarks only test solvable tasks."],
    "1:B": ["Code hallucinations refer to code that cannot execute as expected.", "This remains unexplored for code.", "We test 17 models."],
}


def fake_paper_sources(key, first):
    texts = PAPER_TEXT[key]
    sources = [{"n": first + i, "chunk_id": first + i, "doc_key": key, "text": text, "page": i + 1, "page_end": i + 1, "section": "",
                "score": 1.0, "title": f"Title {key}", "cite": f"{key[-1]} et al., 2026", "apa": f"{key[-1]}.", "year": 2026,
                "venue": "", "has_pdf": True, "pages": 9, "overview": False} for i, text in enumerate(texts)]
    return {"key": key, "title": f"Title {key}"}, sources, len(texts)


def run_named(monkeypatch, question, papers, replies):
    sent, options = [], []

    def fake_stream(model, messages, on_finish=None, **kwargs):
        sent.append(messages)
        options.append(kwargs)
        yield replies[len(sent) - 1]

    monkeypatch.setattr(answer, "named_papers", lambda question, scope: papers)
    monkeypatch.setattr(answer.reading, "paper_sources", fake_paper_sources)
    monkeypatch.setattr(answer, "retrieve", lambda *a, **k: pytest.fail("named papers aren't searched"))
    monkeypatch.setattr(references, "shared", lambda keys: {"papers": 2, "works": [{"title": "Evaluating LLMs trained on code", "cited_by": [], "count": 2}], "missing": []})
    monkeypatch.setattr(llm, "chat_stream", fake_stream)
    checks = []
    monkeypatch.setattr(llm, "chat_once", lambda model, messages, num_predict=128, **kwargs: checks.append(kwargs) or "YES")
    monkeypatch.setattr(llm, "loaded_models", lambda: [])
    monkeypatch.setattr(llm, "list_models", lambda: [config.DEFAULT_LLM])
    options.append(checks)
    job = answer.start(answer.AskRequest(question=question, model=config.DEFAULT_LLM))
    events = []
    while (event := job.events.get(timeout=20)) is not None:
        events.append(event)
    conversation = next(e["id"] for e in events if e["type"] == "conversation")
    saved = store.get_messages(conversation)
    store.delete_conversation(conversation)
    return events, sent, options, saved


def test_named_papers_are_each_read_in_full_then_put_together(monkeypatch):
    replies = [
        'They define it as "ungrounded code generation" [1].',
        'They define code hallucinations as code that "cannot execute as expected" [3].',
        "Both define it, but one by grounding [1] and the other by execution [3]. The shared references are shown below.",
    ]
    events, sent, options, saved = run_named(monkeypatch, "For A and B: their definition, and the references they share.", ["1:A", "1:B"], replies)
    checks, options = options[0], options[1:]

    first, second, together = (messages[-1]["content"] for messages in sent)
    assert "ungrounded code generation" in first and "cannot execute" not in first, "each call holds one paper only"
    assert "[3] (p. 1) Code hallucinations refer to" in second, "numbers run on from one paper to the next"
    assert "[5] (p. 3) We test 17 models." in second, "the whole paper, every passage"
    assert "Folio answers this part itself" in first, "the reference question is left to Folio"
    contexts = {o["num_ctx"] for o in options} | {c["num_ctx"] for c in checks}
    assert len(contexts) == 1 and contexts.pop() >= 4096, "one context for the whole job, so a large local model isn't loaded again and again"
    assert 'They define it as "ungrounded code generation" [1].' in together and "shown below" in together

    text = saved[-1]["content"]
    assert text.index("## A et al., 2026") < text.index("## B et al., 2026") < text.index("## Across the papers")
    check = next(e for e in events if e["type"] == "check")
    assert check["flagged"] == 0, check["flags"]
    assert [s["n"] for s in saved[-1]["sources"]] == [1, 3], "only the passages the answer cites are saved"
    meta = saved[-1]["meta"]
    assert meta["full_read"] == {"papers": 2, "passages": 5, "shortened": []}
    assert [step["passages"] for step in meta["steps"]] == [2, 3]
    assert meta["papers"] == ["1:A", "1:B"] and meta["references"]["papers"] == 2
    assert next(e for e in events if e["type"] == "references")["works"][0]["count"] == 2


def test_too_many_named_papers_fall_back_to_reading_passages(monkeypatch):
    many = [f"1:{n}" for n in range(answer.reading.MAX_PAPERS + 1)]
    monkeypatch.setattr(answer, "named_papers", lambda question, scope: many)
    calls = {}
    hit = {"n": 1, "chunk_id": 1, "doc_key": "1:0", "text": "A passage.", "page": 1, "page_end": 1, "section": "", "score": 1.0, "title": "T",
           "cite": "A, 2026", "apa": "A.", "year": 2026, "venue": "", "has_pdf": True, "pages": 9, "overview": False, "named": True}
    monkeypatch.setattr(answer, "passages_per_paper", lambda keys, aspects, per: calls.update(keys=keys, per=per) or [dict(hit)])
    monkeypatch.setattr(answer.reading, "paper_sources", lambda *a: pytest.fail("too many papers to read whole"))
    monkeypatch.setattr(llm, "chat_stream", lambda model, messages, on_finish=None, **kwargs: iter(["A passage says so [1]."]))
    monkeypatch.setattr(llm, "chat_once", lambda model, messages, num_predict=128: "the definition" if "separate things" in messages[0]["content"] else "YES")
    monkeypatch.setattr(llm, "loaded_models", lambda: [])
    monkeypatch.setattr(llm, "list_models", lambda: [config.DEFAULT_LLM])
    job = answer.start(answer.AskRequest(question="Their definitions?", model=config.DEFAULT_LLM))
    while (event := job.events.get(timeout=20)) is not None:
        if event["type"] == "conversation":
            store.delete_conversation(event["id"])
    assert calls["keys"] == many[: answer._MAX_PAPERS]


def test_papers_to_read():
    from folio.reading import papers_to_read
    assert papers_to_read("answer", ["1:A", "1:B"], search.Scope()) == ["1:A", "1:B"]
    assert papers_to_read("answer", ["1:A"], search.Scope()) == [], "one named paper may be compared with the rest of the library"
    assert papers_to_read("answer", [], search.Scope(doc_keys=["1:A"])) == ["1:A"], "Ask about this paper"
    assert papers_to_read("answer", [], search.Scope(doc_keys=["1:A"], tags=["x"])) == []
    assert papers_to_read("explain", [], search.Scope(doc_keys=["1:A"])) == ["1:A"]
    assert papers_to_read("research", ["1:A", "1:B"], search.Scope()) == []
