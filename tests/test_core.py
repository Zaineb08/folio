"""Search fusion, scoping, the database, citation formatting and the citation checker."""

import numpy as np
import pytest

from folio import answer, cite, config, discover, store, zotero
from folio import search as search_module
from folio.search import Scope, is_about_whole_library, reciprocal_rank_fusion

DOC = {
    "title": "Retrieval-Augmented Generation for NLP Tasks",
    "creators": [
        {"first": "Patrick", "last": "Lewis", "type": "author"},
        {"first": "Ethan", "last": "Perez", "type": "author"},
        {"first": "A. B.", "last": "Piktus", "type": "author"},
    ],
    "year": 2020,
    "venue": "NeurIPS",
    "doi": "10.1000/xyz_1",
    "item_type": "conferencePaper",
}


# ------------------------------------------------------------------- citations


def test_short_cite_forms():
    assert cite.short_cite(DOC) == "Lewis et al., 2020"
    assert cite.short_cite({**DOC, "creators": DOC["creators"][:2]}) == "Lewis & Perez, 2020"
    assert cite.short_cite({**DOC, "creators": DOC["creators"][:1], "year": None}) == "Lewis, n.d."
    assert cite.short_cite({"title": "A Paper With No Authors", "creators": [], "year": 2001}).endswith(", 2001")


def test_apa_reference():
    reference = cite.apa(DOC)
    assert reference.startswith("Lewis, P., Perez, E., & Piktus, A. B. (2020).")
    assert "https://doi.org/10.1000/xyz_1" in reference


def test_apa_initials_cover_every_given_name():
    doc = {**DOC, "creators": [{"first": "W.L.J.M.", "last": "Devillé", "type": "author"}, {"first": "Jean-Paul", "last": "de Vet", "type": "author"}]}
    assert cite.apa(doc).startswith("Devillé, W. L. J. M., & de Vet, J. P. (2020).")


def test_apa_title_ending_in_a_question_mark_gets_no_extra_full_stop():
    reference = cite.apa({**DOC, "title": "Do Users Write More Insecure Code?", "venue": "CCS"})
    assert "Insecure Code? CCS." in reference


def test_bibtex_key_keeps_accented_letters_as_plain_ones():
    doc = {**DOC, "creators": [{"first": "W.", "last": "Devillé", "type": "author"}], "title": "Conducting reviews", "year": 2002}
    assert cite.bibtex_key(doc) == "deville2002conducting"


@pytest.mark.parametrize(
    "name, first, last",
    [
        ("Ada B. Lovelace", "Ada B.", "Lovelace"),
        ("Henrica C. W. de Vet", "Henrica C. W.", "de Vet"),
        ("Daniëlle van der Windt", "Daniëlle", "van der Windt"),
        ("Roberta De Luca", "Roberta", "De Luca"),
        ("Anna Van Der Berg", "Anna", "Van Der Berg"),
        ("De Luca", "De", "Luca"),
        ("Plato", "", "Plato"),
        ("", "", "Unknown"),
    ],
)
def test_names_keep_their_particles_in_the_surname(name, first, last):
    assert discover._split_name(name) == {"first": first, "last": last}


def test_bibtex_entry_escapes_and_types():
    entry = cite.bibtex({**DOC, "title": "Costs & Benefits of 100% RAG"})
    assert entry.startswith("@inproceedings{lewis2020costs,")
    assert r"Costs \& Benefits of 100\% RAG" in entry
    assert "author = {Lewis, Patrick and Perez, Ethan and Piktus, A. B.}" in entry
    assert r"doi = {10.1000/xyz\_1}" in entry


# ---------------------------------------------------------------------- search


def test_rrf_rewards_agreement_between_rankings():
    assert reciprocal_rank_fusion([[1, 2, 3], [3, 4, 1]])[0] in (1, 3)
    assert reciprocal_rank_fusion([[1, 2], [2, 1]])[0:2] in ([1, 2], [2, 1])
    assert reciprocal_rank_fusion([[5], []]) == [5]
    fused = reciprocal_rank_fusion([[9, 7], [7, 8]])
    assert fused[0] == 7


@pytest.mark.parametrize(
    "question, expected",
    [
        ("What are the main findings across my library?", True),
        ("Where do these authors disagree?", True),
        ("Which methods do these articles use?", True),
        ("What do the papers say is unknown?", True),
        ("What is the difference between naive and advanced RAG?", False),
        ("How do teachers adopt technology?", False),
    ],
)
def test_questions_about_the_whole_library_are_recognised(question, expected):
    assert is_about_whole_library(question) is expected


def test_scope_filters():
    docs = [
        {"key": "a", "collections": ["Thesis / Chapter 1"], "tags": ["rag"], "year": 2020},
        {"key": "b", "collections": ["Thesis"], "tags": [], "year": 2015},
        {"key": "c", "collections": ["Other"], "tags": ["rag"], "year": None},
    ]
    assert Scope().resolve(docs) is None
    assert Scope(collections=["Thesis"]).resolve(docs) == {"a", "b"}
    assert Scope(collections=["Thesis / Chapter 1"]).resolve(docs) == {"a"}
    assert Scope(tags=["rag"]).resolve(docs) == {"a", "c"}
    assert Scope(year_min=2018).resolve(docs) == {"a"}
    assert Scope(year_max=2018).resolve(docs) == {"b"}
    assert Scope(doc_keys=["c"]).resolve(docs) == {"c"}
    assert Scope(collections=["Nope"]).resolve(docs) == set()


# -------------------------------------------------------------------- database


def _item(key, title="A title"):
    return zotero.Item(library_id=1, group_id=None, key=key, item_type="journalArticle", title=title)


def _chunk(text, page=1):
    return {"page": page, "page_end": page, "section": "Intro", "text": text, "rects": [[page, 1, 2, 3, 4]]}


def _vector(*values):
    vector = np.zeros(config.EMBED_DIM, dtype=np.float32)
    vector[: len(values)] = values
    return vector / np.linalg.norm(vector)


@pytest.fixture
def indexed():
    store.upsert_docs_metadata([_item("AAA", "Alpha paper"), _item("BBB", "Beta paper")])
    store.save_indexed("1:AAA", "fp", "ok", None, 3, [_chunk("transformers improve translation")], np.stack([_vector(1, 0)]))
    store.save_indexed("1:BBB", "fp", "ok", None, 3, [_chunk("phenomenology of rural teaching")], np.stack([_vector(0, 1)]))
    yield
    store.delete_doc("1:AAA")
    store.delete_doc("1:BBB")


def test_vector_search_ranks_and_respects_scope(indexed):
    first = store.vector_index.search(_vector(1, 0), None, 5)
    assert store.get_chunk(first[0])["doc_key"] == "1:AAA"
    scoped = store.vector_index.search(_vector(1, 0), {"1:BBB"}, 5)
    assert [store.get_chunk(c)["doc_key"] for c in scoped] == ["1:BBB"]


def test_keyword_search_and_scope(indexed):
    hits = store.keyword_search("phenomenology", None, 5)
    assert store.get_chunk(hits[0])["doc_key"] == "1:BBB"
    assert store.keyword_search("phenomenology", {"1:AAA"}, 5) == []


@pytest.mark.parametrize("query", ['"unbalanced', "NOT AND OR", "a*b:c(d)", "-", 'title:"x"', "مرحبا عالم"])
def test_keyword_search_survives_hostile_input(indexed, query):
    assert isinstance(store.keyword_search(query, None, 5), list)


def test_keyword_search_ignores_accents(indexed):
    store.upsert_docs_metadata([_item("CCC")])
    store.save_indexed("1:CCC", "fp", "ok", None, 1, [_chunk("une étude approfondie")], np.stack([_vector(1, 1)]))
    assert store.keyword_search("etude", None, 5)
    store.delete_doc("1:CCC")


def test_reindexing_replaces_old_passages(indexed):
    store.save_indexed("1:AAA", "fp2", "ok", None, 3, [_chunk("completely new words")], np.stack([_vector(1, 0)]))
    assert store.keyword_search("translation", None, 5) == []
    assert store.keyword_search("completely", None, 5)


def test_deleting_a_doc_removes_it_from_every_index(indexed):
    store.delete_doc("1:BBB")
    assert store.keyword_search("phenomenology", None, 5) == []
    remaining = store.vector_index.search(_vector(0, 1), None, 5)
    assert all(store.get_chunk(c)["doc_key"] != "1:BBB" for c in remaining)


def test_conversations_round_trip():
    cid = store.create_conversation("A question")
    store.add_message(cid, "user", "A question")
    store.add_message(cid, "assistant", "An answer [1]", sources=[{"n": 1}], checks={"claims": 1})
    messages = store.get_messages(cid)
    assert messages[1]["sources"] == [{"n": 1}] and messages[1]["checks"] == {"claims": 1}
    store.delete_conversation(cid)
    assert store.get_messages(cid) == []


# ------------------------------------------------------------ citation checking


SOURCES = [
    {"n": 1, "text": "The trial enrolled 240 students and engagement rose by 12.5 percent overall."},
    {"n": 2, "text": "Interviews suggest that teachers value autonomy above pay."},
]


def _word_overlap(claim, passage):
    """Stands in for the model: a claim is supported if most of its words are in the passage."""
    words = set(claim.lower().rstrip(".").split())
    return len(words & set(passage.lower().split())) / max(len(words), 1) > 0.5


def _check(text):
    return answer.check_citations(text, SOURCES, _word_overlap)


def test_supported_claims_pass():
    result = _check("The trial enrolled 240 students [1]. Teachers value autonomy above pay [2].")
    assert result["claims"] == 2 and result["flagged"] == 0 and result["cited"] == [1, 2]


def test_invented_number_is_flagged_without_asking_the_model():
    asked = []
    result = answer.check_citations("The trial enrolled 900 students [1].", SOURCES, lambda c, p: asked.append(c) or True)
    assert result["flags"][0]["kind"] == "number" and "900" in result["flags"][0]["detail"]
    assert not asked


def test_citing_a_missing_source_is_flagged():
    assert _check("Teachers value autonomy above pay [7].")["flags"][0]["kind"] == "invalid"


def test_unsupported_claim_is_flagged():
    assert _check("Quantum gravity explains dark matter halos [2].")["flags"][0]["kind"] == "unsupported"


def test_claim_is_judged_against_all_the_passages_it_cites():
    seen = []
    answer.check_citations("Engagement rose and teachers value autonomy [1][2].", SOURCES, lambda c, p: seen.append(p) or True)
    assert "240 students" in seen[0] and "autonomy" in seen[0]


def test_long_uncited_claim_is_flagged_but_meta_statements_are_not():
    lead_in = "Teachers value autonomy above pay [2]. "
    uncited = "Most researchers now agree that this approach is clearly superior to all others."
    assert _check(lead_in + uncited)["flags"][0]["kind"] == "uncited"
    meta = "The excerpts do not say anything at all about how this was funded or by whom."
    assert _check(lead_in + meta)["flagged"] == 0


def test_an_opening_summary_sentence_needs_no_citation():
    text = "Teachers in these schools identify several barriers to adopting technology overall. Teachers value autonomy above pay [2]."
    assert _check(text)["flagged"] == 0


def test_a_topic_sentence_before_a_cited_claim_is_not_flagged():
    text = (
        "Teachers value autonomy above pay [2]. "
        "Finally, studies on the adoption of technology reveal how teachers navigate policy constraints. "
        "Interviews suggest that teachers value autonomy above pay [2]."
    )
    assert _check(text)["flagged"] == 0


def test_a_sources_publication_year_is_not_an_invented_number():
    sources = [{**SOURCES[1], "year": 2019}]
    text = "Teachers (2019) value autonomy above pay [2]."
    assert answer.check_citations(text, sources, lambda c, p: True)["flagged"] == 0
    assert answer.check_citations("Teachers (1987) value autonomy above pay [2].", sources, lambda c, p: True)["flags"][0]["kind"] == "number"


def test_a_papers_own_reference_numbers_are_hidden_from_the_model():
    source = {"n": 1, "cite": "A, 2020", "page": 3, "section": "", "title": "T", "text": "As shown in [14] and [3, 4] or [7–9], it works [12]."}
    block = answer._excerpt_block([source])
    assert "As shown in and or, it works." in block
    assert 'number="1"' in block


def test_multiple_citation_formats_are_understood():
    result = _check("Engagement rose by 12.5 percent and teachers value autonomy [1, 2].")
    assert result["cited"] == [1, 2] and result["flagged"] == 0
    result = _check("Engagement rose by 12.5 percent [1][2].")
    assert result["cited"] == [1, 2] and result["flagged"] == 0


def test_a_figure_written_differently_is_the_same_figure():
    source = [{"n": 1, "text": "It corrected 77.0% of errors across 1,689 programs."}]
    assert answer.check_citations("It corrected about 77% of errors across 1689 programs [1].", source, lambda c, p: True)["flagged"] == 0


def test_a_quotation_must_be_word_for_word_in_the_cited_passage():
    source = [{"n": 1, "text": "Participants with AI wrote significantly less se-\ncure code [14] than others."}]
    passes = lambda c, p: True  # noqa: E731
    exact = answer.check_citations('They "wrote significantly less se- cure code than others" here [1].', source, passes)
    assert exact["flagged"] == 0, "broken PDF hyphens and the paper's own references shouldn't count as misquotes"
    altered = answer.check_citations('They "wrote far less secure code than others" here [1].', source, passes)
    assert altered["flags"][0]["kind"] == "quote"
    short = answer.check_citations('They call it "less secure code" here [1].', source, passes)
    assert short["flagged"] == 0, "short phrases are too easily reworded to count as quotations"


def test_statements_about_the_excerpts_are_not_checked_against_them():
    asked = []
    text = "Lead sentence here [1]. The excerpts do not describe the labelling techniques used [1]. They do not detail the metrics [1]."
    answer.check_citations(text, SOURCES, lambda c, p: asked.append(c) or True)
    assert asked == ["Lead sentence here."]


def test_the_checker_is_told_who_wrote_each_passage():
    seen = []
    source = [{"n": 1, "cite": "Djire et al., 2026", "text": "This is called slopsquatting."}]
    answer.check_citations("Djire et al. call this slopsquatting [1].", source, lambda c, p: seen.append(p) or True)
    assert seen[0].startswith("From Djire et al., 2026:")


def test_a_question_naming_an_author_finds_their_paper():
    docs = {
        "1:P": {"key": "1:P", "creators": [{"first": "Neil", "last": "Perry", "type": "author"}]},
        "1:L": {"key": "1:L", "creators": [{"first": "Ann", "last": "Long", "type": "author"}]},
        "1:V": {"key": "1:V", "creators": [{"first": "Roberta", "last": "De Luca", "type": "author"}, {"first": "Bo", "last": "Perry", "type": "author"}]},
        "1:N": {"key": "1:N", "creators": []},
    }
    find = search_module._named_docs
    assert find("What did the Perry et al. study find?", docs, None) == ["1:P"]
    assert find("what did perry et al find?", docs, None) == ["1:P"], "lower case still counts when followed by et al"
    assert find("How long does indexing take?", docs, None) == [], "an ordinary word isn't a surname"
    assert find("What did De Luca report?", docs, None) == ["1:V"]
    assert find("What did Perry find?", docs, {"1:L"}) == [], "a paper outside the scope stays out"
    assert find("Compare Perry and Long.", docs, None) == ["1:P", "1:L"]


def test_claims_split_keeps_abbreviations_whole():
    claims = answer.split_claims("Lewis et al. found that it works [1]. A second claim follows here [2].")
    assert len(claims) == 2 and claims[0].startswith("Lewis et al. found")
    assert claims[0].endswith("[1].")


def test_claims_split_handles_markdown_structure():
    text = "## Heading\n\n- First bullet point [1].\n- Second bullet point [2].\n\n| A | B |\n|---|---|\n| x | y [1] |\n"
    claims = answer.split_claims(text)
    assert claims == ["First bullet point [1].", "Second bullet point [2].", "| A | B |", "| x | y [1] |"]


# -------------------------------------------------------------------- discover


def test_openalex_abstract_is_rebuilt_in_order():
    assert discover._abstract({"world": [1], "hello": [0], "again": [2]}) == "hello world again"
    assert discover._abstract(None) == ""


def test_openalex_record_is_normalised():
    work = {
        "id": "W1",
        "doi": "https://doi.org/10.1/ABC",
        "display_name": "A <i>Great</i> Paper",
        "publication_year": 2021,
        "authorships": [{"author": {"display_name": "Ada B. Lovelace"}}],
        "primary_location": {"source": {"display_name": "Nature"}, "landing_page_url": "https://x.org/a"},
        "best_oa_location": {"pdf_url": "https://x.org/a.pdf", "landing_page_url": "https://x.org/a"},
        "open_access": {"is_oa": True},
        "cited_by_count": 7,
        "type": "article",
    }
    result = discover._normalise(work, {"10.1/abc"}, set())
    assert result["title"] == "A Great Paper" and result["doi"] == "10.1/abc"
    assert result["authors"] == [{"first": "Ada B.", "last": "Lovelace"}]
    assert result["pdf_url"] == "https://x.org/a.pdf" and result["in_library"] is True


def test_crossref_record_is_normalised():
    work = {
        "DOI": "10.1/XYZ",
        "title": ["A <i>Crossref</i> Paper"],
        "author": [{"given": "Ada", "family": "Lovelace"}, {"name": "A Consortium"}],
        "issued": {"date-parts": [[2019, 3]]},
        "container-title": ["Nature"],
        "is-referenced-by-count": 4,
        "type": "posted-content",
        "URL": "https://doi.org/10.1/xyz",
        "license": [{"URL": "https://creativecommons.org/licenses/by/4.0/"}],
        "link": [{"URL": "https://x.org/a.html", "content-type": "text/html"}, {"URL": "https://x.org/a.pdf", "content-type": "application/pdf"}],
    }
    result = discover._normalise_crossref(work)
    assert result["title"] == "A Crossref Paper" and result["year"] == 2019 and result["type"] == "preprint"
    assert result["authors"] == [{"first": "Ada", "last": "Lovelace"}, {"first": "", "last": "A Consortium"}]
    assert result["is_oa"] is True and result["pdf_url"] == "https://x.org/a.pdf"
    closed = discover._normalise_crossref({**work, "license": [{"URL": "https://publisher.example/tdm"}]})
    assert closed["is_oa"] is False and closed["pdf_url"] == ""


def test_search_falls_back_to_crossref_when_openalex_is_down(monkeypatch):
    def down(*args):
        raise discover.httpx.HTTPError("HTTP 503")

    monkeypatch.setattr(discover, "_search_openalex", down)
    monkeypatch.setattr(discover, "_search_crossref", lambda *args: [{"doi": "10.1/a", "title": "T"}])
    outcome = discover.search("anything")
    assert outcome["source"] == "Crossref" and outcome["results"][0]["in_library"] is False


def test_search_reports_when_every_source_is_down(monkeypatch):
    def down(*args):
        raise discover.httpx.HTTPError("HTTP 503")

    monkeypatch.setattr(discover, "_search_openalex", down)
    monkeypatch.setattr(discover, "_search_crossref", down)
    with pytest.raises(discover.DiscoverError):
        discover.search("anything")


@pytest.mark.parametrize("url", ["http://127.0.0.1/secret.pdf", "http://localhost:8765/x.pdf", "file:///C:/x.pdf", "ftp://example.com/a.pdf", "http://192.168.1.1/a.pdf"])
def test_pdf_download_refuses_local_targets(url):
    assert discover.download_pdf(url) is None
