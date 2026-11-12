"""The literature matrix, with fake search and a fake model."""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from openpyxl import load_workbook

from folio import config, llm, matrix, store, zotero
from folio import app as app_module


def make_doc(key, title, fingerprint="fp1", year=2023):
    item = zotero.Item(library_id=1, group_id=None, key=key, item_type="journalArticle", title=title, year=year, creators=[{"first": "Ada", "last": f"Author{key[-1]}", "type": "author"}])
    store.upsert_docs_metadata([item])
    chunks = [
        {"page": 1, "page_end": 1, "section": "Abstract", "text": f"{title} studies teachers. We interviewed 12 teachers in 2021.", "rects": []},
        {"page": 4, "page_end": 4, "section": "Limitations", "text": "Our sample was small and drawn from one country.", "rects": []},
    ]
    store.save_indexed(item.doc_key, fingerprint, "ok", None, 1, chunks, np.zeros((2, config.EMBED_DIM), dtype=np.float32))
    return item.doc_key


@pytest.fixture
def library(monkeypatch):
    """Two papers, the default columns, a fake search that returns the passages, and a fake model."""
    for column in store.matrix_columns():
        store.delete_matrix_column(column["id"])
    for label, question in store.MATRIX_DEFAULT_COLUMNS[:2]:
        store.add_matrix_column(label, question)
    keys = [make_doc("MATRIX0A", "Paper A"), make_doc("MATRIX0B", "Paper B")]

    def fake_best_in_paper(doc_key, question, extra_terms="", k=3, skip=None):
        doc = store.get_doc(doc_key)
        rows = store.chunks_of(doc["key"])
        chunk = store.get_chunk(rows[1]["id"])
        return [matrix._hit(chunk, doc, 0.9)]

    replies = {"default": "Interviews with 12 teachers [1]; small sample [2]."}
    calls = []

    def fake_chat(model, messages, num_predict=128):
        calls.append(messages[0]["content"])
        for needle, reply in replies.items():
            if needle != "default" and needle in messages[0]["content"]:
                return reply
        return replies["default"]

    monkeypatch.setattr(matrix, "best_in_paper", fake_best_in_paper)
    monkeypatch.setattr(llm, "chat_once", fake_chat)
    monkeypatch.setattr(llm, "list_models", lambda: [config.DEFAULT_LLM])
    config.save_settings(llm_model=config.DEFAULT_LLM, local_model=config.DEFAULT_LLM)
    yield {"keys": keys, "replies": replies, "calls": calls}
    for key in keys:
        store.delete_doc(key)


def cells_by_key():
    return {(cell["doc_key"], cell["column_id"]): cell for cell in matrix.overview()["cells"]}


def test_new_databases_start_with_the_usual_columns():
    labels = [label for label, _ in store.MATRIX_DEFAULT_COLUMNS]
    assert labels == ["Research question", "Method", "Data and participants", "Main findings", "Limitations"]


def test_filling_writes_every_cell_from_that_papers_passages(library):
    matrix.filler.run_blocking(library["keys"])
    cells = cells_by_key()
    columns = store.matrix_columns()
    assert len(cells) == 4
    cell = cells[(library["keys"][0], columns[0]["id"])]
    assert cell["text"] == "Interviews with 12 teachers [1]; small sample [2]."
    assert [source["page"] for source in cell["sources"]] == [1, 4], "the paper's opening comes first, then the best match"
    assert cell["flags"] == [] and cell["stale"] is False
    assert all("Paper A" in prompt or "Paper B" in prompt for prompt in library["calls"])
    assert matrix.filler.state["done"] == 4 and not matrix.filler.state["running"]


def test_only_empty_or_outdated_cells_are_filled_again(library):
    matrix.filler.run_blocking(library["keys"])
    library["calls"].clear()
    matrix.filler.run_blocking(library["keys"])
    assert library["calls"] == [], "filled cells shouldn't be redone"

    make_doc("MATRIX0A", "Paper A", fingerprint="fp2")  # Re-indexed, so its cells are out of date.
    assert all(cell["stale"] for (key, _), cell in cells_by_key().items() if key == library["keys"][0])
    matrix.filler.run_blocking(library["keys"])
    assert len(library["calls"]) == 2
    assert not any(cell["stale"] for cell in cells_by_key().values())


def test_not_stated_is_kept_as_an_empty_note(library):
    library["replies"]["default"] = "Not stated."
    matrix.filler.run_blocking(library["keys"][:1])
    cell = next(iter(cells_by_key().values()))
    assert cell["text"] == ""
    assert [source["page"] for source in cell["sources"]] == [1, 4], "keep the pages searched, so the student can look"


class OverlapEngine:
    """Scores a passage by the words it shares with the question, however few."""

    rerank_capacity = 40

    def encode_query(self, text):
        vector = np.zeros(config.EMBED_DIM, dtype=np.float32)
        vector[0] = 1.0
        return vector

    def rerank(self, query, passages):
        words = set(query.lower().split())
        return np.array([len(words & set(p.lower().split())) / 100 for p in passages], dtype=np.float32)


def test_the_best_passages_in_a_paper_are_kept_however_low_they_score(monkeypatch):
    from folio import search

    item = zotero.Item(library_id=1, group_id=None, key="INPAPER1", item_type="journalArticle", title="In paper")
    store.upsert_docs_metadata([item])
    chunks = [
        {"page": 1, "page_end": 1, "section": "Abstract", "text": "We study teachers.", "rects": []},
        {"page": 2, "page_end": 2, "section": "II. Related Work", "text": "Smith used interviews with a careful design.", "rects": []},
        {"page": 3, "page_end": 3, "section": "Method", "text": "Our design used interviews.", "rects": []},
    ]
    vectors = np.zeros((3, config.EMBED_DIM), dtype=np.float32)
    vectors[:, 0] = 1.0
    store.save_indexed(item.doc_key, "fp", "ok", None, 3, chunks, vectors)
    monkeypatch.setattr(search, "engine", OverlapEngine())
    searched = []
    original = store.keyword_search
    monkeypatch.setattr(store, "keyword_search", lambda query, keys, limit: searched.append(query) or original(query, keys, limit))
    try:
        hits = search.best_in_paper(item.doc_key, "what design did they use", "methodology", k=3, skip=matrix._about_other_work)
        assert [hit["page"] for hit in hits] == [3, 1], "related work is left out, and low scores are kept"
        assert "methodology" in searched[0], "extra words widen the keyword search"
        only_related = search.best_in_paper(item.doc_key, "what design did they use", k=1, skip=lambda doc, chunk: True)
        assert len(only_related) == 1, "if every passage would be skipped, use them anyway"
    finally:
        store.delete_doc(item.doc_key)


@pytest.mark.parametrize(
    "title, section, text, expected",
    [
        ("A study", "4 Results", "We found that 40% of samples were vulnerable.", False),
        ("A study", "2.3 Evaluations of Copilot", "Our tool works.", False),
        ("A study", "II. RELATED WORK", "Our tool works.", True),
        ("A study", "2.3 Evaluations of Copilot", "Xu et al. (Xu et al., 2022) compared models. Nguyen and Nadi (Nguyen and Nadi, 2022) used LeetCode.", True),
        ("A study", "Intro", "Prior tools [4], [7, 9] and [12] exist.", True),
        ("A study", "Intro", "Tools such as CodeQL [21] exist, as Smith et al. note.", False),
        ("A survey of hallucination", "5 Detection", "Methods [4], [7, 9] and [12] detect it.", False),
        ("A study", "Threats to validity", "Unlike [4], [7] and [12], we recruited students.", False),
        ("A study", "2.3 Evaluations of Copilot", "Xu et al. and Nguyen et al. and Dakhel et al. evaluated it.", True),
    ],
)
def test_passages_about_other_work_are_recognised(title, section, text, expected):
    assert matrix._about_other_work({"title": title}, {"section": section, "text": text}) is expected


def test_the_usual_columns_search_with_the_words_authors_use(library, monkeypatch):
    seen = []
    monkeypatch.setattr(matrix, "best_in_paper", lambda key, question, extra="", **kwargs: seen.append(extra) or [])
    matrix._passages(store.get_doc(library["keys"][0]), {"id": 1, "label": "Method", "question": "What did the authors do?"})
    matrix._passages(store.get_doc(library["keys"][0]), {"id": 2, "label": "Languages", "question": "Which languages?"})
    assert "methodology" in seen[0] and seen[1] == ""


def test_a_figure_missing_from_the_cited_passages_is_flagged(library):
    library["replies"]["default"] = "Interviewed 40 teachers [1]."
    matrix.filler.run_blocking(library["keys"][:1])
    cell = next(iter(cells_by_key().values()))
    assert "40" in cell["flags"][0]["detail"]


def test_the_publication_year_and_figures_in_the_passages_are_fine(library):
    library["replies"]["default"] = "In 2023 they interviewed 12 teachers in 2021 [1]."
    matrix.filler.run_blocking(library["keys"][:1])
    assert next(iter(cells_by_key().values()))["flags"] == []


def test_citations_to_passages_that_dont_exist_are_removed(library):
    library["replies"]["default"] = "Small sample [2, 7]; also [9]."
    matrix.filler.run_blocking(library["keys"][:1])
    assert next(iter(cells_by_key().values()))["text"] == "Small sample [2]; also ."


def test_removing_a_column_removes_its_notes(library):
    matrix.filler.run_blocking(library["keys"])
    first = store.matrix_columns()[0]["id"]
    store.delete_matrix_column(first)
    assert all(column_id != first for _, column_id in cells_by_key())


def test_a_cell_for_a_removed_column_or_paper_is_not_saved(library):
    store.save_matrix_cell(library["keys"][0], 999999, "fp1", "x", [], [], "m")
    store.save_matrix_cell("1:GONE0000", store.matrix_columns()[0]["id"], "fp1", "x", [], [], "m")
    assert cells_by_key() == {}


def test_stopping_keeps_what_was_filled(library, monkeypatch):
    original = matrix.fill_cell

    def fill_then_stop(*args):
        original(*args)
        matrix.filler.stop()

    monkeypatch.setattr(matrix, "fill_cell", fill_then_stop)
    matrix.filler.run_blocking(library["keys"])
    assert len(cells_by_key()) == 1 and matrix.filler.state["stopped"]


def test_it_needs_the_model_on_this_pc(library, monkeypatch):
    monkeypatch.setattr(llm, "list_models", lambda: [])
    with pytest.raises(llm.LLMError):
        matrix.filler.run_blocking(library["keys"])


def test_a_model_failure_is_reported_not_raised(library, monkeypatch):
    def down(*a, **k):
        raise llm.LLMError("Ollama stopped.")

    monkeypatch.setattr(llm, "chat_once", down)
    matrix.filler.run_blocking(library["keys"])
    assert matrix.filler.state["error"] == "Ollama stopped." and not matrix.filler.state["running"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Small sample [2].", "Small sample (p. 4)."),
        ("Both [1, 2].", "Both (pp. 1, 4)."),
        ("Twice [2][2]", "Twice (p. 4) (p. 4)"),
        ("No citation", "No citation"),
    ],
)
def test_spreadsheet_text_uses_page_numbers(text, expected):
    sources = [{"n": 1, "page": 1}, {"n": 2, "page": 4}]
    assert matrix.plain_text(text, sources) == expected


@pytest.fixture
def client():
    return TestClient(app_module.app, base_url="http://127.0.0.1:8799")


def test_the_excel_file_has_one_row_per_paper(library, client):
    library["replies"]["Paper B"] = "=HYPERLINK(\"http://example.com\") 99 teachers [1]."
    matrix.filler.run_blocking(library["keys"])
    response = client.get("/api/matrix/export", params={"keys": ",".join(library["keys"])})
    assert response.status_code == 200 and "attachment" in response.headers["content-disposition"]
    book = load_workbook(io.BytesIO(response.content))
    sheet = book["Matrix"]
    rows = list(sheet.iter_rows(values_only=True))
    assert rows[0] == ("Paper", "Year", "Research question", "Method", "Reference")
    assert len(rows) == 3
    paper_a = next(row for row in rows if "AuthorA" in row[0])
    assert paper_a[2] == "Interviews with 12 teachers (p. 1); small sample (p. 4)."
    library["replies"]["default"] = "Not stated"
    store.delete_matrix_column(store.matrix_columns()[1]["id"])
    store.add_matrix_column("Limitations", "What limitations do they state?")
    matrix.filler.run_blocking(library["keys"][:1])
    book = load_workbook(io.BytesIO(client.get("/api/matrix/export").content))
    assert "Not stated in the passages found (searched pp. 1, 4)" in [cell.value for cell in book["Matrix"][2]]
    paper_b = next(row for row in rows if "AuthorB" in row[0])
    assert sheet.cell(row=rows.index(paper_b) + 1, column=3).data_type == "s", "text from a paper must never become a formula"
    assert sheet.cell(row=rows.index(paper_b) + 1, column=3).comment is not None, "99 isn't in the passage"
    assert "About" in book.sheetnames


def test_the_api_lists_columns_rows_and_cells(library, client):
    matrix.filler.run_blocking(library["keys"][:1])
    data = client.get("/api/matrix").json()
    assert [column["label"] for column in data["columns"]] == ["Research question", "Method"]
    assert {row["key"] for row in data["rows"]} >= set(library["keys"])
    assert len(data["cells"]) == 2 and data["job"]["running"] is False


def test_a_paper_in_zotero_twice_gets_one_row(library):
    copy = zotero.Item(library_id=1, group_id=None, key="MATRIXDA", item_type="journalArticle", title="paper a", abstract="Paper A studies teachers.")
    store.upsert_docs_metadata([copy])
    store.save_indexed(copy.doc_key, "abs", "ok", None, 0, [{"page": 0, "page_end": 0, "section": "", "text": "Paper A studies teachers.", "rects": []}], np.zeros((1, config.EMBED_DIM), dtype=np.float32))
    try:
        data = matrix.overview()
        keys = [row["key"] for row in data["rows"]]
        assert library["keys"][0] in keys and copy.doc_key not in keys, "keep the copy with the PDF"
        assert data["duplicates"] == 1
    finally:
        store.delete_doc(copy.doc_key)


def test_columns_can_be_added_and_removed_through_the_api(library, client):
    added = client.post("/api/matrix/columns", json={"label": "Languages", "question": "Which languages does it cover?"}).json()
    assert added["label"] == "Languages"
    assert client.post("/api/matrix/columns", json={"label": " ", "question": "     x"}).status_code == 400
    assert client.post("/api/matrix/columns", json={"label": "", "question": "What?"}).status_code == 422
    client.delete(f"/api/matrix/columns/{added['id']}")
    assert "Languages" not in [column["label"] for column in store.matrix_columns()]


def test_filling_without_the_local_model_is_explained(library, client, monkeypatch):
    monkeypatch.setattr(llm, "list_models", lambda: [])
    response = client.post("/api/matrix/fill", json={"doc_keys": library["keys"]})
    assert response.status_code == 503 and "Ollama" in response.json()["detail"]
