"""Hybrid retrieval: meaning plus exact terms, combined, then reranked by a cross-encoder."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import cite, store
from .models import engine

_CANDIDATES = 60  # Taken from each of the two searches.
_RRF_K = 60
MIN_SCORE = 0.02  # Below this, the reranker considers a passage unrelated to the question.

# Questions about "these articles" as a whole. Measured: with no topic they score under
# 0.05 at best, while ones that do name a topic ("...about the digital divide") reach 0.45.
_WHOLE_LIBRARY = re.compile(
    r"\b(these|my|the|all|our|this)\s+(articles?|papers?|authors?|studies|sources|library|documents?|readings?|collection)\b",
    re.IGNORECASE,
)
_WEAK_MATCH = 0.1
_ACROSS_NEAREST = 400
_NAMED_CANDIDATES = 6  # Extra candidates from each search, for a paper the question names.
_NAMED_RESERVED = 3  # Places kept for it in the results.
_IN_PAPER_CANDIDATES = 12  # From each of the two searches, within one paper.
_WEAK_EVIDENCE = 0.7  # If the best passage scores below this, the papers probably don't really answer the question.
_PAPER_FACETS = (
    "the research question, aim and motivation of this paper",
    "the method, data, participants and evaluation setup",
    "the main results, findings and answers to the research questions",
    "the limitations and threats to validity",
    "the conclusions and future work",
)
_SUMMING_UP = re.compile(r"abstract|summary|conclu|discussion|limitation|future|findings|implication", re.IGNORECASE)


@dataclass
class Scope:
    collections: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    year_min: int | None = None
    year_max: int | None = None
    doc_keys: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.collections or self.tags or self.year_min or self.year_max or self.doc_keys)

    def resolve(self, docs: list[dict]) -> set[str] | None:
        """The documents this scope allows, or None for the whole library."""
        if self.is_empty():
            return None
        return {doc["key"] for doc in docs if self._allows(doc)}

    def _allows(self, doc: dict) -> bool:
        if self.doc_keys and doc["key"] not in self.doc_keys:
            return False
        if self.collections and not any(
            have == want or have.startswith(f"{want} / ") for have in doc["collections"] for want in self.collections
        ):
            return False
        if self.tags and not set(self.tags) & set(doc["tags"]):
            return False
        year = doc.get("year")
        if self.year_min and (year is None or year < self.year_min):
            return False
        if self.year_max and (year is None or year > self.year_max):
            return False
        return True


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = _RRF_K) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)


def assess_evidence(hits: list[dict]) -> str | None:
    """How well do the passages found match the question: "strong", "weak" or "sample".

    Measured on this library: with 0.7 as the cutoff for the best passage, 12 of 14 questions
    the papers couldn't answer counted as weak, and none of 15 they could answer. The two it
    missed match the topic but lack the specific fact, which scores can't reveal. Passages
    rarely contain their own authors' names, so scores say little when a question names a paper.
    """
    if not hits:
        return None
    if hits[0].get("overview"):
        return "sample"
    if any(hit.get("named") for hit in hits):
        return "strong"
    return "strong" if max(hit["score"] for hit in hits) >= _WEAK_EVIDENCE else "weak"


def paper_passages(doc_key: str, k: int = 12) -> tuple[list[dict], dict]:
    """Passages covering a whole paper: its opening, plus the best match for each part of a study."""
    doc = store.get_doc(doc_key)
    rows = store.chunks_of(doc_key)
    if not doc or not rows:
        return [], {"used": 0, "total": 0}
    chosen = [rows[0]["id"]]
    summing_up = [row["id"] for row in rows if _SUMMING_UP.search(row["section"])]
    chosen += summing_up[:1] + summing_up[-1:]
    ranked = [store.vector_index.search(engine.encode_query(facet), {doc_key}, 3) for facet in _PAPER_FACETS]
    for rank in range(3):  # The best for every part of the study first, so none gets crowded out.
        chosen += [ids[rank] for ids in ranked if rank < len(ids)]
    chosen = list(dict.fromkeys(chosen))[:k]
    chunks = store.get_chunks(chosen)
    ordered = sorted(chunks.values(), key=lambda chunk: chunk["ord"])  # Read in the paper's own order.
    return [_hit(chunk, doc, 1.0) for chunk in ordered], {"used": len(ordered), "total": len(rows)}


def retrieve(query: str, scope: Scope | None = None, k: int = 8, per_doc: int = 3, allow_overview: bool = True) -> list[dict]:
    """Return the k best passages, best first, with their citation details."""
    if not store.counts()["chunks"]:
        return []  # Nothing is indexed yet, and the models may not even be downloaded.
    docs = {doc["key"]: doc for doc in store.list_docs()}
    allowed = (scope or Scope()).resolve(list(docs.values()))
    if allowed is not None and not allowed:
        return []

    query_vector = engine.encode_query(query)
    dense = store.vector_index.search(query_vector, allowed, _CANDIDATES)
    sparse = store.keyword_search(query, allowed, _CANDIDATES)
    candidates = reciprocal_rank_fusion([dense, sparse])[: engine.rerank_capacity]
    named = _named_docs(query, docs, allowed)
    for key in named:
        candidates = (
            store.vector_index.search(query_vector, {key}, _NAMED_CANDIDATES)
            + store.keyword_search(query, {key}, _NAMED_CANDIDATES)
            + candidates
        )
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        return []

    chunks = store.get_chunks(candidates)
    candidates = [chunk_id for chunk_id in candidates if chunk_id in chunks]
    scores = _score(query, candidates, chunks, docs)

    # A narrow scope means the reader wants those papers read closely, so no per-paper cap.
    cap = k if allowed is not None and len(allowed) <= 2 else per_doc
    ranked = sorted(zip(candidates, scores), key=lambda pair: float(pair[1]), reverse=True)

    if allow_overview and not named and is_about_whole_library(query) and (not ranked or float(ranked[0][1]) < _WEAK_MATCH):
        return _read_across(query, query_vector, docs, allowed, k)

    hits: list[dict] = []
    per_doc_count: dict[str, int] = {}
    taken: set[int] = set()

    def take(chunk_id: int, score: float) -> None:
        chunk = chunks[chunk_id]
        doc = docs[chunk["doc_key"]]
        per_doc_count[doc["key"]] = per_doc_count.get(doc["key"], 0) + 1
        taken.add(chunk_id)
        hits.append({**_hit(chunk, doc, score), "named": doc["key"] in named})

    # A paper the question names is included even if its wording doesn't match, so that
    # the model can say what the paper actually covers rather than claim it isn't there.
    for key in named:
        own = [(chunk_id, score) for chunk_id, score in ranked if chunks[chunk_id]["doc_key"] == key]
        for chunk_id, score in own[: min(cap, _NAMED_RESERVED)]:
            take(chunk_id, float(score))

    for chunk_id, score in ranked:
        if len(hits) >= k or float(score) < MIN_SCORE:
            break
        doc_key = chunks[chunk_id]["doc_key"]
        if chunk_id not in taken and per_doc_count.get(doc_key, 0) < cap:
            take(chunk_id, float(score))
    return sorted(hits, key=lambda hit: hit["score"], reverse=True)


def best_in_paper(doc_key: str, question: str, extra_terms: str = "", k: int = 3, skip=None) -> list[dict]:
    """A paper's passages that best answer a question, however well they score.

    Unlike retrieve, nothing is dropped for scoring low: a matrix cell needs the paper's best
    attempt at its question, and the model says "Not stated" when that isn't good enough.
    Extra terms only widen the keyword search, because the reranker scores a bag of words badly.
    skip(doc, chunk) leaves out passages such as related work, unless that's all there is.
    """
    doc = store.get_doc(doc_key)
    if not doc:
        return []
    dense = store.vector_index.search(engine.encode_query(question), {doc_key}, _IN_PAPER_CANDIDATES)
    sparse = store.keyword_search(f"{question} {extra_terms}", {doc_key}, _IN_PAPER_CANDIDATES)
    candidates = reciprocal_rank_fusion([dense, sparse])
    chunks = store.get_chunks(candidates)
    found = [chunk_id for chunk_id in candidates if chunk_id in chunks]
    kept = [chunk_id for chunk_id in found if not (skip and skip(doc, chunks[chunk_id]))]
    candidates = (kept or found)[: engine.rerank_capacity]
    if not candidates:
        return []
    scores = engine.rerank(question, [chunks[chunk_id]["text"] for chunk_id in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda pair: float(pair[1]), reverse=True)[:k]
    return [_hit(chunks[chunk_id], doc, float(score)) for chunk_id, score in ranked]


def is_about_whole_library(question: str) -> bool:
    return bool(_WHOLE_LIBRARY.search(question))


def _named_docs(query: str, docs: dict[str, dict], allowed: set[str] | None) -> list[str]:
    """Papers the question names by first author, as in "the Perry et al. study".

    Passages rarely contain their own authors' names, so ordinary search can't find these.
    A capitalised match avoids mistaking a word for a surname ("long" isn't Long et al.).
    """
    named = []
    for key, doc in docs.items():
        if allowed is not None and key not in allowed:
            continue
        creators = doc.get("creators") or []
        authors = [c for c in creators if c.get("type") == "author"] or creators
        surname = authors[0].get("last", "") if authors else ""
        if len(surname) < 2:
            continue
        escaped = re.escape(surname)
        if re.search(rf"(?<!\w){escaped}(?!\w)", query) or re.search(rf"(?<!\w){escaped}\s+(et al|and|&)", query, re.IGNORECASE):
            named.append(key)
    return named


def _score(query: str, chunk_ids: list[int], chunks: dict[int, dict], docs: dict[str, dict]):
    passages = [f"{docs[chunks[c]['doc_key']]['title']}\n{chunks[c]['text']}" for c in chunk_ids]
    return engine.rerank(query, passages)


def _hit(chunk: dict, doc: dict, score: float, overview: bool = False) -> dict:
    return {
        "chunk_id": chunk["id"],
        "doc_key": doc["key"],
        "text": chunk["text"],
        "page": chunk["page"],
        "page_end": chunk["page_end"],
        "section": chunk["section"],
        "score": round(score, 4),
        "title": doc["title"],
        "cite": cite.short_cite(doc),
        "apa": cite.apa(doc),
        "year": doc.get("year"),
        "venue": doc.get("venue"),
        "has_pdf": bool(doc.get("pdf_path")),
        "pages": doc.get("pages") or 0,
        "overview": overview,
    }


def _read_across(query: str, query_vector, docs: dict[str, dict], allowed: set[str] | None, k: int) -> list[dict]:
    """Answer a question about the library as a whole, such as "what are the main findings?".

    It has no topic to search for, so no passage matches it well. Instead, take each
    article's most relevant passages, favouring the sections where authors sum up.
    """
    nearest = store.vector_index.search(query_vector, allowed, _ACROSS_NEAREST)
    nearest_chunks = store.get_chunks(nearest)
    per_doc: dict[str, list[int]] = {}
    for chunk_id in nearest:
        doc_key = nearest_chunks[chunk_id]["doc_key"]
        if len(per_doc.setdefault(doc_key, [])) < 3:
            per_doc[doc_key].append(chunk_id)
    chosen = list(per_doc)[:k]
    if not chosen:
        return []

    candidates: list[int] = []
    for doc_key in chosen:
        summing_up = [row["id"] for row in store.chunks_of(doc_key) if _SUMMING_UP.search(row["section"])]
        candidates.extend(dict.fromkeys(per_doc[doc_key] + summing_up[:2] + summing_up[-2:]))
    candidates = candidates[: engine.rerank_capacity * 2]

    chunks = store.get_chunks(candidates)
    scores = _score(query, candidates, chunks, docs)
    ranked = sorted(zip(candidates, scores), key=lambda pair: float(pair[1]), reverse=True)
    cap = min(3, max(1, k // len(chosen)))  # With few articles, there's room for more from each.
    counts: dict[str, int] = {}
    hits: list[dict] = []
    for chunk_id, score in ranked:
        chunk = chunks[chunk_id]
        if counts.get(chunk["doc_key"], 0) < cap and len(hits) < k:
            counts[chunk["doc_key"]] = counts.get(chunk["doc_key"], 0) + 1
            hits.append(_hit(chunk, docs[chunk["doc_key"]], float(score), overview=True))
    return hits
