"""Find a source: search the library for passages that back up a sentence from a draft.

This shows the reader the evidence rather than writing anything. A model sorts each passage
into supports, disagrees or related, and has to quote the sentence its verdict rests on. Exact
checks then overrule it, because wrongly calling a passage support is the mistake that costs
the student most. A local model does this when one is installed, so nothing leaves the PC.

Measured on 82 passages against a hand-read key: asking for one word gave 50% of "supports"
labels right, and quoting first with these checks gave 81%, finding 17 of 18 real ones.
"""

from __future__ import annotations

import logging
import re

from . import config, llm
from .answer import _numbers_in, _squash, generation_lock, helper_model
from .models import engine, gpu_lock
from .search import Scope, retrieve

log = logging.getLogger("folio.claims")

_PROMPT = """A student wrote this sentence in a thesis draft:
"{sentence}"

Here is a passage from {cite}:
{passage}

Can the student cite this passage for that sentence? Be strict. Reply in exactly this format:
QUOTE: the one sentence from the passage closest to the student's claim, copied word for word
VERDICT: one of the words below

SUPPORTS: the quoted sentence states the student's whole claim as a finding or fact, even in different words. A research question, an aim, a worry, or a claim about a narrower or different thing does not count.
CONTRADICTS: the quoted sentence states a finding or fact that can't be true if the student's claim is true. Merely being on the same topic, or being about risks in general, does not count.
RELATED: on the same topic, but neither of the above.
UNRELATED: about something else."""

_QUOTE = re.compile(r"QUOTE:[ \t]*(.+)")
_VERDICT = re.compile(r"VERDICT:\W*([A-Za-z]+)")
_VERDICTS = {"SUPPORTS": "supports", "CONTRADICTS": "disagrees", "UNRELATED": "unrelated"}
_REPLY_TOKENS = 160
# Scores measured on the same 82 passages: nothing that really supported or opposed a sentence fell below these.
_MIN_PASSAGE_SCORE = 0.1
_MIN_QUOTE_SCORE = 0.1
# Citing a paper for what it says about another paper is a classic mistake, so point it out.
_REPORTS_OTHER_WORK = re.compile(
    r"\[\d+|\b[A-Z][\w-]+ et al\.|\([A-Z][^()]{0,60}\b(?:19|20)\d{2}[a-z]?\)|\b[Tt]hey (?:found|showed|reported|observed)\b"
)

_ORDER = {"supports": 0, "disagrees": 1, "related": 2}
_MAX_SENTENCES = 40


def _judge(model: str, sentence: str, hit: dict) -> tuple[str, str | None]:
    """The model's verdict on a passage, and the sentence of it that the verdict rests on."""
    prompt = _PROMPT.format(sentence=sentence, cite=hit["cite"], passage=hit["text"])
    try:
        reply = llm.chat_once(model, [{"role": "user", "content": prompt}], num_predict=_REPLY_TOKENS)
    except llm.Declined:
        return "related", None  # A safety filter refusing says nothing about the passage.
    quote = _QUOTE.search(reply)
    verdict = _VERDICT.search(reply)
    return (
        _VERDICTS.get(verdict.group(1).upper(), "related") if verdict else "related",
        quote.group(1).strip().strip("\"“”") if quote else None,
    )


def _missing_figures(sentence: str, hit: dict) -> list[str]:
    allowed = _numbers_in(hit["text"]) | {str(hit.get("year"))}
    return sorted(n for n in _numbers_in(sentence) if (len(n) >= 2 or "." in n) and n not in allowed)


def _closest_sentence(sentence: str, passage: str) -> str:
    """The part of a passage worth reading first."""
    parts = [part for part in re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", passage) if len(part.split()) >= 5][:_MAX_SENTENCES]
    if len(parts) < 2:
        return passage[:500]
    return parts[int(engine.rerank(sentence, parts).argmax())]


def _assess(model: str, sentence: str, hit: dict) -> dict | None:
    """Judge one passage and apply the exact checks. None means it isn't worth showing."""
    verdict, quote = _judge(model, sentence, hit)
    if verdict == "unrelated":
        return None
    if not quote or _squash(quote) not in _squash(hit["text"]):
        # A quote that isn't in the passage (models sometimes echo the student) can't carry a verdict.
        verdict = "related"
        quote = _closest_sentence(sentence, hit["text"])
    if quote.rstrip().endswith("?"):
        verdict = "related"  # A question the paper asks isn't a finding.
    notes = []
    if verdict == "supports":
        missing = _missing_figures(sentence, hit)
        if missing:
            verdict = "related"
            notes.append(f"{', '.join(missing)} isn't in this passage, so check it before using it for that figure.")
        elif float(engine.rerank(sentence, [quote])[0]) < _MIN_QUOTE_SCORE:
            verdict = "related"
    if verdict != "related" and _REPORTS_OTHER_WORK.search(quote):
        notes.append("This is reporting another paper's work. Read and cite that original paper if you can.")
    page = f", p. {hit['page']}" if hit["page"] else ""
    return {**hit, "verdict": verdict, "quote": quote, "note": " ".join(notes) or None, "citation": f"({hit['cite']}{page})"}


def find_sources(sentence: str, scope: Scope | None = None) -> dict:
    """Look for passages that back up a sentence, grouped by how they relate to it."""
    sentence = sentence.strip()
    hits = [hit for hit in retrieve(sentence, scope, k=8, per_doc=2, allow_overview=False) if hit["score"] >= _MIN_PASSAGE_SCORE]
    settings = config.load_settings()
    checker = helper_model(settings["llm_model"], settings)
    results = []
    seen = set()
    with generation_lock, gpu_lock:
        engine.release_gpu()
        for hit in hits:
            result = _assess(checker, sentence, hit)
            if result is None:
                continue
            quote_key = (hit["doc_key"], _squash(result["quote"]))  # Passages overlap, so the same line can turn up twice.
            if quote_key not in seen:
                seen.add(quote_key)
                results.append(result)
    results.sort(key=lambda result: (_ORDER[result["verdict"]], -result["score"]))
    return {"sentence": sentence, "results": results, "checked_by": checker}
