"""Model loading, with fake models: loading must never overlap, and search must stay in full precision."""

import threading
import time

import sentence_transformers
import torch

from folio import models


class Recorder:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.most_at_once = 0
        self.calls = []


def fake_model(recorder):
    class Fake:
        def __init__(self, *args, **kwargs):
            recorder.calls.append(kwargs)
            with recorder.lock:
                recorder.active += 1
                recorder.most_at_once = max(recorder.most_at_once, recorder.active)
            time.sleep(0.05)
            with recorder.lock:
                recorder.active -= 1
            self.max_seq_length = None

    return Fake


def test_models_never_load_at_the_same_time(monkeypatch):
    """While a model loads, the library changes the whole program's default precision, so loads must not overlap."""
    recorder = Recorder()
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", fake_model(recorder))
    monkeypatch.setattr(sentence_transformers, "CrossEncoder", fake_model(recorder))
    engine = models._Engine()
    engine._gpu_ok = True  # So indexing loads its own copy, whatever the computer running the tests has.
    threads = [threading.Thread(target=load) for load in (engine._load_query_embedder, engine._load_reranker, engine._load_bulk)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(recorder.calls) == 3
    assert recorder.most_at_once == 1


def test_the_search_embedder_always_asks_for_full_precision(monkeypatch):
    recorder = Recorder()
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", fake_model(recorder))
    models._Engine()._load_query_embedder()
    assert recorder.calls[0]["model_kwargs"]["dtype"] == torch.float32
