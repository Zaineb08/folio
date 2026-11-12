"""Fitting Folio to the PC: which answer model it starts with, and its memory without a graphics card."""

import numpy as np
import pytest
import sentence_transformers
import torch

from folio import app, config, models


def suited_model(monkeypatch, gpu_gb):
    monkeypatch.setattr(config, "gpu_memory_gb", lambda: gpu_gb)
    config.recommended_model.cache_clear()
    try:
        return config.recommended_model()
    finally:
        config.recommended_model.cache_clear()


def test_a_graphics_card_of_8_gb_gets_the_larger_model(monkeypatch):
    assert suited_model(monkeypatch, 7.6) == config.LARGE_LLM  # Cards sold as 8 GB report a little less.
    assert suited_model(monkeypatch, 24.0) == config.LARGE_LLM


def test_a_pc_without_one_gets_the_smaller_model(monkeypatch):
    assert suited_model(monkeypatch, None) == config.SMALL_LLM
    assert suited_model(monkeypatch, 4.0) == config.SMALL_LLM


@pytest.fixture
def small_pc(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(config, "recommended_model", lambda: config.SMALL_LLM)


def test_the_suited_model_is_the_default(small_pc):
    app.settle_answer_model([config.LARGE_LLM, config.SMALL_LLM])
    assert config.stored_settings() == {}
    assert config.load_settings()["llm_model"] == config.SMALL_LLM


def test_the_other_model_is_used_when_only_it_is_installed(small_pc):
    """Someone who downloaded the larger model on a small PC still gets answers."""
    app.settle_answer_model([config.LARGE_LLM])
    settings = config.load_settings()
    assert settings["llm_model"] == settings["local_model"] == config.LARGE_LLM


def test_a_model_the_student_chose_is_kept(small_pc):
    config.save_settings(llm_model="mistral:7b", local_model="mistral:7b")
    app.settle_answer_model(["mistral:7b", config.LARGE_LLM])
    assert config.load_settings()["llm_model"] == "mistral:7b"


def test_saving_another_setting_leaves_the_model_free_to_follow_the_pc(small_pc):
    config.save_settings(zotero_dir="D:/Zotero")
    assert config.stored_settings() == {"zotero_dir": "D:/Zotero"}
    assert config.load_settings()["llm_model"] == config.SMALL_LLM


def test_without_a_gpu_indexing_reuses_the_question_embedder(monkeypatch):
    """A second copy of the embedding model would cost 2 GB, too much beside a model on a 16 GB PC."""
    built = []

    class FakeEmbedder:
        def __init__(self, *args, **kwargs):
            built.append(kwargs)
            self.max_seq_length = None

        def encode(self, texts, **kwargs):
            return np.ones((len(texts), config.EMBED_DIM), dtype=np.float32)

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeEmbedder)
    engine = models._Engine()
    engine._gpu_ok = False
    vectors = engine.embed_documents(["a passage", "another passage"])
    assert vectors.shape == (2, config.EMBED_DIM)
    assert len(built) == 1
    assert engine._load_bulk() is engine._query_embedder
    assert not engine.gpu_in_use, "indexing on the CPU mustn't count as holding the GPU"


def test_an_old_graphics_card_falls_back_to_the_cpu(monkeypatch):
    """Cards before the GTX 16 series show up as available, then fail on first use with this PyTorch."""
    def no_kernel(*args, **kwargs):
        raise RuntimeError("CUDA error: no kernel image is available for execution on the device")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "ones", no_kernel)
    assert models._Engine()._gpu_usable() is False


def test_no_graphics_card_means_the_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert models._Engine()._gpu_usable() is False
