"""The local search models, and how they share the PC's hardware.

Reranking is many times faster on a GPU than on the CPU, while embedding one question is
quick on the CPU, so the query embedder stays there and leaves the GPU for the answer model.

The reranker lives on the GPU when there is one, loaded before the answer model so
that Ollama sizes itself around it. Bulk embedding borrows the GPU only while indexing,
one article at a time, through `gpu_lock`, which answering also takes. Without a GPU,
indexing reuses the query embedder rather than loading a second 2 GB copy, so Folio fits
in about 5 GB of memory beside a small answer model on a 16 GB PC.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

import numpy as np

from . import config

log = logging.getLogger("folio.models")

_IGNORE = [
    "onnx/*", "*.onnx", "*.onnx_data", "imgs/*", "colbert_linear.pt", "sparse_linear.pt",
    "*.msgpack", "*.h5", "*.ot", "*.md", "*.jpg", "*.png", ".gitattributes",
]
_CPU_THREADS = 8
_EMBED_MAX_TOKENS = 640
_RERANK_MAX_TOKENS = 512

gpu_lock = threading.RLock()


def model_path(name: str) -> Path:
    return config.MODELS_DIR / name.split("/")[-1]


def models_ready() -> bool:
    for name in (config.EMBED_MODEL, config.RERANK_MODEL):
        path = model_path(name)
        if not (path / "config.json").exists():
            return False
        if not any(path.glob("*.safetensors")) and not (path / "pytorch_model.bin").exists():
            return False
    return True


def download_models(report: Callable[[str], None] = lambda _: None) -> None:
    from huggingface_hub import HfApi, snapshot_download

    for name in (config.EMBED_MODEL, config.RERANK_MODEL):
        report(f"Downloading {name}")
        files = HfApi().list_repo_files(name)
        ignore = list(_IGNORE)
        if any(f.endswith(".safetensors") for f in files):
            ignore.append("*.bin")  # Prefer safetensors, which can't execute code when loaded.
        snapshot_download(repo_id=name, local_dir=model_path(name), ignore_patterns=ignore)


class _Engine:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._query_embedder = None
        self._reranker = None
        self._reranker_on_gpu = False
        self._bulk_embedder = None
        self._gpu_ok: bool | None = None

    def _gpu_usable(self) -> bool:
        """Whether PyTorch can really compute on the GPU.

        This PyTorch build only has code for NVIDIA cards from the GTX 16 and RTX 20 series on.
        Older ones, such as a GTX 1060, still show up as available, then fail on first use,
        so try a tiny sum first and use the CPU if it fails.
        """
        if self._gpu_ok is None:
            import torch

            ok = False
            if torch.cuda.is_available():
                try:
                    ok = float((torch.ones(4, device="cuda") * 2).sum()) == 8.0
                except Exception:  # noqa: BLE001  Whatever the failure, the CPU is the safe place.
                    ok = False
                if not ok:
                    log.warning("PyTorch can't use this graphics card, so search runs on the CPU and is slower.")
            self._gpu_ok = ok
        return self._gpu_ok

    # -------------------------------------------------------------- per-question

    def _load_query_embedder(self) -> None:
        with self._lock:
            if self._query_embedder is None:
                import torch
                from sentence_transformers import SentenceTransformer

                torch.set_num_threads(_CPU_THREADS)
                # Explicit, so it can never pick up half precision from another model being loaded.
                model = SentenceTransformer(str(model_path(config.EMBED_MODEL)), device="cpu", model_kwargs={"dtype": torch.float32})
                model.max_seq_length = _EMBED_MAX_TOKENS
                self._query_embedder = model

    def _load_reranker(self, force_cpu: bool = False) -> None:
        with self._lock:
            if self._reranker is not None and not force_cpu:
                return
            import torch
            from sentence_transformers import CrossEncoder

            path = str(model_path(config.RERANK_MODEL))
            self._reranker = None
            if self._gpu_usable() and not force_cpu:
                try:
                    self._reranker = CrossEncoder(
                        path, device="cuda", max_length=_RERANK_MAX_TOKENS, model_kwargs={"dtype": torch.float16}
                    )
                    self._reranker_on_gpu = True
                    return
                except torch.cuda.OutOfMemoryError:
                    log.warning("The GPU is full, so search will run on the CPU and be much slower.")
                    torch.cuda.empty_cache()
            self._reranker = CrossEncoder(path, device="cpu", max_length=_RERANK_MAX_TOKENS, model_kwargs={"dtype": torch.float32})
            self._reranker_on_gpu = False

    def preload(self) -> None:
        if models_ready():
            self._load_reranker()
            self._load_query_embedder()

    @property
    def rerank_capacity(self) -> int:
        """How many passages one search can afford to rerank: on a CPU it's about 0.7 s each."""
        self._load_reranker()
        return 40 if self._reranker_on_gpu else 12

    def encode_query(self, text: str) -> np.ndarray:
        self._load_query_embedder()
        with self._lock:
            vector = self._query_embedder.encode(
                [text], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
            )[0]
        return vector.astype(np.float32)

    def rerank(self, query: str, passages: list[str]) -> np.ndarray:
        """Score each passage against the query, from 0 (unrelated) to 1 (answers it)."""
        if not passages:
            return np.empty(0, dtype=np.float32)
        import torch

        self._load_reranker()
        pairs = [(query, passage) for passage in passages]
        with self._lock:
            try:
                scores = self._reranker.predict(pairs, batch_size=16, show_progress_bar=False)
            except torch.cuda.OutOfMemoryError:
                # Another program filled the GPU. Slower, but it still answers.
                log.warning("The GPU is full, so search has moved to the CPU.")
                self._load_reranker(force_cpu=True)
                torch.cuda.empty_cache()
                scores = self._reranker.predict(pairs, batch_size=16, show_progress_bar=False)
        return np.asarray(scores, dtype=np.float32)

    # ---------------------------------------------------------- bulk indexing, GPU

    @property
    def gpu_in_use(self) -> bool:
        return self._bulk_embedder is not None and self._bulk_embedder is not self._query_embedder

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed passages for indexing. Call while holding gpu_lock."""
        if not texts:
            return np.empty((0, config.EMBED_DIM), dtype=np.float32)
        import torch

        options = {"normalize_embeddings": True, "convert_to_numpy": True, "show_progress_bar": False}
        try:
            bulk = self._load_bulk()
            if bulk is self._query_embedder:
                with self._lock:  # Shared with search, which embeds questions with the same copy.
                    vectors = bulk.encode(texts, batch_size=16, **options)
            else:
                vectors = bulk.encode(texts, batch_size=32, **options)
        except torch.cuda.OutOfMemoryError:
            # Another program is using the GPU. Slower, but it still finishes.
            log.warning("The GPU is out of memory, so this article is being indexed on the CPU.")
            self.release_gpu()
            self._load_query_embedder()
            with self._lock:
                vectors = self._query_embedder.encode(texts, batch_size=16, **options)
        return vectors.astype(np.float32)

    def _load_bulk(self):
        # Under the same lock as the other models: while one loads, the library switches the whole
        # program's default precision, so two loading at once could leave every later one in half precision.
        with self._lock:
            if self._bulk_embedder is not None:
                return self._bulk_embedder
            import torch
            from sentence_transformers import SentenceTransformer

            if not self._gpu_usable():
                log.warning("No usable GPU: indexing on the CPU, which is much slower.")
                self._load_query_embedder()  # The same model, already on the CPU: a second copy would cost 2 GB.
                self._bulk_embedder = self._query_embedder
                return self._bulk_embedder
            path = str(model_path(config.EMBED_MODEL))
            model = SentenceTransformer(path, device="cuda", model_kwargs={"dtype": torch.float16})
            model.max_seq_length = _EMBED_MAX_TOKENS
            self._bulk_embedder = model
            return model

    def release_gpu(self) -> None:
        """Hand the GPU back. Call while holding gpu_lock."""
        if self._bulk_embedder is None:
            return
        self._bulk_embedder = None
        import gc

        import torch

        gc.collect()
        if self._gpu_ok:
            torch.cuda.empty_cache()


engine = _Engine()
