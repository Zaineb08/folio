"""Paths and settings. All app data lives in one per-user folder, apart from the code."""

from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
import threading
from pathlib import Path

HOST = "127.0.0.1"
PORT = int(os.environ.get("FOLIO_PORT", "8765"))

# In the home folder rather than AppData: Windows redirects AppData writes made from
# inside a packaged app into that app's private folder, where Folio started any other
# way wouldn't find them.
DATA_DIR = Path(os.environ.get("FOLIO_DATA") or Path.home() / "Folio")
DB_PATH = DATA_DIR / "folio.db"
MODELS_DIR = DATA_DIR / "models"
SETTINGS_PATH = DATA_DIR / "settings.json"

EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
EMBED_DIM = 1024

OLLAMA_URL = os.environ.get("FOLIO_OLLAMA", "http://127.0.0.1:11434")
ZOTERO_URL = os.environ.get("FOLIO_ZOTERO", "http://127.0.0.1:23119")

# The answer model on the PC. The larger one needs a graphics card with about 8 GB of memory.
# Without one, the smaller one fits in 16 GB beside the search models (about 5 GB) and Windows.
LARGE_LLM = "qwen3.5:9b"
SMALL_LLM = "qwen3.5:4b"
DEFAULT_LLM = LARGE_LLM
# Download sizes from Ollama's registry, shown before a download starts.
LLM_DOWNLOAD_GB = {LARGE_LLM: 6.6, SMALL_LLM: 3.3}
_LARGE_LLM_MIN_GPU_GB = 7.5  # Cards sold as 8 GB report a little less.
# Enough for 12 excerpts, recent history and an answer.
NUM_CTX = 12288

_lock = threading.Lock()


def gpu_memory_gb() -> float | None:
    """Memory on the NVIDIA graphics card, or None without one. Asks nvidia-smi, so torch needn't load."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        output = subprocess.run(
            [exe, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout.split()
        return max(float(value) for value in output) / 1024 if output else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


@functools.lru_cache(maxsize=1)
def recommended_model() -> str:
    """The answer model this PC can run at a usable speed."""
    gpu = gpu_memory_gb()
    return LARGE_LLM if gpu and gpu >= _LARGE_LLM_MIN_GPU_GB else SMALL_LLM


def stored_settings() -> dict:
    """Only what the student chose, without defaults."""
    with _lock:
        try:
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}


def load_settings() -> dict:
    model = recommended_model()
    return {"llm_model": model, "local_model": model, "zotero_dir": None, **stored_settings()}


def save_settings(**changes) -> dict:
    """Store what the student chose. Defaults aren't stored, so they keep following the PC."""
    current = {**stored_settings(), **changes}
    with _lock:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = SETTINGS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(current, indent=2), encoding="utf-8")
        tmp.replace(SETTINGS_PATH)
    return load_settings()
