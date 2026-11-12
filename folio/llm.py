"""Answer models: Ollama, which runs on the PC, and optionally Claude."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Iterator

import httpx

from . import claude, config
from .errors import Declined, LLMError  # noqa: F401  Callers catch these from here.

_TIMEOUT = httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=5.0)
_KEEP_ALIVE = "30m"


def _executable() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found
    for path in (
        config.DATA_DIR / "ollama" / "ollama.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
    ):
        if path.exists():
            return str(path)
    return None


def is_up() -> bool:
    try:
        return httpx.get(f"{config.OLLAMA_URL}/api/version", timeout=1.0).status_code == 200
    except httpx.HTTPError:
        return False


def ensure_running(timeout: float = 20.0) -> bool:
    if is_up():
        return True
    exe = _executable()
    if not exe:
        return False
    subprocess.Popen(
        [exe, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_up():
            return True
        time.sleep(0.3)
    return False


def list_models() -> list[str]:
    try:
        response = httpx.get(f"{config.OLLAMA_URL}/api/tags", timeout=3.0)
        response.raise_for_status()
    except httpx.HTTPError:
        return []
    return sorted(model["name"] for model in response.json().get("models", []))


def pull(model: str, on_progress: Callable[[int, int], None]) -> None:
    """Download a model through Ollama, reporting bytes done and the total as it goes."""
    layers: dict[str, tuple[int, int]] = {}
    try:
        with httpx.stream(
            "POST", f"{config.OLLAMA_URL}/api/pull", json={"model": model, "stream": True},
            timeout=httpx.Timeout(connect=5.0, read=600.0, write=30.0, pool=5.0),
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.strip():
                    continue
                message = json.loads(line)
                if message.get("error"):
                    raise LLMError(f"Ollama couldn't download {model}: {message['error']}")
                if message.get("digest") and message.get("total"):
                    layers[message["digest"]] = (message.get("completed", 0), message["total"])
                    on_progress(sum(done for done, _ in layers.values()), sum(total for _, total in layers.values()))
    except httpx.HTTPError as exc:
        raise LLMError(f"Couldn't download {model}. Check the internet connection and that Ollama is running.") from exc
    if model not in list_models():
        raise LLMError(f"Ollama finished, but {model} isn't installed. Try again.")


def loaded_models() -> list[dict]:
    try:
        response = httpx.get(f"{config.OLLAMA_URL}/api/ps", timeout=2.0)
        response.raise_for_status()
    except httpx.HTTPError:
        return []
    return response.json().get("models", [])


def preload(model: str) -> None:
    """Load the model now, so the first question doesn't wait for it."""
    try:
        httpx.post(
            f"{config.OLLAMA_URL}/api/generate",
            json={"model": model, "keep_alive": _KEEP_ALIVE, "options": {"num_ctx": config.NUM_CTX}},
            timeout=300.0,
        )
    except httpx.HTTPError:
        pass  # The first question will load it instead, and report any real problem.


def unload_all() -> None:
    """Free the GPU, for example before bulk indexing."""
    for model in loaded_models():
        try:
            httpx.post(
                f"{config.OLLAMA_URL}/api/generate",
                json={"model": model["name"], "keep_alive": 0},
                timeout=30.0,
            )
        except httpx.HTTPError:
            pass


def chat_stream(
    model: str,
    messages: list[dict],
    think: bool = False,
    num_ctx: int = config.NUM_CTX,
    num_predict: int = 2048,
    temperature: float = 0.2,
    on_finish: Callable[[dict], None] | None = None,
) -> Iterator[str]:
    """Yield the answer's text as it's generated. on_finish receives usage details, for Claude."""
    if claude.is_claude(model):
        yield from claude.chat_stream(model, messages, on_finish=on_finish)
        return
    if not ensure_running():
        raise LLMError("Ollama isn't running and couldn't be started.")
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "think": think,
        "keep_alive": _KEEP_ALIVE,
        "options": {"num_ctx": num_ctx, "num_predict": num_predict, "temperature": temperature},
    }
    try:
        with httpx.stream("POST", f"{config.OLLAMA_URL}/api/chat", json=payload, timeout=_TIMEOUT) as response:
            if response.status_code == 404:
                raise LLMError(f"The model {model} isn't installed. Install it with: ollama pull {model}")
            if response.status_code == 400 and think is False:
                # Some models reject the "think" option outright.
                response.read()
                payload.pop("think", None)
                yield from _stream_without_think(payload)
                return
            if response.status_code != 200:
                response.read()
                raise LLMError(f"Ollama returned an error: {response.text[:300]}")
            yield from _read_stream(response)
    except httpx.TimeoutException as exc:
        raise LLMError("The model took too long to respond.") from exc
    except httpx.HTTPError as exc:
        raise LLMError(f"Lost the connection to Ollama: {exc}") from exc


def _stream_without_think(payload: dict) -> Iterator[str]:
    with httpx.stream("POST", f"{config.OLLAMA_URL}/api/chat", json=payload, timeout=_TIMEOUT) as response:
        if response.status_code != 200:
            response.read()
            raise LLMError(f"Ollama returned an error: {response.text[:300]}")
        yield from _read_stream(response)


def _read_stream(response: httpx.Response) -> Iterator[str]:
    for line in response.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        if data.get("error"):
            raise LLMError(str(data["error"]))
        text = data.get("message", {}).get("content", "")
        if text:
            yield text
        if data.get("done"):
            return


def chat_once(model: str, messages: list[dict], num_predict: int = 128) -> str:
    if claude.is_claude(model):
        return claude.chat_once(model, messages)
    return "".join(chat_stream(model, messages, num_predict=num_predict, temperature=0.0)).strip()
