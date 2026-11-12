"""Start Folio: `python -m folio`. Add --no-browser to skip opening a browser tab."""

from __future__ import annotations

import logging
import sys
import threading
import time
import webbrowser

import httpx
import uvicorn

from . import config


def _is_running(url: str) -> bool:
    try:
        return httpx.get(url, timeout=1.0).status_code == 200
    except httpx.HTTPError:
        return False


def _open_when_ready(url: str) -> None:
    for _ in range(240):
        if _is_running(url):
            webbrowser.open(url)
            return
        time.sleep(0.25)


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    url = f"http://{config.HOST}:{config.PORT}"
    show_browser = "--no-browser" not in sys.argv
    if _is_running(url):
        print(f"Folio is already running at {url}")
        if show_browser:
            webbrowser.open(url)
        return
    if show_browser:
        threading.Thread(target=_open_when_ready, args=(url,), daemon=True).start()
    print(f"Folio is running at {url}\nClose this window to stop it.")
    uvicorn.run("folio.app:app", host=config.HOST, port=config.PORT, log_level="warning")


if __name__ == "__main__":
    main()
