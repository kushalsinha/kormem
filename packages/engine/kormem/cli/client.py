"""HTTP client for the local engine. Starts the engine in the background when it isn't running.

The engine is a long-lived local server (``kormem serve``) that holds the
index, runs background extraction and daily maintenance. Every CLI command,
the MCP server and the Claude Code hook talk to it over HTTP, so models load
once and work continues between commands.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import httpx

from kormem import __version__
from kormem.core.config import apiKeysSet, loadConfig
from kormem.core.models import snapshot as modelSnapshot

START_TIMEOUT_SECONDS = 90
_startLock = threading.Lock()


class EngineError(RuntimeError):
    def __init__(self, message: str, status: int = 0, detail: Any = None):
        super().__init__(message)
        self.status = status
        self.detail = detail


def kormemHome() -> Path:
    raw = os.environ.get("KORMEM_HOME", "").strip()
    return Path(raw).expanduser().resolve() if raw else Path.home() / ".kormem"


def _configPort() -> Optional[int]:
    path = kormemHome() / "config.yaml"
    if not path.exists():
        return None
    try:
        import yaml
        return int((yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("enginePort") or 0) or None
    except Exception:
        return None


def baseUrl() -> str:
    if os.environ.get("KORMEM_ENGINE_URL"):
        return os.environ["KORMEM_ENGINE_URL"].rstrip("/")
    port = os.environ.get("KORMEM_ENGINE_PORT") or _configPort() or 8787
    return f"http://127.0.0.1:{port}"


def _isLocal(url: str) -> bool:
    return (urlparse(url).hostname or "") in ("127.0.0.1", "localhost", "::1")


def _keysChanged(health: dict) -> bool:
    """A key was added or removed since the running engine started (old engines report none)."""
    return "keys" in health and set(apiKeysSet()) != set(health["keys"])


def _modelsChanged(health: dict) -> bool:
    """A model was changed (``kormem model set``) since the running engine started."""
    return "models" in health and modelSnapshot(loadConfig()) != health["models"]


def logPath() -> Path:
    return kormemHome() / "logs" / "engine.log"


class Engine:
    def __init__(self, namespace: str = "main"):
        self.namespace = namespace
        self.url = baseUrl()
        self._ready = False

    # ------------------------------------------------------------ lifecycle

    def health(self) -> Optional[dict]:
        try:
            r = httpx.get(f"{self.url}/health", timeout=1.5)
            return r.json() if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            return None

    def ensure(self) -> None:
        """Start the engine if needed; restart a local one left running by another kormem version."""
        if self._ready:
            return
        with _startLock:
            self._ensureLocked()

    def _ensureLocked(self) -> None:
        if self._ready:
            return
        h = self.health()
        if h and _isLocal(self.url) and (h.get("version") != __version__ or _keysChanged(h) or _modelsChanged(h)):
            self.stop()  # another version, or the API keys or models changed since it started
            h = None
        if not h:
            if not _isLocal(self.url):
                raise EngineError(f"Engine not reachable at {self.url}")
            self._start()
        self._ready = True

    def _start(self) -> None:
        port = str(urlparse(self.url).port or 8787)
        log = logPath()
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("ab") as out:
            subprocess.Popen(
                [sys.executable, "-m", "kormem.cli", "serve", "--port", port],
                stdout=out, stderr=out, stdin=subprocess.DEVNULL, start_new_session=True,
                env={**os.environ, "KORMEM_HOME": str(kormemHome())},
            )
        deadline = time.time() + START_TIMEOUT_SECONDS
        while time.time() < deadline:
            if self.health():
                return
            time.sleep(0.25)
        raise EngineError(f"The engine did not start within {START_TIMEOUT_SECONDS}s. See {log}")

    def stop(self) -> bool:
        if not self.health():
            return False
        try:
            httpx.post(f"{self.url}/shutdown", timeout=3)
        except httpx.HTTPError:
            pass
        for _ in range(40):
            if not self.health():
                return True
            time.sleep(0.25)
        return False

    # ------------------------------------------------------------- requests

    def _check(self, r: httpx.Response, route: str) -> Any:
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail")
            except ValueError:
                detail = r.text
            msg = detail.get("error") if isinstance(detail, dict) else detail
            raise EngineError(str(msg or f"{route} failed ({r.status_code})"), r.status_code, detail)
        return r.json()

    def get(self, route: str, /, **params: Any) -> Any:
        self.ensure()
        params = {k: v for k, v in params.items() if v is not None}
        return self._check(httpx.get(f"{self.url}{route}", params=params, timeout=600), route)

    def post(self, route: str, body: Optional[dict] = None, /, **params: Any) -> Any:
        self.ensure()
        params = {k: v for k, v in params.items() if v is not None}
        return self._check(httpx.post(f"{self.url}{route}", json=body, params=params, timeout=None), route)
