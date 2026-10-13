"""Token and call accounting for every paid API call.

Each call appends one JSON line to ``$KORMEM_HOME/logs/usage.jsonl``:
``{"ts", "purpose", "model", "input", "cachedInput", "output", "reasoning"}``
(embedding calls record ``texts`` and ``chars`` instead). ``summarize`` rolls
the log up per purpose and model so cost can be checked against a price list.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

from kormem.core.config import getKormemHome

_lock = threading.Lock()


def _logPath() -> Path:
    path = getKormemHome() / "logs" / "usage.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def record(purpose: str, model: str, usage: Optional[Dict[str, Any]] = None, **extra: Any) -> None:
    """Log one call. ``usage`` is an OpenAI-style usage object (chat or responses API)."""
    u = usage or {}
    inDetails = u.get("prompt_tokens_details") or u.get("input_tokens_details") or {}
    outDetails = u.get("completion_tokens_details") or u.get("output_tokens_details") or {}
    row = {
        "ts": time.time(), "purpose": purpose, "model": model,
        "input": u.get("prompt_tokens", u.get("input_tokens", 0)) or 0,
        "cachedInput": inDetails.get("cached_tokens", 0) or 0,
        "output": u.get("completion_tokens", u.get("output_tokens", 0)) or 0,
        "reasoning": outDetails.get("reasoning_tokens", 0) or 0,
        **extra,
    }
    try:
        with _lock, _logPath().open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except OSError:
        pass


def summarize(since: float = 0.0, path: Optional[Path] = None) -> Dict[str, Dict[str, int]]:
    """Totals per ``purpose|model`` for calls logged after ``since`` (epoch seconds)."""
    totals: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    p = path or _logPath()
    if not p.exists():
        return {}
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("ts", 0) < since:
            continue
        t = totals[f"{row.get('purpose')}|{row.get('model')}"]
        t["calls"] += 1
        for k in ("input", "cachedInput", "output", "reasoning", "texts", "chars"):
            t[k] += int(row.get(k) or 0)
    return {k: dict(v) for k, v in totals.items()}
