"""Claude Code hook: ``kormem hook`` reads the hook event from stdin.

- ``SessionStart``: prints the user's lessons (global + this project's);
  Claude Code adds hook stdout to the session's context.
- ``Stop`` (after every turn) and ``SessionEnd``: sync the session transcript
  into kormem; ``SessionEnd`` also closes the session so its last stretch is
  extracted right away.

The project is the git repository (or directory) name of the session's cwd;
it scopes the lessons extraction finds.

Only text is kept: tool calls and tool results are skipped. Turns carry their
position in the transcript, so re-sending the whole transcript is idempotent.
Errors are swallowed so a memory problem never blocks Claude.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from kormem.cli.client import Engine


def projectName(cwd: Optional[str]) -> Optional[str]:
    if not cwd:
        return None
    try:
        root = subprocess.run(["git", "-C", cwd, "rev-parse", "--show-toplevel"], capture_output=True,
                              text=True, timeout=5).stdout.strip()
        if root:
            return Path(root).name
    except (OSError, subprocess.SubprocessError):
        pass
    return Path(cwd).name


def _textOf(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(str(b.get("text") or "") for b in content
                     if isinstance(b, dict) and b.get("type") == "text").strip()


def transcriptTurns(jsonl: str) -> List[Dict[str, Any]]:
    turns: List[Dict[str, Any]] = []
    for line in jsonl.splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        kind = entry.get("type")
        if kind not in ("user", "assistant") or entry.get("isMeta"):
            continue
        text = _textOf((entry.get("message") or {}).get("content"))
        # Slash-command wrappers and local command output are not conversation.
        if not text or text.startswith("<command-") or text.startswith("<local-command"):
            continue
        turns.append({"role": kind, "content": text, "idx": len(turns), "at": entry.get("timestamp")})
    return turns


def run(namespace: str) -> None:
    try:
        event = json.loads(sys.stdin.read() or "{}")
        engine = Engine(namespace)
        project = projectName(event.get("cwd"))
        name = event.get("hook_event_name")
        if name == "SessionStart":
            lessons = engine.get("/lessons", namespaceId=namespace, project=project)["lessons"]
            if lessons:
                print("Lessons from kormem: what this user has asked you to remember about working with them."
                      f"\n\n{lessons}")
            return
        if not event.get("session_id") or not event.get("transcript_path"):
            return
        turns = transcriptTurns(Path(event["transcript_path"]).read_text(encoding="utf-8"))
        if not turns:
            return
        engine.post("/chat/append", {"namespaceId": namespace, "sessionId": f"claude-code:{event['session_id']}",
                                     "turns": turns, "close": name == "SessionEnd", "project": project})
    except Exception as err:  # never block Claude
        if os.environ.get("KORMEM_DEBUG"):
            print(f"kormem hook: {err}", file=sys.stderr)
