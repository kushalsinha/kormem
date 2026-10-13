"""``_meta/lessons.md``: what the user told the assistant to remember about working with them.

Lessons are instructions and corrections ("use bun, not npm", "reply in
short bullets", "never commit without running the tests"). The vault writer
extracts them as ``lesson`` items; this module files them by scope:

    ## Global              applies everywhere
    ## Project: kormem     applies only in that project
    ## Retired             replaced lessons, kept with their dates

A new lesson that replaces an old one retires it. ``lessonsFor`` returns the
global lessons plus one project's, small enough to load at the start of
every session.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

from kormem.vault.note import Note
from kormem.vault.note_sections import addBullet, parseSections, renderSections, supersede

LESSONS_REF = "_meta/lessons.md"
GLOBAL = "Global"
RETIRED = "Retired"
MAX_CHARS = int(os.environ.get("KORMEM_LESSONS_MAX_CHARS", "8000"))
PREAMBLE = "# Lessons\n\nWhat the user asked the assistant to remember about working with them."


def sectionFor(scope: Optional[str]) -> str:
    scope = (scope or "").strip()
    if not scope or scope.lower() in ("global", "all", "everywhere"):
        return GLOBAL
    return f"Project: {scope}"


def _load(vault) -> Tuple[Note, dict]:
    note = vault.readNote(LESSONS_REF)
    if note is None:
        note = Note.create(title="lessons", para="meta")
        note.kormemId = f"meta-lessons-{vault.namespaceId}"
    _, secs = parseSections(note.body)
    return note, {k: v for k, v in secs.items() if v}


def addLessons(vault, lessons: List[Tuple[str, Optional[str], str, Optional[str]]], source: str) -> int:
    """File ``(text, scope, date, supersedes)`` lessons; returns how many were new."""
    if not lessons:
        return 0
    note, secs = _load(vault)
    added = 0
    for text, scope, date, replaces in lessons:
        section = sectionFor(scope)
        if replaces:
            supersede(secs, replaces, date, fromSection=section, toSection=RETIRED)
        added += addBullet(secs, section, f"{text.rstrip('. ')} (since {date})", source)
    ordered = {GLOBAL: secs.pop(GLOBAL, [])}
    ordered.update({k: secs[k] for k in sorted(secs) if k != RETIRED})
    ordered[RETIRED] = secs.get(RETIRED, [])
    note.body = renderSections(PREAMBLE, ordered)
    vault.writeNote(note)
    return added


def lessonsFor(vault, project: Optional[str] = None, maxChars: int = MAX_CHARS) -> str:
    """Global lessons plus ``project``'s, without source links, as a short markdown list.

    It is loaded into every session, so it is capped at ``maxChars`` (about 2k tokens): the
    project's lessons come first, then global ones, newest first within each. Retired lessons
    never load.
    """
    note = vault.readNote(LESSONS_REF)
    if note is None:
        return ""
    _, secs = parseSections(note.body)
    wanted = ([sectionFor(project)] if project else []) + [GLOBAL]
    kept: dict = {}
    used = 0
    for name in wanted:
        lines = [re.sub(r"(\s*\[\[[^\]]+\]\])+\s*$", "", l) for l in secs.get(name) or []]
        order = sorted(range(len(lines)), key=lambda i: _since(lines[i]), reverse=True)
        for i in order:
            if used + len(lines[i]) + 1 > maxChars:
                continue
            kept.setdefault(name, set()).add(i)
            used += len(lines[i]) + 1
        if name in kept:
            kept[name] = [lines[i] for i in sorted(kept[name])]
    parts = [f"{name}:\n" + "\n".join(kept[name]) for name in (GLOBAL, sectionFor(project) if project else None)
             if name in kept]
    return "\n\n".join(parts)


def _since(line: str) -> str:
    m = re.search(r"\(since (\d{4}-\d{2}-\d{2})\)", line)
    return m.group(1) if m else ""
