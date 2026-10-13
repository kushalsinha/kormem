"""Section-structured note bodies.

Topic notes have fixed sections, in this order:

    ## Current facts        what is true now (latest value wins)
    ## Earlier history      condensed older timeline and assistant bullets (maintenance)
    ## Timeline             dated episodes, oldest first; superseded facts land here
    ## How-to               procedures
    ## From the assistant   recommendations, lists and specifics the assistant gave
    ## Related              [[links]] to other notes

Every bullet ends with the ``[[source]]`` links it came from. All edits are
pure functions on the body text, so they are easy to test and never call a
model.
"""

from __future__ import annotations

import difflib
import re
from typing import Dict, List, Optional, Tuple

SECTIONS = ("Current facts", "Earlier history", "Timeline", "How-to", "From the assistant", "Related")
DUPLICATE_RATIO = 0.9       # bullets this similar (after normalising) are the same fact
SUPERSEDE_RATIO = 0.55      # how close "supersedes" text must be to an existing fact

_LINK_RE = re.compile(r"\s*\[\[[^\]]+\]\]")
_DATE_PREFIX_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\s+")


def parseSections(body: str) -> Tuple[str, Dict[str, List[str]]]:
    """Split a body into (preamble, {section: [bullet lines]})."""
    sections: Dict[str, List[str]] = {name: [] for name in SECTIONS}
    preamble: List[str] = []
    current: Optional[str] = None
    for line in body.splitlines():
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            current = m.group(1).strip()
            sections.setdefault(current, [])
            continue
        if current is None:
            preamble.append(line)
        elif line.strip():
            sections[current].append(line.rstrip())
    return "\n".join(preamble).strip(), sections


def renderSections(preamble: str, sections: Dict[str, List[str]]) -> str:
    parts = [preamble] if preamble else []
    order = list(SECTIONS) + [s for s in sections if s not in SECTIONS]
    for name in order:
        lines = sections.get(name) or []
        if lines:
            parts.append(f"## {name}\n" + "\n".join(lines))
    return "\n\n".join(parts).strip() + "\n"


def _norm(text: str) -> str:
    text = _LINK_RE.sub("", text.lstrip("- ").strip())
    text = _DATE_PREFIX_RE.sub("", text)
    return re.sub(r"\W+", " ", text.lower()).strip()


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _withLink(bullet: str, link: str) -> str:
    return bullet if f"[[{link}]]" in bullet else f"{bullet} [[{link}]]"


def addBullet(sections: Dict[str, List[str]], section: str, text: str, source: str,
              date: Optional[str] = None) -> bool:
    """Add a bullet, or just the source link if the same fact is already there.

    Timeline bullets start with their date and stay sorted. Returns True when
    a new bullet was written.
    """
    lines = sections.setdefault(section, [])
    candidate = f"- {date} {text}" if section == "Timeline" and date else f"- {text}"
    for i, existing in enumerate(lines):
        if _ratio(existing, candidate) >= DUPLICATE_RATIO:
            lines[i] = _withLink(existing, source)
            return False
    lines.append(_withLink(candidate, source))
    if section == "Timeline":
        lines.sort(key=lambda l: (_DATE_PREFIX_RE.match(l.lstrip("- ")) or [None, "9999"])[1])
    return True


def supersede(sections: Dict[str, List[str]], oldText: str, newDate: Optional[str],
              fromSection: str = "Current facts", toSection: str = "Timeline") -> Optional[str]:
    """Move the bullet closest to ``oldText`` from ``fromSection`` to ``toSection``, marked superseded."""
    facts = sections.get(fromSection) or []
    best, score = None, 0.0
    for i, line in enumerate(facts):
        r = _ratio(line, oldText)
        if r > score:
            best, score = i, r
    if best is None or score < SUPERSEDE_RATIO:
        return None
    old = facts.pop(best)
    body = old.lstrip("- ").strip()
    m = re.search(r"\((?:as of|since) (\d{4}-\d{2}-\d{2})\)", body)
    when = m.group(1) if m else ""
    marked = f"- {when} {body} — superseded{f' on {newDate}' if newDate else ''}".replace("-  ", "- ")
    timeline = sections.setdefault(toSection, [])
    timeline.append(marked)
    timeline.sort(key=lambda l: (_DATE_PREFIX_RE.match(l.lstrip("- ")) or [None, "9999"])[1])
    return body


def addRelated(sections: Dict[str, List[str]], title: str) -> None:
    lines = sections.setdefault("Related", [])
    link = f"- [[{title}]]"
    if link not in lines:
        lines.append(link)


def mergeSections(into: Dict[str, List[str]], other: Dict[str, List[str]], skipLink: str = "") -> None:
    """Add ``other``'s bullets to ``into`` (same-fact bullets are not repeated); used to merge notes."""
    for name, lines in other.items():
        target = into.setdefault(name, [])
        for line in lines:
            if skipLink and line.strip() == f"- [[{skipLink}]]":
                continue
            if not any(_ratio(line, x) >= DUPLICATE_RATIO for x in target):
                target.append(line)
        if name == "Timeline":
            target.sort(key=lambda l: (_DATE_PREFIX_RE.match(l.lstrip("- ")) or [None, "9999"])[1])
