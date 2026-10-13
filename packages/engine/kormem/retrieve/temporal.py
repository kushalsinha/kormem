"""Temporal resolution and reranking for retrieved evidence.

Two jobs:

1. **Resolve** relative time phrases in retrieved text against the date the
   text was written. A user saying "I started it about a month ago" in a
   session dated 2023/05/28 becomes "about a month ago [≈2023-04-28]", so
   downstream models never have to re-derive (or double-apply) the offset.
   See :func:`resolveRelativeDates`.

2. **Rerank** candidates with a question-aware strategy chosen from the
   query text:

    - ``knowledge_update``   : prefer the *most recent* evidence (the user's
                               latest statement supersedes earlier ones).
    - ``temporal_proximity`` : prefer evidence whose session date is close to
                               the implicit "when" in the question ("last
                               month", "two weeks ago", "in March").
    - ``none``               : no temporal re-weighting (default).

For every candidate we extract a *best anchor date* from either the segment's
``metadata.sessionDate``, a ``[Session date: YYYY/MM/DD ...]`` header in the
plain text, or a trailing ``(meaning YYYY-MM-DD)`` on the first bullet line.
The score is multiplied by ``(1 + alpha * recency_or_proximity_factor)`` so a
match that is both semantically relevant *and* temporally aligned ranks above
one that only wins on cosine.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Literal, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)

Strategy = Literal["knowledge_update", "temporal_proximity", "none"]


class _ScoredItem(Protocol):
    """Any object with a mutable ``score`` and readable ``text`` + ``metadata``."""
    score: float
    text: str
    metadata: dict


# ---------------------------------------------------------------------------
# Query classification
# ---------------------------------------------------------------------------

_KNOWLEDGE_UPDATE_PATTERNS = (
    r"\bnow\b", r"\bcurrent(ly)?\b", r"\blatest\b", r"\bmost recent\b",
    r"\bthese days\b", r"\bnowadays\b", r"\bany more\b", r"\banymore\b",
    r"\btoday\b", r"\bright now\b", r"\bat the moment\b",
)
_TEMPORAL_PATTERNS = (
    r"\bhow long\b", r"\bhow many (days|weeks|months|years)\b",
    r"\bwhen (did|do|was|were|will)\b", r"\bago\b",
    r"\blast (week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday|night)\b",
    r"\bin (january|february|march|april|may|june|july|august|september|october|november|december)\b",
    r"\bin \d{4}\b", r"\bby \d{4}\b", r"\bbefore \d{4}\b", r"\bafter \d{4}\b",
    r"\bthis (morning|afternoon|evening|weekend|week|month|year)\b",
    r"\byesterday\b", r"\btomorrow\b",
)


def classifyQuestion(question: str) -> Strategy:
    """Pick a temporal strategy by cheap regex. Falls back to 'none'."""
    q = question.lower()
    if any(re.search(p, q) for p in _KNOWLEDGE_UPDATE_PATTERNS):
        return "knowledge_update"
    if any(re.search(p, q) for p in _TEMPORAL_PATTERNS):
        return "temporal_proximity"
    return "none"


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

_META_DATE_KEYS = ("sessionDate", "session_date", "scrapedAt", "scraped_at", "date")

_SESSION_HEADER_RE = re.compile(
    r"\[Session date:\s*(\d{4})[/-](\d{2})[/-](\d{2})", re.IGNORECASE,
)
_MEANING_DATE_RE = re.compile(
    r"\((?:meaning|estimated)\s+(\d{4})-(\d{2})-(\d{2})\)", re.IGNORECASE,
)
_SIMPLE_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def parseQueryDate(value: Optional[str]) -> Optional[datetime]:
    """Accept ``2023/05/30 (Tue) 23:40``-style or ISO dates, return UTC datetime."""
    if not value:
        return None
    value = value.strip()
    m = re.match(r"(\d{4})[/-](\d{2})[/-](\d{2})", value)
    if m:
        try:
            return datetime(int(m[1]), int(m[2]), int(m[3]), tzinfo=timezone.utc)
        except ValueError:
            return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def extractAnchorDate(text: str, metadata: Optional[dict]) -> Optional[datetime]:
    """Pick the best date to represent *when* this evidence was produced.

    Preference order:
      1. ``metadata.sessionDate`` (if ingested with explicit date)
      2. ``[Session date: YYYY/MM/DD]`` header anywhere in the text
      3. ``(meaning YYYY-MM-DD)`` trailing anchor on the first bullet
      4. bare ``YYYY-MM-DD`` substring in the first ~400 chars
      5. ``metadata.scrapedAt`` (fallback)
    """
    if metadata:
        for key in _META_DATE_KEYS:
            raw = metadata.get(key)
            if raw and isinstance(raw, str):
                parsed = parseQueryDate(raw)
                if parsed:
                    return parsed
    if not text:
        return None
    head = text[:400]
    m = _SESSION_HEADER_RE.search(text) or _MEANING_DATE_RE.search(head) or _SIMPLE_ISO_RE.search(head)
    if m:
        try:
            return datetime(int(m[1]), int(m[2]), int(m[3]), tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Rescoring
# ---------------------------------------------------------------------------

@dataclass
class RerankConfig:
    knowledgeUpdateMaxDays: int = 365
    knowledgeUpdateBoostFactor: float = 0.30
    proximityHalfLifeDays: int = 30
    proximityBoostFactor: float = 0.25
    # Items with no parseable anchor date get a tiny penalty so dated evidence wins ties
    noAnchorPenalty: float = 0.02


def _knowledgeUpdateFactor(anchor: datetime, questionDate: datetime,
                            cfg: RerankConfig) -> float:
    """Higher = more recent *relative to the question*.

    Items ``cfg.knowledgeUpdateMaxDays`` or more before the question date get
    0; items on/after the question date (same session day counts as newest)
    get 1.
    """
    ageDays = (questionDate - anchor).days
    if ageDays <= 0:
        return 1.0
    if ageDays >= cfg.knowledgeUpdateMaxDays:
        return 0.0
    return 1.0 - (ageDays / cfg.knowledgeUpdateMaxDays)


def _proximityFactor(anchor: datetime, questionDate: datetime,
                      cfg: RerankConfig) -> float:
    """Higher = closer to the question date on either side.

    Uses a two-sided exponential decay with ``proximityHalfLifeDays``.
    """
    distanceDays = abs((questionDate - anchor).days)
    # 2 ** (-d / halfLife) → 1 at d=0, 0.5 at d=halfLife, 0.25 at 2*halfLife...
    return 2.0 ** (-distanceDays / max(1, cfg.proximityHalfLifeDays))


def rerank(
    items: List[_ScoredItem],
    question: str,
    questionDate: Optional[str],
    *,
    strategy: Optional[Strategy] = None,
    config: Optional[RerankConfig] = None,
) -> Tuple[List[_ScoredItem], Strategy]:
    """Rescore ``items`` in-place and return them re-sorted by score desc.

    ``strategy`` can be forced by caller; otherwise inferred from the question.
    Returns ``(items, strategyUsed)`` so callers can log which path was taken.
    """
    cfg = config or RerankConfig()
    chosen = strategy or classifyQuestion(question)
    if chosen == "none":
        return items, chosen
    qDate = parseQueryDate(questionDate)
    if qDate is None:
        logger.debug("temporal rerank skipped — unparseable questionDate=%r", questionDate)
        return items, "none"

    for item in items:
        anchor = extractAnchorDate(getattr(item, "text", "") or "", getattr(item, "metadata", {}) or {})
        if anchor is None:
            item.score *= max(0.0, 1.0 - cfg.noAnchorPenalty)
            continue
        if chosen == "knowledge_update":
            factor = _knowledgeUpdateFactor(anchor, qDate, cfg)
            item.score *= (1.0 + factor * cfg.knowledgeUpdateBoostFactor)
        else:  # temporal_proximity
            factor = _proximityFactor(anchor, qDate, cfg)
            item.score *= (1.0 + factor * cfg.proximityBoostFactor)

    items.sort(key=lambda x: -x.score)
    return items, chosen


# ---------------------------------------------------------------------------
# Relative-date resolution
# ---------------------------------------------------------------------------

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_COUNT_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "a couple of": 2, "a couple": 2, "couple of": 2, "a few": 3, "few": 3, "several": 3,
}
_COUNT = r"(?:\d+|a couple of|a couple|couple of|a few|few|several|an?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
_HEDGE = r"(?:about|around|roughly|approximately|nearly|almost|over|just|like)"
_UNIT_DAYS = {"day": 1, "week": 7, "month": 30.44, "year": 365.25}

_RELATIVE_RE = re.compile(
    r"\b(?:"
    r"(?P<ago>(?:(?P<agoHedge>" + _HEDGE + r")\s+)?(?P<agoN>" + _COUNT + r")\s+(?P<agoUnit>day|week|month|year)s?\s+(?:ago|back|earlier))"
    r"|(?P<infut>in\s+(?:(?P<futHedge>" + _HEDGE + r")\s+)?(?P<futN>" + _COUNT + r")\s+(?P<futUnit>day|week|month|year)s?(?!\s+(?:ago|back|earlier)))"
    r"|(?P<dby>the day before yesterday)"
    r"|(?P<yday>yesterday|last night)"
    r"|(?P<tmrw>tomorrow)"
    r"|(?P<today>today|tonight|this morning|this afternoon|this evening|earlier today)"
    r"|(?P<lastwd>(?:last|this past)\s+(?P<lastDay>" + "|".join(_WEEKDAYS) + r"))"
    r"|(?P<nextwd>(?:next|this coming)\s+(?P<nextDay>" + "|".join(_WEEKDAYS) + r"))"
    r"|(?P<lastwe>(?:last|this past)\s+weekend)"
    r"|(?P<lastper>last\s+(?P<lastUnit>week|month|year))"
    r"|(?P<nextper>next\s+(?P<nextUnit>week|month|year))"
    r")\b(?!\s*\[)",
    re.IGNORECASE,
)
_TURN_RE = re.compile(r"^\[(user|assistant)\]:", re.MULTILINE)


def _count(word: str) -> Optional[int]:
    word = word.lower()
    return int(word) if word.isdigit() else _COUNT_WORDS.get(word)


def _resolveMatch(m: "re.Match[str]", anchor: datetime) -> Optional[str]:
    d0 = anchor.date()
    if m.group("ago") or m.group("infut"):
        n = _count(m.group("agoN") or m.group("futN"))
        unit = (m.group("agoUnit") or m.group("futUnit")).lower()
        if n is None:
            return None
        sign = -1 if m.group("ago") else 1
        target = d0 + timedelta(days=round(sign * n * _UNIT_DAYS[unit]))
        vague = re.search(r"few|couple|several", m.group("agoN") or m.group("futN"), re.I)
        exact = unit in ("day", "week") and not (m.group("agoHedge") or m.group("futHedge") or vague)
        if unit == "year":
            return f"≈{target.year}"
        return f"{'' if exact else '≈'}{target.isoformat()}"
    if m.group("dby"):
        return (d0 - timedelta(days=2)).isoformat()
    if m.group("yday"):
        return (d0 - timedelta(days=1)).isoformat()
    if m.group("tmrw"):
        return (d0 + timedelta(days=1)).isoformat()
    if m.group("today"):
        return d0.isoformat()
    if m.group("lastwd"):
        wd = _WEEKDAYS.index(m.group("lastDay").lower())
        back = (d0.weekday() - wd) % 7 or 7
        return (d0 - timedelta(days=back)).isoformat()
    if m.group("nextwd"):
        wd = _WEEKDAYS.index(m.group("nextDay").lower())
        fwd = (wd - d0.weekday()) % 7 or 7
        return (d0 + timedelta(days=fwd)).isoformat()
    if m.group("lastwe"):
        # On a Saturday or Sunday, "last weekend" is the previous one.
        back = (d0.weekday() - 5) % 7 + (7 if d0.weekday() >= 5 else 0)
        sat = d0 - timedelta(days=back)
        return f"{sat.isoformat()}/{(sat + timedelta(days=1)).isoformat()}"
    if m.group("lastper") or m.group("nextper"):
        unit = (m.group("lastUnit") or m.group("nextUnit")).lower()
        sign = -1 if m.group("lastper") else 1
        if unit == "week":
            monday = d0 - timedelta(days=d0.weekday()) + timedelta(days=7 * sign)
            return f"week of {monday.isoformat()}"
        if unit == "month":
            y, mo = divmod(d0.year * 12 + d0.month - 1 + sign, 12)
            return f"{y}-{mo + 1:02d}"
        return str(d0.year + sign)
    return None


def resolveRelativeDates(text: str, anchor: Optional[datetime], *, userOnly: bool = True) -> str:
    """Annotate relative time phrases with the absolute date they refer to.

    ``anchor`` is when ``text`` was written (its session date). Phrases get a
    bracketed resolution appended — ``"three weeks ago [2023-05-07]"``;
    approximate units (months, hedged counts) are marked ``≈``. With
    ``userOnly`` and a ``[user]:``/``[assistant]:`` transcript, only user turns
    are rewritten: assistant replies use relative phrases generically
    ("a few years ago, research showed...").
    """
    if not text or anchor is None:
        return text

    def sub(chunk: str) -> str:
        def rep(m: "re.Match[str]") -> str:
            resolved = _resolveMatch(m, anchor)
            return f"{m.group(0)} [{resolved}]" if resolved else m.group(0)
        return _RELATIVE_RE.sub(rep, chunk)

    turns = list(_TURN_RE.finditer(text))
    if not userOnly or not turns:
        return sub(text)
    out = [text[: turns[0].start()]]
    for i, t in enumerate(turns):
        end = turns[i + 1].start() if i + 1 < len(turns) else len(text)
        chunk = text[t.start():end]
        out.append(sub(chunk) if t.group(1) == "user" else chunk)
    return "".join(out)


def describeOffset(date: Optional[datetime], questionDate: Optional[datetime]) -> str:
    """Human-readable distance of ``date`` before ``questionDate``."""
    if date is None or questionDate is None:
        return ""
    days = (questionDate.date() - date.date()).days
    if days == 0:
        return "same day as the question"
    if days < 0:
        return f"{-days} days after the question"
    if days < 14:
        return f"{days} days before the question"
    if days < 60:
        return f"{days // 7} weeks before the question"
    return f"{round(days / 30.44)} months before the question"
