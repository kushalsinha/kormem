"""Recency boost for note/guidance scores."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def parseIso(value: str) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return datetime.now(timezone.utc)


def recencyBoost(
    score: float,
    updatedAtIso: str,
    maxAgeDays: int = 90,
    boostFactor: float = 0.2,
    now: Optional[datetime] = None,
) -> float:
    now = now or datetime.now(timezone.utc)
    updated = parseIso(updatedAtIso)
    ageDays = max(0, (now - updated).days)
    if ageDays <= 0:
        factor = 1.0
    elif ageDays >= maxAgeDays:
        factor = 0.0
    else:
        factor = 1.0 - ageDays / maxAgeDays
    return score * (1.0 + factor * boostFactor)
