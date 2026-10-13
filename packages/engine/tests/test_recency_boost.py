import pytest
from datetime import datetime, timedelta, timezone

from kormem.retrieve.recency import recencyBoost


def testFreshNoteBoost():
    now = datetime(2026, 5, 29, tzinfo=timezone.utc)
    updated = now.isoformat()
    boosted = recencyBoost(1.0, updated, now=now)
    assert boosted == pytest.approx(1.2, rel=1e-3)


def testOldNoteNoBoost():
    now = datetime(2026, 5, 29, tzinfo=timezone.utc)
    old = (now - timedelta(days=100)).isoformat()
    boosted = recencyBoost(1.0, old, now=now)
    assert boosted == pytest.approx(1.0, rel=1e-3)


def testMidpointBoost():
    now = datetime(2026, 5, 29, tzinfo=timezone.utc)
    mid = (now - timedelta(days=45)).isoformat()
    boosted = recencyBoost(1.0, mid, now=now)
    assert 1.0 < boosted < 1.2
