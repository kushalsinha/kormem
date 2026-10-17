import pytest


@pytest.fixture(autouse=True)
def _noJev(monkeypatch):
    """Tests never call TypeSafe, even when a developer has the key set; Jev tests fake it."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
