import pytest

from kormem.core.config import KormemConfig
from kormem.store.sqlite_store import SqliteStore
from kormem.vault.note import Note
from kormem.vault.vault_store import VaultStore


@pytest.fixture
def vaultSetup(tmp_path, monkeypatch):
    monkeypatch.setenv("KORMEM_HOME", str(tmp_path))
    index = tmp_path / "index.sqlite"
    guidance = tmp_path / "guidance.sqlite"
    config = KormemConfig(defaultNamespace="test")
    store = SqliteStore(indexPath=index, guidancePath=guidance)
    vault = VaultStore(store=store, config=config, namespaceId="test")
    return vault, store


def testNoteRoundtrip():
    note = Note.create(title="Billing system", para="areas", body="Credits are metered.")
    note.tags = ["billing"]
    note.sources = ["doc_abc"]
    raw = note.serialize()
    parsed = Note.parse(raw)
    assert parsed.title == "Billing system"
    assert parsed.para == "areas"
    assert "Credits" in parsed.body
    assert parsed.tags == ["billing"]
    assert parsed.sources == ["doc_abc"]


def testWikilinks():
    note = Note.create(title="A", para="areas", body="See [[Credit pricing]] and [[Stripe]].")
    assert note.outboundLinks() == ["Credit pricing", "Stripe"]


def testVaultWriteRead(vaultSetup):
    vault, store = vaultSetup
    note = Note.create(title="Billing system", para="areas", body="How credits work.")
    note.sources = ["doc1"]
    kid = vault.writeNote(note)
    loaded = vault.readNote(kid)
    assert loaded is not None
    assert loaded.title == "Billing system"
    row = store.resolveNote("test", kid)
    assert row is not None


def testResolveLinkAndBacklinks(vaultSetup):
    vault, _ = vaultSetup
    n1 = Note.create(title="Billing", para="areas", body="Overview.")
    n2 = Note.create(title="Credits", para="areas", body="See [[Billing]] for context.")
    vault.writeNote(n1)
    vault.writeNote(n2)
    resolved = vault.resolveLink("Billing")
    assert resolved is not None
    backlinks = vault.getBacklinks(n1.kormemId)
    assert any(b.title == "Credits" for b in backlinks)
