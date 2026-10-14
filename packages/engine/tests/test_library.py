"""Collections, change-aware sync, contexts and addresses (kormem://, #docid, titles)."""

import os
import time

import pytest

from kormem import library
from kormem.core.config import KormemConfig, readRawConfig, writeRawConfig
from kormem.embeddings.provider import HashEmbeddingProvider
from kormem.ingest.service import IngestService
from kormem.store.sqlite_store import SqliteStore
from kormem.vault.note import Note
from kormem.vault.vault_store import VaultStore


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("KORMEM_HOME", str(tmp_path / "home"))
    config = KormemConfig(defaultNamespace="t", embeddingDim=64, embeddingProvider="hash")
    store = SqliteStore(indexPath=tmp_path / "i.sqlite", guidancePath=tmp_path / "g.sqlite")
    ingest = IngestService(store=store, config=config)
    ingest.embedder = HashEmbeddingProvider(64)
    notes = tmp_path / "notes"
    (notes / "work").mkdir(parents=True)
    (notes / "deploy.md").write_text("# Deploy\n\nBlue-green releases on Fly.io.\nPriya approves production.\n")
    (notes / "work" / "standup.md").write_text("# Standup\n\nRavi migrates billing.\n")
    (notes / ".hidden.md").write_text("secret")
    (notes / "image.png").write_bytes(b"\x89PNG")
    return ingest, store, config, notes


async def _add(ingest, notes, name="notes", mask="**/*.md"):
    entry = library.addCollection(name, str(notes), mask)
    return await library.syncCollection(ingest, "t", name, entry["path"], entry["pattern"])


@pytest.mark.asyncio
async def testSyncAddsUpdatesRemovesAndSkipsUnchanged(env):
    ingest, store, _, notes = env
    assert await _add(ingest, notes) == {"added": 2, "updated": 0, "unchanged": 0, "removed": 0, "failed": 0}
    again = await library.syncCollection(ingest, "t", "notes", str(notes), "**/*.md")
    assert again["unchanged"] == 2 and again["added"] == 0

    time.sleep(0.01)
    (notes / "work" / "standup.md").write_text("# Standup\n\nRavi migrates billing. Priya owns the rota.\n")
    (notes / "deploy.md").unlink()
    (notes / "new.md").write_text("# New\n\nLisbon offsite.\n")
    stats = await library.syncCollection(ingest, "t", "notes", str(notes), "**/*.md")
    assert (stats["added"], stats["updated"], stats["removed"]) == (1, 1, 1)
    titles = sorted(r["title"] for r in store.documentsForCollection("t", "notes"))
    assert titles == ["new.md", "work/standup.md"]
    # A touched file whose text didn't change is not re-indexed.
    os.utime(notes / "new.md", None)
    assert (await library.syncCollection(ingest, "t", "notes", str(notes), "**/*.md"))["updated"] == 0


def testMaskExpansion():
    assert library._expandMask("**/*.md,docs/**") == ["**/*.md", "docs/**"]
    assert library._expandMask("**/*.{md,txt}") == ["**/*.md", "**/*.txt"]


@pytest.mark.asyncio
async def testResolveByDocidAddressTitleAndPath(env):
    ingest, store, config, notes = env
    await _add(ingest, notes)
    vault = VaultStore(store=store, config=config, namespaceId="t")
    doc = library.resolve(store, vault, "t", "kormem://notes/work/standup.md")
    assert doc["uri"] == "kormem://notes/work/standup.md" and "Ravi" in doc["text"]
    assert library.resolve(store, vault, "t", doc["docid"])["documentId"] == doc["documentId"]
    assert library.resolve(store, vault, "t", str(notes / "deploy.md"))["title"] == "deploy.md"
    assert library.resolve(store, vault, "t", "work/standup.md")["docid"] == doc["docid"]

    note = Note.create(title="Deployment", para="resources", folder="3-Resources")
    note.body = "## Current facts\n- Blue-green on Fly.io.\n"
    vault.writeNote(note)
    got = library.resolve(store, vault, "t", "deployment")
    assert got["kind"] == "note" and got["uri"] == "kormem://vault/3-Resources/Deployment.md"
    assert library.resolve(store, vault, "t", got["uri"])["title"] == "Deployment"

    with pytest.raises(library.RefError) as e:
        library.resolve(store, vault, "t", "Deploymnt")
    assert "Deployment" in e.value.suggestions


@pytest.mark.asyncio
async def testMultiGetGlobAndList(env):
    ingest, store, config, notes = env
    await _add(ingest, notes)
    vault = VaultStore(store=store, config=config, namespaceId="t")
    got = library.multiGet(store, vault, "t", "notes/work/*")
    assert [d["uri"] for d in got] == ["kormem://notes/work/standup.md"]
    first = library.resolve(store, vault, "t", "kormem://notes/deploy.md")["docid"]
    both = library.multiGet(store, vault, "t", f"{first}, missing-thing")
    assert both[0]["docid"] == first and "error" in both[1]
    small = library.multiGet(store, vault, "t", "notes/**", maxBytes=10)
    assert all(d.get("skipped") for d in small)


@pytest.mark.asyncio
async def testContextsNestAndFollowRename(env, tmp_path):
    ingest, store, _, notes = env
    await _add(ingest, notes)
    library.addContext("/", "Everything I know")
    library.addContext("notes", "Team notes")
    assert library.addContext("", "Daily standups", cwd=str(notes / "work")) == "kormem://notes/work"
    assert library.contextFor("kormem://notes/work/standup.md") == "Everything I know\nTeam notes\nDaily standups"
    assert library.contextFor("kormem://notes/deploy.md") == "Everything I know\nTeam notes"
    with pytest.raises(ValueError):
        library.normalizeContextPath(str(tmp_path / "elsewhere"))

    assert library.renameCollection(store, "t", "notes", "team") == 2
    assert set(library.contexts()) == {"/", "kormem://team", "kormem://team/work"}
    assert {r["collectionId"] for r in store.findDocuments("t", urlSuffix=".md")} == {"team"}
    assert library.removeCollection(store, "t", "team") == 2
    assert set(library.contexts()) == {"/"} and library.collections() == {}


def testConfigEditsKeepUnknownKeys(env):
    writeRawConfig({"openaiApiKey": "keep-me", "enginePort": 9999})
    library.addContext("/", "x")
    raw = readRawConfig()
    assert raw["openaiApiKey"] == "keep-me" and raw["contexts"] == {"/": "x"}


def testSplitRef():
    from kormem.cli.main import _splitRef
    assert _splitRef("#abc123:120:40") == ("#abc123", 120, 40)
    assert _splitRef("kormem://notes/a.md:5") == ("kormem://notes/a.md", 5, None)
    assert _splitRef("kormem://notes/a.md") == ("kormem://notes/a.md", None, None)
