"""Keeping memory small: storage format, month shards, retirement, archive, merge, condense, lessons cap."""

import json
from datetime import datetime, timezone

import numpy as np
import pytest

from kormem.core.config import KormemConfig
from kormem.extract import maintenance as maintMod
from kormem.extract.lessons import addLessons, lessonsFor
from kormem.extract.maintenance import MaintenanceService
from kormem.extract.memory_index import appendMemoryLines, months, readMemory, retireLine
from kormem.extract.note_embed import embedNote
from kormem.ingest.chat_sessions import ChatSessionService
from kormem.ingest.service import IngestService
from kormem.store.sqlite_store import SqliteStore, _packEmbedding, _unpackEmbedding
from kormem.vault.note import Note
from kormem.vault.note_sections import parseSections
from kormem.vault.vault_store import VaultStore


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("KORMEM_HOME", str(tmp_path))
    config = KormemConfig(defaultNamespace="t", embeddingDim=64, embeddingProvider="hash", rerankEnabled=False)
    store = SqliteStore(indexPath=tmp_path / "i.sqlite", guidancePath=tmp_path / "g.sqlite")
    vault = VaultStore(store=store, config=config, namespaceId="t")
    return store, config, vault


def _cos(a, b):
    return float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))


# ------------------------------------------------------------- storage

def testInt8VectorsKeepCosineAndOldFormatsStillRead():
    v = np.random.default_rng(0).normal(size=768).astype(np.float32)
    blob = _packEmbedding(v.tolist())
    assert len(blob) == 768 + 4
    assert _cos(_unpackEmbedding(blob, 768), v) > 0.999
    assert np.allclose(_unpackEmbedding(np.float16(v).tobytes(), 768), v, atol=1e-2)
    assert np.allclose(_unpackEmbedding(v.tobytes(), 768), v)


@pytest.mark.asyncio
async def testChunksStoreSpansNotText(env):
    store, config, _ = env
    ingest = IngestService(store=store, config=config)
    doc = "\n\n".join(f"Paragraph {i}: " + "word " * 60 for i in range(12))
    r = await ingest.ingestText(doc, namespaceId="t", title="doc", enqueue=False)
    chats = ChatSessionService(ingest=ingest, config=config)
    await chats.append("t", "s1", [{"idx": i, "role": "user" if i % 2 == 0 else "assistant",
                                    "content": f"turn {i} " + "talk " * 80} for i in range(6)], "2023/05/01 (Mon) 10:00")
    rows = store.segmentRows("t", kind="segment")
    assert len(rows) > 4 and all(r["plainText"] is None and r["encryptedText"] is None for r in rows)
    for row in rows:
        text = store.getSegmentText(row)
        assert len(text) > 50 and ("Paragraph" in text or "turn" in text)
    assert r["segmentCount"] >= 2


@pytest.mark.asyncio
async def testCompactionMovesOldRowsToSpansAndRepacks(env):
    store, config, _ = env
    ingest = IngestService(store=store, config=config)
    doc = "\n\n".join(f"Section {i}: " + "alpha " * 70 for i in range(8))
    r = await ingest.ingestText(doc, namespaceId="t", title="doc", enqueue=False)
    # Rewrite the rows the way older versions stored them: full text, float16 vectors.
    rows = store.segmentRows("t", kind="segment")
    store.rewriteSegments([{
        "segmentKey": x["segmentKey"], "plainText": store.getSegmentText(x),
        "embedding": np.float16(_unpackEmbedding(x["embedding"], x["embeddingDim"])).tobytes(),
        "metadata": {k: v for k, v in json.loads(x["metadata"]).items() if k != "span"},
    } for x in rows])
    before = {x["segmentKey"]: store.getSegmentText(x) for x in store.segmentRows("t", kind="segment")}
    service = MaintenanceService(store, config)
    assert service.compactSegments("t") == len(rows)
    assert service.repackVectors("t") == len(rows)
    after = store.segmentRows("t", kind="segment")
    assert all(x["plainText"] is None and len(x["embedding"]) == x["embeddingDim"] + 4 for x in after)
    assert {x["segmentKey"]: store.getSegmentText(x) for x in after} == before
    assert r["documentId"]


# -------------------------------------------------------- memory index

def testMemoryIsShardedByMonthAndLegacyFileMigrates(env):
    store, config, vault = env
    legacy = Note.create(title="memory", para="meta", body="Date: 2023-03-02\n\n🔴 Old line. [[A]] [[S0]]\n")
    legacy.kormemId = "meta-memory-t"
    vault.writeNote(legacy)
    appendMemoryLines(vault, "2023-05-28", [("2023-05-28", "🔴", "User bought a bike", "[[Bikes]] [[S1]]")])
    appendMemoryLines(vault, "2023-06-02", [("2023-06-02", "🔴", "User sold a bike", "[[Bikes]] [[S2]]")])
    assert months(vault) == ["2023-03", "2023-05", "2023-06"]
    assert "Date:" not in vault.readNote("_meta/memory.md").body          # now just the month index
    assert "[[memory 2023-06]]" in vault.readNote("_meta/memory.md").body
    text = readMemory(vault)
    assert text.index("Old line") < text.index("bought") < text.index("sold")
    assert "Old line" not in readMemory(vault, since="2023-05")


@pytest.mark.asyncio
async def testRetiredLineLeavesRecallAndLosesItsVector(env):
    store, config, vault = env
    from kormem.embeddings.provider import getInternalEmbedder
    from kormem.extract.memory_index import cleanLine, lineEmbeddings

    added = appendMemoryLines(vault, "2023-05-01", [("2023-05-01", "🔴", "User works at Acme as an analyst",
                                                     "[[Work]] [[S1]]")])
    await lineEmbeddings(store, getInternalEmbedder(config), "t", [cleanLine(l) for l in added])
    assert len(store.segmentRows("t", kind="memory")) == 1
    assert retireLine(vault, store, "Work", "User works at Acme as analyst", "2023-09-01")
    assert "Acme" not in readMemory(vault)
    assert "⚫ User works at Acme as an analyst. (replaced 2023-09-01) [[Work]] [[S1]]" in readMemory(vault, includeRetired=True)
    assert store.segmentRows("t", kind="memory") == []


# ------------------------------------------------------ archive, merge

def _topic(vault, title, body, folder="2-Areas"):
    note = Note.create(title=title, para="areas", folder=folder, body=body)
    vault.writeNote(note)
    return note


@pytest.mark.asyncio
async def testColdNotesArchiveAndComeBackWithNewKnowledge(env, monkeypatch):
    store, config, vault = env
    old = _topic(vault, "Guitar", "## Timeline\n- 2022-01-05 User took a guitar lesson [[S1]]\n")
    _topic(vault, "Running", "## Timeline\n- 2023-05-20 User ran 10k [[S2]]\n")
    await embedNote(vault.readNote("Guitar"), "t", store, config)
    clock = datetime(2023, 6, 1, tzinfo=timezone.utc)
    assert MaintenanceService(store, config).archiveCold(vault, clock) == 1
    g = vault.readNote("Guitar")
    assert g.para == "archive" and g.folder == "4-Archive" and g.archivedFrom == "2-Areas"
    meta = json.loads(store.getSegmentsByKeys([f"note:{old.kormemId}"])[0]["metadata"])
    assert meta["para"] == "archive"
    assert vault.readNote("Running").para == "areas"

    from kormem.extract.vault_linker import ExtractionUnit, VaultLinker
    linker = VaultLinker(store, config)

    async def write(prompt):
        return {"source": {"title": "guitar again", "summary": ""},
                "items": [{"kind": "event", "speaker": "user", "text": "User restarted guitar lessons",
                           "date": "2023-06-03", "note": "Guitar"}]}
    monkeypatch.setattr(linker, "_write", write)
    await linker.link(ExtractionUnit(namespaceId="t", kind="chat", title="c", date="2023-06-03", text="..."))
    g = vault.readNote("Guitar")
    assert g.para == "areas" and g.folder == "2-Areas" and not g.archivedFrom
    assert not (vault.vaultRoot / "4-Archive" / "Guitar.md").exists()


@pytest.mark.asyncio
async def testNearDuplicateNotesMergeAndLinksFollow(env, monkeypatch):
    store, config, vault = env
    _topic(vault, "Running", "## Current facts\n- User runs 3 times a week [[S1]]\n")
    _topic(vault, "Jogging", "## Current facts\n- User runs 3 times a week [[S1]]\n- User's best 5k is 24 min [[S2]]\n")
    src = Note.create(title="S2", para="source", folder="_sources", body="## Feeds\n- [[Jogging]]\n")
    vault.writeNote(src)
    for t in ("Running", "Jogging"):
        await embedNote(vault.readNote(t), "t", store, config)
    monkeypatch.setattr(maintMod, "MERGE_SIMILARITY", -1.0)
    assert await MaintenanceService(store, config).mergeDuplicates(vault) == 1
    keep = vault.readNote("Jogging")
    assert keep and "Running" in keep.aliases
    facts = parseSections(keep.body)[1]["Current facts"]
    assert len(facts) == 2                                         # the shared fact is not repeated
    assert vault.readNote("Running").title == "Jogging"            # old title resolves as an alias
    assert len([n for n in vault.listNotes() if n.para == "areas"]) == 1
    assert "[[Jogging]]" in vault.readNote("S2").body


@pytest.mark.asyncio
async def testLongNotesCondenseButKeepEverySource(env, monkeypatch):
    store, config, vault = env
    timeline = "\n".join(f"- 2023-01-{i + 1:02d} User ran {i} km [[S{i}]]" for i in range(25))
    _topic(vault, "Running", f"## Current facts\n- User runs weekly [[S0]]\n\n## Timeline\n{timeline}\n")
    monkeypatch.setattr(maintMod, "NOTE_MAX_CHARS", 100)
    monkeypatch.setattr(maintMod, "MIN_TO_CONDENSE", 100)

    async def fakeLlm(prompt, system, **kw):
        assert "User ran 0 km" in prompt and "User ran 24 km" not in prompt
        return json.dumps({"bullets": ["- 2023-01-01 to 2023-01-10: ran 0-9 km [[S0]] [[S1]]"]})
    monkeypatch.setattr(maintMod, "completeLlm", fakeLlm)
    assert await MaintenanceService(store, config).condenseLong(vault) == 1
    _, secs = parseSections(vault.readNote("Running").body)
    assert len(secs["Timeline"]) == maintMod.KEEP_RECENT
    assert secs["Earlier history"][0].startswith("- 2023-01-01 to 2023-01-10")
    assert all(f"[[S{i}]]" in secs["Earlier history"][1] for i in range(2, 10))   # dropped links kept


def testLongAssistantListsAreKeptByCharacterBudget():
    lines = [f"- Assistant listed option {i}: " + "x" * 900 + f" [[S{i}]]" for i in range(8)]
    start = maintMod._keepFrom(lines)
    assert start == 8 - maintMod.KEEP_MIN                     # 3 newest stay verbatim, 5 are condensed
    assert maintMod._keepFrom(lines[:2]) == 0                 # short sections stay whole


# ------------------------------------------------------------- lessons

def testLessonsLoadedAreCappedNewestAndProjectFirst(env):
    _, _, vault = env
    rules = ["Reply in short bullets", "Run the tests before saying a change is done", "Never commit without asking",
             "Use bun, not npm", "Prefer TypeScript strict mode", "Write dates as YYYY-MM-DD",
             "Keep pull requests under 300 lines", "Explain trade-offs before coding", "Use metric units",
             "Quote prices in euros"]
    addLessons(vault, [(f"Rule {i}: {r}", None, f"2023-01-{i + 1:02d}", None) for i, r in enumerate(rules)], "S")
    addLessons(vault, [("Use bun here", "kormem", "2023-01-01", None)], "S")
    out = lessonsFor(vault, "kormem", maxChars=200)
    assert "Use bun here" in out and len(out) < 260
    assert "Rule 9:" in out and "Rule 0:" not in out


def testLegacyKeywordRowsFoldOnceAndStillMatch(tmp_path, monkeypatch):
    monkeypatch.setenv("KORMEM_HOME", str(tmp_path))
    store = SqliteStore(indexPath=tmp_path / "i.sqlite", guidancePath=tmp_path / "g.sqlite")
    store.upsertSegments("t", "d1", "s1", [{"segmentKey": "k1", "segmentIndex": 0, "plainText": "x",
                                            "keywordHashes": []}])
    with store._connectIndex() as conn:   # the old layout: a row per keyword, arbitrary rowids, user_version 0
        conn.executemany("INSERT INTO segments_fts (rowid, segmentKey, keywordHash) VALUES (?, ?, ?)",
                         [(50, "k1", "aaa111"), (51, "k1", "bbb222"), (52, "k1", "aaa111"), (53, "gone", "aaa111")])
        conn.execute("PRAGMA user_version = 0")
        conn.commit()
    store = SqliteStore(indexPath=tmp_path / "i.sqlite", guidancePath=tmp_path / "g.sqlite")
    with store._connectIndex() as conn:
        rows = conn.execute("SELECT rowid, segmentKey FROM segments_fts").fetchall()
        segRowid = conn.execute("SELECT rowid FROM segments WHERE segmentKey='k1'").fetchone()[0]
    assert [(r[0], r[1]) for r in rows] == [(segRowid, "k1")]
    from kormem.core import keywords
    monkeypatch.setattr(keywords, "queryKeywordGroups", lambda q: [["aaa111"], ["bbb222", "zzz999"]])
    assert store.keywordSearch("t", "anything") == [("k1", 2.0)]


def testKeywordSearchMatchesPluralAndTenseForms(tmp_path, monkeypatch):
    monkeypatch.setenv("KORMEM_HOME", str(tmp_path))
    from kormem.core.keywords import hashKeywords
    store = SqliteStore(indexPath=tmp_path / "i.sqlite", guidancePath=tmp_path / "g.sqlite")
    store.upsertSegments("t", "d1", "s1", [{"segmentKey": "k1", "segmentIndex": 0, "plainText": "x",
                                            "keywordHashes": hashKeywords("Rollbacks deployed nightly")}])
    assert store.keywordSearch("t", "rollback deploy") == [("k1", 2.0)]
    assert store.keywordSearch("t", "rollbacks rollback") == [("k1", 2.0)]
    assert store.keywordSearch("t", "unrelated") == []
