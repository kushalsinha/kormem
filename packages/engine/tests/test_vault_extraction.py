import json

import pytest

from kormem.core.config import KormemConfig
from kormem.extract.memory_index import formatLine, stripLinks, trailingLinks
from kormem.extract.service import ExtractionService, splitSections
from kormem.extract.vault_linker import VaultLinker
from kormem.ingest.chat_sessions import ChatSessionService
from kormem.ingest.service import IngestService
from kormem.store.sqlite_store import SqliteStore
from kormem.vault import folders as folderMod
from kormem.extract.memory_index import readMemory
from kormem.vault.note_sections import addBullet, parseSections, renderSections, supersede
from kormem.vault.vault_store import VaultStore


# ---------------------------------------------------------------- sections

def testDuplicateBulletOnlyAddsSourceLink():
    secs = {}
    assert addBullet(secs, "Current facts", "User owns 3 bikes (as of 2023-05-01)", "src A")
    assert not addBullet(secs, "Current facts", "User owns 3 bikes (as of 2023-05-01).", "src B")
    assert secs["Current facts"] == ["- User owns 3 bikes (as of 2023-05-01) [[src A]] [[src B]]"]


def testTimelineStaysSorted():
    secs = {}
    addBullet(secs, "Timeline", "Finished season 8", "s2", "2023-06-14")
    addBullet(secs, "Timeline", "Started watching", "s1", "2023-04-28")
    assert [l[:12] for l in secs["Timeline"]] == ["- 2023-04-28", "- 2023-06-14"]


def testSupersedeMovesOldFactToTimeline():
    secs = {}
    addBullet(secs, "Current facts", "User works at Acme as an analyst (as of 2023-01-10)", "s1")
    old = supersede(secs, "User works at Acme as analyst", "2023-05-02")
    addBullet(secs, "Current facts", "User works at Globex as a manager (as of 2023-05-02)", "s2")
    assert old and "Acme" in old
    assert len(secs["Current facts"]) == 1 and "Globex" in secs["Current facts"][0]
    assert "superseded on 2023-05-02" in secs["Timeline"][0]
    body = renderSections("", secs)
    assert parseSections(body)[1]["Current facts"] == secs["Current facts"]


def testMemoryLineLinks():
    line = formatLine("2023-05-28", "2023-04-28", "🔴", "User started Game of Thrones",
                      "[[Game of Thrones]] [[2023-05-28 chat · HBO]]")
    assert line == "🔴 User started Game of Thrones. (meaning 2023-04-28) [[Game of Thrones]] [[2023-05-28 chat · HBO]]"
    assert stripLinks(line).endswith("(meaning 2023-04-28)")
    assert trailingLinks(line) == ["Game of Thrones", "2023-05-28 chat · HBO"]


def testSplitSectionsOnHeadings():
    text = "# A\n" + "a" * 50 + "\n# B\n" + "b" * 50
    assert splitSections(text, maxChars=60) == ["# A\n" + "a" * 50 + "\n", "# B\n" + "b" * 50]


# --------------------------------------------------------------- pipeline

@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("KORMEM_HOME", str(tmp_path))
    config = KormemConfig(defaultNamespace="t", embeddingDim=64, embeddingProvider="hash",
                          rerankEnabled=False)
    store = SqliteStore(indexPath=tmp_path / "i.sqlite", guidancePath=tmp_path / "g.sqlite")
    ingest = IngestService(store=store, config=config)
    extraction = ExtractionService(store=store, config=config)

    return store, ingest, extraction, config


def _fakeWriter(monkeypatch, linker: VaultLinker, replies):
    calls = []

    async def write(prompt):
        calls.append(prompt)
        return replies[min(len(calls), len(replies)) - 1]
    monkeypatch.setattr(linker, "_write", write)
    return calls


@pytest.mark.asyncio
async def testChatSettlesOnCloseAndBuildsLinkedVault(env, monkeypatch):
    store, ingest, extraction, config = env
    chats = ChatSessionService(ingest=ingest, config=config)
    calls = _fakeWriter(monkeypatch, extraction.linker, [{
        "source": {"title": "HBO picks", "summary": "User asked for shows like Game of Thrones."},
        "items": [
            {"kind": "event", "speaker": "user", "text": "User started watching Game of Thrones",
             "date": "2023-04-28", "note": "Game of Thrones",
             "new_note": {"para": "resources", "folder": "3-Resources/TV & film", "aliases": ["GoT"],
                          "folder_purpose": "shows and films", "folder_reason": "several shows"}},
            {"kind": "suggestion", "speaker": "assistant",
             "text": "Assistant listed shows like Game of Thrones: 1. Westworld 2. Vikings",
             "date": "2023-05-28", "note": "Game of Thrones"},
            {"kind": "fact", "speaker": "user", "text": "User subscribes to HBO Max", "date": "2023-05-28",
             "note": "HBO Max", "new_note": {"para": "resources", "folder": "3-Resources"}},
        ],
        "related": [["Game of Thrones", "HBO Max"]],
    }])

    await chats.append("t", "s1", [
        {"role": "user", "content": "I started GoT about a month ago, any similar shows?"},
        {"role": "assistant", "content": "1. Westworld 2. Vikings"},
    ], sessionDate="2023/05/28 (Sun) 09:52")
    assert await extraction.processPending() == 0          # still live: nothing extracted yet
    chats.close("t", "s1")
    assert await extraction.processPending() == 1
    assert len(calls) == 1 and "[user]: I started GoT" in calls[0]

    vault = VaultStore(store=store, config=config, namespaceId="t")
    got = vault.readNote("GoT")                             # alias resolves
    assert got.title == "Game of Thrones" and got.folder == "3-Resources"   # folder only proposed
    _, secs = parseSections(got.body)
    assert secs["Timeline"][0].startswith("- 2023-04-28 User started watching Game of Thrones")
    assert "Westworld" in secs["From the assistant"][0]
    assert "- [[HBO Max]]" in secs["Related"]
    source = "2023-05-28 chat · HBO picks"
    assert f"[[{source}]]" in secs["Timeline"][0]
    src = vault.readNote(source)
    assert src.folder == "_sources" and "[[Game of Thrones]]" in src.body
    memory = readMemory(vault)
    assert "Date: 2023-05-28" in memory and "(meaning 2023-04-28) [[Game of Thrones]]" in memory
    backlinks = {b.title for b in vault.getBacklinks(got.kormemId)}
    assert {"HBO Max", source} <= backlinks

    # A resumed chat extracts only the new stretch, into the same source note.
    await chats.append("t", "s1", [{"role": "user", "content": "Finished season 8 today."}])
    chats.close("t", "s1")
    await extraction.processPending()
    assert len(calls) == 2 and "[new turns]" in calls[1] and "Finished season 8" in calls[1]
    assert "Westworld" not in calls[1].split("[new turns]")[1]


@pytest.mark.asyncio
async def testFolderCreatedOnlyAfterThreeNotesAsk(env, monkeypatch):
    store, ingest, extraction, config = env
    replies = [{"source": {"title": f"doc {i}"}, "items": [{
        "kind": "fact", "speaker": "user", "text": f"Fact about show {i}", "date": "2023-05-01",
        "note": f"Show {i}", "new_note": {"para": "resources", "folder": "3-Resources/TV & film",
                                          "folder_purpose": "TV shows", "folder_reason": "many shows"}}]}
        for i in range(3)]
    _fakeWriter(monkeypatch, extraction.linker, replies)
    for i in range(3):
        await ingest.ingestText(f"Notes about show {i} " * 20, namespaceId="t", title=f"show {i}")
        await extraction.processPending()
        active = [f["path"] for f in store.listFolders("t", "active")]
        assert active == ([] if i < 2 else ["3-Resources/TV & film"])
    vault = VaultStore(store=store, config=config, namespaceId="t")
    assert all(vault.readNote(f"Show {i}").folder == "3-Resources/TV & film" for i in range(3))
    assert "3-Resources/TV & film" in vault.readNote("_meta/folders.md").body


@pytest.mark.asyncio
async def testDeepFolderRejected(env, monkeypatch):
    store, *_ = env
    vault = VaultStore(store=store, config=env[3], namespaceId="t")
    reg = folderMod.FolderRegistry(store, vault, env[1].embedder)
    folder, proposal = await reg.place("2-Areas/Health/Running/Shoes", "areas", "Shoes", "running shoes", "x")
    assert folder == "2-Areas" and proposal is None


@pytest.mark.asyncio
async def testUnchangedDocumentSectionsAreNotReExtracted(env, monkeypatch):
    store, ingest, extraction, _ = env
    calls = _fakeWriter(monkeypatch, extraction.linker, [{"source": {"title": "x"}, "items": []}])
    text = "# Setup\n" + "Install with pip. " * 20
    await ingest.ingestText(text, namespaceId="t", title="README")
    await extraction.processPending()
    await ingest.ingestText(text, namespaceId="t", title="README")
    await extraction.processPending()
    assert len(calls) == 1


@pytest.mark.asyncio
async def testRememberedFactsAreBatched(env, monkeypatch):
    store, ingest, extraction, _ = env
    monkeypatch.setattr("kormem.extract.service.FACT_DEBOUNCE_MINUTES", 0.0)
    calls = _fakeWriter(monkeypatch, extraction.linker, [{"source": {"title": "facts"}, "items": []}])
    for fact in ("I prefer aisle seats", "My dog is called Max"):
        await ingest.ingestText(fact, namespaceId="t", role="user")
    await extraction.processPending()
    assert len(calls) == 1 and "aisle seats" in calls[0] and "Max" in calls[0]


@pytest.mark.asyncio
async def testRecallWalksVaultAndResolvesDates(env, monkeypatch):
    store, ingest, extraction, config = env
    chats = ChatSessionService(ingest=ingest, config=config)
    _fakeWriter(monkeypatch, extraction.linker, [{
        "source": {"title": "coupon"},
        "items": [{"kind": "event", "speaker": "user", "text": "User redeemed a $5 coffee creamer coupon at Target",
                   "date": "2023-05-28", "note": "Groceries", "new_note": {"para": "areas", "folder": "2-Areas"}}],
    }])
    await chats.append("t", "c1", [{"role": "user", "content": "I redeemed a $5 creamer coupon yesterday at Target."}],
                       sessionDate="2023/05/29 (Mon) 13:28")
    chats.close("t", "c1")
    await extraction.processPending()

    from kormem.retrieve import recall as recallMod
    seen = {}

    async def fakeAgent(question, briefing, tools, valid, **kw):
        seen["briefing"] = briefing
        seen["note"] = await tools["read_note"]("Groceries")
        seen["source"] = await tools["read_source"]("S1")
        from kormem.retrieve.recursive_agent import RecallResult
        return RecallResult(sources=["S1"], findings=["Redeemed at Target on 2023-05-28 (S1)"], rounds=2,
                            notesRead=["Groceries"])
    monkeypatch.setattr(recallMod, "recursiveRecall", fakeAgent)
    pipe = recallMod.RecallPipeline(store=store, config=config)
    monkeypatch.setattr(pipe.reranker, "_load", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    ctx = await pipe.recall("Where did I redeem the creamer coupon?", "t", "2023/05/30 (Tue) 20:42", effort="high")

    assert "S1" in seen["briefing"] and "Target" in seen["note"] and "(S1)" in seen["note"]
    assert "yesterday [2023-05-28]" in seen["source"]
    assert ctx.findings and ctx.sources[0]["sid"] == "S1" and "yesterday [2023-05-28]" in ctx.sources[0]["text"]
    assert ctx.notes[0]["title"] == "Groceries"
    assert "Target" in ctx.relevantMemory and "[[" not in ctx.relevantMemory


def testLessonsScopeAndRetire(tmp_path, monkeypatch):
    monkeypatch.setenv("KORMEM_HOME", str(tmp_path))
    from kormem.extract.lessons import addLessons, lessonsFor
    config = KormemConfig(defaultNamespace="t", embeddingDim=64, embeddingProvider="hash")
    store = SqliteStore(indexPath=tmp_path / "i.sqlite", guidancePath=tmp_path / "g.sqlite")
    vault = VaultStore(store=store, config=config, namespaceId="t")
    addLessons(vault, [("Use npm for installs", "kormem", "2026-10-01", None),
                       ("Reply in short bullets", "global", "2026-10-01", None)], "src 1")
    addLessons(vault, [("Use bun, not npm", "kormem", "2026-10-12", "Use npm for installs")], "src 2")
    text = lessonsFor(vault, "kormem")
    assert "Global:\n- Reply in short bullets (since 2026-10-01)" in text
    assert "Project: kormem:\n- Use bun, not npm (since 2026-10-12)" in text
    assert "npm for installs" not in text and "[[" not in text
    assert "superseded on 2026-10-12" in vault.readNote("_meta/lessons.md").body
    assert lessonsFor(vault, "other") == "Global:\n- Reply in short bullets (since 2026-10-01)"


async def testLessonFromChatIsScopedToItsProject(env, monkeypatch):
    store, ingest, extraction, config = env
    chats = ChatSessionService(ingest=ingest, config=config)
    calls = _fakeWriter(monkeypatch, extraction.linker, [{
        "source": {"title": "tooling"},
        "items": [{"kind": "lesson", "speaker": "user", "text": "Use bun, not npm", "date": "2026-10-12",
                   "scope": "kormem"}],
    }])
    await chats.append("t", "cc1", [{"role": "user", "content": "Stop using npm here, use bun."}],
                       sessionDate="2026/10/12 (Mon) 10:00", project="kormem")
    chats.close("t", "cc1")
    await extraction.processPending()
    from kormem.extract.lessons import lessonsFor
    vault = VaultStore(store=store, config=config, namespaceId="t")
    assert "Project: kormem" in calls[0]
    assert lessonsFor(vault, "kormem") == "Project: kormem:\n- Use bun, not npm (since 2026-10-12)"
    assert "User asked the assistant: Use bun, not npm (in kormem)" in readMemory(vault)
    assert "[[lessons]]" in vault.readNote("2026-10-12 chat · tooling").body


def testAgentSkippedOnlyForClearSimpleLookups():
    from kormem.retrieve.recall import RecallPipeline
    skip = RecallPipeline._skipReason
    assert skip("What is my dog's name?", [0.97, 0.2])
    assert skip("What is my dog's name?", [0.97, 0.8]) is None          # two strong candidates
    assert skip("What is my dog's name?", [0.6, 0.1]) is None           # nothing clearly answers
    assert skip("How many bikes do I own?", [0.99, 0.1]) is None        # counting
    assert skip("What is my current job?", [0.99, 0.1]) is None         # latest value
    assert skip("When did I move to Lisbon?", [0.99, 0.1]) is None      # time


async def testExcerptsKeepMatchingWindowsInOrder(env, monkeypatch):
    store, ingest, extraction, config = env
    from kormem.retrieve.recall import RecallPipeline
    pipe = RecallPipeline(store=store, config=config)
    monkeypatch.setattr(pipe.reranker, "_load", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    text = ("filler text about nothing. " * 80) + "The coupon was redeemed at Target store. " + ("more filler. " * 300)
    out = await pipe._excerpts(text, "where was the coupon redeemed")
    assert "redeemed at Target" in out and len(out) <= 4200
