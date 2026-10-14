"""Vault linker: turn one settled source into linked vault notes.

Cost: one writing call per extraction unit. Everything around it is free:

1. Shortlist  — embed a few chunks of the unit and take the nearest topic
   notes (best similarity over the chunks).
2. Write      — one model call sees the unit plus the shortlisted notes'
   current facts and returns structured items (facts, events, preferences,
   procedures, assistant suggestions, decisions) with their target notes.
3. Apply      — code writes a ``_sources/`` note, adds bullets to topic
   notes (duplicates merge, superseded facts move to the Timeline), enforces
   the folder gate, links notes to each other and appends memory lines.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from kormem.core.config import KormemConfig, loadConfig
from kormem.embeddings.provider import getInternalEmbedder
from kormem.extract.llm_client import completeLlm
from kormem.extract.lessons import addLessons
from kormem.extract.memory_index import appendMemoryLines, cleanLine, lineEmbeddings, retireLine
from kormem.extract.note_embed import embedNote
from kormem.store.sqlite_store import SqliteStore
from kormem.vault.folders import FolderRegistry, PARA_ROOTS
from kormem.vault.note import Note, PARA_FOLDERS
from kormem.vault.note_sections import addBullet, addRelated, parseSections, renderSections, supersede
from kormem.vault.paths import noteFileName
from kormem.vault.vault_store import VaultStore

logger = logging.getLogger(__name__)

SHORTLIST_NOTES = 5
NOTE_FACTS_CHARS = 1200       # per shortlisted note, shown to the writer
UNIT_CHARS = 60_000           # cap on the text one call sees

KIND_SECTION = {
    "fact": "Current facts", "preference": "Current facts", "decision": "Current facts",
    "event": "Timeline", "procedure": "How-to", "suggestion": "From the assistant",
}
KIND_PRIORITY = {
    "fact": "🔴", "preference": "🔴", "decision": "🔴", "event": "🔴",
    "suggestion": "🟡", "procedure": "🟢",
}
SKIP_PARA = {"meta", "source"}
# One unit at a time per namespace: notes and memory.md are read-modify-write.
_namespaceLocks: Dict[str, asyncio.Lock] = {}


@dataclass
class ExtractionUnit:
    namespaceId: str
    kind: str                       # document | chat | fact | trace
    title: str                      # human title of the source
    date: str                       # YYYY-MM-DD the source was written / took place
    text: str
    documentId: Optional[str] = None
    location: str = ""              # path, URL or chat id
    project: str = ""               # project the source came from (e.g. a repo), for lesson scope
    sourceNote: Optional[str] = None  # existing _sources note to extend (chat stretches)
    span: str = ""                  # e.g. "turns 12–31"
    context: str = ""               # user's description of the collection / folder it came from


@dataclass
class LinkResult:
    sourceNote: str
    notesTouched: List[str] = field(default_factory=list)
    items: int = 0
    proposedFolders: List[str] = field(default_factory=list)
    seconds: Dict[str, float] = field(default_factory=dict)   # shortlist / write / apply


WRITER_SYSTEM = """You file knowledge from one source into a personal Obsidian-style vault organised by PARA:
- 1-Projects: efforts with a goal and an end (planning a trip, buying a home, a job search, a renovation, a build).
- 2-Areas: the user's own ongoing life with no end date: where they live and their home, their job and employer, health and fitness (including exercise like running), finances, family, friends and pets, hobbies they practise.
- 3-Resources: external topics and reference: media they watch or read, places, products, recipes, tools, how-tos.
- 4-Archive: finished projects and inactive material.

Extract EVERY durable piece of knowledge as an item. Be exhaustive about specifics; they are what people ask about later:
- fact: a true statement about the user's life, possessions, people, numbers, places ("User owns 3 bikes").
- event: something that happened or is scheduled, with its absolute date ("User attended a Valentine's Day fundraiser dinner").
- preference: likes, dislikes, habits, constraints, and how the user wants things done.
- procedure: a reusable how-to or workflow that worked.
- suggestion: what the ASSISTANT recommended, listed, explained or computed. Keep names, numbers and list items verbatim and in order ("Assistant listed 5 shows similar to Game of Thrones: 1. Westworld 2. Vikings ...").
- decision: something decided or agreed in the source.
- lesson: an instruction or correction the user gives the assistant about how to work with them: style, format, tools, workflow, things to always or never do ("Use bun, not npm", "Reply in short bullets", "Run the tests before saying a change is done"). Write it as an imperative. "scope" is "global" unless the user ties the lesson to the current project ("here", "in this repo", "for this project"); then use the project name. Lessons need no note.
Also capture what the user asked for help with as an event ("User asked for tips on styling a vintage band t-shirt").

Rules:
- speaker: "user" for what the user said or did, "assistant" for what the assistant said, "both" for agreed decisions. An assistant suggestion is never a user fact.
- status: "done", "planned" (intentions, bookings not yet taken), "ongoing", "tried", "rejected", or null.
- Resolve relative dates against the source date and write absolute YYYY-MM-DD dates ("last Tuesday" in a 2023-05-28 source -> 2023-05-23). "date" is when the item happened, not when it was mentioned; if only the month is known, use the month's first day and name the month in the text.
- Write each item's text so it stands alone without the source (name the thing, the place, the amount).
- note: the vault note the item belongs to. REUSE an existing note whenever one fits; shortlisted notes are listed with their current facts. Prefer a few broad notes over many narrow ones: one note per area of life or subject ("Fitness" holds running, kickboxing, cardio and nutrition; "Home" holds the apartment, rent and furniture; "Work" holds the job and colleagues). Create a new note only for a distinct subject the user will come back to: a project, a person, a pet, a place they live or travel to, a show or book series. Note titles are short noun phrases ("Fitness", "Lisbon trip", "Game of Thrones").
- new_note: required on every item whose note is not an existing note: "para" is one of projects, areas, resources, archive; "folder" is one from the folder list. Prefer a PARA root or an existing subfolder. Request a new subfolder only if several notes clearly belong together and nothing existing fits; give folder_purpose and folder_reason.
- supersedes: when an item replaces an older current fact in a shortlisted note (a new job, a new count, a changed preference), copy that older fact's text here.
- related: pairs of note titles this source connects.
- If the source has no durable knowledge, return an empty items list.

Return JSON only:
{"source": {"title": "<= 8 words", "summary": "2-4 sentences"},
 "items": [{"kind": "...", "speaker": "...", "status": null, "text": "...", "date": "YYYY-MM-DD or null",
            "note": "...", "new_note": {"para": "...", "folder": "...", "aliases": [], "folder_purpose": "", "folder_reason": ""} or null,
            "scope": null, "supersedes": null}],
 "related": [["Note A", "Note B"]]}"""

KIND_HINT = {
    "chat": "This is a stretch of a chat between the user and an AI assistant. Lines under [earlier context] were already filed; use them only to understand the new turns.",
    "document": "This is a document the user added (a file or web page). Its knowledge is about the topic, not about the user, unless it says otherwise.",
    "fact": "These are facts the user explicitly asked to remember.",
    "trace": "This is the trace of an agent run. File only reusable procedures and decisions.",
}


def _para(value: str) -> str:
    """'areas', '2-Areas' or '2-Areas/Health' -> 'areas' (default 'resources')."""
    v = value.strip().lower()
    for name in ("projects", "areas", "resources", "archive"):
        if name in v.split("/")[0]:
            return name
    return "resources"


def _jsonFrom(raw: str) -> Dict[str, Any]:
    m = re.search(r"\{.*\}", raw or "", re.S)
    return json.loads(m.group(0)) if m else {}


def _chunks(text: str, n: int = 6, size: int = 1500) -> List[str]:
    if len(text) <= size:
        return [text]
    step = max(size, len(text) // n)
    return [text[i:i + size] for i in range(0, len(text), step)][:n]


class VaultLinker:
    def __init__(self, store: Optional[SqliteStore] = None, config: Optional[KormemConfig] = None):
        self.store = store or SqliteStore()
        self.config = config or loadConfig()
        self.embedder = getInternalEmbedder(self.config)  # shortlisting is local and free
        # Existing notes updated during a run are re-embedded once, at flush time.
        self._dirty: Dict[str, set] = {}

    async def flushEmbeddings(self, namespaceId: Optional[str] = None) -> int:
        """Re-embed notes updated since the last flush (once each, at their latest text)."""
        done = 0
        for ns in [namespaceId] if namespaceId else list(self._dirty):
            titles = self._dirty.pop(ns, set())
            vault = VaultStore(store=self.store, config=self.config, namespaceId=ns)
            for t in titles:
                note = vault.readNote(t)
                if note:
                    await embedNote(note, ns, self.store, self.config)
                    done += 1
        return done

    # ------------------------------------------------------------ shortlist

    async def _shortlist(self, unit: ExtractionUnit, vault: VaultStore) -> List[Note]:
        chunks = _chunks(unit.text)
        embs = await self.embedder.encodeBatch(chunks)
        best: Dict[str, float] = {}
        rows: Dict[str, Any] = {}
        for emb in embs:
            for row, sim in self.store.vectorSearch(unit.namespaceId, emb, limit=15, kind="note"):
                meta = json.loads(row["metadata"] or "{}")
                if meta.get("para") in SKIP_PARA:
                    continue
                key = meta.get("kormemId")
                if key and sim > best.get(key, -1):
                    best[key], rows[key] = sim, meta
        if not best:
            return []
        top = sorted(best, key=lambda k: -best[k])[:SHORTLIST_NOTES]
        return [n for n in (vault.readNote(k) for k in top) if n]

    # ---------------------------------------------------------------- write

    def _prompt(self, unit: ExtractionUnit, shortlist: List[Note], folders: List[str]) -> str:
        noteBlocks = []
        for n in shortlist:
            _, secs = parseSections(n.body)
            facts = "\n".join((secs.get("Current facts") or [])[:20])[:NOTE_FACTS_CHARS]
            noteBlocks.append(f"### {n.title}  (folder: {n.folder or PARA_FOLDERS.get(n.para, '')})\n{facts or '(no facts yet)'}")
        return (
            f"{KIND_HINT.get(unit.kind, '')}\n"
            f"Source: {unit.title} ({unit.kind}{', ' + unit.span if unit.span else ''})\n"
            f"Source date: {unit.date}\n"
            + (f"Project: {unit.project}\n" if unit.project else "")
            + (f"What the user says this collection holds: {unit.context}\n" if unit.context else "")
            + "\n"
            f"Folders you may use:\n" + "\n".join(f"- {f}" for f in folders) + "\n\n"
            f"Shortlisted existing notes:\n" + ("\n\n".join(noteBlocks) or "(none yet)") + "\n\n"
            f"--- SOURCE ---\n{unit.text[:UNIT_CHARS]}\n--- END SOURCE ---"
        )

    async def _write(self, prompt: str) -> Dict[str, Any]:
        import os
        effort = os.environ.get("KORMEM_EXTRACTION_REASONING_EFFORT", "low")
        for attempt in range(3):
            try:
                raw = await completeLlm(prompt, WRITER_SYSTEM, config=self.config,
                                        maxTokens=16000, reasoningEffort=effort,
                                        purpose="extract.vault", jsonMode=True)
                return _jsonFrom(raw)
            except Exception as e:
                if attempt == 2:
                    raise
                logger.warning("vault writer failed (%s); retrying", str(e)[:120])
                await asyncio.sleep(2 ** (attempt + 1))
        return {}

    # ---------------------------------------------------------------- apply

    def _sourceTitle(self, unit: ExtractionUnit, topic: str) -> str:
        label = {"chat": "chat", "document": "doc", "fact": "remembered", "trace": "agent run"}.get(unit.kind, unit.kind)
        return noteFileName(f"{unit.date} {label} · {topic or unit.title}")

    async def _writeSourceNote(self, vault: VaultStore, unit: ExtractionUnit, data: Dict[str, Any],
                               feeds: List[str]) -> str:
        src = data.get("source") or {}
        title = unit.sourceNote or self._sourceTitle(unit, str(src.get("title") or "").strip())
        note = vault.readNote(title) or Note.create(title=title, para="source", folder="_sources")
        summary = str(src.get("summary") or "").strip()
        preamble, secs = parseSections(note.body)
        if not preamble:
            meta = [f"type: {unit.kind}", f"date: {unit.date}"]
            if unit.location:
                meta.append(f"location: {unit.location}")
            if unit.documentId:
                meta.append(f"document: {unit.documentId}")
            preamble = "\n".join(meta)
        if summary:
            secs.setdefault("Summary", []).append(f"- {unit.span + ': ' if unit.span else ''}{summary}")
        for f in feeds:
            line = f"- [[{f}]]"
            if line not in secs.setdefault("Feeds", []):
                secs["Feeds"].append(line)
        note.body = renderSections(preamble, secs)
        if unit.documentId and unit.documentId not in note.sources:
            note.sources.append(unit.documentId)
        vault.writeNote(note)
        return title

    async def link(self, unit: ExtractionUnit) -> LinkResult:
        async with _namespaceLocks.setdefault(unit.namespaceId, asyncio.Lock()):
            return await self._link(unit)

    async def _link(self, unit: ExtractionUnit) -> LinkResult:
        vault = VaultStore(store=self.store, config=self.config, namespaceId=unit.namespaceId)
        vault.ensureMetaNotes()
        folders = FolderRegistry(self.store, vault, self.embedder)
        t0 = time.monotonic()
        shortlist = await self._shortlist(unit, vault)
        t1 = time.monotonic()
        data = await self._write(self._prompt(unit, shortlist, folders.tree()))
        t2 = time.monotonic()
        items = [i for i in data.get("items") or [] if isinstance(i, dict) and str(i.get("text") or "").strip()]

        # The source note title is needed for links before topic notes are written.
        sourceTitle = unit.sourceNote or self._sourceTitle(unit, str((data.get("source") or {}).get("title") or ""))
        touched: Dict[str, Note] = {}
        proposed: List[str] = []
        memory: List[Tuple[str, str, str, str]] = []  # (date, priority, text, links)
        created: set = set()

        # The writer may give a new note's placement on only one of its items.
        specFor: Dict[str, Dict[str, Any]] = {}
        for item in items:
            if isinstance(item.get("new_note"), dict):
                specFor.setdefault(str(item.get("note") or "").strip().lower(), item["new_note"])

        lessonItems = [i for i in items if str(i.get("kind") or "").lower() == "lesson"]
        items = [i for i in items if i not in lessonItems]
        lessons = []
        for item in lessonItems:
            date = item.get("date") or unit.date
            text = str(item["text"]).strip()
            scope = item.get("scope") or None
            lessons.append((text, scope, date, item.get("supersedes")))
            where = f" (in {scope})" if scope and str(scope).lower() != "global" else ""
            memory.append((date, "🔴", f"User asked the assistant: {text}{where}", f"[[lessons]] [[{sourceTitle}]]"))
        if lessons:
            addLessons(vault, lessons, sourceTitle)

        for item in items:
            title = str(item.get("note") or "").strip() or "Inbox"
            note = touched.get(title.lower())
            if note is None:
                note = vault.readNote(title)
                if note is None or note.para in SKIP_PARA:
                    spec = item.get("new_note") or specFor.get(title.lower()) or {}
                    para = _para(str(spec.get("para") or spec.get("folder") or ""))
                    folder, proposal = await folders.place(
                        str(spec.get("folder") or ""), para, title,
                        str(spec.get("folder_purpose") or ""), str(spec.get("folder_reason") or ""),
                    )
                    if proposal:
                        proposed.append(proposal)
                    note = Note.create(title=title, para=para, folder=folder)
                    note.aliases = [str(a) for a in spec.get("aliases") or []][:5]
                    created.add(title.lower())
                elif note.archivedFrom:  # new knowledge brings an archived note back
                    vault.moveNote(note, note.archivedFrom)
                    note.archivedFrom = ""
                touched[title.lower()] = note
            preamble, secs = parseSections(note.body)
            kind = str(item.get("kind") or "fact").lower()
            if kind not in KIND_SECTION:
                kind = "fact"
            date = item.get("date") or unit.date
            text = str(item["text"]).strip()
            status = item.get("status")
            if status and status not in ("done", None):
                text = f"{text} ({status})"
            if item.get("supersedes"):
                old = supersede(secs, str(item["supersedes"]), date)
                # The replaced fact's memory line stops being scored by recall.
                retireLine(vault, self.store, note.title, old or str(item["supersedes"]), date)
            bulletText = text if kind != "fact" and kind != "preference" else f"{text} (as of {date})"
            addBullet(secs, KIND_SECTION[kind], bulletText, sourceTitle, date)
            note.body = renderSections(preamble, secs)
            if unit.documentId and unit.documentId not in note.sources:
                note.sources.append(unit.documentId)
            memory.append((date, KIND_PRIORITY[kind], text, f"[[{note.title}]] [[{sourceTitle}]]"))

        for pair in data.get("related") or []:
            if isinstance(pair, list) and len(pair) == 2:
                a, b = (touched.get(str(x).lower()) for x in pair)
                if a and b and a is not b:
                    for x, y in ((a, b), (b, a)):
                        _, secs = parseSections(x.body)
                        addRelated(secs, y.title)
                        x.body = renderSections(parseSections(x.body)[0], secs)

        for key, note in touched.items():
            vault.writeNote(note)
            if key in created:  # new notes must be findable by the next unit right away
                await embedNote(note, unit.namespaceId, self.store, self.config)
            else:
                self._dirty.setdefault(unit.namespaceId, set()).add(note.title)
        await self._writeSourceNote(vault, unit, data,
                                    [n.title for n in touched.values()] + (["lessons"] if lessons else []))

        for path, titles in await folders.promoteReady():
            for t in titles:
                n = vault.readNote(t)
                if n and n.folder != path and n.para == PARA_ROOTS.get(path.split("/")[0]):
                    vault.moveNote(n, path)
        if proposed:
            folders.writeRegistryNote()
        newLines = appendMemoryLines(vault, unit.date, memory)
        await lineEmbeddings(self.store, self.embedder, unit.namespaceId, [cleanLine(l) for l in newLines])
        return LinkResult(sourceNote=sourceTitle, notesTouched=[n.title for n in touched.values()],
                          items=len(items) + len(lessons), proposedFolders=proposed,
                          seconds={"shortlist": t1 - t0, "write": t2 - t1, "apply": time.monotonic() - t2})
