"""Extraction: settled sources -> linked vault notes, lessons and the memory index.

Every source kind goes through one paid call:
:class:`~kormem.extract.vault_linker.VaultLinker` files knowledge into
``_sources/`` and PARA topic notes, lessons into ``_meta/lessons.md``, and
linked lines into the monthly memory index (``_meta/memory/``).

Units per kind:

- document (file, URL, text): one unit per ~8k-token section; sections whose
  text was already extracted in this namespace are skipped.
- chat: the settled stretch of turns plus the 4 turns before it as context.
- fact (``remember``): every settled fact of a namespace in one unit.
- trace (agent run): one unit, procedures and decisions only.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, List, Optional

from kormem.core.config import KormemConfig, loadConfig
from kormem import library
from kormem.core.ids import contentHash
from kormem.extract.vault_linker import ExtractionUnit, VaultLinker
from kormem.ingest.chat_sessions import chatTurns, documentText, enqueueSettledChats
from kormem.store.sqlite_store import SqliteStore

logger = logging.getLogger(__name__)

SECTION_CHARS = 32_000          # ~8k tokens
CHAT_CONTEXT_TURNS = 4
FACT_DEBOUNCE_MINUTES = float(os.environ.get("KORMEM_FACT_DEBOUNCE_MINUTES", "5"))
FACT_BATCH = int(os.environ.get("KORMEM_FACT_BATCH", "20"))
_HEADING_RE = re.compile(r"(?m)^(?=#{1,3}\s)")


def splitSections(text: str, maxChars: int = SECTION_CHARS) -> List[str]:
    """Split on markdown headings, packing consecutive parts up to ``maxChars``."""
    parts = [p for p in _HEADING_RE.split(text) if p.strip()] or [text]
    sections: List[str] = []
    cur = ""
    for part in parts:
        while len(part) > maxChars:  # one huge heading-less part: hard split
            sections.append(part[:maxChars])
            part = part[maxChars:]
        if cur and len(cur) + len(part) > maxChars:
            sections.append(cur)
            cur = ""
        cur += part
    if cur.strip():
        sections.append(cur)
    return sections


def _date(meta: dict, fallback: str) -> str:
    m = re.search(r"(\d{4})[/-](\d{2})[/-](\d{2})", str(meta.get("date") or fallback or ""))
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else (fallback or "")[:10]


class ExtractionService:
    def __init__(self, store: Optional[SqliteStore] = None, config: Optional[KormemConfig] = None):
        self.store = store or SqliteStore()
        self.config = config or loadConfig()
        self.linker = VaultLinker(self.store, self.config)

    # -------------------------------------------------------------- units

    def _docText(self, documentId: Optional[str]) -> tuple[str, Any]:
        doc = self.store.getDocument(documentId) if documentId else None
        return documentText(self.store, doc, self.config.encryptionKey), doc

    def _documentUnits(self, job) -> List[ExtractionUnit]:
        text, doc = self._docText(job["documentId"])
        if not text.strip():
            return []
        meta = json.loads(doc["metadata"] or "{}")
        sections = splitSections(text)
        hashes = [contentHash(s) for s in sections]
        done = self.store.extractedSectionHashes(job["namespaceId"], hashes)
        context = library.contextFor(library.documentUri(doc))
        units = []
        for i, (sec, h) in enumerate(zip(sections, hashes)):
            if h in done:
                continue
            units.append(ExtractionUnit(
                namespaceId=job["namespaceId"], kind="document", title=doc["title"] or "Untitled",
                date=_date(meta, doc["createdAt"]), text=sec, documentId=doc["id"],
                location=doc["canonicalUrl"] or "", sourceNote=meta.get("sourceNote"), span=f"part {i + 1}/{len(sections)}" if len(sections) > 1 else "",
                context=context,
            ))
            units[-1].sectionHash = h  # type: ignore[attr-defined]
        return units

    def _chatUnit(self, job) -> Optional[ExtractionUnit]:
        payload = json.loads(job["payload"] or "{}")
        sessionId = payload.get("sessionId")
        session = self.store.getChatSession(job["namespaceId"], sessionId)
        turns = chatTurns(self.store, job["namespaceId"], sessionId, self.config.encryptionKey)
        start, end = job["spanStart"], job["spanEnd"]
        context = [t for t in turns if start - CHAT_CONTEXT_TURNS <= t["idx"] < start]
        fresh = [t for t in turns if start <= t["idx"] <= end]
        if not fresh:
            return None
        lines = []
        if context:
            lines.append("[earlier context]")
            lines += [f"[{t['role']}]: {t['content']}" for t in context]
            lines.append("[new turns]")
        lines += [f"[{t['role']}]: {t['content']}" for t in fresh]
        doc = self.store.getDocument(session["documentId"]) if session and session["documentId"] else None
        meta = json.loads(doc["metadata"] or "{}") if doc else {}
        return ExtractionUnit(
            namespaceId=job["namespaceId"], kind="chat", title=f"chat {sessionId}",
            date=_date({"date": session["sessionDate"] if session else None}, job["createdAt"]),
            text="\n\n".join(lines), documentId=session["documentId"] if session else None,
            location=f"chat://{job['namespaceId']}/{sessionId}", sourceNote=meta.get("sourceNote"),
            project=meta.get("project") or "",
            span=f"turns {start + 1}–{end + 1}",
        )

    def _factUnit(self, jobs) -> Optional[ExtractionUnit]:
        lines, date = [], ""
        for j in jobs:
            text, doc = self._docText(j["documentId"])
            if text.strip():
                d = _date(json.loads(doc["metadata"] or "{}"), doc["createdAt"])
                date = date or d
                lines.append(f"({d}) {text.strip()}")
        if not lines:
            return None
        return ExtractionUnit(namespaceId=jobs[0]["namespaceId"], kind="fact", title="remembered facts",
                              date=date, text="\n".join(lines))

    def _traceUnit(self, job) -> Optional[ExtractionUnit]:
        payload = json.loads(job["payload"] or "{}")
        text = str(payload.get("text") or "")
        if not text.strip():
            return None
        return ExtractionUnit(namespaceId=job["namespaceId"], kind="trace",
                              title=str(payload.get("title") or "agent run"),
                              date=str(payload.get("date") or job["createdAt"])[:10], text=text)

    # ------------------------------------------------------------ process

    async def _extractUnit(self, unit: ExtractionUnit) -> str:
        """Link one unit into the vault; returns its source note title."""
        result = await self.linker.link(unit)
        if unit.documentId and not unit.sourceNote:
            self.store.mergeDocumentMeta(unit.documentId, {"sourceNote": result.sourceNote})
        h = getattr(unit, "sectionHash", None)
        if h:
            self.store.markSectionsExtracted(unit.namespaceId, [h])
        logger.info("extracted %s %s: %d items -> %s | %s", unit.kind, unit.title, result.items,
                    ", ".join(result.notesTouched[:6]),
                    " ".join(f"{k}={v:.1f}s" for k, v in result.seconds.items()))
        return result.sourceNote

    async def processJob(self, job) -> None:
        kind = (job["kind"] if "kind" in job.keys() else None) or "document"
        if kind == "chat":
            unit = self._chatUnit(job)
            units = [unit] if unit else []
        elif kind == "trace":
            unit = self._traceUnit(job)
            units = [unit] if unit else []
        else:
            units = self._documentUnits(job)
        sourceNote = None
        for unit in units:  # sections of one document share its source note
            unit.sourceNote = unit.sourceNote or sourceNote
            sourceNote = await self._extractUnit(unit)
        self.store.completeExtraction(job["id"])

    async def processPending(self, limit: int = 3, namespaceId: Optional[str] = None) -> int:
        """Queue settled chats, then extract up to ``limit`` jobs. Returns jobs handled."""
        enqueueSettledChats(self.store)
        handled = 0
        for batch in self.store.claimFactBatches(FACT_DEBOUNCE_MINUTES, FACT_BATCH):
            if namespaceId and batch[0]["namespaceId"] != namespaceId:
                self.store.setExtractionJobsStatus([j["id"] for j in batch], "pending")
                continue
            try:
                unit = self._factUnit(batch)
                if unit:
                    await self._extractUnit(unit)
                for j in batch:
                    self.store.completeExtraction(j["id"])
            except Exception as e:
                logger.exception("fact batch failed")
                for j in batch:
                    self.store.failExtraction(j["id"], str(e))
            handled += len(batch)
        for job in self.store.claimExtractionJobs(limit, namespaceId, excludeKinds=("fact",)):
            try:
                await self.processJob(job)
            except Exception as e:
                logger.exception("extraction job %s failed", job["id"])
                self.store.failExtraction(job["id"], str(e))
            handled += 1
        await self.linker.flushEmbeddings(namespaceId)
        return handled
