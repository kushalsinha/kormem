"""Maintenance: keep a growing memory small, current and fast to recall.

Runs per namespace (daily from the server's worker, or ``kormem maintain``).
Every step but one is free:

1. Memory index    — a single-file ``_meta/memory.md`` is split into month
                     shards; vectors of memory lines older than the recall
                     window are dropped (recomputed on demand if needed).
2. Index storage   — chunk text stored on a row becomes a span into its
                     document; note bodies are read from the vault; vectors are
                     re-packed as int8.
3. Archive         — topic notes with no activity for ``KORMEM_ARCHIVE_DAYS``
                     (180) move to ``4-Archive``. They stay searchable, recall
                     reaches them when no current note matches, and new
                     knowledge moves them back.
4. Merge           — near-duplicate topic notes (vector similarity
                     ``KORMEM_MERGE_SIMILARITY``, 0.92) are merged into the
                     longer one; the shorter title becomes an alias and every
                     link to it is rewritten.
5. Condense (paid) — a note over ``KORMEM_NOTE_MAX_CHARS`` (6000) gets its
                     older Timeline and assistant bullets condensed into
                     "Earlier history" by one cheap model call. Source links
                     are always kept, so the agent can go back to the
                     original chat.
6. Optional        — ``KORMEM_DROP_VECTORS_AFTER_DAYS`` (0 = off) drops chunk
                     vectors of sources older than that; they stay keyword
                     searchable.

"Time" is the namespace's activity clock (its newest source date), so
imported history ages the same way as live use.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from kormem.core.config import KormemConfig, loadConfig
from kormem.embeddings.provider import getInternalEmbedder
from kormem.extract.llm_client import completeLlm
from kormem.extract.memory_index import (
    cleanLine, migrateLegacy, months, parseBlocks, shardTitle, _textKey,
)
from kormem.extract.note_embed import embedNote
from kormem.ingest.service import chunkSpans
from kormem.retrieve.temporal import parseQueryDate
from kormem.store.sqlite_store import NOTE_SENTINEL, SqliteStore, _packEmbedding, _unpackEmbedding
from kormem.vault.note_sections import mergeSections, parseSections, renderSections
from kormem.vault.vault_store import VaultStore

logger = logging.getLogger(__name__)

ARCHIVE_DAYS = int(os.environ.get("KORMEM_ARCHIVE_DAYS", "180"))
MERGE_SIMILARITY = float(os.environ.get("KORMEM_MERGE_SIMILARITY", "0.92"))
NOTE_MAX_CHARS = int(os.environ.get("KORMEM_NOTE_MAX_CHARS", "6000"))
DROP_VECTORS_DAYS = int(os.environ.get("KORMEM_DROP_VECTORS_AFTER_DAYS", "0"))
KEEP_RECENT = 15                # at most this many newest bullets per section stay verbatim ...
KEEP_RECENT_CHARS = 3000        # ... within this many characters (but at least KEEP_MIN)
KEEP_MIN = 3
MIN_TO_CONDENSE = 2000          # fewer older characters than this: not worth a call
TOPIC_PARA = ("projects", "areas", "resources")
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")

CONDENSE_SYSTEM = """You condense the older bullets of one note in a personal knowledge vault. The original chats stay linked, so detail is never lost; this summary is what a reader skims first.
Aim for about a third of the input length, in at most 10 bullets:
- Keep what is specific to this user and what they might ask about later: names, numbers, dates (YYYY-MM-DD), places, products, and list items in their original order ("Assistant listed 7 beginner yoga poses: Mountain, Downward Dog, Tree, ...").
- Keep who said it: "User ..." for the user's own facts, "Assistant suggested/listed ..." for the assistant's.
- Cut generic advice wording, explanations and filler; keep the item names.
- Merge bullets about the same thing.
End each bullet with the [[source]] links of the bullets it covers, copied exactly.
Return JSON only: {"bullets": ["- ... [[source]]", ...]}"""


def _day(value: str) -> Optional[datetime]:
    d = parseQueryDate(value) if value else None
    return d.replace(tzinfo=timezone.utc) if d and d.tzinfo is None else d


class MaintenanceService:
    def __init__(self, store: Optional[SqliteStore] = None, config: Optional[KormemConfig] = None):
        self.store = store or SqliteStore()
        self.config = config or loadConfig()

    async def run(self, namespaceId: str, condense: bool = True) -> Dict[str, Any]:
        vault = VaultStore(store=self.store, config=self.config, namespaceId=namespaceId)
        clock = self.activityClock(namespaceId)
        report: Dict[str, Any] = {"namespaceId": namespaceId, "clock": clock.date().isoformat(), "seconds": {}}
        steps = [
            ("memoryLinesMoved", lambda: migrateLegacy(vault)),
            ("memoryVectorsDropped", lambda: self.dropOldMemoryVectors(vault, clock)),
            ("segmentsCompacted", lambda: self.compactSegments(namespaceId)),
            ("vectorsRepacked", lambda: self.repackVectors(namespaceId)),
            ("notesArchived", lambda: self.archiveCold(vault, clock)),
            ("notesMerged", lambda: self.mergeDuplicates(vault)),
            ("notesCondensed", lambda: self.condenseLong(vault) if condense else 0),
            ("chunkVectorsDropped", lambda: self.dropOldChunkVectors(namespaceId, clock) if DROP_VECTORS_DAYS else 0),
        ]
        for name, step in steps:
            t = time.monotonic()
            value = step()
            report[name] = await value if asyncio.iscoroutine(value) else value
            report["seconds"][name] = round(time.monotonic() - t, 2)
        return report

    # ----------------------------------------------------------- the clock

    def activityClock(self, namespaceId: str) -> datetime:
        """The namespace's 'now': its newest source date, else the wall clock."""
        newest = None
        for r in self.store.documentsForNamespace(namespaceId):
            d = _day(str(json.loads(r["metadata"] or "{}").get("date") or ""))
            if d and (newest is None or d > newest):
                newest = d
        return newest or datetime.now(timezone.utc)

    # -------------------------------------------------------- memory index

    def dropOldMemoryVectors(self, vault: VaultStore, clock: datetime) -> int:
        """Memory lines older than recall's scan window are not scored; drop their vectors."""
        from kormem.retrieve.recall import MEMORY_SCAN_MONTHS

        if MEMORY_SCAN_MONTHS <= 0:
            return 0
        y, m = clock.year, clock.month - MEMORY_SCAN_MONTHS
        while m < 1:
            y, m = y - 1, m + 12
        cutoff = f"{y:04d}-{m:02d}"
        old = set()
        for month in months(vault):
            if month >= cutoff:
                break
            shard = vault.readNote(shardTitle(month))
            for lines in parseBlocks(shard.body if shard else "")[1].values():
                old.update(_textKey(cleanLine(line)) for line in lines)
        return self.store.deleteMemoryVectors(vault.namespaceId, old)

    # ------------------------------------------------------- index storage

    def compactSegments(self, namespaceId: str) -> int:
        """Rows that still carry their text: chunks keep a span into the document, notes keep nothing."""
        key = self.config.encryptionKey
        byDoc: Dict[str, List[Any]] = {}
        updates: List[Dict[str, Any]] = []
        for r in self.store.segmentRows(namespaceId):
            if not (r["plainText"] or r["encryptedText"]):
                continue
            if r["kind"] == "note":
                meta = json.loads(r["metadata"] or "{}")
                if self.store._noteBody(namespaceId, meta.get("kormemId") or ""):
                    updates.append({"segmentKey": r["segmentKey"], "plainText": None, "encryptedText": None})
            elif r["kind"] == "segment" and r["documentId"] != NOTE_SENTINEL:
                byDoc.setdefault(r["documentId"], []).append(r)
        for docId, rows in byDoc.items():
            rows.sort(key=lambda r: r["segmentIndex"])
            text = self.store.documentTextCached(docId, key)
            if not text:
                continue
            texts = [self.store.getSegmentText(r, key) for r in rows]
            for r, t, span in zip(rows, texts, chunkSpans(text, texts)):
                if span is None or text[span[0]:span[1]] != t:
                    continue
                meta = {k: v for k, v in json.loads(r["metadata"] or "{}").items() if k not in ("startPos", "endPos")}
                meta["span"] = list(span)
                updates.append({"segmentKey": r["segmentKey"], "plainText": None, "encryptedText": None,
                                "metadata": meta})
        self.store.rewriteSegments(updates)
        return len(updates)

    def repackVectors(self, namespaceId: str) -> int:
        """float16 / float32 vectors from older versions become int8."""
        updates = []
        for r in self.store.segmentRows(namespaceId, columns="segmentKey, embedding, embeddingDim"):
            blob, dim = r["embedding"], r["embeddingDim"]
            if blob and dim and len(blob) in (2 * dim, 4 * dim):
                updates.append({"segmentKey": r["segmentKey"],
                                "embedding": _packEmbedding(_unpackEmbedding(blob, dim))})
        self.store.rewriteSegments(updates)
        return len(updates)

    def dropOldChunkVectors(self, namespaceId: str, clock: datetime) -> int:
        """Opt-in: chunks of sources older than ``DROP_VECTORS_DAYS`` keep keyword search only."""
        cutoff = clock - timedelta(days=DROP_VECTORS_DAYS)
        old = {r["id"] for r in self.store.documentsForNamespace(namespaceId)
               if (d := _day(str(json.loads(r["metadata"] or "{}").get("date") or ""))) and d < cutoff}
        updates = [{"segmentKey": r["segmentKey"], "embedding": None, "embeddingDim": 0}
                   for r in self.store.segmentRows(namespaceId, kind="segment",
                                                   columns="segmentKey, documentId, embedding")
                   if r["documentId"] in old and r["embedding"]]
        self.store.rewriteSegments(updates)
        return len(updates)

    # ------------------------------------------------------------- archive

    def _topicNotes(self, vault: VaultStore, para=TOPIC_PARA):
        for meta in vault.listNotes():
            if meta.para in para:
                note = vault.readNote(meta.kormemId)
                if note:
                    yield note

    def archiveCold(self, vault: VaultStore, clock: datetime) -> int:
        """Topic notes whose newest dated bullet is older than ``ARCHIVE_DAYS`` move to 4-Archive."""
        cutoff = (clock - timedelta(days=ARCHIVE_DAYS)).date().isoformat()
        moved = 0
        for note in list(self._topicNotes(vault)):
            dates = _DATE_RE.findall(note.body)
            if not dates or max(dates) >= cutoff:
                continue
            note.archivedFrom = note.folder or vault.store.resolveNote(vault.namespaceId, note.kormemId)["relPath"].rsplit("/", 1)[0]
            vault.moveNote(note, "4-Archive")
            self._refreshNoteMeta(vault, note)
            moved += 1
        return moved

    def _refreshNoteMeta(self, vault: VaultStore, note) -> None:
        """Keep the note vector's metadata (para, path) in step without re-embedding."""
        from kormem.vault.paths import relPathFor

        for r in self.store.getSegmentsByKeys([f"note:{note.kormemId}"]):
            meta = json.loads(r["metadata"] or "{}")
            meta.update({"para": note.para, "relPath": relPathFor(note)})
            self.store.rewriteSegments([{"segmentKey": r["segmentKey"], "metadata": meta}])

    # --------------------------------------------------------------- merge

    async def mergeDuplicates(self, vault: VaultStore) -> int:
        rows = []
        for r in self.store.segmentRows(vault.namespaceId, kind="note"):
            meta = json.loads(r["metadata"] or "{}")
            if r["embedding"] and meta.get("para") in TOPIC_PARA:
                rows.append((meta, _unpackEmbedding(r["embedding"], r["embeddingDim"])))
        if len(rows) < 2:
            return 0
        mat = np.stack([v for _, v in rows])
        mat /= np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9)
        sims = mat @ mat.T
        np.fill_diagonal(sims, 0)
        gone: set = set()
        merged = 0
        iu, ju = np.triu_indices(len(rows), 1)
        pairs = [(int(i), int(j)) for i, j in zip(iu, ju) if sims[i, j] >= MERGE_SIMILARITY]
        for i, j in sorted(pairs, key=lambda p: -sims[p]):
            a, b = rows[i][0], rows[j][0]
            if a["kormemId"] in gone or b["kormemId"] in gone or a.get("para") != b.get("para"):
                continue
            na, nb = vault.readNote(a["kormemId"]), vault.readNote(b["kormemId"])
            if not na or not nb:
                continue
            keep, drop = (na, nb) if len(na.body) >= len(nb.body) else (nb, na)
            await self._merge(vault, keep, drop)
            gone.add(drop.kormemId)
            merged += 1
        return merged

    async def _merge(self, vault: VaultStore, keep, drop) -> None:
        logger.info("merging note %r into %r", drop.title, keep.title)
        backlinks = vault.getBacklinks(drop.kormemId)
        pre, secs = parseSections(keep.body)
        _, other = parseSections(drop.body.replace(f"[[{keep.title}]]", ""))
        mergeSections(secs, other, skipLink=keep.title)
        keep.body = renderSections(pre, secs)
        keep.aliases = list(dict.fromkeys(keep.aliases + [drop.title] + drop.aliases))[:20]
        keep.sources = list(dict.fromkeys(keep.sources + drop.sources))
        vault.writeNote(keep)
        vault.deleteNote(drop)
        self.store.repointLinks(drop.kormemId, keep.kormemId)
        pattern = re.compile(r"\[\[" + re.escape(drop.title) + r"(\|[^\]]*)?\]\]", re.I)
        for meta in backlinks:
            if meta.kormemId == keep.kormemId:
                continue
            n = vault.readNote(meta.kormemId)
            if n:
                n.body = pattern.sub(f"[[{keep.title}]]", n.body)
                vault.writeNote(n)
        await embedNote(keep, vault.namespaceId, self.store, self.config)

    # ------------------------------------------------------------ condense

    async def condenseLong(self, vault: VaultStore) -> int:
        done = 0
        for note in list(self._topicNotes(vault, TOPIC_PARA + ("archive",))):
            if len(note.body) <= NOTE_MAX_CHARS:
                continue
            pre, secs = parseSections(note.body)
            older: List[str] = []
            keepFrom: Dict[str, int] = {}
            for name in ("Timeline", "From the assistant"):
                lines = secs.get(name) or []
                keepFrom[name] = _keepFrom(lines)
                older += lines[:keepFrom[name]]
            if sum(map(len, older)) < MIN_TO_CONDENSE:
                continue
            earlier = secs.get("Earlier history") or []
            try:
                raw = await completeLlm(
                    f"Note: {note.title}\n\nBullets to condense:\n" + "\n".join(earlier + older),
                    CONDENSE_SYSTEM, config=self.config, maxTokens=4000, reasoningEffort="low",
                    purpose="maintain.condense", jsonMode=True)
                m = re.search(r"\{.*\}", raw or "", re.S)
                bullets = [str(b).strip() for b in (json.loads(m.group(0)).get("bullets") if m else []) if str(b).strip()]
            except Exception as e:
                logger.warning("condensing %r failed: %s", note.title, str(e)[:200])
                continue
            if not bullets:
                continue
            bullets = [b if b.startswith("- ") else f"- {b}" for b in bullets]
            # Every source the old bullets cited stays reachable from the note.
            cited = set(_LINK_RE.findall("\n".join(bullets)))
            missing = [l for l in dict.fromkeys(_LINK_RE.findall("\n".join(earlier + older))) if l not in cited]
            if missing:
                bullets.append("- Sources: " + " ".join(f"[[{l}]]" for l in missing))
            for name, start in keepFrom.items():
                secs[name] = (secs.get(name) or [])[start:]
            secs["Earlier history"] = bullets
            note.body = renderSections(pre, secs)
            vault.writeNote(note)
            await embedNote(note, vault.namespaceId, self.store, self.config)
            done += 1
        return done


def _keepFrom(lines: List[str]) -> int:
    """Index of the first bullet kept verbatim: the newest ones, within KEEP_RECENT and KEEP_RECENT_CHARS."""
    used, start = 0, len(lines)
    while start > 0 and len(lines) - start < KEEP_RECENT:
        n = len(lines[start - 1])
        if len(lines) - start >= KEEP_MIN and used + n > KEEP_RECENT_CHARS:
            break
        used += n
        start -= 1
    return start


async def maintainAll(store: SqliteStore, config: KormemConfig, namespaces: List[str],
                      condense: bool = True, vacuum: bool = False) -> List[Dict[str, Any]]:
    service = MaintenanceService(store, config)
    reports = [await service.run(ns, condense=condense) for ns in namespaces]
    if vacuum:
        store.vacuum()
    return reports
