"""Guidance retrieval from PARA vault notes + semantic memory search."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import List, Optional

from kormem.core.config import KormemConfig, loadConfig
from kormem.core.schemas import GuidanceItem
from kormem.embeddings.provider import getEmbeddingProvider
from kormem.extract.memory_utils import enrich_memory_with_relative_time
from kormem.retrieve.memory_search import search_memory_for_query
from kormem.retrieve.recency import recencyBoost
from kormem.store.sqlite_store import SqliteStore
from kormem.extract.lessons import lessonsFor
from kormem.extract.memory_index import readMemory
from kormem.vault.vault_store import VaultStore

SEMANTIC_MEMORY_THRESHOLD = 300


class GuidanceSearch:
    def __init__(
        self,
        store: Optional[SqliteStore] = None,
        config: Optional[KormemConfig] = None,
    ):
        self.store = store or SqliteStore()
        self.config = config or loadConfig()
        self.embedder = getEmbeddingProvider(self.config)

    async def search(
        self,
        query: str,
        namespaceId: Optional[str] = None,
        limit: int = 5,
        queryDate: Optional[str] = None,
    ) -> List[GuidanceItem]:
        namespaceId = namespaceId or self.config.defaultNamespace
        vault = VaultStore(store=self.store, config=self.config, namespaceId=namespaceId)
        vault.ensureMetaNotes()

        items: List[GuidanceItem] = []
        queryWords = set(re.findall(r"\w+", query.lower()))

        def scoreText(text: str, itemType: str, updated: str = "") -> float:
            if not text:
                return 0.0
            words = set(re.findall(r"\w+", text.lower()))
            overlap = len(queryWords & words) / max(len(queryWords), 1)
            sc = 0.5 + overlap
            return recencyBoost(sc, updated) if updated else sc

        memoryNote = vault.readNote("_meta/memory.md")
        memoryBody = readMemory(vault)
        if memoryNote and len(memoryBody) > SEMANTIC_MEMORY_THRESHOLD:
            memoryText = memoryBody
            if queryDate:
                memoryText = enrich_memory_with_relative_time(memoryText, queryDate)
            filtered = await search_memory_for_query(
                query=query,
                memory_text=memoryText,
                memory_char_limit=8000,
                reference_date=queryDate,
            )
            if filtered and filtered != "__NO_RELEVANT_MEMORY__":
                items.append(GuidanceItem(
                    itemType="memory",
                    phrase=filtered[:2000],
                    score=0.95,
                    lastUpdated=datetime.fromisoformat(memoryNote.updated.replace("Z", "+00:00"))
                    if memoryNote.updated else None,
                    metadata={
                        "kormemId": memoryNote.kormemId,
                        "relPath": "_meta/memory.md",
                        "semanticSearch": True,
                    },
                ))
        elif memoryNote:
            for line in memoryBody.split("\n"):
                line = line.strip()
                if len(line) < 10:
                    continue
                sc = scoreText(line, "memory", memoryNote.updated)
                if sc > 0.5 or not queryWords:
                    items.append(GuidanceItem(
                        itemType="memory",
                        phrase=line[:500],
                        score=sc,
                        lastUpdated=datetime.fromisoformat(memoryNote.updated.replace("Z", "+00:00"))
                        if memoryNote.updated else None,
                        metadata={"kormemId": memoryNote.kormemId, "relPath": "_meta/memory.md"},
                    ))

        for name, itemType in [("lessons", "lesson")]:
            note = vault.readNote(f"_meta/{name}.md")
            if not note:
                continue
            for line in lessonsFor(vault).split("\n"):  # global lessons; retired ones never load
                line = line.strip()
                if len(line) < 10:
                    continue
                sc = scoreText(line, itemType, note.updated)
                if sc > 0.5 or not queryWords:
                    items.append(GuidanceItem(
                        itemType=itemType,
                        phrase=line[:500],
                        score=sc,
                        lastUpdated=datetime.fromisoformat(note.updated.replace("Z", "+00:00"))
                        if note.updated else None,
                        metadata={"kormemId": note.kormemId, "relPath": f"_meta/{name}.md"},
                    ))

        qEmb = await self.embedder.encodeQuery(query)
        noteHits = self.store.vectorSearch(
            namespaceId, qEmb, limit=limit, kind="note",
        )
        for row, sim in noteHits:
            meta = json.loads(row["metadata"] or "{}")
            text = self.store.getSegmentText(row, self.config.encryptionKey)
            updated = meta.get("updated") or row["updatedAt"] or ""
            sc = recencyBoost(sim, updated)
            items.append(GuidanceItem(
                itemType="memory",
                phrase=text[:500],
                score=sc,
                metadata={
                    "kormemId": meta.get("kormemId"),
                    "relPath": meta.get("relPath"),
                    "title": meta.get("title"),
                    "entryNote": True,
                },
            ))

        items.sort(key=lambda x: -x.score)
        return items[:limit]
