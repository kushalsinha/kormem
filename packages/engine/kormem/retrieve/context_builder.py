"""Structured context builder for integrators."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from kormem.core.config import KormemConfig, loadConfig
from kormem.core.schemas import GuidanceItem, KnowledgeItem
from kormem.extract.memory_utils import enrich_memory_with_relative_time
from kormem.retrieve.guidance import GuidanceSearch
from kormem.retrieve.knowledge import KnowledgeSearch
from kormem.retrieve.memory_search import search_memory_for_query
from kormem.store.sqlite_store import SqliteStore
from kormem.extract.lessons import lessonsFor
from kormem.extract.memory_index import readMemory
from kormem.vault.vault_store import VaultStore

_PREFERENCE_RE = re.compile(
    r"\b(recommend|suggest|prefer|like|enjoy|favorite|favourite|interested in|"
    r"what should i|help me (choose|decide|pick)|any tips)\b",
    re.I,
)
_TEMPORAL_RE = re.compile(
    r"\b(when|how long|how many|how often|latest|recent|current|now|before|after|"
    r"updated|changed|total|count)\b",
    re.I,
)


@dataclass
class GuidanceBrief:
    behavioralGuidance: str = ""
    relevantMemory: str = ""
    relevantSkills: str = ""
    tacticalPlaybooks: str = ""
    rawPersona: str = ""
    rawMemory: str = ""
    rawPlaybooks: str = ""


@dataclass
class ContextBundle:
    query: str
    queryDate: Optional[str]
    contextOrdering: Literal["memory-before-sources", "sources-before-memory"]
    guidanceBrief: GuidanceBrief
    knowledge: List[KnowledgeItem] = field(default_factory=list)
    guidance: List[GuidanceItem] = field(default_factory=list)


class ContextBuilder:
    def __init__(
        self,
        store: Optional[SqliteStore] = None,
        config: Optional[KormemConfig] = None,
    ):
        self.config = config or loadConfig()
        self.store = store or SqliteStore()
        self.knowledge = KnowledgeSearch(self.store, self.config)
        self.guidance = GuidanceSearch(self.store, self.config)

    def resolveContextOrdering(self, query: str) -> Literal["memory-before-sources", "sources-before-memory"]:
        if _PREFERENCE_RE.search(query):
            return "memory-before-sources"
        if _TEMPORAL_RE.search(query):
            return "sources-before-memory"
        return "memory-before-sources"

    async def produceGuidanceBrief(
        self,
        query: str,
        namespaceId: str,
        queryDate: Optional[str] = None,
    ) -> GuidanceBrief:
        vault = VaultStore(store=self.store, config=self.config, namespaceId=namespaceId)
        vault.ensureMetaNotes()
        memory = readMemory(vault)
        lessons = lessonsFor(vault)

        if memory and queryDate:
            memory = enrich_memory_with_relative_time(memory, queryDate)

        relevantMemory = ""
        if len(memory) > 300:
            relevantMemory = await search_memory_for_query(
                query=query,
                memory_text=memory,
                memory_char_limit=100000,
                reference_date=queryDate,
            )
            if relevantMemory == "__NO_RELEVANT_MEMORY__":
                relevantMemory = ""
        elif memory:
            relevantMemory = memory[:100000]

        return GuidanceBrief(
            behavioralGuidance=lessons[:1500],
            relevantMemory=relevantMemory,
            relevantSkills="",
            tacticalPlaybooks="",
            rawPersona="",
            rawMemory=memory,
            rawPlaybooks="",
        )

    def _labelKnowledgeItems(
        self,
        items: List[KnowledgeItem],
        queryDate: Optional[str],
    ) -> List[KnowledgeItem]:
        if not queryDate or not items:
            return items
        labeled: List[KnowledgeItem] = []
        for i, item in enumerate(items):
            tag = "[LATEST]" if i == 0 else "[EARLIER]"
            labeled.append(item.model_copy(update={
                "text": f"{tag} {item.text}",
            }))
        return labeled

    async def buildContext(
        self,
        query: str,
        namespaceId: str,
        queryDate: Optional[str] = None,
        knowledgeLimit: int = 8,
    ) -> ContextBundle:
        ordering = self.resolveContextOrdering(query)
        brief = await self.produceGuidanceBrief(query, namespaceId, queryDate)
        knowledge, _ = await self.knowledge.search(
            query=query,
            namespaceId=namespaceId,
            limit=knowledgeLimit,
        )
        knowledge = self._labelKnowledgeItems(knowledge, queryDate)
        guidance = await self.guidance.search(
            query=query,
            namespaceId=namespaceId,
            queryDate=queryDate,
        )
        return ContextBundle(
            query=query,
            queryDate=queryDate,
            contextOrdering=ordering,
            guidanceBrief=brief,
            knowledge=knowledge,
            guidance=guidance,
        )
