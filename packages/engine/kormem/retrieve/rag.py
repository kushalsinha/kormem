"""RAG orchestrator: knowledge + guidance + citations + context ordering."""

from __future__ import annotations

import asyncio
from typing import Optional

from kormem.core.config import KormemConfig, loadConfig
from kormem.core.schemas import Citation, SearchRequest, SearchResponse
from kormem.retrieve.context_builder import ContextBuilder
from kormem.retrieve.guidance import GuidanceSearch
from kormem.retrieve.knowledge import KnowledgeSearch
from kormem.store.sqlite_store import SqliteStore


class RagService:
    def __init__(
        self,
        store: Optional[SqliteStore] = None,
        config: Optional[KormemConfig] = None,
    ):
        self.store = store or SqliteStore()
        self.config = config or loadConfig()
        self.knowledge = KnowledgeSearch(self.store, self.config)
        self.guidance = GuidanceSearch(self.store, self.config)
        self.contextBuilder = ContextBuilder(self.store, self.config)

    async def search(self, request: SearchRequest) -> SearchResponse:
        knowledgeTask = self.knowledge.search(
            query=request.query,
            namespaceId=request.namespaceId,
            collectionId=request.collectionId,
            limit=request.limit,
            mode=request.mode if request.mode != "hybrid" else "hybrid",
            minScore=request.minScore,
            expand=request.expand,
            liveFetch=request.liveFetch,
            window=request.window,
            rerank=request.rerank,
            sourceIdsFilter=request.sourceIdsFilter,
            queryDate=request.queryDate,
        )
        if request.includeGuidance:
            guidanceTask = self.guidance.search(
                query=request.query,
                namespaceId=request.namespaceId,
                limit=request.limit,
                queryDate=request.queryDate,
            )
            (knowledge, ms), guidance = await asyncio.gather(
                knowledgeTask,
                guidanceTask,
            )
        else:
            knowledge, ms = await knowledgeTask
            guidance = []

        citations: list[Citation] = []
        for i, item in enumerate(knowledge, start=1):
            citations.append(item.toCitation(i))

        ordering = self.contextBuilder.resolveContextOrdering(request.query)

        return SearchResponse(
            query=request.query,
            knowledge=knowledge,
            guidance=guidance,
            citations=citations,
            retrievalTimeMs=ms,
            contextOrdering=ordering,
        )

    async def buildContext(self, request):
        return await self.contextBuilder.buildContext(
            query=request.query,
            namespaceId=request.namespaceId,
            queryDate=request.queryDate,
            knowledgeLimit=request.knowledgeLimit,
        )
