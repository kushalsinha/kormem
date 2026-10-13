"""Hybrid knowledge search: multi-query vector + keyword + RRF + CE rerank."""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Dict, List, Optional, Tuple

from kormem.core.config import KormemConfig, loadConfig
from kormem.core.schemas import KnowledgeItem
from kormem.embeddings.provider import getEmbeddingProvider
from kormem.retrieve.enrichment import Enricher
from kormem.retrieve.query_optimizer import QueryOptimizer
from kormem.retrieve.reranker import CrossEncoderReranker
from kormem.retrieve.rrf import reciprocalRankFusion
from kormem.retrieve.temporal import rerank as temporalRerank
from kormem.store.sqlite_store import SqliteStore

STAGE1_LIMIT = 80

_PREFERENCE_BOOST_KEYWORDS = (
    "recommend", "suggest", "advice", "advise", "tips",
    "what should", "help me", "activity", "activities",
    "resource", "resources", "options", "what would you",
)
_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "what", "how", "when", "where",
    "who", "why", "do", "does", "did", "my", "me", "i", "you", "your", "of", "in",
    "on", "at", "to", "for", "and", "or", "with", "about",
}


def extractSearchKeywords(query: str) -> str:
    words = [w for w in re.findall(r"\w+", query.lower()) if w not in _STOP_WORDS and len(w) > 2]
    return " ".join(words[:15])


class KnowledgeSearch:
    def __init__(
        self,
        store: Optional[SqliteStore] = None,
        config: Optional[KormemConfig] = None,
    ):
        self.store = store or SqliteStore()
        self.config = config or loadConfig()
        self.embedder = getEmbeddingProvider(self.config)
        self.optimizer = QueryOptimizer(self.config)
        self.reranker = CrossEncoderReranker(self.config)
        self.enricher = Enricher(self.store, self.config)

    async def search(
        self,
        query: str,
        namespaceId: Optional[str] = None,
        collectionId: Optional[str] = None,
        limit: int = 8,
        mode: str = "hybrid",
        minScore: float = 0.0,
        expand: bool = False,
        liveFetch: bool = False,
        window: int = 2,
        rerank: Optional[bool] = None,
        stage1Limit: Optional[int] = None,
        sourceIdsFilter: Optional[List[str]] = None,
        queryDate: Optional[str] = None,
    ) -> tuple[List[KnowledgeItem], float]:
        t0 = time.perf_counter()
        namespaceId = namespaceId or self.config.defaultNamespace
        stage1Limit = stage1Limit or STAGE1_LIMIT
        if rerank is not None:
            self.reranker.enabled = rerank
        elif len(query.split()) <= 2:
            self.reranker.enabled = False

        if mode == "keyword":
            keywordRanked = self._keywordSearch(
                namespaceId, query, stage1Limit, collectionId,
            )
            fused = keywordRanked
        elif mode == "vector":
            qEmb = await self.embedder.encodeQuery(query)
            vecResults = self.store.vectorSearch(
                namespaceId, qEmb, limit=stage1Limit, collectionId=collectionId,
                minScore=minScore, kind="segment",
            )
            fused = [(r["segmentKey"], score) for r, score in vecResults]
        else:
            opt = await self.optimizer.optimize(query)
            variants = opt.queryVariants
            embs = await self.embedder.encodeBatch(variants) if variants else []

            async def vecSearch(emb: List[float]) -> List[Tuple[str, float]]:
                results = self.store.vectorSearch(
                    namespaceId, emb, limit=stage1Limit, collectionId=collectionId,
                    minScore=minScore, kind="segment",
                )
                return [(r["segmentKey"], score) for r, score in results]

            tasks = [vecSearch(e) for e in embs]
            if mode in ("hybrid", "keyword"):
                tasks.append(asyncio.to_thread(
                    self._keywordSearch,
                    namespaceId,
                    query,
                    stage1Limit,
                    collectionId,
                ))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            vectorLists: List[List[Tuple[str, float]]] = []
            keywordList: List[Tuple[str, float]] = []
            for i, res in enumerate(results):
                if isinstance(res, Exception):
                    continue
                if i < len(embs):
                    vectorLists.append(res)
                else:
                    keywordList = res

            queryLower = query.lower()
            if any(kw in queryLower for kw in _PREFERENCE_BOOST_KEYWORDS):
                try:
                    prefQuery = f"user likes prefers enjoys wants favorites {query}"
                    prefEmb = await self.embedder.encodeQuery(prefQuery)
                    prefResults = self.store.vectorSearch(
                        namespaceId, prefEmb, limit=15, collectionId=collectionId,
                        minScore=max(minScore - 0.05, 0.2), kind="segment",
                    )
                    prefList = [(r["segmentKey"], score) for r, score in prefResults]
                    existingKeys = {k for vl in vectorLists for k, _ in vl}
                    offsetList = [
                        (k, s) for k, s in prefList if k not in existingKeys
                    ]
                    if offsetList:
                        vectorLists.append(offsetList)
                except Exception:
                    pass

            rankInputs = vectorLists + ([keywordList] if keywordList else [])
            fused = reciprocalRankFusion(rankInputs, k=60)

        if sourceIdsFilter:
            rows = {r["segmentKey"]: r for r in self.store.getSegmentsByKeys([k for k, _ in fused[:stage1Limit]])}
            fused = [
                (k, s) for k, s in fused
                if rows.get(k, {}).get("sourceId") in sourceIdsFilter
            ]

        topKeys = [k for k, _ in fused[: max(limit * 2, 16)]]
        scoreMap = {k: s for k, s in fused}

        pairs = self.store.docIndexPairsFor(topKeys)
        adj = self.store.getAdjacentSegments(pairs, window=window)
        parentScores = {k: scoreMap.get(k, 0.5) for k in topKeys}
        adjKeys: List[str] = []
        for a in adj:
            key = a["segmentKey"]
            if key not in topKeys:
                parent = parentScores.get(next((pk for pk in topKeys if pk), ""), 0.4)
                scoreMap[key] = max(scoreMap.get(key, 0.0), parent * 0.9)
                adjKeys.append(key)
        allKeys = list(dict.fromkeys(topKeys + adjKeys))

        rows = {r["segmentKey"]: r for r in self.store.getSegmentsByKeys(allKeys)}
        items: List[KnowledgeItem] = []
        for key in allKeys:
            row = rows.get(key)
            if not row or row["kind"] == "note":
                continue
            meta = json.loads(row["metadata"] or "{}")
            text = self.store.getSegmentText(row, self.config.encryptionKey)
            items.append(KnowledgeItem(
                segmentKey=key,
                text=text,
                score=scoreMap.get(key, 0.0),
                sourceUrl=meta.get("sourceUrl"),
                sourceTitle=meta.get("sourceTitle"),
                sourcePlatform=meta.get("sourcePlatform"),
                documentId=row["documentId"],
                sourceId=row["sourceId"],
                metadata=meta,
            ))

        items = await self.reranker.rerank(query, items)
        if queryDate:
            items, strategy = temporalRerank(items, query, queryDate)
            if strategy != "none":
                import logging
                logging.getLogger(__name__).debug(
                    "temporal rerank applied strategy=%s items=%d queryDate=%s",
                    strategy, len(items), queryDate,
                )
        items = self._applyScoreGapFilter(items)
        items = self._diversityCap(items, perSource=8)
        if expand or liveFetch:
            items = await self.enricher.enrich(items, fullDoc=expand, liveFetch=liveFetch)

        elapsed = (time.perf_counter() - t0) * 1000
        return items[:limit], elapsed

    def _keywordSearch(
        self,
        namespaceId: str,
        query: str,
        limit: int,
        collectionId: Optional[str],
    ) -> List[Tuple[str, float]]:
        kwQuery = extractSearchKeywords(query)
        return self.store.keywordSearch(
            namespaceId, kwQuery or query, limit=limit, collectionId=collectionId, kind="segment",
        )

    def _applyScoreGapFilter(
        self,
        items: List[KnowledgeItem],
        minGap: float = 0.10,
        lowScoreThreshold: float = 0.30,
    ) -> List[KnowledgeItem]:
        if not items or len(items) == 1:
            return items
        # Thresholds below are calibrated for sigmoid cross-encoder scores; RRF
        # scores top out near 1/61, so applying them there always truncated
        # results to 3. Skip the filter when the reranker didn't score items.
        if items[0].ceScore is None:
            return items
        topScore = items[0].ceScore
        if topScore < lowScoreThreshold:
            return items[:3]
        for i in range(1, len(items)):
            prevScore = items[i - 1].ceScore if items[i - 1].ceScore is not None else items[i - 1].score
            currScore = items[i].ceScore if items[i].ceScore is not None else items[i].score
            gap = prevScore - currScore
            if gap >= minGap:
                isPlateau = True
                if i + 1 < len(items):
                    remaining = [
                        items[j].ceScore if items[j].ceScore is not None else items[j].score
                        for j in range(i, min(i + 3, len(items)))
                    ]
                    if len(remaining) > 1:
                        isPlateau = (max(remaining) - min(remaining)) < 0.08
                if isPlateau:
                    return items[: max(i, 3)]
        return items

    def _diversityCap(self, items: List[KnowledgeItem], perSource: int = 8) -> List[KnowledgeItem]:
        counts: Dict[str, int] = {}
        out: List[KnowledgeItem] = []
        for it in items:
            src = it.sourceTitle or it.metadata.get("sourceTitle") or it.sourceId or it.documentId or "unknown"
            if counts.get(src, 0) >= perSource:
                continue
            counts[src] = counts.get(src, 0) + 1
            out.append(it)
        return out
