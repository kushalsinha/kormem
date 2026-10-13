"""Cross-encoder reranking via fastembed with sigmoid calibration."""

from __future__ import annotations

import asyncio
import logging
import math
from typing import List, Optional

from kormem.core.config import KormemConfig, getKormemHome, loadConfig
from kormem.core.schemas import KnowledgeItem

logger = logging.getLogger(__name__)


_MODEL_ALIASES = {
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "Xenova/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-MiniLM-L-12-v2": "Xenova/ms-marco-MiniLM-L-12-v2",
}


def sigmoidScore(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    expX = math.exp(x)
    return expX / (1.0 + expX)


class CrossEncoderReranker:
    def __init__(self, config: Optional[KormemConfig] = None):
        self.config = config or loadConfig()
        self.enabled = self.config.rerankEnabled
        self.modelName = self.config.rerankModel
        self._ce = None

    def _load(self):
        if self._ce is not None:
            return self._ce
        try:
            from fastembed import TextCrossEncoder
        except ImportError:  # fastembed>=0.9 no longer re-exports it
            from fastembed.rerank.cross_encoder import TextCrossEncoder

        cacheDir = str(getKormemHome() / "models")
        # fastembed ships the HF cross-encoder/* checkpoints under Xenova/*.
        modelName = _MODEL_ALIASES.get(self.modelName, self.modelName)
        self._ce = TextCrossEncoder(model_name=modelName, cache_dir=cacheDir)
        return self._ce

    async def rerank(self, query: str, items: List[KnowledgeItem]) -> List[KnowledgeItem]:
        if not self.enabled or not items:
            return items
        try:
            ce = await asyncio.to_thread(self._load)
            texts = [it.text for it in items]
            scores = await asyncio.to_thread(lambda: list(ce.rerank(query, texts)))
            for it, sc in zip(items, scores):
                it.ceScore = sigmoidScore(float(sc))
            items.sort(key=lambda x: x.ceScore or 0.0, reverse=True)
            return self._applyCeFilters(items)
        except Exception:
            logger.warning("cross-encoder rerank failed; returning unreranked items", exc_info=True)
            return items

    def _applyCeFilters(self, items: List[KnowledgeItem]) -> List[KnowledgeItem]:
        if not items:
            return items
        top = items[0].ceScore or 0.0
        if top < 0.10:
            return []
        out = [items[0]]
        for prev, cur in zip(items, items[1:]):
            curScore = cur.ceScore or 0.0
            prevScore = prev.ceScore or 0.0
            if curScore < 0.05:
                break
            if prevScore >= 10 * curScore:
                break
            out.append(cur)
        return out
