"""Embedding provider base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @property
    def providerName(self) -> str:
        return "unknown"

    @property
    def modelName(self) -> str:
        return "unknown"

    @abstractmethod
    async def encodeBatch(self, texts: List[str]) -> List[List[float]]:
        pass

    async def encodeQuery(self, text: str) -> List[float]:
        vecs = await self.encodeBatch([text])
        return vecs[0] if vecs else []
