"""Gemini embedding provider using gemini-embedding-2."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from kormem.embeddings.base import EmbeddingProvider
from kormem.core import usage
from kormem.core.config import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_MODEL,
)

if TYPE_CHECKING:
    from google.genai import types

logger = logging.getLogger(__name__)

TASK_RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_RETRIEVAL_QUERY = "RETRIEVAL_QUERY"

_EMBED_CONCURRENCY = int(os.environ.get("GEMINI_EMBED_CONCURRENCY", "5"))
_embedSemaphore: Optional[asyncio.Semaphore] = None
_MAX_RETRIES_429 = int(os.environ.get("GEMINI_EMBED_MAX_RETRIES", "6"))


def _getSemaphore() -> asyncio.Semaphore:
    global _embedSemaphore
    if _embedSemaphore is None:
        _embedSemaphore = asyncio.Semaphore(_EMBED_CONCURRENCY)
    return _embedSemaphore


def _parseRetryAfter(err: Exception, attempt: int) -> float:
    """Pick a backoff delay for a 429 error. Honour server's retry hint when
    available; otherwise exponential with jitter."""
    msg = str(err)
    import re as _re
    m = _re.search(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'", msg) \
        or _re.search(r"retry in\s+(\d+(?:\.\d+)?)s", msg)
    if m:
        return float(m.group(1)) + 0.5
    return min(60.0, 2.0 ** attempt)


def _is429(err: Exception) -> bool:
    """Worth retrying: rate limits, transient server errors (5xx) and timeouts."""
    if isinstance(err, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True
    s = str(err)
    return ("429" in s or "RESOURCE_EXHAUSTED" in s or ("rate" in s.lower() and "limit" in s.lower())
            or any(code in s for code in ("500", "502", "503", "504", "UNAVAILABLE", "INTERNAL", "DEADLINE")))


class GeminiEmbeddingProvider(EmbeddingProvider):
    """768-dim Gemini embeddings via google-genai."""

    def __init__(
        self,
        apiKey: str,
        modelName: str = DEFAULT_EMBEDDING_MODEL,
        dimension: int = DEFAULT_EMBEDDING_DIM,
        batchSize: int = 100,
    ):
        from google import genai

        self._modelName = modelName
        self._dimension = dimension
        # google-genai's `embed_content(contents=[...])` returns only ONE
        # embedding regardless of list size; the rest silently become zero
        # vectors. Force one-text-per-call (concurrency is handled by the
        # module-level `GEMINI_EMBED_CONCURRENCY` semaphore).
        self.batchSize = 1
        self._client = genai.Client(api_key=apiKey)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def providerName(self) -> str:
        return "gemini"

    @property
    def modelName(self) -> str:
        return self._modelName

    def _embedConfig(self, taskType: Optional[str] = None):
        from google.genai import types

        return types.EmbedContentConfig(
            task_type=taskType,
            output_dimensionality=self._dimension,
        )

    def _normalize(self, values: List[float]) -> List[float]:
        if not values or self._dimension == 3072:
            return values
        arr = np.array(values, dtype=np.float64)
        norm = np.linalg.norm(arr)
        if norm <= 0:
            return values
        return (arr / norm).tolist()

    async def _embedOne(self, text: str, taskType: Optional[str] = None) -> List[float]:
        if not text or not text.strip():
            return [0.0] * self._dimension
        timeout = int(os.environ.get("GEMINI_EMBED_CALL_TIMEOUT", "30"))
        for attempt in range(_MAX_RETRIES_429 + 1):
            try:
                async with _getSemaphore():
                    response = await asyncio.wait_for(
                        self._client.aio.models.embed_content(
                            model=self._modelName,
                            contents=text.strip(),
                            config=self._embedConfig(taskType),
                        ),
                        timeout=timeout,
                    )
                if response.embeddings and len(response.embeddings) > 0:
                    return self._normalize(list(response.embeddings[0].values))
                raise ValueError("Empty embedding response from Gemini")
            except Exception as e:
                if _is429(e) and attempt < _MAX_RETRIES_429:
                    delay = _parseRetryAfter(e, attempt)
                    logger.warning("Gemini 429, retrying in %.1fs (attempt %d/%d)",
                                   delay, attempt + 1, _MAX_RETRIES_429)
                    await asyncio.sleep(delay)
                    continue
                raise
        raise RuntimeError("Gemini embed_content exhausted retries after 429")

    async def encodeBatch(
        self,
        texts: List[str],
        taskType: str = TASK_RETRIEVAL_DOCUMENT,
    ) -> List[List[float]]:
        if not texts:
            return []
        results: List[List[float]] = [[0.0] * self._dimension for _ in texts]
        validPairs = [
            (i, t.strip()) for i, t in enumerate(texts) if t and t.strip()
        ]
        if not validPairs:
            return results

        config = self._embedConfig(taskType)
        timeout = int(os.environ.get("GEMINI_EMBED_CALL_TIMEOUT", "30"))

        async def _one(idx: int, text: str) -> None:
            # google-genai `embed_content(contents=[...])` silently returns a
            # single embedding regardless of input list size, so we go one at
            # a time. Concurrency is bounded by `_getSemaphore()`.
            for attempt in range(_MAX_RETRIES_429 + 1):
                try:
                    async with _getSemaphore():
                        resp = await asyncio.wait_for(
                            self._client.aio.models.embed_content(
                                model=self._modelName,
                                contents=text,
                                config=config,
                            ),
                            timeout=timeout,
                        )
                    if resp.embeddings:
                        results[idx] = self._normalize(list(resp.embeddings[0].values))
                    return
                except Exception as e:
                    if _is429(e) and attempt < _MAX_RETRIES_429:
                        delay = _parseRetryAfter(e, attempt)
                        logger.warning(
                            "Gemini 429 on batch item, retrying in %.1fs (attempt %d/%d)",
                            delay, attempt + 1, _MAX_RETRIES_429,
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise

        await asyncio.gather(*[_one(i, t) for i, t in validPairs])
        usage.record("embed", self._modelName, texts=len(validPairs), chars=sum(len(t) for _, t in validPairs))
        return results

    async def encodeQuery(self, text: str) -> List[float]:
        usage.record("embed", self._modelName, texts=1, chars=len(text))
        return await self._embedOne(text, taskType=TASK_RETRIEVAL_QUERY)
