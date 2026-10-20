"""OpenAI embedding provider (text-embedding-3-small by default), shortened to the index dimension.

The text-embedding-3 models are trained so a vector can be cut to fewer dimensions
(the ``dimensions`` parameter), so 768-d OpenAI vectors take the same space as
Gemini's. Text only: kormem embeds no images.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from kormem.core import usage
from kormem.core.config import DEFAULT_OPENAI_EMBED_MODEL
from kormem.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

_URL = "https://api.openai.com/v1/embeddings"
_BATCH = 128                    # inputs per request (the API allows 2,048 and 300k tokens)
_MAX_CHARS = 24_000             # ~6k tokens, under the model's 8,192-token input limit
_MAX_RETRIES = int(os.environ.get("OPENAI_EMBED_MAX_RETRIES", "6"))
_CONCURRENCY = int(os.environ.get("OPENAI_EMBED_CONCURRENCY", "4"))
_semaphore: Optional[asyncio.Semaphore] = None


def _getSemaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(_CONCURRENCY)
    return _semaphore


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, apiKey: str, modelName: str = DEFAULT_OPENAI_EMBED_MODEL, dimension: int = 768):
        self._apiKey = apiKey
        self._modelName = modelName
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def providerName(self) -> str:
        return "openai"

    @property
    def modelName(self) -> str:
        return self._modelName

    async def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        for attempt in range(_MAX_RETRIES + 1):
            try:
                async with _getSemaphore(), httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.post(_URL, json=payload,
                                             headers={"Authorization": f"Bearer {self._apiKey}"})
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise httpx.HTTPStatusError(f"{resp.status_code}", request=resp.request, response=resp)
                if resp.status_code >= 400:
                    raise ValueError(f"OpenAI embeddings rejected the request: {resp.text[:300]}")
                return resp.json()
            except (httpx.HTTPStatusError, httpx.TransportError) as e:
                if attempt == _MAX_RETRIES:
                    raise
                delay = min(60.0, 2.0 ** attempt)
                logger.warning("OpenAI embeddings: %s, retrying in %.0fs (%d/%d)", e, delay, attempt + 1,
                               _MAX_RETRIES)
                await asyncio.sleep(delay)
        raise RuntimeError("unreachable")

    async def encodeBatch(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = [[0.0] * self._dimension for _ in texts]
        valid = [(i, t.strip()[:_MAX_CHARS]) for i, t in enumerate(texts) if t and t.strip()]

        async def batch(pairs) -> None:
            data = await self._post({"model": self._modelName, "input": [t for _, t in pairs],
                                     "dimensions": self._dimension, "encoding_format": "float"})
            for item in data["data"]:
                out[pairs[item["index"]][0]] = item["embedding"]
            usage.record("embed", self._modelName, data.get("usage"), texts=len(pairs),
                         chars=sum(len(t) for _, t in pairs))

        await asyncio.gather(*(batch(valid[i:i + _BATCH]) for i in range(0, len(valid), _BATCH)))
        return out
