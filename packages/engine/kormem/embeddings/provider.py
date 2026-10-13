"""Pluggable embedding providers."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import List, Optional

from kormem.embeddings.base import EmbeddingProvider
from kormem.core.config import (
    INTERNAL_EMBED_DIMS,
    LOCAL_INTERNAL_EMBED_MODEL,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LOCAL_EMBED_MODEL,
    KormemConfig,
    getKormemHome,
    loadConfig,
)
from kormem.embeddings.gemini import GeminiEmbeddingProvider


class HashEmbeddingProvider(EmbeddingProvider):
    """Deterministic embeddings for unit tests only."""

    def __init__(self, dim: int = DEFAULT_EMBEDDING_DIM):
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def providerName(self) -> str:
        return "hash"

    @property
    def modelName(self) -> str:
        return "hash-test"

    def _embedOne(self, text: str) -> List[float]:
        vec = [0.0] * self._dim
        tokens = text.lower().split()
        for i, tok in enumerate(tokens):
            h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
            idx = h % self._dim
            sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
            vec[idx] += sign * (1.0 + math.log1p(i % 10))
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    async def encodeBatch(self, texts: List[str]) -> List[List[float]]:
        return [self._embedOne(t) for t in texts]


class LocalEmbeddingProvider(EmbeddingProvider):
    """Offline ONNX embeddings via fastembed (QMD-style local fallback)."""

    def __init__(
        self,
        modelName: str = DEFAULT_LOCAL_EMBED_MODEL,
        cacheDir: Optional[Path] = None,
        dimension: int = DEFAULT_EMBEDDING_DIM,
    ):
        from fastembed import TextEmbedding

        cacheDir = cacheDir or (getKormemHome() / "models")
        cacheDir.mkdir(parents=True, exist_ok=True)
        self._modelName = modelName
        self._cacheDir = cacheDir
        self._model = TextEmbedding(model_name=modelName, cache_dir=str(cacheDir))
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def providerName(self) -> str:
        return "local"

    @property
    def modelName(self) -> str:
        return self._modelName

    async def encodeBatch(self, texts: List[str]) -> List[List[float]]:
        import asyncio

        loop = asyncio.get_event_loop()

        # Batches of similar length: padding to the longest text in a batch
        # dominates CPU time when short and long texts are mixed.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))

        def _run():
            return list(self._model.embed([texts[i] for i in order], batch_size=16))

        embeddings = await loop.run_in_executor(None, _run)
        out: List[List[float]] = [[] for _ in texts]
        for i, e in zip(order, embeddings):
            out[i] = e.tolist()
        return out

    async def encodeQuery(self, text: str) -> List[float]:
        """Query-side embedding (BGE models add their retrieval instruction)."""
        import asyncio

        def _run():
            return list(self._model.query_embed([text]))[0].tolist()

        return await asyncio.get_event_loop().run_in_executor(None, _run)


_internalCache: dict = {}


def getInternalEmbedder(config: Optional[KormemConfig] = None) -> EmbeddingProvider:
    """Embedder for kormem's own ranking: notes, memory lines, folders.

    ``internalEmbedModel="auto"`` (default) uses the configured provider when
    it is an API (Gemini): it parallelises without loading the CPU. Without
    an API key it falls back to a small local model (fastembed, free). A
    model name forces that local model.
    """
    config = config or loadConfig()
    if (config.embeddingProvider or "").lower() == "hash":
        return HashEmbeddingProvider(config.embeddingDim)
    key = config.internalEmbedModel
    if key == "auto":
        main = getEmbeddingProvider(config)
        if main.providerName != "local":
            return main
        key = LOCAL_INTERNAL_EMBED_MODEL
    if key not in _internalCache:
        try:
            _internalCache[key] = LocalEmbeddingProvider(
                modelName=key, dimension=INTERNAL_EMBED_DIMS.get(key, config.embeddingDim))
        except Exception:
            _internalCache[key] = getEmbeddingProvider(config)
    return _internalCache[key]


def getEmbeddingProvider(config: Optional[KormemConfig] = None) -> EmbeddingProvider:
    config = config or loadConfig()
    provider = (config.embeddingProvider or "gemini").lower()

    if provider == "hash":
        return HashEmbeddingProvider(config.embeddingDim)

    if provider == "local":
        return _createLocalProvider(config)

    if provider == "gemini":
        if config.geminiApiKey:
            return _createGeminiProvider(config)
        return _createLocalProvider(config)

    # auto / unknown: gemini when key present, else local
    if config.geminiApiKey:
        return _createGeminiProvider(config)
    return _createLocalProvider(config)


def _createGeminiProvider(config: KormemConfig) -> GeminiEmbeddingProvider:
    if not config.geminiApiKey:
        raise ValueError("GEMINI_API_KEY required for gemini embedding provider")
    return GeminiEmbeddingProvider(
        apiKey=config.geminiApiKey,
        modelName=config.embeddingModel,
        dimension=config.embeddingDim,
    )


def _createLocalProvider(config: KormemConfig) -> LocalEmbeddingProvider:
    try:
        return LocalEmbeddingProvider(
            modelName=config.localEmbedModel,
            dimension=config.embeddingDim,
        )
    except ImportError as e:
        raise ImportError(
            "fastembed is required for local embeddings. "
            "Install with: pip install fastembed"
        ) from e


def describeActiveProvider(config: Optional[KormemConfig] = None) -> dict:
    """Return metadata about which provider would be selected."""
    config = config or loadConfig()
    provider = getEmbeddingProvider(config)
    return {
        "provider": provider.providerName,
        "model": provider.modelName,
        "dimension": provider.dimension,
        "geminiApiKeySet": bool(config.geminiApiKey),
        "configuredProvider": config.embeddingProvider,
    }
