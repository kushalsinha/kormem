from kormem.embeddings.base import EmbeddingProvider
from kormem.embeddings.provider import (
    HashEmbeddingProvider,
    LocalEmbeddingProvider,
    describeActiveProvider,
    getEmbeddingProvider,
)

__all__ = [
    "EmbeddingProvider",
    "HashEmbeddingProvider",
    "LocalEmbeddingProvider",
    "describeActiveProvider",
    "getEmbeddingProvider",
]


def __getattr__(name: str):
    if name == "GeminiEmbeddingProvider":
        from kormem.embeddings.gemini import GeminiEmbeddingProvider
        return GeminiEmbeddingProvider
    raise AttributeError(name)
