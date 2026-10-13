import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from kormem.core.config import KormemConfig
from kormem.embeddings.provider import (
    HashEmbeddingProvider,
    describeActiveProvider,
    getEmbeddingProvider,
)


def _mockEmbedResponse(values: list[float]):
    emb = MagicMock()
    emb.values = values
    resp = MagicMock()
    resp.embeddings = [emb]
    return resp


@pytest.mark.asyncio
async def testGeminiNormalize768():
    from kormem.embeddings.gemini import GeminiEmbeddingProvider

    provider = GeminiEmbeddingProvider(
        apiKey="test-key",
        modelName="gemini-embedding-2",
        dimension=768,
    )
    raw = [3.0, 4.0] + [0.0] * 766
    with patch.object(provider, "_client") as mockClient:
        mockClient.aio.models.embed_content = AsyncMock(
            return_value=_mockEmbedResponse(raw)
        )
        vec = await provider.encodeQuery("hello world")
    assert len(vec) == 768
    norm = sum(x * x for x in vec) ** 0.5
    assert abs(norm - 1.0) < 0.01


@pytest.mark.asyncio
async def testGeminiBatchUsesDocumentTaskType():
    from kormem.embeddings.gemini import GeminiEmbeddingProvider

    provider = GeminiEmbeddingProvider(apiKey="test-key", dimension=768)
    with patch.object(provider, "_client") as mockClient:
        mockClient.aio.models.embed_content = AsyncMock(
            return_value=_mockEmbedResponse([1.0] * 768)
        )
        await provider.encodeBatch(["a", "b"], taskType="RETRIEVAL_DOCUMENT")
        call = mockClient.aio.models.embed_content.call_args
        assert call.kwargs["config"].task_type == "RETRIEVAL_DOCUMENT"


def testProviderResolutionGeminiWhenKeySet():
    config = KormemConfig(
        embeddingProvider="gemini",
        geminiApiKey="key",
        embeddingDim=768,
    )
    with patch("kormem.embeddings.provider._createGeminiProvider") as mockCreate:
        mockCreate.return_value = MagicMock(providerName="gemini", modelName="gemini-embedding-2", dimension=768)
        getEmbeddingProvider(config)
        mockCreate.assert_called_once()


def testProviderResolutionLocalWhenNoKey():
    config = KormemConfig(
        embeddingProvider="gemini",
        geminiApiKey=None,
        embeddingDim=768,
    )
    with patch("kormem.embeddings.provider._createLocalProvider") as mockCreate:
        mockCreate.return_value = MagicMock(providerName="local", modelName="BAAI/bge-base-en-v1.5", dimension=768)
        getEmbeddingProvider(config)
        mockCreate.assert_called_once()


def testHashProviderForTests():
    config = KormemConfig(embeddingProvider="hash", embeddingDim=768)
    provider = getEmbeddingProvider(config)
    assert isinstance(provider, HashEmbeddingProvider)
    assert provider.dimension == 768
