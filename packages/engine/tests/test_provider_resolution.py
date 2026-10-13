from kormem.core.config import KormemConfig
from kormem.embeddings.provider import describeActiveProvider, getEmbeddingProvider
from kormem.embeddings.provider import HashEmbeddingProvider


def testDescribeActiveProviderHash():
    config = KormemConfig(embeddingProvider="hash", embeddingDim=768)
    info = describeActiveProvider(config)
    assert info["provider"] == "hash"
    assert info["dimension"] == 768


def testGetEmbeddingProviderExplicitHash():
    config = KormemConfig(embeddingProvider="hash", embeddingDim=768)
    p = getEmbeddingProvider(config)
    assert isinstance(p, HashEmbeddingProvider)
