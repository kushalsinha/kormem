import pytest

from kormem.core.config import KormemConfig
from kormem.core.schemas import SearchRequest
from kormem.embeddings.provider import HashEmbeddingProvider
from kormem.ingest.service import IngestService
from kormem.retrieve.rag import RagService
from kormem.retrieve.rrf import reciprocalRankFusion
from kormem.store.sqlite_store import SqliteStore


def testRrfK60():
    listA = [("a", 1.0), ("b", 0.9), ("c", 0.8)]
    listB = [("b", 1.0), ("d", 0.7)]
    fused = reciprocalRankFusion([listA, listB], k=60)
    keys = [k for k, _ in fused]
    assert "b" in keys
    assert keys.index("b") <= keys.index("d")


@pytest.fixture
def multiQuerySetup(tmp_path):
    index = tmp_path / "index.sqlite"
    guidance = tmp_path / "guidance.sqlite"
    config = KormemConfig(defaultNamespace="test", embeddingDim=768, embeddingProvider="hash")
    store = SqliteStore(indexPath=index, guidancePath=guidance)
    ingest = IngestService(store=store, config=config)
    ingest.embedder = HashEmbeddingProvider(768)
    rag = RagService(store=store, config=config)
    rag.knowledge.embedder = HashEmbeddingProvider(768)
    rag.guidance.embedder = HashEmbeddingProvider(768)
    rag.knowledge.reranker.enabled = False
    return ingest, rag


@pytest.mark.asyncio
async def testMultiQueryHybridSearch(multiQuerySetup):
    ingest, rag = multiQuerySetup
    await ingest.ingestText(
        content="Segment zero about alpha. " * 5 + "Segment one about beta deployment.",
        namespaceId="test",
        title="Doc",
    )
    resp = await rag.search(SearchRequest(
        query="alpha beta deployment",
        namespaceId="test",
        limit=5,
        rerank=False,
    ))
    assert len(resp.knowledge) >= 1
