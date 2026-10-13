import tempfile
from pathlib import Path

import pytest

from kormem.core.config import KormemConfig
from kormem.core.schemas import SearchRequest
from kormem.embeddings.provider import HashEmbeddingProvider
from kormem.ingest.service import IngestService
from kormem.retrieve.rag import RagService
from kormem.store.sqlite_store import SqliteStore


@pytest.fixture
def tempHome(tmp_path):
    return tmp_path


@pytest.fixture
def services(tempHome):
    index = tempHome / "index.sqlite"
    guidance = tempHome / "guidance.sqlite"
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
async def testIngestAndSearch(services):
    ingest, rag = services
    await ingest.ingestText(
        content="The deployment process uses Kubernetes and Helm charts for staging.",
        namespaceId="test",
        title="Deploy doc",
    )
    resp = await rag.search(SearchRequest(
        query="kubernetes helm deployment",
        namespaceId="test",
        limit=5,
        rerank=False,
    ))
    assert len(resp.knowledge) >= 1
    assert "kubernetes" in resp.knowledge[0].text.lower() or "helm" in resp.knowledge[0].text.lower()
