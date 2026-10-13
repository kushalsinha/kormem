import pytest

from kormem.core.config import KormemConfig
from kormem.core.document_content import packDocumentContent, unpackDocumentContent
from kormem.core.schemas import KnowledgeItem
from kormem.retrieve.enrichment import Enricher
from kormem.store.sqlite_store import SqliteStore


@pytest.fixture
def enricherSetup(tmp_path):
    index = tmp_path / "index.sqlite"
    guidance = tmp_path / "guidance.sqlite"
    config = KormemConfig(defaultNamespace="test")
    store = SqliteStore(indexPath=index, guidancePath=guidance)
    enricher = Enricher(store=store, config=config)
    return enricher, store, config


@pytest.mark.asyncio
async def testFullDocAttach(enricherSetup):
    enricher, store, config = enricherSetup
    src = store.createSource("test", "file", url="file:///tmp/x.txt")
    blob = packDocumentContent("Full document body text here.", config.encryptionKey)
    docId = store.createDocument(
        namespaceId="test",
        sourceId=src,
        title="Doc",
        platform="file",
        encryptedData=blob,
    )
    item = KnowledgeItem(segmentKey="s1", text="chunk", documentId=docId, sourcePlatform="file")
    out = await enricher.enrich([item], fullDoc=True, liveFetch=False)
    assert "Full document body" in (out[0].fullDocumentText or "")
