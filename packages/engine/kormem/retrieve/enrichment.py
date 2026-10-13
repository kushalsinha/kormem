"""Post-retrieval enrichment: full document attach + live re-fetch."""

from __future__ import annotations

import asyncio
import json
from typing import List, Optional
from urllib.parse import urlparse

from kormem.connectors.file_connector import FileConnector
from kormem.connectors.web_connector import WebConnector
from kormem.core.config import KormemConfig, loadConfig
from kormem.ingest.chat_sessions import documentText
from kormem.core.schemas import KnowledgeItem
from kormem.store.sqlite_store import SqliteStore

MAX_CHARS = 50_000


class Enricher:
    def __init__(
        self,
        store: Optional[SqliteStore] = None,
        config: Optional[KormemConfig] = None,
    ):
        self.store = store or SqliteStore()
        self.config = config or loadConfig()
        self.web = WebConnector(firecrawlApiKey=self.config.firecrawlApiKey)
        self.file = FileConnector()

    async def enrich(
        self,
        items: List[KnowledgeItem],
        fullDoc: bool = False,
        liveFetch: bool = False,
    ) -> List[KnowledgeItem]:
        tasks = [self._enrichOne(it, fullDoc, liveFetch) for it in items]
        return list(await asyncio.gather(*tasks))

    async def _enrichOne(
        self,
        item: KnowledgeItem,
        fullDoc: bool,
        liveFetch: bool,
    ) -> KnowledgeItem:
        if fullDoc and item.documentId:
            doc = self.store.getDocument(item.documentId)
            if doc:
                text = documentText(self.store, doc, self.config.encryptionKey)
                if text:
                    item.fullDocumentText = text[:MAX_CHARS]
        if liveFetch:
            platform = item.sourcePlatform or item.metadata.get("sourcePlatform")
            try:
                if platform == "web" and item.sourceUrl:
                    item.liveText = (await self.web.fetchContent(item.sourceUrl))[:MAX_CHARS]
                elif platform == "file" and item.sourceUrl:
                    path = self._filePath(item.sourceUrl)
                    if path:
                        item.liveText = (await self.file.fetchContent(path))[:MAX_CHARS]
            except Exception:
                pass
        return item

    def _filePath(self, sourceUrl: str) -> Optional[str]:
        if sourceUrl.startswith("file://"):
            return urlparse(sourceUrl).path
        return sourceUrl if sourceUrl.startswith("/") else None
