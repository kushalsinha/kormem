"""Connector orchestration for web and file ingestion."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, List, Optional

from kormem.connectors.file_connector import FileConnector
from kormem.connectors.web_connector import WebConnector
from kormem.core.config import KormemConfig, loadConfig
from kormem.ingest.service import IngestService


class ConnectorService:
    def __init__(
        self,
        ingest: Optional[IngestService] = None,
        config: Optional[KormemConfig] = None,
    ):
        self.config = config or loadConfig()
        self.ingest = ingest or IngestService(config=self.config)
        self.web = WebConnector(firecrawlApiKey=self.config.firecrawlApiKey)
        self.file = FileConnector()

    async def ingestUrl(
        self,
        url: str,
        namespaceId: Optional[str] = None,
        collectionId: Optional[str] = None,
        title: Optional[str] = None,
    ) -> dict[str, Any]:
        content = await self.web.fetchContent(url)
        return await self.ingest.ingestText(
            content=content,
            namespaceId=namespaceId,
            collectionId=collectionId,
            sourceUrl=url,
            title=title or url,
            platform="web",
        )

    async def ingestPath(
        self,
        path: str,
        namespaceId: Optional[str] = None,
        collectionId: Optional[str] = None,
        recursive: bool = False,
    ) -> List[dict[str, Any]]:
        target = path
        if recursive and Path(path).is_dir():
            target = str(Path(path) / "**" / "*")
        files = await self.file.listFiles(target)
        results = []
        for sf in files:
            content = await self.file.fetchContent(sf.path)
            result = await self.ingest.ingestText(
                content=content,
                namespaceId=namespaceId,
                collectionId=collectionId,
                sourceUrl=f"file://{sf.path}",
                title=sf.name,
                platform="file",
            )
            results.append({**result, "path": sf.path})
        return results

    async def enqueueWebJob(
        self,
        url: str,
        namespaceId: Optional[str] = None,
        collectionId: Optional[str] = None,
    ) -> str:
        namespaceId = namespaceId or self.config.defaultNamespace
        jobId = str(uuid.uuid4())
        sourceId = self.ingest.store.createSource(
            namespaceId=namespaceId,
            platform="web",
            url=url,
            collectionId=collectionId,
        )
        self.ingest.store.enqueueJob(
            jobId,
            "web_url",
            namespaceId,
            {"url": url, "collectionId": collectionId},
            sourceId=sourceId,
        )
        return jobId
