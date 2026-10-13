"""Background worker for web_url and file ingestion jobs."""

from __future__ import annotations

import json
import logging

from kormem.connectors.service import ConnectorService
from kormem.core.config import loadConfig
from kormem.extract.service import ExtractionService
from kormem.store.sqlite_store import SqliteStore

logger = logging.getLogger(__name__)


class IngestionWorker:
    def __init__(self):
        self.config = loadConfig()
        self.store = SqliteStore()
        self.connectors = ConnectorService(config=self.config)
        self.extraction = ExtractionService(self.store, self.config)

    async def processBatch(self, limit: int = 5) -> int:
        jobs = self.store.claimJobs(limit)
        for job in jobs:
            try:
                payload = json.loads(job["payload"] or "{}")
                if job["jobType"] == "web_url":
                    await self.connectors.ingestUrl(
                        url=payload["url"],
                        namespaceId=job["namespaceId"],
                        collectionId=payload.get("collectionId"),
                    )
                elif job["jobType"] == "file":
                    await self.connectors.ingestPath(
                        path=payload["path"],
                        namespaceId=job["namespaceId"],
                        collectionId=payload.get("collectionId"),
                    )
                self.store.completeJob(job["id"])
            except Exception as e:
                logger.exception("Job %s failed", job["id"])
                self.store.failJob(job["id"], str(e))
        await self.extraction.processPending()
        return len(jobs)
