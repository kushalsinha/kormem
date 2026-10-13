"""Public API schemas for retrieval and ingestion."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class Citation(BaseModel):
    index: int
    segmentKey: str = "unknown"
    textSnippet: Optional[str] = None
    sourceUrl: Optional[str] = None
    sourceTitle: Optional[str] = None
    timestampStart: Optional[float] = None
    timestampEnd: Optional[float] = None
    timestampedUrl: Optional[str] = None
    platform: Optional[str] = None

    def toReference(self) -> str:
        return f"[{self.index}]"

    @classmethod
    def buildTimestampedUrl(cls, baseUrl: str, timestampSeconds: float) -> Optional[str]:
        if not baseUrl or timestampSeconds is None:
            return baseUrl
        ts = int(timestampSeconds)
        if "youtube.com" in baseUrl or "youtu.be" in baseUrl:
            sep = "&" if "?" in baseUrl else "?"
            return f"{baseUrl}{sep}t={ts}"
        if "spotify.com" in baseUrl:
            return f"{baseUrl}#t={ts}"
        return f"{baseUrl}#t={ts}"


class KnowledgeItem(BaseModel):
    segmentKey: str
    text: str
    score: float = 0.0
    ceScore: Optional[float] = None
    sourceUrl: Optional[str] = None
    sourceTitle: Optional[str] = None
    sourcePlatform: Optional[str] = None
    documentId: Optional[str] = None
    sourceId: Optional[str] = None
    fullDocumentText: Optional[str] = None
    liveText: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def toCitation(self, index: int) -> Citation:
        ts = self.metadata.get("startSeconds")
        url = self.sourceUrl
        tsUrl = Citation.buildTimestampedUrl(url, ts) if url and ts is not None else url
        return Citation(
            index=index,
            segmentKey=self.segmentKey,
            textSnippet=(self.text[:200] + "...") if len(self.text) > 200 else self.text,
            sourceUrl=url,
            sourceTitle=self.sourceTitle,
            timestampStart=ts,
            platform=self.sourcePlatform,
            timestampedUrl=tsUrl,
        )


class GuidanceItem(BaseModel):
    itemType: Literal["lesson", "memory", "preference", "trait"]
    phrase: str
    score: float = 1.0
    lastUpdated: Optional[datetime] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str
    namespaceId: str = "main"
    collectionId: Optional[str] = None
    limit: int = 8
    mode: Literal["hybrid", "keyword", "vector"] = "hybrid"
    minScore: float = 0.0
    includeGuidance: bool = True
    expand: bool = False
    liveFetch: bool = False
    window: int = 2
    rerank: Optional[bool] = None
    sourceIdsFilter: Optional[list[str]] = None
    queryDate: Optional[str] = None


class NoteReadRequest(BaseModel):
    ref: str
    namespaceId: str = "main"


class DocumentReadRequest(BaseModel):
    documentId: str
    namespaceId: str = "main"


class SearchResponse(BaseModel):
    query: str
    knowledge: list[KnowledgeItem] = Field(default_factory=list)
    guidance: list[GuidanceItem] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    retrievalTimeMs: float = 0.0
    contextOrdering: Optional[str] = None


class IngestTextRequest(BaseModel):
    content: str
    namespaceId: str = "main"
    collectionId: Optional[str] = None
    sourceUrl: Optional[str] = None
    title: Optional[str] = None
    platform: str = "text"
    role: Optional[Literal["user", "assistant", "system"]] = None


class IngestSessionRequest(BaseModel):
    turns: list[dict[str, Any]]
    namespaceId: str = "main"
    sessionId: str
    sessionDate: Optional[str] = None
    collectionId: Optional[str] = None


class ChatAppendRequest(BaseModel):
    namespaceId: str = "main"
    sessionId: str
    turns: list[dict[str, Any]]      # {role, content, idx?, at?}; re-sent idx values are ignored
    sessionDate: Optional[str] = None
    project: Optional[str] = None    # e.g. the repository a coding session ran in; scopes its lessons
    close: bool = False              # also end the session (extract its last stretch now)


class ChatCloseRequest(BaseModel):
    namespaceId: str = "main"
    sessionId: str


class ExtractChatRequest(BaseModel):
    namespaceId: str = "main"
    turns: list[dict[str, Any]] = Field(default_factory=list)
    sessionId: Optional[str] = None
    trace: Optional[dict[str, Any]] = None
    scratchpadText: Optional[str] = None


class InitRequest(BaseModel):
    namespaceId: str = "main"


class ExtractRunRequest(BaseModel):
    namespaceId: Optional[str] = None  # None: every namespace
    maxJobs: int = 50


class MaintainRequest(BaseModel):
    namespaceId: Optional[str] = None   # None: every namespace starting with namespacePrefix
    namespacePrefix: str = ""
    condense: bool = True               # the one paid step: condensing long notes
    vacuum: bool = False                # shrink the index file afterwards (slow on large indexes)


class ExtractQueueRequest(BaseModel):
    action: Literal["park", "resume"]
    namespacePrefix: str = ""  # "" = every namespace


class ContextRequest(BaseModel):
    query: str
    namespaceId: str = "main"
    queryDate: Optional[str] = None
    knowledgeLimit: int = 8


class RecallRequest(BaseModel):
    query: str
    namespaceId: str = "main"
    queryDate: Optional[str] = None
    effort: Literal["low", "medium"] = "low"   # low: CLI/MCP; medium: deeper agent
    project: Optional[str] = None              # include this project's lessons


class IngestPathRequest(BaseModel):
    path: str
    namespaceId: str = "main"
    collectionId: Optional[str] = None
    recursive: bool = False


class IngestUrlRequest(BaseModel):
    url: str
    namespaceId: str = "main"
    collectionId: Optional[str] = None
    scrapeMode: Literal["single", "crawl"] = "single"


class StatusResponse(BaseModel):
    home: str
    namespaceId: str
    segmentCount: int
    documentCount: int
    sourceCount: int
    pendingEmbeds: int
    pendingJobs: int
    pendingExtractions: int = 0
    engineVersion: str
