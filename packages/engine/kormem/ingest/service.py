"""Document ingestion: chunk, embed, store."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from kormem.chunking.text_chunker import TextChunker
from kormem.core.config import KormemConfig, loadConfig
from kormem.core.document_content import packDocumentContent
from kormem.core.encryption import encryptText
from kormem.core.ids import generateSegmentKey, contentHash
from kormem.core.keywords import extractTfidfKeywords, hashKeywords
from kormem.embeddings.provider import getEmbeddingProvider
from kormem.store.sqlite_store import SqliteStore


def chunkSpans(content: str, texts: List[str]) -> List[Optional[Tuple[int, int]]]:
    """Where each chunk sits in ``content`` (chunks are overlapping substrings, in order); None if not found."""
    spans: List[Optional[Tuple[int, int]]] = []
    start = 0
    for t in texts:
        i = content.find(t, start)
        if i < 0:
            i = content.find(t)
        spans.append((i, i + len(t)) if i >= 0 and t else None)
        if i >= 0:
            start = i + 1
    return spans


class IngestService:
    def __init__(
        self,
        store: Optional[SqliteStore] = None,
        config: Optional[KormemConfig] = None,
    ):
        self.store = store or SqliteStore()
        self.config = config or loadConfig()
        self.embedder = getEmbeddingProvider(self.config)

    async def ingestText(
        self,
        content: str,
        namespaceId: Optional[str] = None,
        collectionId: Optional[str] = None,
        sourceUrl: Optional[str] = None,
        title: Optional[str] = None,
        platform: str = "text",
        role: Optional[str] = None,
        kind: Optional[str] = None,
        sourceDate: Optional[str] = None,
        enqueue: bool = True,
        storeContent: bool = True,
        metadata: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Store, chunk and embed ``content``, then queue it for extraction.

        ``kind`` picks how extraction treats it: ``document`` (files, URLs,
        text; the default), ``fact`` (``remember``; batched) or ``chat``
        (managed by :class:`ChatSessionService`, which queues its own jobs).
        ``storeContent=False`` keeps only the chunks: chats keep their text
        once, as turns, and rebuild the transcript on demand.
        """
        namespaceId = namespaceId or self.config.defaultNamespace
        kind = kind or ("fact" if role else "document")
        if role:
            content = f"[{role}]: {content}"
        sourceDate = sourceDate or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        sourceId = self.store.createSource(
            namespaceId=namespaceId,
            platform=platform,
            url=sourceUrl,
            collectionId=collectionId,
        )
        docId = self.store.createDocument(
            namespaceId=namespaceId,
            sourceId=sourceId,
            title=title or "Untitled",
            canonicalUrl=sourceUrl,
            platform=platform,
            collectionId=collectionId,
            contentHash=contentHash(content),
            encryptedData=packDocumentContent(content, self.config.encryptionKey) if storeContent else None,
            metadata={"kind": kind, "date": sourceDate, **(metadata or {})},
        )
        segmentCount = await self._syncSegments(
            content=content,
            namespaceId=namespaceId,
            documentId=docId,
            sourceId=sourceId,
            collectionId=collectionId,
            metadata={"sourceUrl": sourceUrl, "sourceTitle": title, "sourcePlatform": platform,
                      "sourceDate": sourceDate},
        )
        if enqueue and kind != "chat":
            self.store.enqueueExtractionJob(str(uuid.uuid4()), namespaceId, kind, documentId=docId)
        return {
            "sourceId": sourceId,
            "documentId": docId,
            "segmentCount": segmentCount,
        }

    async def updateText(self, documentId: str, content: str, storeContent: bool = True) -> int:
        """Replace a document's text (a growing chat), re-using embeddings of unchanged chunks."""
        doc = self.store.getDocument(documentId)
        if doc is None:
            raise KeyError(documentId)
        self.store.updateDocumentContent(
            documentId, packDocumentContent(content, self.config.encryptionKey) if storeContent else None,
            contentHash(content),
        )
        reuse = self.store.embeddingsByTextHash(documentId)
        meta = json.loads(doc["metadata"] or "{}")
        self.store.replaceDocumentSegments(documentId)
        return await self._syncSegments(
            content=content, namespaceId=doc["namespaceId"], documentId=documentId,
            sourceId=doc["sourceId"], collectionId=doc["collectionId"],
            metadata={"sourceUrl": doc["canonicalUrl"], "sourceTitle": doc["title"],
                      "sourcePlatform": doc["platform"], "sourceDate": meta.get("date")},
            reuse=reuse,
        )

    async def _syncSegments(
        self,
        content: str,
        namespaceId: str,
        documentId: str,
        sourceId: str,
        collectionId: Optional[str],
        metadata: Optional[dict] = None,
        reuse: Optional[Dict[str, List[float]]] = None,
    ) -> int:
        meta = metadata or {}
        if len(content.strip()) < 100:
            chunkDicts = [{
                "text": content.strip(),
                "kind": "paragraph",
                "metadata": meta,
            }]
        else:
            # ~800-char overlapping chunks for chats too: whole-turn windows
            # (up to ~12k chars) retrieve less precisely.
            chunker = TextChunker(1000, 200)
            chunkDicts = await chunker.chunkContent(content, meta)

        texts = [c["text"] for c in chunkDicts]
        hashes = [contentHash(t) for t in texts]
        reuse = reuse or {}
        fresh = [t for t, h in zip(texts, hashes) if h not in reuse]
        freshEmb = iter(await self.embedder.encodeBatch(fresh) if fresh else [])
        embeddings = [reuse[h] if h in reuse else next(freshEmb) for h in hashes]
        encKey = self.config.encryptionKey

        spans = chunkSpans(content, texts)
        segments: List[dict] = []
        for idx, (chunk, emb, textHash) in enumerate(zip(chunkDicts, embeddings, hashes)):
            text = chunk["text"]
            span = spans[idx]
            tfidf = extractTfidfKeywords(text)
            from kormem.core.keywords import computeHash
            kwList = [
                {"hash": computeHash(word), "score": score}
                for word, score in sorted(tfidf.items(), key=lambda x: -x[1])[:15]
            ]

            # A chunk found in the document stores only its span; the text lives once, in the document.
            encPayload = encryptText(text, encKey) if span is None else None
            segKey = generateSegmentKey(namespaceId, documentId, idx)
            chunkMeta = {k: v for k, v in chunk.get("metadata", {}).items() if k not in ("startPos", "endPos")}
            segments.append({
                "segmentKey": segKey,
                "segmentIndex": idx,
                "embedding": emb,
                "embeddingDim": len(emb),
                "plainText": text if span is None and not encKey else None,
                "encryptedText": json.dumps(encPayload) if encPayload else None,
                "keywords": kwList,
                "keywordHashes": [k["hash"] for k in kwList],
                "metadata": {**meta, **chunkMeta, "textHash": textHash,
                             **({"span": list(span)} if span else {})},
            })

        self.store.upsertSegments(
            namespaceId, documentId, sourceId, segments, collectionId
        )
        return len(segments)

    async def embedPending(
        self,
        namespaceId: Optional[str] = None,
        limit: int = 100,
        force: bool = False,
    ) -> int:
        namespaceId = namespaceId or self.config.defaultNamespace
        rows = self.store.getSegmentsForEmbed(force=force, limit=limit)
        count = 0
        texts: List[str] = []
        keys: List[str] = []
        for row in rows:
            if namespaceId and row["namespaceId"] != namespaceId:
                continue
            text = self.store.getSegmentText(row, self.config.encryptionKey)
            if text:
                texts.append(text)
                keys.append(row["segmentKey"])
        if not texts:
            return 0
        embeddings = await self.embedder.encodeBatch(texts)
        for key, emb in zip(keys, embeddings):
            self.store.updateSegmentEmbedding(key, emb)
            count += 1
        return count
