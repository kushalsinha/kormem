"""Embed vault notes into index.sqlite for semantic entry points."""

from __future__ import annotations

from typing import Optional

from kormem.core.config import KormemConfig, loadConfig
from kormem.core.keywords import computeHash, extractTfidfKeywords, hashKeywords
from kormem.embeddings.provider import getInternalEmbedder
from kormem.store.sqlite_store import SqliteStore
from kormem.vault.note import Note
from kormem.vault.paths import relPathFor


async def embedNote(
    note: Note,
    namespaceId: str,
    store: Optional[SqliteStore] = None,
    config: Optional[KormemConfig] = None,
) -> str:
    store = store or SqliteStore()
    config = config or loadConfig()
    embedder = getInternalEmbedder(config)  # notes are ranked locally, for free
    relPath = relPathFor(note)
    text = f"{note.title}\n{note.body}".strip()
    embedding = (await embedder.encodeBatch([text]))[0]
    tfidf = extractTfidfKeywords(text)
    kwList = [
        {"hash": computeHash(word), "score": score}
        for word, score in sorted(tfidf.items(), key=lambda x: -x[1])[:25]
    ]
    # The body is not copied into the index: the vault file is the one copy (see SqliteStore.getSegmentText).
    segment = {
        "segmentKey": f"note:{note.kormemId}",
        "kind": "note",
        "segmentIndex": 0,
        "embedding": embedding,
        "embeddingDim": len(embedding),
        "plainText": None,
        "encryptedText": None,
        "keywords": kwList,
        "keywordHashes": list(hashKeywords(text)),
        "metadata": {
            "kormemId": note.kormemId,
            "relPath": relPath,
            "title": note.title,
            "updated": note.updated,
            "para": note.para,
            "sources": note.sources,
        },
    }
    store.upsertNoteSegment(namespaceId, segment)
    return segment["segmentKey"]
