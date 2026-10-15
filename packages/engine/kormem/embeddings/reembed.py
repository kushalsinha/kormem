"""Keep every stored vector on the embedding model in use.

Vectors from different models can't be compared, and two models can share a
dimension (local ``bge-base`` and Gemini are both 768-d), so a switch is never
mixed in: when the provider, model or dimension changes (a Gemini key added
or removed, or ``kormem model set embedding``), every vector is dropped at engine start and the background worker re-embeds the
index with the new model. Keyword search works throughout; vector search
covers more of the index as the re-embed progresses.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from kormem.core.config import KormemConfig, getEmbeddingPinPath, loadConfig
from kormem.core.models import embeddingIdentity
from kormem.embeddings.provider import getEmbeddingProvider, getInternalEmbedder

logger = logging.getLogger(__name__)


def switchIfNeeded(store, config: KormemConfig, home: Optional[Path] = None) -> Optional[str]:
    """Drop the vectors when the selected embedding model differs from the one that built them.

    The pin holds ``provider:model:dim``. Returns the previous pin when it switched. The pin is
    written after the reset, so a crash in between re-runs the reset rather than mixing models.
    """
    selected = embeddingIdentity(config)
    pin = getEmbeddingPinPath(home)
    recorded = pin.read_text(encoding="utf-8").strip() if pin.exists() else None
    previous = recorded
    if recorded and ":" not in recorded and recorded == selected.split(":")[0]:
        previous = selected  # a provider-only pin from 0.1.x: same provider, same defaults
    switched = None
    if previous and previous != selected:
        n = store.resetEmbeddings()
        logger.info("embeddings switched from %s to %s: re-embedding %d vectors in the background",
                    previous, selected, n)
        switched = previous
    if recorded != selected:
        pin.parent.mkdir(parents=True, exist_ok=True)
        pin.write_text(selected + "\n", encoding="utf-8")
    return switched


async def reembedPending(store, config: Optional[KormemConfig] = None, limit: int = 64,
                         namespaceId: Optional[str] = None) -> int:
    """Embed up to ``limit`` segments (and any folders) that have no vector. Returns how many."""
    config = config or loadConfig()
    rows = store.segmentsWithoutVectors(limit, namespaceId)
    main, internal = getEmbeddingProvider(config), getInternalEmbedder(config)
    done = 0
    for embedder, group in ((internal, [r for r in rows if r["kind"] == "note"]),
                            (main, [r for r in rows if r["kind"] != "note"])):
        keys, texts = [], []
        for r in group:
            text = store.getSegmentText(r, config.encryptionKey)
            if r["kind"] == "note":
                title = json.loads(r["metadata"] or "{}").get("title") or ""
                text = f"{title}\n{text}".strip()  # as note_embed embeds it
            if text:
                keys.append(r["segmentKey"])
                texts.append(text)
            else:
                store.markEmbeddingDropped(r["segmentKey"])  # nothing to embed: don't retry forever
        if texts:
            for key, vec in zip(keys, await embedder.encodeBatch(texts)):
                store.updateSegmentEmbedding(key, vec)
            done += len(keys)
    for f in store.foldersWithoutVectors()[:limit]:
        store.setFolderEmbedding(f["namespaceId"], f["path"], await internal.encodeQuery(f["purpose"]))
        done += 1
    return done
