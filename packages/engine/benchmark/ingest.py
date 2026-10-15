"""Build each LongMemEval question's memory through kormem's product path.

For every question, its haystack sessions are sent to a fresh namespace
(``lme_<qid>``) as chat sessions (``ChatSessionService.append`` + ``close``,
exactly what ``/chat/append`` does), then the extraction queue for that
namespace is drained: one vault-writer call per session.
The result is the same vault a user would get.

Usage:
    python -m benchmark.ingest --subset questions.json [--qid a,b] [--concurrency 12]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import sys
import time
from typing import Any, Dict, List

from benchmark.common import NAMESPACE_PREFIX, _loadDotEnv, datasetPath
from kormem.core.config import loadConfig
from kormem.embeddings.base import EmbeddingProvider
from kormem.extract.service import ExtractionService
from kormem.ingest.chat_sessions import ChatSessionService
from kormem.ingest.service import IngestService
from kormem.vault.paths import vaultDir

logger = logging.getLogger(__name__)

# Chunk text -> vector from an earlier build of the same namespace (same chunker, same embedding
# model), so a rebuild only pays for text it hasn't embedded before.
_REUSE: Dict[str, Any] = {}  # text -> float32 vector
_REFS: Dict[str, int] = {}   # namespaces being rebuilt that hold each text (filler sessions repeat)


class ReuseEmbedder(EmbeddingProvider):
    """Serves vectors from ``_REUSE`` and sends only the misses to the wrapped embedder."""

    def __init__(self, inner: EmbeddingProvider):
        self.inner, self.hits, self.misses = inner, 0, 0

    @property
    def dimension(self) -> int:
        return self.inner.dimension

    @property
    def providerName(self) -> str:
        return self.inner.providerName

    @property
    def modelName(self) -> str:
        return self.inner.modelName

    async def encodeBatch(self, texts: List[str]) -> List[List[float]]:
        known = {t: _REUSE[t] for t in texts if t in _REUSE}  # taken before any await
        missing = [t for t in dict.fromkeys(texts) if t not in known]
        fresh = dict(zip(missing, await self.inner.encodeBatch(missing))) if missing else {}
        self.hits += len(texts) - len(missing)
        self.misses += len(missing)
        return [known[t].tolist() if t in known else fresh[t] for t in texts]

    async def encodeQuery(self, text: str) -> List[float]:
        return await self.inner.encodeQuery(text)


def _rememberVectors(store, ns: str, dim: int, encryptionKey) -> List[str]:
    """Cache the namespace's chunk vectors by text before it is purged and rebuilt; returns the texts.

    ``KORMEM_REUSE_FROM`` names another index (read-only), e.g. the store this run's home was copied
    from, whose vectors are used too: an interrupted rebuild has already purged its own.
    """
    import os
    import sqlite3
    from kormem.store.sqlite_store import _unpackEmbedding
    sql = ("SELECT * FROM segments WHERE namespaceId=? AND kind='segment' "
           "AND embedding IS NOT NULL AND embeddingDim=?")
    with store._connectIndex() as conn:
        rows = conn.execute(sql, (ns, dim)).fetchall()
    if os.environ.get("KORMEM_REUSE_FROM"):
        src = sqlite3.connect(f"file:{os.environ['KORMEM_REUSE_FROM']}?mode=ro", uri=True)
        src.row_factory = sqlite3.Row
        try:
            rows += src.execute(sql, (ns, dim)).fetchall()
        finally:
            src.close()
    texts = []
    for r in rows:
        text = store.getSegmentText(r, encryptionKey)
        if text:
            _REUSE[text] = _unpackEmbedding(r["embedding"], r["embeddingDim"]).astype("float32")
            _REFS[text] = _REFS.get(text, 0) + 1
            texts.append(text)
    return texts


async def buildOne(item: Dict[str, Any], chats: ChatSessionService, extraction: ExtractionService,
                   fresh: bool = False) -> Dict[str, int]:
    ns = f"{NAMESPACE_PREFIX}{item['question_id'].lower()}"
    store = chats.store
    cached: List[str] = []
    ids = item.get("haystack_session_ids") or [f"s{i}" for i in range(len(item["haystack_sessions"]))]
    with store._connectIndex() as conn:
        have = {r["sessionId"] for r in conn.execute(
            "SELECT sessionId FROM chat_sessions WHERE namespaceId=? AND documentId IS NOT NULL", (ns,))}
    if not fresh and set(ids) <= have:
        # Already ingested: retry failed extraction jobs and finish whatever is queued.
        with store._connectIndex() as conn:
            conn.execute("UPDATE extraction_jobs SET status='pending' WHERE namespaceId=? AND status='failed'", (ns,))
            conn.commit()
        ids = []
    else:
        if not fresh:
            cached = _rememberVectors(store, ns, chats.ingest.embedder.dimension, chats.config.encryptionKey)
            logger.info("%s: %d chunk vectors kept for reuse", ns, len(cached))
        store.purgeNamespace(ns)
        shutil.rmtree(vaultDir(ns), ignore_errors=True)
        store.ensureNamespace(ns)
    for sid, date, turns in zip(ids, item["haystack_dates"], item["haystack_sessions"]):
        await chats.append(ns, sid, [{"role": t.get("role"), "content": t.get("content")} for t in turns],
                           sessionDate=date)
        chats.close(ns, sid)
    for text in cached:
        _REFS[text] -= 1
        if not _REFS[text]:
            del _REFS[text]
            _REUSE.pop(text, None)
    jobs = 0
    while True:
        n = await extraction.processPending(limit=3, namespaceId=ns)
        if not n:
            break
        jobs += n
    with store._connectIndex() as conn:
        failed = conn.execute("SELECT COUNT(*) AS c FROM extraction_jobs WHERE namespaceId=? AND status='failed'",
                              (ns,)).fetchone()["c"]
        notes = conn.execute("SELECT COUNT(*) AS c FROM notes WHERE namespaceId=?", (ns,)).fetchone()["c"]
    return {"sessionsAdded": len(ids), "jobs": jobs, "failed": failed, "notes": notes}


async def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _loadDotEnv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=None, help="Questions JSON (default: $LONGMEMEVAL_DATASET)")
    ap.add_argument("--qid", default=None, help="Comma-separated qids to build (default: all in the file)")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--fresh", action="store_true", help="Rebuild namespaces that were already ingested")
    args = ap.parse_args()

    items = json.load(open(args.subset or datasetPath(), encoding="utf-8"))
    if args.qid:
        wanted = {q.strip().lower() for q in args.qid.split(",")}
        items = [it for it in items if it["question_id"].lower() in wanted]
    config = loadConfig()
    ingest = IngestService(config=config)
    ingest.embedder = ReuseEmbedder(ingest.embedder)
    chats = ChatSessionService(ingest=ingest, config=config)
    extraction = ExtractionService(store=ingest.store, config=config)
    ingest.store.requeueStaleExtractions()  # jobs a killed run left half-done
    sem = asyncio.Semaphore(args.concurrency)
    results: Dict[str, Any] = {}

    async def bounded(it: Dict[str, Any]) -> None:
        qid = it["question_id"].lower()
        async with sem:
            t0 = time.monotonic()
            try:
                results[qid] = await buildOne(it, chats, extraction, args.fresh)
            except Exception as e:
                logger.exception("build failed qid=%s", qid)
                results[qid] = {"error": str(e)[:300]}
            logger.info("[%d/%d] %s %s %.0fs", len(results), len(items), qid, results[qid], time.monotonic() - t0)

    await asyncio.gather(*[bounded(it) for it in items])
    bad = {q: r for q, r in results.items() if r.get("error") or r.get("failed")}
    logger.info("chunk vectors reused %d, embedded %d", ingest.embedder.hits, ingest.embedder.misses)
    print(json.dumps({"built": len(results) - len(bad), "problems": bad}, indent=2))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
