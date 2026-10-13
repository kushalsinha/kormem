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
from kormem.extract.service import ExtractionService
from kormem.ingest.chat_sessions import ChatSessionService
from kormem.ingest.service import IngestService
from kormem.vault.paths import vaultDir

logger = logging.getLogger(__name__)


async def buildOne(item: Dict[str, Any], chats: ChatSessionService, extraction: ExtractionService,
                   fresh: bool = False) -> Dict[str, int]:
    ns = f"{NAMESPACE_PREFIX}{item['question_id'].lower()}"
    store = chats.store
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
        store.purgeNamespace(ns)
        shutil.rmtree(vaultDir(ns), ignore_errors=True)
        store.ensureNamespace(ns)
    for sid, date, turns in zip(ids, item["haystack_dates"], item["haystack_sessions"]):
        await chats.append(ns, sid, [{"role": t.get("role"), "content": t.get("content")} for t in turns],
                           sessionDate=date)
        chats.close(ns, sid)
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
    print(json.dumps({"built": len(results) - len(bad), "problems": bad}, indent=2))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
