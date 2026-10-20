"""Retrieval-only LongMemEval check: run recall, score sessions, no answer model or judge.

For every question, ``RecallPipeline.recall`` runs on its ingested namespace
(``benchmark.ingest``) and the gold sessions are looked up in

- the ranked candidates (``debug.candidates``): ``any@k`` / ``all@k``;
- the sources sent to the answer model: ``ctx any`` / ``ctx all``.

Sessions are identified exactly, by the ``chat <session id>`` source title.

Usage:
    python -m benchmark.recall_only --subset questions.json --output recall.json [--concurrency 4]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import statistics
import time
from typing import Any, Dict

from benchmark.common import NAMESPACE_PREFIX, _loadDotEnv
from kormem.core import usage

logger = logging.getLogger(__name__)

KS = (1, 3, 5, 10)


def _session(title: str) -> str:
    return re.sub(r"^chat ", "", title)


async def _one(pipeline, item: Dict[str, Any], effort: str) -> Dict[str, Any]:
    qid = item["question_id"].lower()
    ctx = await pipeline.recall(item["question"], f"{NAMESPACE_PREFIX}{qid}", item.get("question_date"),
                                effort=effort)
    return {
        "questionId": qid, "questionType": item["question_type"],
        "gold": item["answer_session_ids"],
        "ranked": [_session(c["title"]) for c in ctx.debug.get("candidates", [])],
        "context": [_session(s["title"]) for s in ctx.sources],
        "agent": ctx.debug.get("agent"),
    }


def summarise(rows) -> Dict[str, float]:
    n = len(rows)
    out: Dict[str, float] = {"n": n}
    for k in KS:
        out[f"any@{k}"] = sum(bool(set(r["gold"]) & set(r["ranked"][:k])) for r in rows) / n
        out[f"all@{k}"] = sum(set(r["gold"]) <= set(r["ranked"][:k]) for r in rows) / n
    out["ctx any"] = sum(bool(set(r["gold"]) & set(r["context"])) for r in rows) / n
    out["ctx all"] = sum(set(r["gold"]) <= set(r["context"]) for r in rows) / n
    out["ctx sessions"] = statistics.mean(len(r["context"]) for r in rows)
    return out


async def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _loadDotEnv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    from kormem.retrieve.recall import RecallPipeline

    with open(args.subset, encoding="utf-8") as f:
        items = json.load(f)
    pipeline = RecallPipeline()
    effort = os.environ.get("KORMEM_RECALL_EFFORT", "high")
    sem = asyncio.Semaphore(args.concurrency)

    async def run(item):
        async with sem:
            return await _one(pipeline, item, effort)

    t0 = time.time()
    rows = await asyncio.gather(*(run(it) for it in items))
    summary = summarise(rows)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "usage": usage.summarize(since=t0),
                   "rows": rows}, f, indent=1)
    for key, value in summary.items():
        print(f"{key:13} {value:.3f}" if key != "n" else f"{key:13} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
