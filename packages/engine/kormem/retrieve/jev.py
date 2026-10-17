"""Jev (TypeSafe System One): fast yes/no decisions for recall.

Jev answers a question about a small ``state`` with a calibrated probability
in about 0.1 s, for $0.042 per million input tokens (output is free). Recall
asks it one thing per candidate source: does this preview help answer the
question, even partly? One call per preview, all in parallel.

On LongMemEval (500 questions), fusing Jev's ranking of the first pass's top
30 previews with the first-pass ranking put every needed session in the top
5 for 97.8% of questions, against 95.6% for the first pass alone, at about
$0.0009 a question. Needed sources scored a median 0.9 and the rest 0.06.

Previews of the user's sources leave the machine only when ``TYPESAFE_API_KEY``
is set. Without the key, or when Jev errors or is slow, :func:`scoreSources`
returns ``None`` and recall ranks locally as before.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

import httpx

from kormem.core import usage
from kormem.core.config import KormemConfig
from kormem.core.models import decisionKey, decisionModel

logger = logging.getLogger(__name__)

JEV_URL = "https://api.typesafe.ai/v1/systemone"
TIMEOUT_SEC = 3.0       # whole batch; previews not scored by then count as unscored
CONCURRENCY = 16
RETRIES = 3
NEEDED = ("Does `source` contain information that helps answer `question`, even partly: one item of a count, "
          "one event's date, an earlier or later value, a stated preference, or something the assistant said?")

_warned = False


def available(config: KormemConfig) -> bool:
    return bool(decisionKey(config))


async def _one(client: httpx.AsyncClient, sem: asyncio.Semaphore, model: str,
               question: str, questionDate: str, preview: str) -> Optional[float]:
    body = {"model": model,
            "state": {"question": question, "question_date": questionDate, "source": preview},
            "questions": {"needed": {"type": "noul", "instructions": NEEDED}}}
    async with sem:
        for attempt in range(RETRIES):
            try:
                r = await client.post(JEV_URL, json=body)
            except httpx.HTTPError:
                await asyncio.sleep(0.2 * (attempt + 1))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                await asyncio.sleep(0.2 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            usage.record("recall.jev", model, data.get("usage"))
            return float(data["answers"]["needed"]["noul"])
    return None


async def scoreSources(question: str, questionDate: str, previews: List[str],
                       config: KormemConfig) -> Optional[List[Optional[float]]]:
    """Jev's probability that each preview helps answer ``question``; ``None`` per preview it could not
    score, and ``None`` overall without a key or when nothing was scored."""
    global _warned
    key = decisionKey(config)
    if not key or not previews:
        return None
    model = decisionModel(config)[0]
    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient(timeout=TIMEOUT_SEC, headers={"Authorization": f"Bearer {key}"}) as client:
        tasks = [asyncio.create_task(_one(client, sem, model, question, questionDate, p)) for p in previews]
        done, pending = await asyncio.wait(tasks, timeout=TIMEOUT_SEC)
        for t in pending:
            t.cancel()
        scores: List[Optional[float]] = []
        errors = []
        for t in tasks:
            if t in done and t.exception() is None:
                scores.append(t.result())
            else:
                scores.append(None)
                if t in done:
                    errors.append(t.exception())
    if all(s is None for s in scores):
        if not _warned:
            _warned = True
            logger.warning("Jev unavailable (%s); recall ranks locally", errors[0] if errors else "timeout")
        return None
    return scores
