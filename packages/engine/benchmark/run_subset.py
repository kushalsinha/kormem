"""Answer and judge LongMemEval questions with kormem recall.

Run after ``benchmark.ingest`` has built each question's vault. For every
question, ``RecallPipeline.recall`` (the same code ``/recall`` runs; effort
``KORMEM_RECALL_EFFORT``, default medium) returns notes, sources and
findings, the answer model writes notes then an ``ANSWER:`` line, and
kormem's judge scores it. Re-score with ``benchmark.official_judge`` for
numbers comparable with published results.

With ``KORMEM_CONTEXT_CACHE=<dir>`` each question's recall output is saved
to and reused from ``<dir>/<qid>.json``, so answer models can be compared on
identical input.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from benchmark.common import NAMESPACE_PREFIX, _loadDotEnv
from benchmark.config import loadBenchmarkConfig
from benchmark.judge import LlmJudge
from kormem.core import usage

logger = logging.getLogger(__name__)

_CONTEXT_CACHE = os.environ.get("KORMEM_CONTEXT_CACHE")
_RECALL_EFFORT = os.environ.get("KORMEM_RECALL_EFFORT", "medium")


ANSWER_PROMPT = """You are answering a question about the user's own past conversations, using ONLY the context below.

Today's date (authoritative): {queryDate}

Sources:
- CHAT SESSIONS are the raw transcripts of the past conversations most relevant to the question; each starts with its [Session date: ...]. They are the ground truth.
- RECALL FINDINGS are dated facts a recall agent collected by reading the sessions and notes, with quotes. Use them as a checklist of instances and dates, and confirm each against the transcripts.
- VAULT NOTES are topic notes built from all past conversations: current facts, a dated timeline and what the assistant suggested. They help you see every instance, but if a note conflicts with a transcript, trust the transcript.
- RELEVANT MEMORY holds dated lines extracted from the conversations. Like the notes, they can be incomplete.
- Relative dates in user turns are resolved in brackets, e.g. "three weeks ago [2023-05-07]"; "≈" marks approximate ones.

Output format:
NOTES:
- one line per relevant fact or candidate item: what it is, its date, the session it came from, and a status tag when relevant — [counts], [duplicate of ...], [planned/not done], [suggested by assistant], [outside time window], [superseded by ...], [different entity]
ANSWER: <the final answer, one or two sentences>

Keep NOTES brief (only what bears on the question). The ANSWER line must stand on its own.

Rules:
1. For counts, totals and orderings, the NOTES must list every candidate item with its tag; the ANSWER gives the result and briefly names the items counted.
2. Dating events: an event happens on the date of the session where the user reports it, shifted by any relative phrase in that session ("yesterday", "last weekend", "three weeks ago" are relative to THAT session's date, not today). Plans, intentions and questions ("I'm going to", "I'm thinking of", "can you suggest") are not completed events unless a later session confirms them.
3. Date arithmetic: first write down each event's exact date (year included), then compute. Day differences are calendar days between the two dates; "N weeks/months" questions round to the nearest whole unit. "How long ago" / "how many days ago" is measured from today's date above — unless the question names another reference event ("how many days ago did I do X when I did Y", "how long before Y"), in which case measure between the two events.
4. Counting and totals: list every candidate instance across ALL sessions, keep only those satisfying every constraint in the question (time window relative to today, the specific action or role asked about — e.g. "led" vs "took part in", "bought" vs "considered"), merge mentions of the same item across sessions, then count or sum. Do not count things the assistant suggested or the user only planned or asked about. When OR-joined actions are asked ("pick up or return"), count each action separately.
5. "The Nth time I did X" / ordering questions: list every instance chronologically with its date, then pick or order.
6. Knowledge updates: "current/now/latest" means the value from the most recent session; "previous/before/used to" means the value immediately before the latest change.
7. Details from assistant replies ("what was item N", "which brand did you mention"): find the assistant's reply in the transcripts and quote the item.
8. Preference / recommendation questions: give a specific, personalised suggestion grounded in the user's stated tastes, routines and past experiences. Do not answer "I don't know".
9. "Did I do X with Y" questions: if the event is recorded without Y, answer that it was not with Y rather than abstaining.
10. Mismatched details: if the question describes an event slightly differently from the transcripts (different wording, a minor attribute such as a person's description, an approximate date), answer about the matching event. But never substitute a different entity for the one asked about (tennis is not table tennis; a different store, person, city or course is not the one asked about). Only treat the question as a false premise when its core subject clearly never happened or is contradicted (e.g. asks about "my job as a nurse" when the user was always a teacher); then reply "I don't know — you mentioned <actual fact>, not <premise>."
11. Missing information: if the event is recorded but the specific detail asked for (a number, a place, a price) is never stated, say "I don't know" and say what was mentioned — never guess or fill it in from general knowledge. If a "how many X" or "did I X" question finds no instance of X at all, answer "You didn't mention any X" (for the period asked), rather than a bare "I don't know" or a bare number.
12. Do not cite sources or add disclaimers.

[RELEVANT MEMORY — extracted notes; lines may carry (meaning DATE) or (estimated DATE) anchors]
{relevantMemory}

[LESSONS — how the user asked the assistant to work with them]
{lessons}

[RECALL FINDINGS]
{findings}

[VAULT NOTES]
{notes}

[CHAT SESSIONS — raw transcripts of the past sessions selected as relevant to this question, oldest first]
{sessionsText}

Question: {question}
"""


def _answerEndpoint(model: str) -> tuple[str, str]:
    """Return (base_url, api_key_env) for the provider that owns ``model``."""
    m = model.lower()
    if m.startswith("grok") or m.startswith("xai"):
        return ("https://api.x.ai/v1/chat/completions", "XAI_API_KEY")
    if m.startswith("gemini"):
        return ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "GEMINI_API_KEY")
    if m.startswith("claude"):
        return ("https://api.anthropic.com/v1/chat/completions", "ANTHROPIC_API_KEY")
    return ("https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY")


async def _answer(client: httpx.AsyncClient, apiKey: str, model: str,
                  effort: Optional[str], prompt: str, maxTokens: int = 4000) -> str:
    url, keyEnv = _answerEndpoint(model)
    # If apiKey didn't come from the right provider's env, try the correct one.
    chosenKey = os.environ.get(keyEnv) or apiKey
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": maxTokens,
    }
    if effort and keyEnv == "ANTHROPIC_API_KEY":
        # Anthropic's OpenAI-compatible endpoint ignores reasoning_effort; map it
        # to an extended-thinking budget instead.
        budget = {"low": 2000, "medium": 6000, "high": 10000}.get(effort, 6000)
        body["thinking"] = {"type": "enabled", "budget_tokens": min(budget, maxTokens - 1000)}
    elif effort:
        body["reasoning_effort"] = effort
    headers = {"Authorization": f"Bearer {chosenKey}"}
    if keyEnv == "ANTHROPIC_API_KEY" and os.environ.get("ANTHROPIC_WORKSPACE_ID"):
        # Keys not scoped to a workspace must name one per request.
        headers["anthropic-workspace-id"] = os.environ["ANTHROPIC_WORKSPACE_ID"]
    resp = await client.post(url, headers=headers, json=body)
    resp.raise_for_status()
    data = resp.json()
    usage.record("bench.answer", model, data.get("usage"))
    return data["choices"][0]["message"]["content"].strip()


async def _recallContext(pipeline, item: Dict[str, Any]) -> Dict[str, Any]:
    qid = item["question_id"].lower()
    cachePath = Path(_CONTEXT_CACHE) / f"{qid}.json" if _CONTEXT_CACHE else None
    if cachePath and cachePath.exists():
        return json.loads(cachePath.read_text())
    ctx = await pipeline.recall(item["question"], f"{NAMESPACE_PREFIX}{qid}", item.get("question_date"),
                                effort=_RECALL_EFFORT)
    from kormem.extract.lessons import lessonsFor
    from kormem.vault.vault_store import VaultStore
    vault = VaultStore(store=pipeline.store, config=pipeline.config, namespaceId=f"{NAMESPACE_PREFIX}{qid}")
    out = {"relevantMemory": ctx.relevantMemory, "lessons": lessonsFor(vault),
           "notes": ctx.notes, "sources": ctx.sources, "findings": ctx.findings, "debug": ctx.debug}
    if cachePath:
        cachePath.parent.mkdir(parents=True, exist_ok=True)
        cachePath.write_text(json.dumps(out))
    return out


async def _evalOne(pipeline, item: Dict[str, Any], config, judge: LlmJudge, openaiApiKey: str) -> Dict[str, Any]:
    qid = item["question_id"].lower()
    qType = item.get("question_type", "unknown")
    question = item["question"]
    t0 = time.monotonic()
    try:
        ctx = await _recallContext(pipeline, item)
    except Exception as e:
        logger.exception("recall failed qid=%s", qid)
        return {"questionId": qid, "questionType": qType, "score": 0, "error": f"recall: {e!r}"}

    sessionsText = "\n\n---\n\n".join(s["text"] for s in ctx.get("sources", [])) or "(none)"
    notesText = "\n\n".join(f"### {n['title']} ({n['folder']})\n{n['body']}" for n in ctx.get("notes", [])) or "(none)"
    prompt = ANSWER_PROMPT.format(
        queryDate=item.get("question_date") or "unknown",
        relevantMemory=(ctx.get("relevantMemory") or "").strip() or "(none)",
        lessons=(ctx.get("lessons") or "").strip()[:2000] or "(none)",
        findings="\n".join(f"- {f}" for f in ctx.get("findings") or []) or "(none)",
        notes=notesText,
        sessionsText=sessionsText,
        question=question,
    )
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            raw = await _answer(client, openaiApiKey, config.answerModel, config.answerReasoningEffort,
                                prompt, maxTokens=12000)
        m = re.search(r"ANSWER:\s*(.+)", raw, re.S)
        answer = (m.group(1) if m else raw).strip()
        notes = raw[: m.start()].strip() if m else ""
    except Exception as e:
        logger.exception("answer failed qid=%s", qid)
        return {"questionId": qid, "questionType": qType, "score": 0, "error": f"answer: {e!r}"}

    verdict = await judge.judge(question=question, expected=item["answer"], generated=answer, questionType=qType)
    score = 1 if verdict.correct else 0
    logger.info("%s %s %s | %s", "OK " if score else "   ", qid, verdict.reason[:16], answer[:140])
    gold = set(item.get("answer_session_ids") or [])
    inContext = {re.sub(r"^chat ", "", s["title"]) for s in ctx.get("sources", [])}
    dbg = ctx.get("debug", {})
    return {
        "questionId": qid, "questionType": qType, "question": question, "expected": item["answer"],
        "generated": answer, "score": score, "judgeReason": verdict.reason,
        "latencySec": time.monotonic() - t0, "notes": notes, "promptLen": len(prompt),
        "goldSessionsInContext": f"{len(gold & inContext)}/{len(gold)}",
        "selectedSources": dbg.get("selected"), "agent": dbg.get("agent"),
    }


async def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _loadDotEnv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--output", default=None)
    ap.add_argument("--results-jsonl", default=None,
                    help="Append each result as it completes; with --resume, skip qids already in it")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    config = loadBenchmarkConfig()
    judge = LlmJudge(config)
    openaiApiKey = os.environ.get("OPENAI_API_KEY") or config.openaiApiKey or ""
    with open(args.subset, encoding="utf-8") as f:
        items = json.load(f)

    prior: List[Dict[str, Any]] = []
    if args.results_jsonl and args.resume and Path(args.results_jsonl).exists():
        with open(args.results_jsonl, encoding="utf-8") as f:
            prior = [json.loads(line) for line in f if line.strip()]
        # Errored results are retried rather than kept.
        prior = [r for r in prior if "error" not in r]
        doneQids = {r["questionId"] for r in prior}
        items = [it for it in items if it["question_id"].lower() not in doneQids]
        logger.info("resume: %d done, %d remaining", len(prior), len(items))

    sem = asyncio.Semaphore(args.concurrency)
    from kormem.retrieve.recall import RecallPipeline
    pipeline = RecallPipeline()

    async def _bounded(idx, it):
        async with sem:
            r = await _evalOne(pipeline, it, config, judge, openaiApiKey)
        if args.results_jsonl:
            with open(args.results_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
        return r

    t0 = time.monotonic()
    results = prior + list(await asyncio.gather(*[_bounded(i, it) for i, it in enumerate(items)]))

    byType: Dict[str, List[int]] = defaultdict(list)
    for r in results:
        byType[r.get("questionType", "unknown")].append(r.get("score", 0))
    total = len(results)
    correct = sum(r.get("score", 0) for r in results)
    acc = correct / total if total else 0
    byTypeAcc = {
        qt: {"total": len(s), "correct": sum(s), "accuracy": sum(s)/len(s) if s else 0}
        for qt, s in byType.items()
    }
    summary = {
        "runId": str(uuid.uuid4()),
        "subset": args.subset,
        "total": total, "correct": correct, "accuracy": acc,
        "byType": byTypeAcc,
        "elapsedSec": time.monotonic() - t0,
        "recallEffort": _RECALL_EFFORT,
        "models": {
            "answer": config.answerModel,
            "answerReasoning": config.answerReasoningEffort,
            "judge": config.judgeModel,
            "judgeReasoning": config.judgeReasoningEffort,
        },
        "results": results,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "accuracy": acc, "correct": correct, "total": total,
        "byType": byTypeAcc, "elapsedSec": summary["elapsedSec"],
    }, indent=2))
    return 0 if acc >= 0.5 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
