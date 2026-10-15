"""Re-judge saved LongMemEval results with Mem0's judge, to compare with Mem0's published score.

Mem0 (mem0ai/memory-benchmarks) scores LongMemEval with a single unified prompt for
every question type, ``benchmarks/longmemeval/prompts.py::JUDGE_PROMPT``, which
accepts semantic matches, partial abstentions and loose preference matches, judged by
``gpt-5`` by default. The prompt is loaded from a local clone of that repository
rather than copied here, and the protocol follows its ``LLMClient``: no system
prompt, the model's default temperature for gpt-5 / o-series (0 otherwise),
4,096 completion tokens, and the verdict parsed by ``_parse_yes_no_judgment``.

These scores are comparable with Mem0's 93.4-94.4%, not with the official
``evaluate_qa.py`` scores (``benchmark.official_judge``).

Usage:
    git clone https://github.com/mem0ai/memory-benchmarks /tmp/mem0
    python -m benchmark.mem0_judge out.json --mem0-repo /tmp/mem0 [--model gpt-5] [--output out.mem0.json]
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import httpx

from benchmark.common import _loadDotEnv, datasetPath
from kormem.core import usage


def _loadPrompts(repo: Path):
    path = repo / "benchmarks" / "longmemeval" / "prompts.py"
    spec = importlib.util.spec_from_file_location("mem0_longmemeval_prompts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parseVerdict(raw: str) -> bool:
    """Mem0's ``LLMClient._parse_yes_no_judgment``, verbatim in behaviour."""
    text = raw.strip()
    if not text:
        return False
    afterCot = re.split(r"</judge_thinking>|</thinking>", text, flags=re.IGNORECASE)
    region = afterCot[-1].strip() if afterCot else text
    for line in reversed([ln.strip().lower() for ln in region.splitlines() if ln.strip()]):
        if line in ("yes", "no"):
            return line == "yes"
    tokens = re.findall(r"\b(yes|no)\b", region.lower())
    if tokens:
        return tokens[-1] == "yes"
    return text.lower().startswith("yes")


async def _judge(client: httpx.AsyncClient, model: str, prompt: str) -> str:
    body: Dict[str, Any] = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if model.lower().startswith(("gpt-5", "o1", "o3", "o4")):
        body["max_completion_tokens"] = 4096
    else:
        body.update(temperature=0, max_tokens=4096)
    for attempt in range(5):
        try:
            resp = await client.post("https://api.openai.com/v1/chat/completions", json=body)
            resp.raise_for_status()
            data = resp.json()
            usage.record("bench.judge.mem0", model, data.get("usage"))
            return (data["choices"][0]["message"]["content"] or "").strip()
        except (httpx.HTTPError, KeyError):
            if attempt == 4:
                raise
            await asyncio.sleep(2 * (attempt + 1))
    return ""


async def main() -> int:
    _loadDotEnv()
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--mem0-repo", required=True)
    ap.add_argument("--model", default="gpt-5")
    ap.add_argument("--subset", default=None)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    prompts = _loadPrompts(Path(args.mem0_repo))
    items = {it["question_id"].lower(): it for it in json.load(open(datasetPath(args.subset)))}
    results = json.load(open(args.results))["results"]
    sem = asyncio.Semaphore(args.concurrency)
    headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}

    async with httpx.AsyncClient(timeout=300, headers=headers) as client:
        async def one(r: Dict[str, Any]) -> Dict[str, Any]:
            it = items[r["questionId"].lower()]
            prompt = prompts.get_judge_prompt(
                question_type=it["question_type"], question_id=it["question_id"], question=it["question"],
                answer=it["answer"], response=r.get("generated") or "", question_date=it.get("question_date", ""))
            async with sem:
                raw = await _judge(client, args.model, prompt)
            return {"questionId": r["questionId"], "questionType": it["question_type"],
                    "mem0": parseVerdict(raw), "raw": raw[-400:]}

        judged = await asyncio.gather(*(one(r) for r in results))

    byType: Dict[str, list] = defaultdict(list)
    for j in judged:
        byType[j["questionType"]].append(j["mem0"])
    correct = sum(j["mem0"] for j in judged)
    out = {"judgeModel": args.model, "judgePrompt": "mem0ai/memory-benchmarks longmemeval JUDGE_PROMPT",
           "correct": correct, "total": len(judged), "accuracy": correct / len(judged),
           "byType": {t: {"correct": sum(v), "total": len(v)} for t, v in sorted(byType.items())},
           "results": judged}
    path = args.output or str(Path(args.results).with_suffix(f".mem0_{args.model}.json"))
    Path(path).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
