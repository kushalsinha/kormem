"""Re-judge saved LongMemEval results with the official evaluate_qa.py protocol.

Prompts are copied verbatim from LongMemEval's
``src/evaluation/evaluate_qa.py::get_anscheck_prompt`` (xiaowu0162/LongMemEval);
protocol: judge ``gpt-4o-2024-08-06``, temperature 0, max_tokens 10, verdict
``'yes' in response.lower()``, abstention prompt for ``_abs`` question ids.

Scores from this judge are comparable with published LongMemEval numbers;
``benchmark.judge.LlmJudge`` is kormem's own (more lenient on preferences)
judge and is not.

Usage:
    python -m benchmark.official_judge benchmark/runs/full500_curated.json \\
        [--model gpt-4o-2024-08-06] [--concurrency 16] [--output out.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import httpx

from kormem.core import usage

from benchmark.common import _loadDotEnv, datasetPath

OFFICIAL_JUDGE_MODEL = "gpt-4o-2024-08-06"

_STANDARD = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
_TEMPORAL = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
_KNOWLEDGE_UPDATE = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
_PREFERENCE = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
_ABSTENTION = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."


def getAnscheckPrompt(task: str, question: str, answer: str, response: str, abstention: bool = False) -> str:
    if abstention:
        return _ABSTENTION.format(question, answer, response)
    if task in ("single-session-user", "single-session-assistant", "multi-session"):
        return _STANDARD.format(question, answer, response)
    if task == "temporal-reasoning":
        return _TEMPORAL.format(question, answer, response)
    if task == "knowledge-update":
        return _KNOWLEDGE_UPDATE.format(question, answer, response)
    if task == "single-session-preference":
        return _PREFERENCE.format(question, answer, response)
    raise NotImplementedError(task)


async def _judgeOne(client: httpx.AsyncClient, key: str, model: str, prompt: str) -> str:
    body: Dict[str, Any] = {"model": model, "messages": [{"role": "user", "content": prompt}], "n": 1}
    if model.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4")):
        body["max_completion_tokens"] = 2000  # reasoning models reject temperature/max_tokens
    else:
        body["temperature"] = 0
        body["max_tokens"] = 10
    for attempt in range(6):
        try:
            r = await client.post("https://api.openai.com/v1/chat/completions",
                                  headers={"Authorization": f"Bearer {key}"}, json=body, timeout=120)
            r.raise_for_status()
            data = r.json()
            usage.record("bench.judge.official", model, data.get("usage"))
            return data["choices"][0]["message"]["content"] or ""
        except Exception:
            if attempt == 5:
                raise
            await asyncio.sleep(2 ** attempt)
    return ""


async def main() -> int:
    _loadDotEnv()
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--model", default=OFFICIAL_JUDGE_MODEL)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    key = os.environ["OPENAI_API_KEY"]
    data = json.load(open(args.results))
    ds = {x["question_id"].lower(): x for x in json.load(open(datasetPath()))}
    sem = asyncio.Semaphore(args.concurrency)

    async def one(client: httpx.AsyncClient, r: Dict[str, Any]) -> Dict[str, Any]:
        qid = r["questionId"]
        it = ds[qid]
        if "error" in r:
            return {"questionId": qid, "questionType": it["question_type"], "official": 0, "error": r["error"]}
        prompt = getAnscheckPrompt(it["question_type"], it["question"], it["answer"],
                                   r.get("generated") or "", abstention="_abs" in qid)
        async with sem:
            resp = await _judgeOne(client, key, args.model, prompt)
        return {"questionId": qid, "questionType": it["question_type"],
                "official": int("yes" in resp.lower()), "officialRaw": resp.strip(),
                "kormemJudge": r.get("score")}

    async with httpx.AsyncClient() as client:
        out = await asyncio.gather(*[one(client, r) for r in data["results"]])

    byType: Dict[str, list] = defaultdict(list)
    for o in out:
        byType[o["questionType"]].append(o["official"])
    absRows = [o["official"] for o in out if o["questionId"].endswith("_abs")]
    total = len(out)
    correct = sum(o["official"] for o in out)
    agree = sum(1 for o in out if o.get("kormemJudge") is not None and o["official"] == o["kormemJudge"])
    summary = {
        "source": args.results, "judgeModel": args.model, "total": total, "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "taskAveragedAccuracy": sum(sum(v) / len(v) for v in byType.values()) / len(byType) if byType else 0.0,
        "abstention": {"correct": sum(absRows), "total": len(absRows)},
        "byType": {t: {"correct": sum(v), "total": len(v), "accuracy": sum(v) / len(v)} for t, v in sorted(byType.items())},
        "agreementWithKormemJudge": agree / total if total else 0.0,
        "flips": {
            "kormemYes_officialNo": [o["questionId"] for o in out if o.get("kormemJudge") == 1 and o["official"] == 0],
            "kormemNo_officialYes": [o["questionId"] for o in out if o.get("kormemJudge") == 0 and o["official"] == 1],
        },
        "results": out,
    }
    outPath = args.output or str(Path(args.results).with_suffix("")) + f".official_{args.model}.json"
    Path(outPath).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("results",)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
