"""LongMemEval LLM judge."""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

from kormem.core import usage

from benchmark.config import KormemBenchmarkConfig

_JUDGE_DEFAULT = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no.\n\n"
    "Question: {question}\nCorrect Answer: {expected}\nModel Response: {generated}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_PROMPT_MAP = {
    "temporal-reasoning": _JUDGE_DEFAULT,
    "knowledge-update": _JUDGE_DEFAULT,
    "single-session-preference": (
        "I will give you a question, a rubric describing what a good personalized response "
        "should contain, and a response from a model. The rubric lists ideal preferences; "
        "answer YES if the response reflects the SPIRIT of the rubric — i.e. it addresses "
        "the user's situation with relevant personalized advice drawn from their stated "
        "preferences, routines, or past context — even if it doesn't cover every detail in "
        "the rubric or phrases things differently. Answer NO only if the response is "
        "generic / unrelated / clearly contradicts the user's stated preferences, OR if the "
        "model refused to answer (e.g. 'I don't know') when a reasonable personalized "
        "response was possible.\n\n"
        "Question: {question}\nRubric: {expected}\nModel Response: {generated}\n\n"
        "Does the response substantially address the user's preferences? Answer yes or no only."
    ),
    "absent": (
        "I will give you an unanswerable question, an explanation, and a response from a model. "
        "Please answer yes if the model correctly identifies the question as unanswerable.\n\n"
        "Question: {question}\nExplanation: {expected}\nModel Response: {generated}\n\n"
        "Does the model correctly identify the question as unanswerable? Answer yes or no only."
    ),
}


@dataclass
class JudgeResult:
    correct: bool
    reason: str
    rawResponse: str = ""


class LlmJudge:
    def __init__(self, config: KormemBenchmarkConfig):
        self.config = config
        self.apiKey = config.openaiApiKey or os.environ.get("OPENAI_API_KEY")

    async def judge(
        self,
        question: str,
        expected: str,
        generated: str,
        questionType: str,
    ) -> JudgeResult:
        template = _PROMPT_MAP.get(questionType, _JUDGE_DEFAULT)
        prompt = template.format(question=question, expected=expected, generated=generated)
        if not self.apiKey:
            ok = expected.lower() in generated.lower()
            return JudgeResult(correct=ok, reason="keyword_fallback", rawResponse="")
        model = self.config.judgeModel
        ml = model.lower()
        if ml.startswith("grok") or ml.startswith("xai"):
            url = "https://api.x.ai/v1/chat/completions"
            apiKey = os.environ.get("XAI_API_KEY") or self.apiKey
        else:
            url = "https://api.openai.com/v1/chat/completions"
            apiKey = self.apiKey
        async with httpx.AsyncClient(timeout=120.0) as client:
            body: dict = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": 2000,
            }
            if self.config.judgeReasoningEffort:
                body["reasoning_effort"] = self.config.judgeReasoningEffort
            resp = await client.post(
                url,
                headers={"Authorization": f"Bearer {apiKey}"},
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()
            usage.record("bench.judge", body.get("model", ""), data.get("usage"))
            raw = data["choices"][0]["message"]["content"].strip().lower()
        correct = raw.startswith("yes")
        return JudgeResult(correct=correct, reason=raw, rawResponse=raw)
