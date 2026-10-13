"""Kormem LongMemEval benchmark configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class KormemBenchmarkConfig:
    judgeModel: str = "gpt-6-luna"
    judgeReasoningEffort: str | None = "medium"
    answerModel: str = "gpt-6.1-sol"
    answerReasoningEffort: str | None = "medium"
    openaiApiKey: str | None = None


def loadBenchmarkConfig() -> KormemBenchmarkConfig:
    return KormemBenchmarkConfig(
        judgeModel=os.environ.get("KORMEM_JUDGE_MODEL", "gpt-6-luna"),
        judgeReasoningEffort=os.environ.get("KORMEM_JUDGE_REASONING_EFFORT", "medium") or None,
        answerModel=os.environ.get("KORMEM_ANSWER_MODEL", "gpt-6.1-sol"),
        answerReasoningEffort=os.environ.get("KORMEM_ANSWER_REASONING_EFFORT", "medium") or None,
        openaiApiKey=os.environ.get("OPENAI_API_KEY"),
    )
