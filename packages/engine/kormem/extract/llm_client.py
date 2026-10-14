"""Shared LLM completion helper for extraction."""

from __future__ import annotations

import os
from typing import Optional

import httpx

from kormem.core import usage
from kormem.core.config import KormemConfig, loadConfig


DEFAULT_OPENAI_MODEL = "gpt-6-luna"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


def defaultModel(config: KormemConfig) -> str:
    """Gemini when its key is set (one key runs everything), else OpenAI."""
    if os.environ.get("GEMINI_API_KEY") or config.geminiApiKey:
        return DEFAULT_GEMINI_MODEL
    if os.environ.get("OPENAI_API_KEY") or config.openaiApiKey:
        return DEFAULT_OPENAI_MODEL
    return DEFAULT_GEMINI_MODEL


def geminiThinking(effort: Optional[str]) -> dict:
    return {"thinkingConfig": {"thinkingLevel": effort}} if effort in ("minimal", "low", "medium", "high") else {}


async def completeLlm(
    prompt: str,
    systemPrompt: str,
    *,
    config: Optional[KormemConfig] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    maxTokens: int = 8000,
    reasoningEffort: Optional[str] = None,
    purpose: str = "llm",
    jsonMode: bool = False,
) -> str:
    """One completion.

    ``reasoningEffort`` overrides ``KORMEM_QUERY_REASONING_EFFORT`` for
    reasoning models (others ignore it). ``purpose`` labels the call in the
    usage log. ``jsonMode`` asks the API for a valid JSON object.
    """
    config = config or loadConfig()
    geminiKey = os.environ.get("GEMINI_API_KEY") or config.geminiApiKey
    openaiKey = os.environ.get("OPENAI_API_KEY") or config.openaiApiKey
    extractionModel = (
        model
        or os.environ.get("KORMEM_EXTRACTION_MODEL")
        or config.extractionModel
        or defaultModel(config)
    )

    if geminiKey and "gemini" in extractionModel.lower():
        return await _completeGemini(
            prompt, systemPrompt, geminiKey, extractionModel, temperature, maxTokens, purpose, jsonMode,
            reasoningEffort or os.environ.get("KORMEM_QUERY_REASONING_EFFORT"),
        )
    if openaiKey:
        return await _completeOpenAi(
            prompt, systemPrompt, openaiKey, extractionModel, temperature, maxTokens, reasoningEffort, purpose,
            jsonMode,
        )
    raise RuntimeError("No LLM API key configured (GEMINI_API_KEY or OPENAI_API_KEY)")


async def _completeOpenAi(
    prompt: str,
    systemPrompt: str,
    apiKey: str,
    model: str,
    temperature: float,
    maxTokens: int,
    reasoningEffort: Optional[str] = None,
    purpose: str = "llm",
    jsonMode: bool = False,
) -> str:
    # Reasoning / gpt-5+/gpt-6+ models reject `temperature != 1` and `max_tokens`.
    isReasoningModel = model.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4"))
    body: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": systemPrompt},
            {"role": "user", "content": prompt},
        ],
    }
    if isReasoningModel:
        body["max_completion_tokens"] = maxTokens
        reasoningEffort = reasoningEffort or os.environ.get("KORMEM_QUERY_REASONING_EFFORT")
        if reasoningEffort:
            body["reasoning_effort"] = reasoningEffort
    else:
        body["temperature"] = temperature
        body["max_tokens"] = maxTokens
    if jsonMode:
        body["response_format"] = {"type": "json_object"}
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {apiKey}"},
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()
        usage.record(purpose, model, data.get("usage"))
        return data["choices"][0]["message"]["content"]


async def _completeGemini(
    prompt: str,
    systemPrompt: str,
    apiKey: str,
    model: str,
    temperature: float,
    maxTokens: int,
    purpose: str = "llm",
    jsonMode: bool = False,
    reasoningEffort: Optional[str] = None,
) -> str:
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": apiKey},  # never in the URL: URLs end up in logs
            json={
                "systemInstruction": {"parts": [{"text": systemPrompt}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": maxTokens,
                    **({"responseMimeType": "application/json"} if jsonMode else {}),
                    **geminiThinking(reasoningEffort),
                },
            },
        )
        resp.raise_for_status()
        data = resp.json()
        meta = data.get("usageMetadata") or {}
        usage.record(purpose, model, {"input_tokens": meta.get("promptTokenCount", 0),
                                      "output_tokens": meta.get("candidatesTokenCount", 0),
                                      "output_tokens_details": {"reasoning_tokens": meta.get("thoughtsTokenCount", 0)}})
        parts = data["candidates"][0].get("content", {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts if not p.get("thought"))
