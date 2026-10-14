"""Recursive recall agent: walks the vault and the raw sources to find the
knowledge a question needs.

It starts from a free first pass (sources shortlisted by the cross-encoder,
plus the best-matching topic notes) and decides what to open next from what
it has read. Notes are the map: a topic note's Timeline lists every dated
instance with a ``[[source]]`` link, so counting, ordering and "latest value"
questions become "read the note, then verify the linked sources".

Tools:

    search(query)              chunks, notes and memory lines in one call
    read_note(title)           a topic or source note
    note_links(title)          outbound links and backlinks
    list_folder(path)          notes in a vault folder ("" = the folder tree)
    list_sources()             every source: id, date, kind, one-line preview
    read_source(source_id)     a source's full text, relative dates resolved
    finish(sources, findings)  the sources the answer needs + dated findings

The agent does not answer. Its findings state what the evidence supports,
quoting it, so the answer model is not talked out of a supported answer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx

from kormem.core import usage
from kormem.core.config import KormemConfig, loadConfig
from kormem.extract.llm_client import defaultModel, geminiThinking

logger = logging.getLogger(__name__)

EFFORT = {
    # effort: (reasoning effort, tool rounds)
    "low": ("low", 4),
    "medium": ("medium", 8),
}


@dataclass
class RecallResult:
    sources: List[str]
    findings: List[str]
    rounds: int
    toolCalls: List[Dict[str, Any]] = field(default_factory=list)
    notesRead: List[str] = field(default_factory=list)
    sourcesRead: List[str] = field(default_factory=list)


def _fn(name: str, description: str, properties: Optional[dict] = None, required: Optional[list] = None) -> dict:
    return {"type": "function", "name": name, "description": description,
            "parameters": {"type": "object", "properties": properties or {}, "required": required or []}}


TOOLS = [
    _fn("search", "Search everything at once: raw source chunks (with source ids), topic notes and "
        "memory lines. Use several wordings to find instances the first pass missed.",
        {"query": {"type": "string"}}, ["query"]),
    _fn("read_note", "Read a vault note by title. Topic notes have Current facts, a dated Timeline, "
        "How-to, From the assistant and Related sections; every bullet links its [[source]].",
        {"title": {"type": "string"}}, ["title"]),
    _fn("note_links", "Outbound links and backlinks of a note: related topics and the sources that mention it.",
        {"title": {"type": "string"}}, ["title"]),
    _fn("list_folder", "List the notes in a vault folder, or the folder tree when path is empty.",
        {"path": {"type": "string"}}),
    _fn("list_sources", "List every source (chats, documents, remembered facts) with id, date, kind and a preview."),
    _fn("read_source", "Read the parts of a source that match a query (about 4k characters), or the whole "
        "source with full=true. Relative dates in user turns are resolved in brackets.",
        {"source_id": {"type": "string", "description": "e.g. S12"},
         "query": {"type": "string", "description": "What to look for in this source"},
         "full": {"type": "boolean", "description": "Whole source; only when you need all of it (e.g. a full list)"}},
        ["source_id"]),
    _fn("finish", "Return the sources the answer needs and the findings. Call exactly once, last.", {
        "sources": {"type": "array", "items": {"type": "string"},
                    "description": "Every source id holding evidence for the answer."},
        "findings": {"type": "array", "items": {"type": "string"}, "description": (
            "One line per relevant fact or instance: what it is, its absolute date, its source id, and a "
            "short quote. Tag [counts], [planned/not done], [suggested by assistant], [superseded by Sx] or "
            "[outside time window] where it applies. State what the evidence supports; mark anything "
            "inferred as (inferred).")},
    }, ["sources", "findings"]),
]

SYSTEM_PROMPT = """You are a recall agent for a personal memory vault. You find the evidence another model needs to answer a question about the user's history. You do not answer the question.

The vault is Obsidian-style: topic notes (in PARA folders) summarise knowledge, with dated Timeline entries and [[links]] to the sources they came from; sources are the raw chats and documents.

How to work:
- Notes first. The briefing shows the best-matching notes' facts and timelines with source ids; their bullets usually list every instance. Read a note's full text or follow its links when you need more.
- Then confirm in the sources: read_source with a query returns just the matching part. Ask for full=true only when you need a whole conversation (an assistant's complete list, for example).
- Counting, totals, lists, ordering, "Nth time": find EVERY instance. Use the note's Timeline, backlinks and several search wordings; one missed instance makes the answer wrong.
- Current / latest / previous value: find where the value was set and every later change; the most recent source wins.
- Time between events, "how long ago", "when": record each event's absolute date. A relative phrase is relative to its own source's date; resolved dates appear in [brackets].
- Assistant-provided details ("the third item you listed"): open the source and copy the exact item.
- Preferences / recommendations: find the user's relevant tastes, constraints and past experiences.
- Report what the evidence supports, with a short quote. If something is likely but not stated outright, say so and mark it (inferred); do not turn a well-supported answer into doubt.
- If the history never mentions the subject, say so in one finding.
- Stop when the evidence is complete. You may call several tools in one turn. Finish by calling finish exactly once."""


ToolFn = Callable[..., Awaitable[str]]


async def _respond(model: str, apiKey: str, input: List[Dict[str, Any]], previousId: Optional[str],
                   effort: str, toolChoice: Any = "auto") -> Dict[str, Any]:
    body: Dict[str, Any] = {"model": model, "input": input, "tools": TOOLS, "tool_choice": toolChoice,
                            "max_output_tokens": 16000, "reasoning": {"effort": effort}}
    if previousId:
        body["previous_response_id"] = previousId
    for attempt in range(4):
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post("https://api.openai.com/v1/responses",
                                         headers={"Authorization": f"Bearer {apiKey}"}, json=body)
                if resp.status_code == 400:
                    raise ValueError(f"recall agent request rejected: {resp.text[:300]}")
                resp.raise_for_status()
                data = resp.json()
                usage.record("recall.agent", model, data.get("usage"))
                return data
        except ValueError:
            raise
        except Exception as e:
            if attempt == 3:
                raise
            logger.warning("recall agent call failed (%s); retrying", str(e)[:120])
            await asyncio.sleep(2 ** (attempt + 1))
    raise RuntimeError("unreachable")


def _geminiTools() -> List[Dict[str, Any]]:
    decls = []
    for t in TOOLS:
        d: Dict[str, Any] = {"name": t["name"], "description": t["description"]}
        if t["parameters"]["properties"]:  # Gemini rejects an object schema with no properties
            d["parameters"] = t["parameters"]
        decls.append(d)
    return [{"functionDeclarations": decls}]


async def _respondGemini(model: str, apiKey: str, contents: List[Dict[str, Any]], effort: str,
                         final: bool) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
        "tools": _geminiTools(),
        "toolConfig": {"functionCallingConfig": (
            {"mode": "ANY", "allowedFunctionNames": ["finish"]} if final else {"mode": "AUTO"})},
        "generationConfig": {"maxOutputTokens": 16000, **geminiThinking(effort)},
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    for attempt in range(4):
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(url, headers={"x-goog-api-key": apiKey}, json=body)
                if resp.status_code == 400:
                    raise ValueError(f"recall agent request rejected: {resp.text[:300]}")
                resp.raise_for_status()
                data = resp.json()
                meta = data.get("usageMetadata") or {}
                usage.record("recall.agent", model, {
                    "input_tokens": meta.get("promptTokenCount", 0),
                    "output_tokens": meta.get("candidatesTokenCount", 0),
                    "output_tokens_details": {"reasoning_tokens": meta.get("thoughtsTokenCount", 0)}})
                return data
        except ValueError:
            raise
        except Exception as e:
            if attempt == 3:
                raise
            logger.warning("recall agent call failed (%s); retrying", str(e)[:120])
            await asyncio.sleep(2 ** (attempt + 1))
    raise RuntimeError("unreachable")


class _OpenAiTurns:
    """Conversation state on OpenAI's Responses API (the server keeps the history)."""

    def __init__(self, model: str, apiKey: str, question: str, briefing: str):
        self.model, self.apiKey = model, apiKey
        self.previousId: Optional[str] = None
        self.pending: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\n{briefing}"},
        ]

    def say(self, text: str) -> None:
        self.pending.append({"role": "user", "content": text})

    async def step(self, effort: str, final: bool) -> List[Dict[str, Any]]:
        resp = await _respond(self.model, self.apiKey, self.pending, self.previousId, effort,
                              {"type": "function", "name": "finish"} if final else "auto")
        self.previousId, self.pending = resp["id"], []
        calls = []
        for o in resp.get("output", []):
            if o.get("type") == "function_call":
                try:
                    args = json.loads(o.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                calls.append({"name": o["name"], "args": args, "id": o["call_id"]})
        return calls

    def results(self, done: List[tuple]) -> None:
        for call, result in done:
            self.pending.append({"type": "function_call_output", "call_id": call["id"], "output": result})


class _GeminiTurns:
    """Conversation state for Gemini: the full history is resent, model turns verbatim
    (their thought signatures must come back unchanged)."""

    def __init__(self, model: str, apiKey: str, question: str, briefing: str):
        self.model, self.apiKey = model, apiKey
        self.contents: List[Dict[str, Any]] = [
            {"role": "user", "parts": [{"text": f"Question: {question}\n\n{briefing}"}]}]
        self.userParts: List[Dict[str, Any]] = []

    def say(self, text: str) -> None:
        self.userParts.append({"text": text})

    async def step(self, effort: str, final: bool) -> List[Dict[str, Any]]:
        if self.userParts:
            self.contents.append({"role": "user", "parts": self.userParts})
            self.userParts = []
        resp = await _respondGemini(self.model, self.apiKey, self.contents, effort, final)
        content = ((resp.get("candidates") or [{}])[0].get("content") or {})
        parts = content.get("parts") or []
        self.contents.append({"role": "model", "parts": parts})
        return [{"name": p["functionCall"]["name"], "args": p["functionCall"].get("args") or {},
                 "id": p["functionCall"].get("id")} for p in parts if "functionCall" in p]

    def results(self, done: List[tuple]) -> None:
        for call, result in done:
            fr: Dict[str, Any] = {"name": call["name"], "response": {"result": result}}
            if call.get("id"):
                fr["id"] = call["id"]
            self.userParts.append({"functionResponse": fr})


async def recursiveRecall(
    question: str,
    briefing: str,
    tools: Dict[str, ToolFn],
    validSources: set,
    *,
    effort: str = "low",
    model: Optional[str] = None,
    config: Optional[KormemConfig] = None,
) -> RecallResult:
    """Run the agent until it calls ``finish`` or its round budget runs out.

    ``briefing`` is the first-pass context (question date, shortlisted
    sources and notes); ``tools`` maps tool names to async implementations.
    """
    config = config or loadConfig()
    reasoning, maxRounds = EFFORT.get(effort, EFFORT["low"])
    model = model or os.environ.get("KORMEM_RECALL_MODEL") or config.queryModel or defaultModel(config)
    if "gemini" in model.lower():
        apiKey = os.environ.get("GEMINI_API_KEY") or config.geminiApiKey or ""
        turns: Any = _GeminiTurns(model, apiKey, question, briefing)
    else:
        apiKey = os.environ.get("OPENAI_API_KEY") or config.openaiApiKey or ""
        turns = _OpenAiTurns(model, apiKey, question, briefing)
    if not apiKey:
        raise RuntimeError("the recall agent needs GEMINI_API_KEY (or OPENAI_API_KEY)")

    calls: List[Dict[str, Any]] = []
    notesRead: List[str] = []
    sourcesRead: List[str] = []

    for roundIdx in range(maxRounds + 1):
        last = roundIdx == maxRounds
        if last:
            turns.say("Tool budget used up. Call finish now with what you have.")
        fnCalls = await turns.step(reasoning, last)
        if not fnCalls:
            turns.say("Call finish to return your result.")
            continue
        parsed = []
        for call in fnCalls:
            args = call["args"] if isinstance(call["args"], dict) else {}
            if call["name"] == "finish":
                srcs = [str(s).strip().strip("[]") for s in args.get("sources", [])]
                return RecallResult(
                    sources=[s for s in dict.fromkeys(srcs) if s in validSources],
                    findings=[str(f).strip() for f in args.get("findings", []) if str(f).strip()],
                    rounds=roundIdx + 1, toolCalls=calls, notesRead=notesRead, sourcesRead=sourcesRead,
                )
            parsed.append((call, args))

        async def run(name: str, args: Dict[str, Any]) -> str:
            fn = tools.get(name)
            try:
                return await fn(**args) if fn else f"(unknown tool {name})"
            except TypeError as e:
                return f"(bad arguments for {name}: {e})"

        # Tool calls from one turn are independent reads: run them together.
        results = await asyncio.gather(*[run(c["name"], a) for c, a in parsed])
        for (call, args), result in zip(parsed, results):
            name = call["name"]
            if name == "read_note":
                notesRead.append(str(args.get("title")))
            elif name == "read_source":
                sourcesRead.append(str(args.get("source_id")))
            calls.append({"name": name, "args": args, "resultChars": len(result)})
        turns.results([(call, result) for (call, _), result in zip(parsed, results)])

    return RecallResult(sources=[s for s in sourcesRead if s in validSources], findings=[],
                        rounds=maxRounds, toolCalls=calls, notesRead=notesRead, sourcesRead=sourcesRead)
