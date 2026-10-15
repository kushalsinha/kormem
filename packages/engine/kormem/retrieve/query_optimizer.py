"""Query optimization: rewrite, HyDE, decomposition."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from kormem.core.config import KormemConfig, loadConfig
from kormem.core.models import llmModel
from kormem.extract.llm_client import completeLlm

_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "what", "how", "when", "where",
    "who", "why", "do", "does", "did", "my", "me", "i", "you", "your", "of", "in",
    "on", "at", "to", "for", "and", "or", "with", "about",
}
_RECOMMENDATION_WORDS = {"recommend", "suggest", "best", "should", "prefer", "gift", "idea"}
_PERSONAL_WORDS = {"my", "me", "i", "mine", "our", "we"}
_COUNTING_PHRASES = (
    "how many", "how often", "how much", "how long",
    "total number", "number of times", "how frequently",
    "count of", "how regularly",
)
_PREFERENCE_SINGLE_WORDS = frozenset([
    "suggest", "recommend", "recommendation", "advice",
])
_PREFERENCE_MULTI_PHRASES = (
    "what should", "should i", "tips for", "gift for", "present for",
    "something to watch", "something to read", "something to listen",
    "to try", "options for", "ideas for",
)
_PERSONAL_MEMORY_PRONOUNS = re.compile(
    r"\b(my|i |i'm|i've|i'd|me\b|mine\b|myself\b)", re.IGNORECASE,
)
_REASONING_WORDS = {"compare", "explain", "analyze", "why", "difference", "versus", "vs"}
_MULTI_INTENT = re.compile(
    r"\b(and also|as well as|compare .+ and|in addition|furthermore)\b", re.I,
)


class QueryComplexity(str, Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"


@dataclass
class QueryOptimizationResult:
    originalQuery: str
    queryVariants: List[str]
    complexity: QueryComplexity
    hydeDocument: Optional[str] = None
    subQuestions: List[str] = field(default_factory=list)
    optimizationTimeMs: float = 0.0


class QueryOptimizer:
    def __init__(self, config: Optional[KormemConfig] = None):
        self.config = config or loadConfig()
        self.maxVariants = 3
        self.simpleQueryMaxWords = 15
        self.timeoutSeconds = 8.0

    def _hasLlm(self) -> bool:
        return bool(
            os.environ.get("OPENAI_API_KEY")
            or self.config.openaiApiKey
            or os.environ.get("GEMINI_API_KEY")
            or self.config.geminiApiKey
        )

    def _queryModel(self) -> str:
        return llmModel("recall", self.config)[0]

    def _isPersonalMemory(self, query: str) -> bool:
        return bool(_PERSONAL_MEMORY_PRONOUNS.search(query))

    def classifyQuery(self, query: str) -> QueryComplexity:
        q = query.strip()
        queryLower = q.lower()
        wordCount = len(q.split())
        if wordCount > self.simpleQueryMaxWords:
            return QueryComplexity.COMPLEX
        if _MULTI_INTENT.search(q) or q.count("?") > 1:
            return QueryComplexity.COMPLEX
        reasoningPatterns = [
            r"\b(explain|describe|elaborate|detail)\b.+\b(how|why|what)\b",
            r"\b(step by step|in detail|thoroughly)\b",
            r"\b(relationship between|connection between)\b",
        ]
        for pattern in reasoningPatterns:
            if re.search(pattern, queryLower):
                return QueryComplexity.COMPLEX
        codeComplexPatterns = [
            r"\b(implement|create|build|develop)\b.+\b(with|using|that)\b",
            r"\b(architecture|design pattern|best practice)\b",
            r"\b(integrate|connect|combine)\b.+\b(with|and)\b",
        ]
        for pattern in codeComplexPatterns:
            if re.search(pattern, queryLower):
                return QueryComplexity.COMPLEX
        if self._isPersonalMemory(q):
            if any(re.search(r"\b" + re.escape(p) + r"\b", queryLower) for p in _COUNTING_PHRASES):
                return QueryComplexity.COMPLEX
            if any(w in queryLower for w in _PREFERENCE_SINGLE_WORDS):
                return QueryComplexity.COMPLEX
            if any(re.search(r"\b" + re.escape(p) + r"\b", queryLower) for p in _PREFERENCE_MULTI_PHRASES):
                return QueryComplexity.COMPLEX
        if any(w in queryLower for w in _REASONING_WORDS):
            return QueryComplexity.COMPLEX
        return QueryComplexity.SIMPLE

    def _extractKeywords(self, query: str) -> str:
        words = [w for w in re.findall(r"\w+", query.lower()) if w not in _STOP_WORDS]
        return " ".join(words[:12])

    def _questionToStatement(self, query: str) -> str:
        q = query.strip().rstrip("?")
        for prefix in ("what is ", "what are ", "how does ", "how do ", "who is ", "where is "):
            if q.lower().startswith(prefix):
                return q[len(prefix):].strip()
        return q

    def _expandQuery(self, query: str) -> List[str]:
        variants = [query]
        kw = self._extractKeywords(query)
        if kw and kw != query.lower():
            variants.append(kw)
        stmt = self._questionToStatement(query)
        if stmt and stmt not in variants:
            variants.append(stmt)
        lower = query.lower()
        if any(w in lower for w in _RECOMMENDATION_WORDS):
            variants.append(f"user preferences interests {kw or query}")
        if self._isPersonalMemory(query) and any(w in lower for w in _RECOMMENDATION_WORDS):
            variants.append(f"user likes prefers wants favorites {kw or query}")
        return variants[:4]

    async def rewriteQuery(self, query: str, context: Optional[str] = None) -> str:
        systemPrompt = """You are a search query optimizer. Your job is to rewrite user queries to improve retrieval from a knowledge base.

RULES:
1. Keep the core intent intact
2. Expand abbreviations and acronyms
3. Add synonyms for key technical terms in parentheses if helpful
4. Normalize casual language to technical terminology
5. Keep the rewritten query concise (under 50 words)
6. Do NOT add information not implied by the original query
7. Output ONLY the rewritten query, nothing else

Examples:
- "how to use py for ml?" → "how to use Python for machine learning (ML, deep learning, neural networks)"
- "k8s deployment" → "Kubernetes deployment configuration (k8s, container orchestration)"
- "fix npm error" → "troubleshoot npm error (Node.js package manager, dependency issues)"
- "react hooks" → "React hooks (useState, useEffect, custom hooks, functional components)\""""
        userPrompt = f"Rewrite this search query:\n\n{query}"
        if context:
            userPrompt += f"\n\nConversation context: {context}"
        try:
            rewritten = await asyncio.wait_for(
                completeLlm(
                    userPrompt, systemPrompt,
                    config=self.config, model=self._queryModel(), temperature=0,
                ),
                timeout=self.timeoutSeconds,
            )
            rewritten = rewritten.strip().strip('"').strip("'")
            return rewritten if rewritten and len(rewritten) >= 3 else query
        except Exception:
            return query

    async def generateHyde(self, query: str, sourceType: str = "auto") -> Optional[str]:
        if sourceType == "auto":
            codeIndicators = [
                "function", "method", "class", "implement", "code", "api",
                "error", "bug", "fix", "debug", "syntax", "import", "module",
            ]
            queryLower = query.lower()
            if any(ind in queryLower for ind in codeIndicators):
                sourceType = "code"
            elif self._isPersonalMemory(query):
                sourceType = "personal_memory"
            else:
                sourceType = "documentation"

        if sourceType == "personal_memory":
            systemPrompt = """You are generating a hypothetical chat-log excerpt for search retrieval.

TASK: Write a short conversation in [user]: / [assistant]: format where the user naturally reveals the answer to the given question.

REQUIREMENTS:
1. Always produce exactly this two-line format:
   [user]: <user statement that naturally reveals the answer as a personal fact>
   [assistant]: <brief assistant acknowledgement that paraphrases the user's fact>
2. The [user]: line must include the specific personal detail that answers the question.
3. Use first-person language in the [user]: line ("I", "my", "me").
4. Keep each line short: under 40 words.
5. Do NOT add any text outside the two lines above.
6. Do NOT mention this is hypothetical or that you are generating text.

EXAMPLE — Question: "What phone does the user have?"
[user]: I just got the iPhone 13 Pro last month and I love the camera.
[assistant]: That's great! The iPhone 13 Pro has an excellent camera system."""
        elif sourceType == "code":
            systemPrompt = """You are an expert technical writer creating search-optimized code documentation.

TASK: Generate a hypothetical code documentation snippet that would perfectly answer the given question.

REQUIREMENTS:
1. Write as if this is from official documentation or a well-maintained codebase
2. Include realistic function/method signatures if applicable
3. Include brief code examples (2-4 lines max)
4. Use proper technical terminology
5. Do NOT mention this is hypothetical
6. Do NOT add disclaimers or caveats
7. Keep it concise: 2-3 short paragraphs
8. Include relevant import statements if applicable

FORMAT:
Write a documentation snippet that a developer would find when searching for this topic."""
        else:
            systemPrompt = """You are an expert technical writer creating search-optimized documentation.

TASK: Generate a hypothetical document that would perfectly answer the given question.

REQUIREMENTS:
1. Write as if this is from an authoritative technical knowledge base
2. Be factual, specific, and detailed
3. Use proper technical terminology
4. Include concrete examples or specifications where relevant
5. Do NOT mention this is hypothetical
6. Do NOT add opinions or disclaimers
7. Keep it concise: 2-3 focused paragraphs
8. Structure information clearly with key points

FORMAT:
Write a knowledge base article excerpt that directly answers the question."""

        userPrompt = f"Question: {query}\n\nGenerate the hypothetical document:"
        try:
            hydeDoc = await asyncio.wait_for(
                completeLlm(
                    userPrompt, systemPrompt,
                    config=self.config, model=self._queryModel(), temperature=0,
                ),
                timeout=self.timeoutSeconds,
            )
            hydeDoc = hydeDoc.strip()
            return hydeDoc if hydeDoc and len(hydeDoc) >= 20 else None
        except Exception:
            return None

    async def decomposeQuery(self, query: str) -> List[str]:
        systemPrompt = """You are a query decomposition expert. Break down complex questions into simpler sub-questions.

RULES:
1. Split compound questions into 2-3 atomic sub-questions
2. Each sub-question should be searchable independently
3. Preserve the technical terms from the original
4. Do NOT add questions that weren't implied
5. If the query is already simple, return just that query
6. Output as a JSON array of strings

Examples:
Input: "How do I set up authentication and implement rate limiting in FastAPI?"
Output: ["How to set up authentication in FastAPI?", "How to implement rate limiting in FastAPI?"]

Input: "What is the difference between REST and GraphQL, and when should I use each?"
Output: ["What is REST API?", "What is GraphQL?", "When to use REST vs GraphQL?"]

Input: "How to deploy to AWS?"
Output: ["How to deploy to AWS?"]"""
        userPrompt = f"Decompose this query into sub-questions:\n\n{query}"
        try:
            raw = await asyncio.wait_for(
                completeLlm(
                    userPrompt, systemPrompt,
                    config=self.config, model=self._queryModel(), temperature=0,
                ),
                timeout=self.timeoutSeconds,
            )
            data = json.loads(raw)
            if isinstance(data, list):
                subQuestions = data
            elif isinstance(data, dict):
                subQuestions = data.get("questions", data.get("sub_questions", [query]))
            else:
                subQuestions = [query]
            subQuestions = [q.strip() for q in subQuestions if isinstance(q, str) and q.strip()]
            return subQuestions[:3] if subQuestions else [query]
        except Exception:
            return [query]

    async def optimize(
        self,
        query: str,
        context: Optional[str] = None,
        forceComplex: bool = False,
    ) -> QueryOptimizationResult:
        startTime = time.time()
        complexity = QueryComplexity.COMPLEX if forceComplex else self.classifyQuery(query)
        isPersonal = self._isPersonalMemory(query)

        if not self._hasLlm() or not self.config.queryOptimize:
            return QueryOptimizationResult(
                originalQuery=query,
                queryVariants=self._expandQuery(query)[: self.maxVariants],
                complexity=complexity,
                optimizationTimeMs=(time.time() - startTime) * 1000,
            )

        rewritten = await self.rewriteQuery(query, context)
        queryVariants = [rewritten]
        hydeDocument: Optional[str] = None
        subQuestions: List[str] = []

        if complexity == QueryComplexity.COMPLEX:
            hydeRes, subRes = await asyncio.gather(
                self.generateHyde(rewritten, sourceType="auto"),
                self.decomposeQuery(query),
                return_exceptions=True,
            )
            if isinstance(hydeRes, str):
                hydeDocument = hydeRes
            if isinstance(subRes, list):
                subQuestions = subRes
            if hydeDocument:
                queryVariants.append(hydeDocument)
            for sq in subQuestions:
                if sq.lower().strip() != rewritten.lower().strip() and sq not in queryVariants:
                    queryVariants.append(sq)
                    if len(queryVariants) >= self.maxVariants:
                        break
        elif isPersonal:
            hydeDocument = await self.generateHyde(rewritten, sourceType="personal_memory")
            if hydeDocument:
                queryVariants.append(hydeDocument)

        if not queryVariants:
            queryVariants = [query]

        return QueryOptimizationResult(
            originalQuery=query,
            queryVariants=queryVariants[: self.maxVariants],
            complexity=complexity,
            hydeDocument=hydeDocument,
            subQuestions=subQuestions,
            optimizationTimeMs=(time.time() - startTime) * 1000,
        )
