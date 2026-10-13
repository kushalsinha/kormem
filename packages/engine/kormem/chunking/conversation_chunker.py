"""Conversation turn chunking."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from kormem.chunking.text_chunker import TextChunker


def isConversationText(text: str) -> bool:
    if not text:
        return False
    patterns = [
        r"\[user\]",
        r"\[assistant\]",
        r"^User:",
        r"^Assistant:",
        r"^Human:",
        r"^AI:",
    ]
    for p in patterns:
        if re.search(p, text, re.IGNORECASE | re.MULTILINE):
            return True
    return False


class ConversationChunker:
    def __init__(self, maxChars: int = 500, overlapTurns: int = 1):
        self.maxChars = maxChars
        self.overlapTurns = overlapTurns
        self.fallback = TextChunker()

    def _splitTurns(self, text: str) -> List[str]:
        parts = re.split(
            r"(?=\[(?:user|assistant)\]|\b(?:User|Assistant|Human|AI):)",
            text,
            flags=re.IGNORECASE,
        )
        return [p.strip() for p in parts if p.strip()]

    async def chunkContent(
        self,
        textContent: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not isConversationText(textContent):
            return await self.fallback.chunkContent(textContent, metadata)
        turns = self._splitTurns(textContent)
        if not turns:
            return await self.fallback.chunkContent(textContent, metadata)
        chunks: List[Dict[str, Any]] = []
        window: List[str] = []
        windowLen = 0
        idx = 0
        for turn in turns:
            if windowLen + len(turn) > self.maxChars and window:
                chunks.append({
                    "text": "\n".join(window),
                    "kind": "conversation",
                    "metadata": {**(metadata or {}), "chunkIndex": idx, "chunker": "conversation"},
                })
                idx += 1
                if self.overlapTurns and len(window) > self.overlapTurns:
                    window = window[-self.overlapTurns :]
                    windowLen = sum(len(t) for t in window)
                else:
                    window = []
                    windowLen = 0
            window.append(turn)
            windowLen += len(turn)
        if window:
            chunks.append({
                "text": "\n".join(window),
                "kind": "conversation",
                "metadata": {**(metadata or {}), "chunkIndex": idx, "chunker": "conversation"},
            })
        return chunks
