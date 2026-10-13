"""Text chunking."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter


class TextChunker:
    def __init__(
        self,
        chunkSize: int = 1000,
        chunkOverlap: int = 200,
        minChunkSize: int = 100,
    ):
        self.chunkSize = chunkSize
        self.chunkOverlap = chunkOverlap
        self.minChunkSize = minChunkSize
        self.markdownSplitter = MarkdownTextSplitter(
            chunk_size=chunkSize,
            chunk_overlap=chunkOverlap,
        )
        self.recursiveSplitter = RecursiveCharacterTextSplitter(
            chunk_size=chunkSize,
            chunk_overlap=chunkOverlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _isMarkdown(self, text: str) -> bool:
        return bool(re.search(r"^#{1,6}\s", text, re.MULTILINE)) or "**" in text

    async def chunkContent(
        self,
        textContent: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not textContent or not textContent.strip():
            return []
        isMd = self._isMarkdown(textContent)
        raw = (
            self.markdownSplitter.split_text(textContent)
            if isMd
            else self.recursiveSplitter.split_text(textContent)
        )
        chunks: List[Dict[str, Any]] = []
        pos = 0
        for i, chunkText in enumerate(raw):
            if len(chunkText.strip()) < self.minChunkSize and i < len(raw) - 1:
                continue
            end = pos + len(chunkText)
            chunks.append({
                "text": chunkText,
                "kind": "markdown" if isMd else "paragraph",
                "metadata": {
                    **(metadata or {}),
                    "startPos": pos,
                    "endPos": end,
                    "chunkIndex": i,
                    "isMarkdown": isMd,
                    "chunker": "text",
                },
            })
            pos = end
        return chunks
