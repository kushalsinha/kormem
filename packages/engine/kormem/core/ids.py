"""Stable ID generation for kormem entities."""

from __future__ import annotations

import hashlib
import uuid
from typing import Optional


def generateSourceId() -> str:
    return str(uuid.uuid4())


def generateDocumentId() -> str:
    return str(uuid.uuid4())


def generateSegmentKey(
    namespaceId: str,
    documentId: str,
    idx: int,
    filePath: Optional[str] = None,
) -> str:
    if filePath:
        pathHash = hashlib.md5(filePath.encode()).hexdigest()[:8]
        return f"seg:{namespaceId}:{documentId}:{pathHash}:{idx}"
    return f"seg:{namespaceId}:{documentId}:{idx}"


def contentHash(content: str) -> str:
    return hashlib.md5(content.encode("utf-8")).hexdigest()
