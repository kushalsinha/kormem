"""Base connector interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SourceFile:
    path: str
    name: str
    mimeType: Optional[str] = None
    sizeBytes: Optional[int] = None


class BaseConnector(ABC):
    sourceType: str = "unknown"

    @abstractmethod
    async def fetchContent(self, target: str) -> str:
        pass

    @abstractmethod
    async def listFiles(self, target: str) -> List[SourceFile]:
        pass
