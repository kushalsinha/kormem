"""Obsidian-compatible note model with YAML frontmatter."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List, Optional

import yaml

WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")

PARA_FOLDERS = {
    "projects": "1-Projects",
    "areas": "2-Areas",
    "resources": "3-Resources",
    "archive": "4-Archive",
    "meta": "_meta",
    "source": "_sources",
}


@dataclass
class NoteMeta:
    kormemId: str
    title: str
    para: str
    relPath: str
    tags: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    updatedAt: str = ""


@dataclass
class Note:
    kormemId: str
    title: str
    para: str
    body: str
    tags: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    created: str = ""
    updated: str = ""
    folder: str = ""            # vault-relative directory, e.g. "2-Areas/Health"; "" = PARA root
    aliases: List[str] = field(default_factory=list)
    archivedFrom: str = ""      # folder a cold note was archived from; it moves back when it gets new knowledge

    @classmethod
    def create(cls, title: str, para: str = "resources", body: str = "", folder: str = "") -> "Note":
        now = datetime.now(timezone.utc).isoformat()
        return cls(
            kormemId=str(uuid.uuid4()),
            title=title,
            para=para,
            body=body,
            created=now,
            updated=now,
            folder=folder,
        )

    @classmethod
    def parse(cls, raw: str) -> "Note":
        raw = raw.strip()
        if not raw.startswith("---"):
            return cls(kormemId=str(uuid.uuid4()), title="Untitled", para="resources", body=raw)
        parts = raw.split("---", 2)
        if len(parts) < 3:
            return cls(kormemId=str(uuid.uuid4()), title="Untitled", para="resources", body=raw)
        fm: dict[str, Any] = yaml.safe_load(parts[1]) or {}
        body = parts[2].lstrip("\n")
        return cls(
            kormemId=str(fm.get("kormemId") or uuid.uuid4()),
            title=str(fm.get("title") or "Untitled"),
            para=str(fm.get("para") or "resources"),
            body=body,
            tags=list(fm.get("tags") or []),
            sources=[str(s) for s in (fm.get("sources") or [])],
            created=str(fm.get("created") or ""),
            updated=str(fm.get("updated") or ""),
            folder=str(fm.get("folder") or ""),
            aliases=[str(a) for a in (fm.get("aliases") or [])],
            archivedFrom=str(fm.get("archivedFrom") or ""),
        )

    def serialize(self) -> str:
        fm = {
            "kormemId": self.kormemId,
            "title": self.title,
            "para": self.para,
            "tags": self.tags,
            "sources": self.sources,
            "created": self.created,
            "updated": self.updated,
        }
        if self.folder:
            fm["folder"] = self.folder
        if self.aliases:
            fm["aliases"] = self.aliases
        if self.archivedFrom:
            fm["archivedFrom"] = self.archivedFrom
        header = yaml.safe_dump(fm, default_flow_style=False, sort_keys=False).strip()
        return f"---\n{header}\n---\n\n{self.body.rstrip()}\n"

    def outboundLinks(self) -> List[str]:
        return WIKILINK_RE.findall(self.body)

    def toMeta(self, relPath: str) -> NoteMeta:
        return NoteMeta(
            kormemId=self.kormemId,
            title=self.title,
            para=self.para,
            relPath=relPath,
            tags=self.tags,
            sources=self.sources,
            updatedAt=self.updated,
        )
