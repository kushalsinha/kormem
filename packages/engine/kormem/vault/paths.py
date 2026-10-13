"""Vault path helpers."""

from __future__ import annotations

import re
from pathlib import Path

from kormem.core.config import getKormemHome
from kormem.vault.note import Note, PARA_FOLDERS


def vaultDir(namespaceId: str, home: Path | None = None) -> Path:
    return (home or getKormemHome()) / "vault" / namespaceId


def paraFolder(para: str) -> str:
    return PARA_FOLDERS.get(para.lower(), "3-Resources")


def slugify(title: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", title.lower())
    slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
    return slug or "untitled"


def noteFileName(title: str) -> str:
    """File name Obsidian resolves ``[[title]]`` to: the title minus unsafe characters."""
    name = re.sub(r'[\\/:*?"<>|#^\[\]]+', "-", title).strip(" .-")
    return name[:120] or "untitled"


def relPathFor(note: Note) -> str:
    folder = note.folder or paraFolder(note.para)
    return f"{folder}/{noteFileName(note.title)}.md"


def ensureVaultLayout(namespaceId: str, home: Path | None = None) -> Path:
    root = vaultDir(namespaceId, home)
    for folder in list(PARA_FOLDERS.values()):
        (root / folder).mkdir(parents=True, exist_ok=True)
    return root
