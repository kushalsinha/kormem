"""File-backed PARA vault with SQLite note index."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from kormem.core.config import KormemConfig, loadConfig
from kormem.core.ids import contentHash
from kormem.store.sqlite_store import SqliteStore
from kormem.vault.note import Note, NoteMeta
from kormem.vault.paths import ensureVaultLayout, relPathFor, vaultDir


def _atomicWrite(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmpPath = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmpPath, path)
    except Exception:
        try:
            os.unlink(tmpPath)
        except OSError:
            pass
        raise


class VaultStore:
    def __init__(
        self,
        store: Optional[SqliteStore] = None,
        config: Optional[KormemConfig] = None,
        namespaceId: Optional[str] = None,
    ):
        self.store = store or SqliteStore()
        self.config = config or loadConfig()
        self.namespaceId = namespaceId or self.config.defaultNamespace
        self.vaultRoot = vaultDir(self.namespaceId)
        ensureVaultLayout(self.namespaceId)

    def writeNote(self, note: Note) -> str:
        note.updated = datetime.now(timezone.utc).isoformat()
        if not note.created:
            note.created = note.updated
        relPath = relPathFor(note)
        fullPath = self.vaultRoot / relPath
        _atomicWrite(fullPath, note.serialize())
        self.store.upsertNoteIndex(
            namespaceId=self.namespaceId,
            note=note,
            relPath=relPath,
            contentHash=contentHash(note.body),
            outboundLinks=note.outboundLinks(),
        )
        self._regenerateMoc()
        return note.kormemId

    def readNote(self, ref: str) -> Optional[Note]:
        row = self.store.resolveNote(self.namespaceId, ref)
        if not row:
            # allow direct _meta/persona style paths
            candidate = self.vaultRoot / ref
            if candidate.exists():
                return Note.parse(candidate.read_text(encoding="utf-8"))
            return None
        fullPath = self.vaultRoot / row["relPath"]
        if not fullPath.exists():
            return None
        return Note.parse(fullPath.read_text(encoding="utf-8"))

    def listNotes(self, para: Optional[str] = None) -> List[NoteMeta]:
        rows = self.store.listNoteIndex(self.namespaceId, para)
        return [
            NoteMeta(
                kormemId=r["kormemId"],
                title=r["title"],
                para=r["para"],
                relPath=r["relPath"],
                tags=__import__("json").loads(r["tags"] or "[]"),
                sources=__import__("json").loads(r["sources"] or "[]"),
                updatedAt=r["updatedAt"] or "",
            )
            for r in rows
        ]

    def resolveLink(self, title: str) -> Optional[NoteMeta]:
        row = self.store.resolveNote(self.namespaceId, title)
        if not row:
            return None
        return NoteMeta(
            kormemId=row["kormemId"],
            title=row["title"],
            para=row["para"],
            relPath=row["relPath"],
            updatedAt=row["updatedAt"] or "",
        )

    def getBacklinks(self, kormemId: str) -> List[NoteMeta]:
        rows = self.store.getBacklinks(self.namespaceId, kormemId)
        return [
            NoteMeta(
                kormemId=r["kormemId"],
                title=r["title"],
                para=r["para"],
                relPath=r["relPath"],
                updatedAt=r["updatedAt"] or "",
            )
            for r in rows
        ]

    def getLinks(self, ref: str) -> dict:
        note = self.readNote(ref)
        if not note:
            return {"outbound": [], "backlinks": []}
        outbound = []
        for title in note.outboundLinks():
            resolved = self.resolveLink(title)
            outbound.append({
                "title": title,
                "kormemId": resolved.kormemId if resolved else None,
                "relPath": resolved.relPath if resolved else None,
            })
        backlinks = [
            {"kormemId": b.kormemId, "title": b.title, "relPath": b.relPath}
            for b in self.getBacklinks(note.kormemId)
        ]
        return {"outbound": outbound, "backlinks": backlinks}

    def getMoc(self) -> str:
        mocPath = self.vaultRoot / "_meta" / "_moc.md"
        if mocPath.exists():
            return mocPath.read_text(encoding="utf-8")
        return self._regenerateMoc()

    def moveNote(self, note: Note, folder: str) -> str:
        """Move a note into another folder (same title, same id); the PARA root follows the folder."""
        from kormem.vault.folders import PARA_ROOTS

        old = self.vaultRoot / relPathFor(note)
        note.folder = folder
        note.para = PARA_ROOTS.get(folder.split("/")[0], note.para)
        kormemId = self.writeNote(note)
        if old.exists() and old != self.vaultRoot / relPathFor(note):
            old.unlink()
        return kormemId

    def deleteNote(self, note: Note) -> None:
        path = self.vaultRoot / relPathFor(note)
        if path.exists():
            path.unlink()
        self.store.deleteNoteIndex(self.namespaceId, note.kormemId)
        self._regenerateMoc()

    def _regenerateMoc(self) -> str:
        notes = self.listNotes()
        byFolder: dict[str, list] = {}
        for n in notes:
            if n.para in ("meta", "source"):
                continue
            byFolder.setdefault(n.relPath.rsplit("/", 1)[0], []).append(n)
        lines = ["# Map of Content", ""]
        for folder in sorted(byFolder):
            lines.append(f"## {folder}")
            for n in sorted(byFolder[folder], key=lambda x: x.title.lower()):
                lines.append(f"- [[{n.title}]]")
            lines.append("")
        lines += ["## Meta", "- [[memory]]", "- [[lessons]]", "- [[folders]]", ""]
        content = "\n".join(lines).rstrip() + "\n"
        moc = Note.create(title="_moc", para="meta", body=content)
        mocPath = self.vaultRoot / "_meta" / "_moc.md"
        _atomicWrite(mocPath, moc.serialize())
        return content

    def ensureMetaNotes(self) -> None:
        for name in ("memory", "lessons"):
            ref = f"_meta/{name}.md"
            if not self.readNote(ref):
                note = Note.create(title=name, para="meta", body="")
                note.kormemId = f"meta-{name}-{self.namespaceId}"
                self.writeNote(note)
