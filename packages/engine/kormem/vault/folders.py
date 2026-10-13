"""Vault folder registry and the strict new-folder gate.

PARA roots (``1-Projects`` .. ``4-Archive``) always exist. A subfolder is
created only when all of these hold, checked in code:

- at least ``MIN_NOTES`` notes have asked for it (until then each waits in
  the closest existing folder and the request is recorded as *proposed*),
- its purpose is less than ``MAX_SIMILARITY`` similar to every active folder,
- it is at most ``MAX_DEPTH`` levels below its PARA root,
- the extractor gave a reason, which is kept in ``_meta/folders.md``.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

from kormem.vault.note import Note, PARA_FOLDERS

logger = logging.getLogger(__name__)

MIN_NOTES = 3
MAX_SIMILARITY = 0.80
MAX_DEPTH = 2
PARA_ROOTS = {v: k for k, v in PARA_FOLDERS.items() if k in ("projects", "areas", "resources", "archive")}


def splitFolder(path: str) -> Tuple[Optional[str], List[str]]:
    """``"2-Areas/Health/Running"`` -> (``"areas"``, ``["Health", "Running"]``)."""
    parts = [p.strip() for p in (path or "").strip("/").split("/") if p.strip()]
    if not parts or parts[0] not in PARA_ROOTS:
        return None, parts
    return PARA_ROOTS[parts[0]], parts[1:]


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    va, vb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    return float(va @ vb / max(np.linalg.norm(va) * np.linalg.norm(vb), 1e-9))


class FolderRegistry:
    def __init__(self, store, vault, embedder):
        self.store = store
        self.vault = vault
        self.embedder = embedder
        self.ns = vault.namespaceId

    def active(self) -> List[dict]:
        return [dict(r) for r in self.store.listFolders(self.ns, "active")]

    def tree(self) -> List[str]:
        """Every folder the extractor may file into: PARA roots plus active subfolders."""
        return sorted(set(PARA_ROOTS) | {f["path"] for f in self.active()})

    async def closest(self, purpose: str, para: str) -> str:
        """Best active folder under ``para``'s root for this purpose, or the root itself."""
        root = PARA_FOLDERS.get(para, "3-Resources")
        candidates = [f for f in self.active() if f["path"].startswith(root + "/") and f["embedding"]]
        if not candidates or not purpose:
            return root
        q = await self.embedder.encodeQuery(purpose)
        best = max(candidates, key=lambda f: _cos(q, json.loads(f["embedding"])))
        return best["path"] if _cos(q, json.loads(best["embedding"])) >= 0.5 else root

    async def place(self, folder: str, para: str, noteTitle: str, purpose: str = "",
                    reason: str = "") -> Tuple[str, Optional[str]]:
        """Folder a note goes into now, and the folder it may move to later.

        Returns ``(folderNow, proposedPath)``. ``proposedPath`` is set when the
        requested folder does not exist yet and was recorded as a proposal.
        """
        root = PARA_FOLDERS.get(para, "3-Resources")
        rootPara, sub = splitFolder(folder)
        if rootPara is None or not sub:
            return root, None
        path = "/".join([PARA_FOLDERS[rootPara], *sub])
        if any(f["path"] == path for f in self.active()):
            return path, None
        if len(sub) > MAX_DEPTH:
            logger.info("folder %s rejected: deeper than %d levels", path, MAX_DEPTH)
            return await self.closest(purpose or path, rootPara), None
        rows = {r["path"]: dict(r) for r in self.store.listFolders(self.ns, "proposed")}
        prior = rows.get(path)
        titles = list(dict.fromkeys(json.loads(prior["noteTitles"]) if prior else []) | {noteTitle: None}.keys())
        self.store.upsertFolder(self.ns, path, purpose or (prior or {}).get("purpose") or path,
                                "proposed", reason or (prior or {}).get("reason"), titles)
        return await self.closest(purpose or path, rootPara), path

    async def promoteReady(self) -> List[Tuple[str, List[str]]]:
        """Activate proposals that pass the gate; returns (path, notes to move) for each."""
        promoted: List[Tuple[str, List[str]]] = []
        active = self.active()
        for prop in [dict(r) for r in self.store.listFolders(self.ns, "proposed")]:
            titles = json.loads(prop["noteTitles"] or "[]")
            if len(titles) < MIN_NOTES or not prop.get("reason"):
                continue
            emb = await self.embedder.encodeQuery(prop["purpose"])
            clash = [f["path"] for f in active if f["embedding"]
                     and _cos(emb, json.loads(f["embedding"])) >= MAX_SIMILARITY]
            if clash:
                logger.info("folder %s not created: too close to %s", prop["path"], clash[0])
                continue
            self.store.upsertFolder(self.ns, prop["path"], prop["purpose"], "active",
                                    prop["reason"], [], list(map(float, emb)))
            active.append({**prop, "embedding": json.dumps(list(map(float, emb)))})
            promoted.append((prop["path"], titles))
        if promoted:
            self.writeRegistryNote()
        return promoted

    def writeRegistryNote(self) -> None:
        """Human-readable registry at ``_meta/folders.md``."""
        lines = ["# Folders", "", "Subfolders exist only when at least "
                 f"{MIN_NOTES} notes need them. Proposals wait below until then.", ""]
        for f in self.active():
            lines.append(f"- **{f['path']}** — {f['purpose']}" + (f" _(reason: {f['reason']})_" if f.get("reason") else ""))
        proposed = [dict(r) for r in self.store.listFolders(self.ns, "proposed")]
        if proposed:
            lines += ["", "## Proposed", ""]
            for f in proposed:
                n = len(json.loads(f["noteTitles"] or "[]"))
                lines.append(f"- {f['path']} — {f['purpose']} ({n}/{MIN_NOTES} notes)")
        note = self.vault.readNote("_meta/folders.md")
        if note is None:
            note = Note.create(title="folders", para="meta")
            note.kormemId = f"meta-folders-{self.ns}"
        note.body = "\n".join(lines)
        self.vault.writeNote(note)
