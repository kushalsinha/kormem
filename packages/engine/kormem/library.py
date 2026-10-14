"""Collections, contexts and addresses: the QMD-compatible layer over the index and the vault.

Every source and note has an address:

- ``kormem://<collection>/<path>``: a file in a collection
- ``kormem://vault/<folder>/<note>.md``: a vault note
- ``#abc123``: any source (file, URL, chat, remembered fact) by docid, the
  first 6 characters of its document id

Collections and contexts live in ``config.yaml``::

    collections:
      notes: {path: ~/notes, pattern: "**/*.md"}
    contexts:
      /: Knowledge base for my projects
      kormem://notes: Personal notes and ideas

A context describes a collection, a folder in it, the vault, or everything
(``/``). It is returned with matching search and recall results and is shown
to extraction, so notes are filed with that description in mind.
"""

from __future__ import annotations

import difflib
import fnmatch
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from kormem.connectors.file_connector import SUPPORTED_EXTENSIONS, FileConnector
from kormem.core.config import readRawConfig, writeRawConfig
from kormem.core.ids import contentHash
from kormem.ingest.chat_sessions import documentText

SCHEME = "kormem://"
VAULT = "vault"
DEFAULT_MASK = "**/*"
SKIP_DIRS = {"node_modules", "__pycache__", "venv", ".venv"}


class RefError(LookupError):
    """A reference matched nothing, or more than one thing."""

    def __init__(self, message: str, suggestions: Optional[List[str]] = None):
        super().__init__(message)
        self.suggestions = suggestions or []


# ------------------------------------------------------------------ config


def collections() -> Dict[str, Dict[str, str]]:
    raw = readRawConfig().get("collections") or {}
    return {name: {"path": str(c.get("path", "")), "pattern": str(c.get("pattern") or DEFAULT_MASK)}
            for name, c in raw.items() if isinstance(c, dict)}


def contexts() -> Dict[str, str]:
    return {str(k): str(v) for k, v in (readRawConfig().get("contexts") or {}).items()}


def _editConfig(fn) -> Any:
    data = readRawConfig()
    out = fn(data)
    writeRawConfig(data)
    return out


def docid(documentId: str) -> str:
    return "#" + documentId.replace("-", "")[:6]


def noteUri(relPath: str) -> str:
    return f"{SCHEME}{VAULT}/{relPath}"


def _rootOf(path: str) -> Path:
    return Path(path).expanduser().resolve()


def documentUri(doc, cols: Optional[Dict[str, Dict[str, str]]] = None) -> Optional[str]:
    """``kormem://<collection>/<path>`` for a file inside a collection, else None."""
    coll = doc["collectionId"] if "collectionId" in doc.keys() else None
    url = doc["canonicalUrl"] or ""
    if not coll or not url.startswith("file://"):
        return None
    cols = collections() if cols is None else cols
    path = Path(url[len("file://"):])
    root = cols.get(coll, {}).get("path")
    if root:
        try:
            return f"{SCHEME}{coll}/{path.resolve().relative_to(_rootOf(root)).as_posix()}"
        except ValueError:
            pass
    return f"{SCHEME}{coll}/{path.name}"


def contextFor(uri: Optional[str], ctx: Optional[Dict[str, str]] = None) -> str:
    """Every context that covers ``uri``, from the most general to the most specific."""
    ctx = contexts() if ctx is None else ctx
    hits = []
    for key, text in ctx.items():
        k = key.rstrip("/")
        if key == "/" or (uri and (uri == k or uri.startswith(k + "/"))):
            hits.append((0 if key == "/" else len(k), text))
    return "\n".join(t for _, t in sorted(hits))


# ------------------------------------------------------------- collections


def _expandMask(mask: str) -> List[str]:
    """Comma-separated masks, each with an optional ``{a,b}`` group."""
    out: List[str] = []
    for part in re.split(r",(?![^{]*\})", mask or DEFAULT_MASK):
        part = part.strip()
        m = re.search(r"\{([^}]*)\}", part)
        if m:
            out += [part[:m.start()] + alt.strip() + part[m.end():] for alt in m.group(1).split(",")]
        elif part:
            out.append(part)
    return out or [DEFAULT_MASK]


def collectionFiles(root: str, mask: str) -> List[Path]:
    base = _rootOf(root)
    if base.is_file():
        return [base]
    seen: Dict[str, Path] = {}
    for pattern in _expandMask(mask):
        for p in base.glob(pattern):
            rel = p.relative_to(base).parts
            if any(x.startswith(".") or x in SKIP_DIRS for x in rel[:-1]) or rel[-1].startswith("."):
                continue
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                seen[str(p)] = p
    return [seen[k] for k in sorted(seen)]


async def syncCollection(ingest, namespaceId: str, name: str, root: str, mask: str,
                         prune: bool = True) -> Dict[str, int]:
    """Index new files, re-index changed ones and drop deleted ones. Unchanged files are skipped."""
    store = ingest.store
    reader = FileConnector()
    existing = {r["canonicalUrl"]: r for r in store.documentsForCollection(namespaceId, name)}
    stats = {"added": 0, "updated": 0, "unchanged": 0, "removed": 0, "failed": 0}
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    base = _rootOf(root)
    for fp in collectionFiles(root, mask):
        url = f"file://{fp}"
        mtime = fp.stat().st_mtime
        row = existing.pop(url, None)
        meta = json.loads(row["metadata"] or "{}") if row is not None else {}
        if row is not None and meta.get("mtime") == mtime:
            stats["unchanged"] += 1
            continue
        try:
            content = await reader.fetchContent(str(fp))
        except Exception:
            stats["failed"] += 1
            continue
        if not content.strip():
            continue
        if row is None:
            rel = fp.relative_to(base).as_posix() if base.is_dir() else fp.name
            await ingest.ingestText(content=content, namespaceId=namespaceId, collectionId=name,
                                    sourceUrl=url, title=rel, platform="file",
                                    metadata={"mtime": mtime})
            stats["added"] += 1
        elif row["contentHash"] == contentHash(content):
            store.mergeDocumentMeta(row["id"], {"mtime": mtime})
            stats["unchanged"] += 1
        else:
            await ingest.updateText(row["id"], content)
            store.mergeDocumentMeta(row["id"], {"mtime": mtime, "date": today})
            store.enqueueExtractionJob(str(uuid.uuid4()), namespaceId, "document", documentId=row["id"])
            stats["updated"] += 1
    if prune:
        for row in existing.values():
            if (row["canonicalUrl"] or "").startswith("file://"):
                store.deleteDocumentFully(row["id"])
                stats["removed"] += 1
    return stats


def addCollection(name: str, path: str, mask: str = DEFAULT_MASK) -> Dict[str, str]:
    root = _rootOf(path)
    if not root.exists():
        raise FileNotFoundError(f"No such file or folder: {path}")
    entry = {"path": str(root), "pattern": mask or DEFAULT_MASK}

    def put(data):
        data.setdefault("collections", {})
        data["collections"] = data["collections"] or {}
        data["collections"][name] = entry
    _editConfig(put)
    return entry


def removeCollection(store, namespaceId: str, name: str) -> int:
    rows = store.documentsForCollection(namespaceId, name)
    for r in rows:
        store.deleteDocumentFully(r["id"])

    def drop(data):
        (data.get("collections") or {}).pop(name, None)
        ctx = data.get("contexts") or {}
        for k in [k for k in ctx if k == f"{SCHEME}{name}" or k.startswith(f"{SCHEME}{name}/")]:
            ctx.pop(k)
    _editConfig(drop)
    return len(rows)


def renameCollection(store, namespaceId: str, old: str, new: str) -> int:
    cols = collections()
    if old not in cols:
        raise KeyError(old)
    if new in cols:
        raise ValueError(f"A collection named {new!r} already exists")
    n = store.renameCollection(namespaceId, old, new)

    def move(data):
        c = data["collections"]
        data["collections"] = {(new if k == old else k): v for k, v in c.items()}
        ctx = data.get("contexts") or {}
        data["contexts"] = {
            (f"{SCHEME}{new}" + k[len(f"{SCHEME}{old}"):]
             if k == f"{SCHEME}{old}" or k.startswith(f"{SCHEME}{old}/") else k): v
            for k, v in ctx.items()
        }
    _editConfig(move)
    return n


# ------------------------------------------------------------------ contexts


def normalizeContextPath(path: str, cwd: str = "") -> str:
    """``/``, ``kormem://…``, a collection name, or a folder on disk inside a collection."""
    path = (path or "").strip()
    if not path or path == ".":
        path = cwd
    if path == "/":
        return "/"
    if path.startswith(SCHEME):
        return path.rstrip("/")
    cols = collections()
    head = path.split("/", 1)[0]
    if head in cols or head == VAULT:
        return f"{SCHEME}{path}".rstrip("/")
    disk = Path(path if Path(path).is_absolute() else Path(cwd or ".") / path).expanduser().resolve()
    for name, c in cols.items():
        try:
            rel = disk.relative_to(_rootOf(c["path"])).as_posix()
        except ValueError:
            continue
        return f"{SCHEME}{name}" + ("" if rel == "." else f"/{rel}")
    raise ValueError(f"{path!r} is not '/', a kormem:// address, a collection, or a folder inside one")


def addContext(path: str, text: str, cwd: str = "") -> str:
    key = normalizeContextPath(path, cwd)

    def put(data):
        data["contexts"] = data.get("contexts") or {}
        data["contexts"][key] = text
    _editConfig(put)
    return key


def removeContext(path: str, cwd: str = "") -> bool:
    key = normalizeContextPath(path, cwd)
    return bool(_editConfig(lambda data: (data.get("contexts") or {}).pop(key, None)))


# --------------------------------------------------------------- addresses


def _notePayload(vault, row) -> Dict[str, Any]:
    full = vault.vaultRoot / row["relPath"]
    uri = noteUri(row["relPath"])
    return {"kind": "note", "uri": uri, "title": row["title"], "path": str(full),
            "date": (row["updatedAt"] or "")[:10],
            "text": full.read_text(encoding="utf-8") if full.exists() else "",
            "context": contextFor(uri)}


def _docPayload(store, doc, encryptionKey: Optional[str], cols) -> Dict[str, Any]:
    meta = json.loads(doc["metadata"] or "{}")
    uri = documentUri(doc, cols)
    url = doc["canonicalUrl"] or ""
    return {"kind": "document", "uri": uri, "docid": docid(doc["id"]), "documentId": doc["id"],
            "title": doc["title"] or "Untitled", "source": meta.get("kind") or doc["platform"],
            "path": url[len("file://"):] if url.startswith("file://") else url,
            "date": str(meta.get("date") or doc["createdAt"] or "")[:10],
            "text": documentText(store, doc, encryptionKey), "context": contextFor(uri)}


def _one(rows, ref: str):
    if len(rows) > 1:
        raise RefError(f"{ref} matches {len(rows)} sources", [docid(r["id"]) + " " + (r["title"] or "") for r in rows])
    return rows[0] if rows else None


def resolve(store, vault, namespaceId: str, ref: str, encryptionKey: Optional[str] = None) -> Dict[str, Any]:
    """A note or source by address, docid, note title / alias / path, file path or title."""
    ref = ref.strip()
    cols = collections()
    if ref.startswith("#"):
        doc = _one(store.findDocuments(namespaceId, idPrefix=ref[1:].replace("-", "")[:8]), ref)
        if doc:
            return _docPayload(store, doc, encryptionKey, cols)
        raise RefError(f"No source with docid {ref}")
    rest = ref[len(SCHEME):] if ref.startswith(SCHEME) else ref
    coll, _, rel = rest.partition("/")
    if coll == VAULT and rel:
        rest = rel
    elif coll in cols and rel:
        doc = _one(store.findDocuments(namespaceId, url=f"file://{_rootOf(cols[coll]['path']) / rel}"), ref) \
            or _one([d for d in store.findDocuments(namespaceId, urlSuffix="/" + rel) if d["collectionId"] == coll], ref)
        if doc:
            return _docPayload(store, doc, encryptionKey, cols)
    for candidate in (rest, rest[:-3] if rest.endswith(".md") else None, Path(rest).stem):
        row = store.resolveNote(namespaceId, candidate) if candidate else None
        if row:
            return _notePayload(vault, row)
    disk = Path(ref).expanduser()
    if disk.exists():
        doc = _one(store.findDocuments(namespaceId, url=f"file://{disk.resolve()}"), ref)
        if doc:
            return _docPayload(store, doc, encryptionKey, cols)
    doc = _one(store.findDocuments(namespaceId, urlSuffix="/" + rest.lstrip("/")), ref) \
        or _one(store.findDocuments(namespaceId, title=rest), ref)
    if doc:
        return _docPayload(store, doc, encryptionKey, cols)
    names = [n.title for n in vault.listNotes()] + [r["title"] for r in store.documentsForNamespace(namespaceId)]
    raise RefError(f"Nothing matches {ref}", difflib.get_close_matches(rest, names, n=5, cutoff=0.5))


def _allAddresses(store, vault, namespaceId: str, cols) -> List[Dict[str, Any]]:
    out = [{"uri": noteUri(n.relPath), "alt": n.relPath, "title": n.title, "kind": "note",
            "date": (n.updatedAt or "")[:10]} for n in vault.listNotes()]
    for r in store.documentsForNamespace(namespaceId):
        doc = store.getDocument(r["id"])
        uri = documentUri(doc, cols)
        meta = json.loads(r["metadata"] or "{}")
        out.append({"uri": uri or docid(r["id"]), "alt": uri[len(SCHEME):] if uri else (r["title"] or ""),
                    "title": r["title"] or "Untitled", "kind": meta.get("kind") or r["platform"],
                    "docid": docid(r["id"]), "date": str(meta.get("date") or r["createdAt"] or "")[:10]})
    return out


def multiGet(store, vault, namespaceId: str, pattern: str, maxBytes: int = 65_536,
             encryptionKey: Optional[str] = None) -> List[Dict[str, Any]]:
    """Several notes or sources: a glob over addresses, or a comma-separated list of refs."""
    if not any(ch in pattern for ch in "*?[") or "," in pattern:
        refs = [p.strip() for p in pattern.split(",") if p.strip()]
    else:
        pat = pattern[len(SCHEME):] if pattern.startswith(SCHEME) else pattern
        refs = [a["uri"] for a in _allAddresses(store, vault, namespaceId, collections())
                if fnmatch.fnmatch(a["alt"], pat) or fnmatch.fnmatch(a["uri"][len(SCHEME):], pat)]
    out = []
    for ref in refs:
        try:
            item = resolve(store, vault, namespaceId, ref, encryptionKey)
        except RefError as e:
            out.append({"ref": ref, "error": str(e), "suggestions": e.suggestions})
            continue
        size = len(item["text"].encode("utf-8"))
        if maxBytes and size > maxBytes:
            item = {**item, "text": "", "skipped": f"{size} bytes > --max-bytes {maxBytes}"}
        out.append(item)
    return out


def listing(store, vault, namespaceId: str, path: str = "") -> Dict[str, Any]:
    """Collections and vault folders, or the entries inside one of them."""
    cols = collections()
    path = (path[len(SCHEME):] if path.startswith(SCHEME) else path).strip("/")
    if not path:
        counts = store.collectionCounts(namespaceId)
        folders: Dict[str, int] = {}
        for n in vault.listNotes():
            folder = n.relPath.rsplit("/", 1)[0] if "/" in n.relPath else ""
            folders[folder] = folders.get(folder, 0) + 1
        return {"collections": [{"name": k, "uri": f"{SCHEME}{k}", "path": v["path"], "pattern": v["pattern"],
                                 "documents": counts.get(k, 0), "context": contextFor(f"{SCHEME}{k}")}
                                for k, v in cols.items()],
                "vault": [{"folder": f, "uri": noteUri(f), "notes": c} for f, c in sorted(folders.items())]}
    head = path.split("/", 1)[0]
    if head in cols:
        entries = [a for a in _allAddresses(store, vault, namespaceId, cols)
                   if a["uri"].startswith(f"{SCHEME}{path}/") or a["uri"] == f"{SCHEME}{path}"]
    else:
        folder = path[len(VAULT) + 1:] if head == VAULT else path
        entries = [a for a in _allAddresses(store, vault, namespaceId, cols)
                   if a["kind"] == "note" and (not folder or a["alt"].startswith(folder.rstrip("/") + "/"))]
    return {"path": path, "entries": entries}
