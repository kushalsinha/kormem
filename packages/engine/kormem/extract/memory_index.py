"""The dated memory index: every extracted item as one linked line, filed by month.

Lines live in ``_meta/memory/YYYY-MM.md`` (one note per month, titled
``memory YYYY-MM``) so the index never becomes one ever-growing file that is
rewritten on every extraction. ``_meta/memory.md`` lists the months. Layout of
a month:

    Date: 2023-05-28

    🔴 User started watching Game of Thrones. (meaning 2023-04-28) [[Game of Thrones]] [[2023-05-28 chat · HBO picks]]

Lines sit under the date of the source they came from. When the item itself
happened on another date, a ``(meaning DATE)`` anchor carries it. The links
point at the topic note and the source note.

A line whose fact was later replaced is retired: its marker becomes ⚫ and its
vector is dropped, so recall stops scoring it. The topic note's Timeline keeps
the history.
"""

from __future__ import annotations

import difflib
import hashlib
import re
from typing import Dict, List, Optional, Sequence, Tuple

from kormem.vault.note import Note

LEGACY_REF = "_meta/memory.md"
SHARD_FOLDER = "_meta/memory"
RETIRED = "⚫"
RETIRE_RATIO = 0.55
_SHARD_RE = re.compile(r"^memory (\d{4}-\d{2})$")
_MONTHS_RE = re.compile(r"^Months:.*$", re.M)

_HEADER_RE = re.compile(r"^Date:\s*(\d{4}-\d{2}-\d{2})\s*$")
TRAILING_LINKS_RE = re.compile(r"(\s*\[\[[^\]]+\]\])+\s*$")


def stripLinks(line: str) -> str:
    """Memory line without its trailing ``[[links]]``."""
    return TRAILING_LINKS_RE.sub("", line)


def trailingLinks(line: str) -> List[str]:
    m = TRAILING_LINKS_RE.search(line)
    return re.findall(r"\[\[([^\]]+)\]\]", m.group(0)) if m else []


def _norm(line: str) -> str:
    return re.sub(r"\W+", " ", stripLinks(line).lower()).strip()


def parseBlocks(body: str) -> Tuple[str, Dict[str, List[str]]]:
    preamble: List[str] = []
    blocks: Dict[str, List[str]] = {}
    current = None
    for line in body.splitlines():
        m = _HEADER_RE.match(line.strip())
        if m:
            current = m.group(1)
            blocks.setdefault(current, [])
        elif current is None:
            preamble.append(line)
        elif line.strip():
            blocks[current].append(line.rstrip())
    return "\n".join(preamble).strip(), blocks


def renderBlocks(preamble: str, blocks: Dict[str, List[str]]) -> str:
    parts = [preamble] if preamble else []
    for date in sorted(blocks):
        if blocks[date]:
            parts.append(f"Date: {date}\n\n" + "\n".join(blocks[date]))
    return "\n\n".join(parts).strip() + "\n"


def formatLine(sourceDate: str, itemDate: str, priority: str, text: str, links: str) -> str:
    text = text.rstrip(". ") + "."
    anchor = f" (meaning {itemDate})" if itemDate and itemDate != sourceDate else ""
    return f"{priority} {text}{anchor} {links}".strip()


def shardTitle(month: str) -> str:
    return f"memory {month}"


def months(vault) -> List[str]:
    """Months that have a memory shard, oldest first."""
    found = []
    for n in vault.listNotes(para="meta"):
        m = _SHARD_RE.match(n.title)
        if m:
            found.append(m.group(1))
    return sorted(set(found))


def _readShard(vault, month: str) -> Optional[Note]:
    return vault.readNote(shardTitle(month))


def _newShard(vault, month: str) -> Note:
    note = Note.create(title=shardTitle(month), para="meta", folder=SHARD_FOLDER)
    note.kormemId = f"meta-memory-{vault.namespaceId}-{month}"
    return note


def _writeIndex(vault, preamble: str = "") -> None:
    """``_meta/memory.md``: free text kept from older versions, then links to every month."""
    note = vault.readNote(LEGACY_REF)
    if note is None:
        note = Note.create(title="memory", para="meta")
        note.kormemId = f"meta-memory-{vault.namespaceId}"
        keep = preamble
    else:
        keep = preamble or _MONTHS_RE.sub("", parseBlocks(note.body)[0]).strip()
    links = " · ".join(f"[[{shardTitle(m)}]]" for m in reversed(months(vault)))
    note.body = "\n\n".join(x for x in (keep, f"Months: {links}" if links else "") if x) + "\n"
    vault.writeNote(note)


def migrateLegacy(vault) -> int:
    """Move ``Date:`` blocks of a single-file ``_meta/memory.md`` into month shards; returns lines moved."""
    note = vault.readNote(LEGACY_REF)
    if note is None:
        return 0
    preamble, blocks = parseBlocks(note.body)
    if not any(blocks.values()):
        return 0
    byMonth: Dict[str, Dict[str, List[str]]] = {}
    for date, lines in blocks.items():
        byMonth.setdefault(date[:7], {})[date] = lines
    moved = 0
    for month, monthBlocks in byMonth.items():
        shard = _readShard(vault, month) or _newShard(vault, month)
        pre, have = parseBlocks(shard.body)
        for date, lines in monthBlocks.items():
            have.setdefault(date, []).extend(lines)
            moved += len(lines)
        shard.body = renderBlocks(pre, have)
        vault.writeNote(shard)
    _writeIndex(vault, _MONTHS_RE.sub("", preamble).strip())
    return moved


def appendMemoryLines(vault, sourceDate: str, entries: Sequence[Tuple[str, str, str, str]]) -> List[str]:
    """Add ``(itemDate, priority, text, links)`` entries under ``sourceDate``; skip duplicates.

    Returns the lines that were added."""
    if not entries:
        return []
    migrateLegacy(vault)
    month = sourceDate[:7]
    note = _readShard(vault, month)
    isNew = note is None
    note = note or _newShard(vault, month)
    preamble, blocks = parseBlocks(note.body)
    lines = blocks.setdefault(sourceDate, [])
    seen = {_norm(l) for l in lines}
    added: List[str] = []
    for itemDate, priority, text, links in entries:
        line = formatLine(sourceDate, itemDate, priority, text, links)
        if _norm(line) in seen:
            continue
        seen.add(_norm(line))
        lines.append(line)
        added.append(line)
    note.body = renderBlocks(preamble, blocks)
    vault.writeNote(note)
    if isNew:
        _writeIndex(vault)
    return added


def readMemory(vault, since: Optional[str] = None, includeRetired: bool = False) -> str:
    """The memory index as one ``Date:``-block text, from month ``since`` (YYYY-MM) on."""
    migrateLegacy(vault)
    parts: List[str] = []
    legacy = vault.readNote(LEGACY_REF)
    if legacy:
        pre = _MONTHS_RE.sub("", parseBlocks(legacy.body)[0]).strip()
        if pre:
            parts.append(pre)
    for month in months(vault):
        if since and month < since:
            continue
        shard = _readShard(vault, month)
        if not shard:
            continue
        _, blocks = parseBlocks(shard.body)
        if not includeRetired:
            blocks = {d: [l for l in v if not isRetired(l)] for d, v in blocks.items()}
        body = renderBlocks("", blocks).strip()
        if body:
            parts.append(body)
    return "\n\n".join(parts).strip()


def isRetired(line: str) -> bool:
    return line.lstrip().startswith(RETIRED)


def retireLine(vault, store, noteTitle: str, oldText: str, date: str) -> Optional[str]:
    """Retire the live memory line about ``noteTitle`` closest to ``oldText``; returns its text.

    Its vector is deleted so recall no longer scores it.
    """
    want = _norm(oldText)
    best: Tuple[float, str, int, str] = (0.0, "", -1, "")
    for month in reversed(months(vault)):
        shard = _readShard(vault, month)
        if not shard:
            continue
        _, blocks = parseBlocks(shard.body)
        for d, lines in blocks.items():
            for i, l in enumerate(lines):
                if isRetired(l) or f"[[{noteTitle.lower()}]]" not in l.lower():
                    continue
                r = difflib.SequenceMatcher(None, _norm(l), want).ratio()
                if r > best[0]:
                    best = (r, month, i, d)
    score, month, idx, d = best
    if score < RETIRE_RATIO:
        return None
    shard = _readShard(vault, month)
    preamble, blocks = parseBlocks(shard.body)
    old = blocks[d][idx]
    marker = old.lstrip().split(" ", 1)
    rest = marker[1] if len(marker) > 1 and not marker[0][:1].isalnum() else old.lstrip()
    links = trailingLinks(rest)
    head = stripLinks(rest).rstrip()
    blocks[d][idx] = f"{RETIRED} {head} (replaced {date}) " + " ".join(f"[[{x}]]" for x in links)
    shard.body = renderBlocks(preamble, blocks)
    vault.writeNote(shard)
    store.deleteMemoryVectors(vault.namespaceId, {_textKey(cleanLine(old))})
    return cleanLine(old)


_ANCHOR_RE = re.compile(r"\s*\((meaning|estimated)\s+([^)]+)\)\s*\.?\s*$", re.I)


def cleanLine(line: str) -> str:
    """Memory line as readers see it: no links, and '(meaning D)' as '(date: D)'."""
    line = stripLinks(line).strip()
    if line.startswith(RETIRED):
        line = line[len(RETIRED):].strip()
    m = _ANCHOR_RE.search(line)
    if not m:
        return line
    tag = "date" if m.group(1).lower() == "meaning" else "approx. date"
    return f"{line[:m.start()].rstrip(' .')}. ({tag}: {m.group(2).strip()})"


async def lineEmbeddings(store, embedder, namespaceId: str, texts: Sequence[str]):
    """Unit-normalised embeddings for memory lines, computed once per line and kept in the index.

    Extraction warms this as it writes lines, so recall normally only reads.
    """
    import numpy as np

    from kormem.store.sqlite_store import _unpackEmbedding

    tag = hashlib.md5(embedder.modelName.encode()).hexdigest()[:6]  # a new model never reuses old vectors
    keys = [f"mem:{namespaceId}:{tag}:{_textKey(t)}" for t in texts]
    unique = list(set(keys))
    have = {}
    for i in range(0, len(unique), 500):  # stay under SQLite's bound-parameter limit
        for r in store.getSegmentsByKeys(unique[i:i + 500]):
            if r["embedding"]:
                have[r["segmentKey"]] = _unpackEmbedding(r["embedding"], r["embeddingDim"])
    missing = list(dict.fromkeys((k, t) for k, t in zip(keys, texts) if k not in have))
    if missing:
        vecs = await embedder.encodeBatch([t for _, t in missing])
        store.upsertSegments(namespaceId, "__memory__", "__memory__", [
            {"segmentKey": k, "segmentIndex": 0, "embedding": v, "embeddingDim": len(v), "kind": "memory",
             "plainText": None, "metadata": {"model": embedder.modelName}}
            for (k, _), v in zip(missing, vecs)
        ])
        for (k, _), v in zip(missing, vecs):
            have[k] = np.asarray(v, dtype=np.float32)
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    arr = np.stack([have[k] for k in keys]).astype(np.float32)
    return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9)


def _textKey(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", "replace")).hexdigest()[:16]
