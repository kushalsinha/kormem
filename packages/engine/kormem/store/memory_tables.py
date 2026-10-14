"""Tables for chat sessions, extraction bookkeeping and the vault folder registry.

Mixed into :class:`kormem.store.sqlite_store.SqliteStore`.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    namespaceId TEXT NOT NULL,
    sessionId TEXT NOT NULL,
    documentId TEXT,
    sessionDate TEXT,
    lastTurnAt TEXT NOT NULL,
    extractedThrough INTEGER NOT NULL DEFAULT -1,
    closed INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (namespaceId, sessionId)
);
CREATE TABLE IF NOT EXISTS chat_turns (
    namespaceId TEXT NOT NULL,
    sessionId TEXT NOT NULL,
    idx INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    at TEXT,
    PRIMARY KEY (namespaceId, sessionId, idx)
);
CREATE TABLE IF NOT EXISTS extracted_sections (
    namespaceId TEXT NOT NULL,
    sectionHash TEXT NOT NULL,
    PRIMARY KEY (namespaceId, sectionHash)
);
CREATE TABLE IF NOT EXISTS vault_folders (
    namespaceId TEXT NOT NULL,
    path TEXT NOT NULL,
    purpose TEXT NOT NULL,
    reason TEXT,
    status TEXT NOT NULL,          -- 'active' | 'proposed'
    noteTitles TEXT,               -- proposed folders: notes waiting to move in
    embedding TEXT,
    createdAt TEXT NOT NULL,
    PRIMARY KEY (namespaceId, path)
);
"""

_EXTRACTION_COLUMNS = {
    "kind": "TEXT DEFAULT 'document'",
    "spanStart": "INTEGER",
    "spanEnd": "INTEGER",
    "payload": "TEXT",
}
_NOTE_COLUMNS = {"aliases": "TEXT", "folder": "TEXT"}


class MemoryTablesMixin:
    """Requires ``_connectIndex`` and ``_now`` from the host store."""

    def _initMemoryTables(self, conn: sqlite3.Connection) -> None:
        conn.executescript(_SCHEMA)
        for table, wanted in (("extraction_jobs", _EXTRACTION_COLUMNS), ("notes", _NOTE_COLUMNS)):
            have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
            for col, ddl in wanted.items():
                if col not in have:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")

    # ------------------------------------------------------------------ chat

    def appendChatTurns(self, namespaceId: str, sessionId: str, turns: List[Dict[str, Any]],
                        sessionDate: Optional[str] = None) -> int:
        """Insert turns (idempotent by index); returns the session's turn count."""
        now = self._now()
        with self._connectIndex() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM chat_turns WHERE namespaceId=? AND sessionId=?",
                (namespaceId, sessionId),
            ).fetchone()
            nextIdx = row["n"]
            for t in turns:
                idx = t.get("idx")
                idx = nextIdx if idx is None else int(idx)
                conn.execute(
                    "INSERT OR IGNORE INTO chat_turns (namespaceId, sessionId, idx, role, content, at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (namespaceId, sessionId, idx, t["role"], t["content"], t.get("at")),
                )
                nextIdx = max(nextIdx, idx + 1)
            conn.execute(
                """
                INSERT INTO chat_sessions (namespaceId, sessionId, sessionDate, lastTurnAt, closed)
                VALUES (?, ?, ?, ?, 0)
                ON CONFLICT(namespaceId, sessionId) DO UPDATE SET
                    lastTurnAt=excluded.lastTurnAt, closed=0,
                    sessionDate=COALESCE(chat_sessions.sessionDate, excluded.sessionDate)
                """,
                (namespaceId, sessionId, sessionDate, now),
            )
            conn.commit()
            return conn.execute(
                "SELECT COUNT(*) AS n FROM chat_turns WHERE namespaceId=? AND sessionId=?",
                (namespaceId, sessionId),
            ).fetchone()["n"]

    def getChatTurns(self, namespaceId: str, sessionId: str) -> List[sqlite3.Row]:
        with self._connectIndex() as conn:
            return conn.execute(
                "SELECT * FROM chat_turns WHERE namespaceId=? AND sessionId=? ORDER BY idx",
                (namespaceId, sessionId),
            ).fetchall()

    def getChatSession(self, namespaceId: str, sessionId: str) -> Optional[sqlite3.Row]:
        with self._connectIndex() as conn:
            return conn.execute(
                "SELECT * FROM chat_sessions WHERE namespaceId=? AND sessionId=?",
                (namespaceId, sessionId),
            ).fetchone()

    def setChatSessionDocument(self, namespaceId: str, sessionId: str, documentId: str) -> None:
        with self._connectIndex() as conn:
            conn.execute(
                "UPDATE chat_sessions SET documentId=? WHERE namespaceId=? AND sessionId=?",
                (documentId, namespaceId, sessionId),
            )
            conn.commit()

    def closeChatSession(self, namespaceId: str, sessionId: str) -> None:
        with self._connectIndex() as conn:
            conn.execute(
                "UPDATE chat_sessions SET closed=1 WHERE namespaceId=? AND sessionId=?",
                (namespaceId, sessionId),
            )
            conn.commit()

    def settleChatSessions(self, idleMinutes: float, maxTurns: int) -> List[Dict[str, Any]]:
        """Claim the unextracted stretch of every settled session.

        A session settles when it is closed, idle for ``idleMinutes``, or has
        ``maxTurns`` unextracted turns. The watermark moves when the stretch
        is claimed, so each turn is extracted once.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=idleMinutes)).isoformat()
        out: List[Dict[str, Any]] = []
        with self._connectIndex() as conn:
            rows = conn.execute(
                """
                SELECT s.*, (SELECT MAX(idx) FROM chat_turns t
                             WHERE t.namespaceId=s.namespaceId AND t.sessionId=s.sessionId) AS lastIdx
                FROM chat_sessions s
                """
            ).fetchall()
            for r in rows:
                lastIdx = r["lastIdx"]
                if lastIdx is None or lastIdx <= r["extractedThrough"]:
                    continue
                pending = lastIdx - r["extractedThrough"]
                if not (r["closed"] or r["lastTurnAt"] <= cutoff or pending >= maxTurns):
                    continue
                conn.execute(
                    "UPDATE chat_sessions SET extractedThrough=? WHERE namespaceId=? AND sessionId=?",
                    (lastIdx, r["namespaceId"], r["sessionId"]),
                )
                out.append({
                    "namespaceId": r["namespaceId"], "sessionId": r["sessionId"],
                    "documentId": r["documentId"], "spanStart": r["extractedThrough"] + 1,
                    "spanEnd": lastIdx,
                })
            conn.commit()
        return out

    # ---------------------------------------------------------- extraction

    def enqueueExtractionJob(self, jobId: str, namespaceId: str, kind: str,
                             documentId: Optional[str] = None, spanStart: Optional[int] = None,
                             spanEnd: Optional[int] = None, payload: Optional[dict] = None) -> None:
        now = self._now()
        with self._connectIndex() as conn:
            conn.execute(
                """
                INSERT INTO extraction_jobs
                    (id, namespaceId, documentId, status, textPreview, kind, spanStart, spanEnd,
                     payload, createdAt, updatedAt)
                VALUES (?, ?, ?, 'pending', '', ?, ?, ?, ?, ?, ?)
                """,
                (jobId, namespaceId, documentId, kind, spanStart, spanEnd,
                 json.dumps(payload) if payload else None, now, now),
            )
            conn.commit()

    def claimFactBatches(self, debounceMinutes: float, maxBatch: int) -> List[List[sqlite3.Row]]:
        """Pending ``fact`` jobs grouped per namespace, once a namespace's facts settle."""
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=debounceMinutes)).isoformat()
        batches: List[List[sqlite3.Row]] = []
        with self._connectIndex() as conn:
            rows = conn.execute(
                "SELECT * FROM extraction_jobs WHERE status='pending' AND kind='fact' ORDER BY createdAt"
            ).fetchall()
            byNs: Dict[str, List[sqlite3.Row]] = {}
            for r in rows:
                byNs.setdefault(r["namespaceId"], []).append(r)
            for jobs in byNs.values():
                if len(jobs) < maxBatch and jobs[-1]["createdAt"] > cutoff:
                    continue
                batch = jobs[:maxBatch]
                for j in batch:
                    conn.execute("UPDATE extraction_jobs SET status='processing', updatedAt=? WHERE id=?",
                                 (self._now(), j["id"]))
                batches.append(batch)
            conn.commit()
        return batches

    def markSectionsExtracted(self, namespaceId: str, hashes: List[str]) -> None:
        with self._connectIndex() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO extracted_sections (namespaceId, sectionHash) VALUES (?, ?)",
                [(namespaceId, h) for h in hashes],
            )
            conn.commit()

    def extractedSectionHashes(self, namespaceId: str, hashes: List[str]) -> set:
        if not hashes:
            return set()
        with self._connectIndex() as conn:
            marks = ",".join("?" * len(hashes))
            rows = conn.execute(
                f"SELECT sectionHash FROM extracted_sections WHERE namespaceId=? AND sectionHash IN ({marks})",
                (namespaceId, *hashes),
            ).fetchall()
        return {r["sectionHash"] for r in rows}

    # -------------------------------------------------------------- folders

    def listFolders(self, namespaceId: str, status: Optional[str] = None) -> List[sqlite3.Row]:
        with self._connectIndex() as conn:
            if status:
                return conn.execute(
                    "SELECT * FROM vault_folders WHERE namespaceId=? AND status=? ORDER BY path",
                    (namespaceId, status),
                ).fetchall()
            return conn.execute(
                "SELECT * FROM vault_folders WHERE namespaceId=? ORDER BY path", (namespaceId,),
            ).fetchall()

    def upsertFolder(self, namespaceId: str, path: str, purpose: str, status: str,
                     reason: Optional[str] = None, noteTitles: Optional[List[str]] = None,
                     embedding: Optional[List[float]] = None) -> None:
        with self._connectIndex() as conn:
            conn.execute(
                """
                INSERT INTO vault_folders (namespaceId, path, purpose, reason, status, noteTitles, embedding, createdAt)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespaceId, path) DO UPDATE SET
                    purpose=excluded.purpose, reason=COALESCE(excluded.reason, vault_folders.reason),
                    status=excluded.status, noteTitles=excluded.noteTitles,
                    embedding=COALESCE(excluded.embedding, vault_folders.embedding)
                """,
                (namespaceId, path, purpose, reason, status,
                 json.dumps(noteTitles or []), json.dumps(embedding) if embedding else None, self._now()),
            )
            conn.commit()

    # ------------------------------------------------------------ documents

    def updateDocumentContent(self, documentId: str, encryptedData: str, contentHash: str,
                              metadata: Optional[dict] = None) -> None:
        with self._connectIndex() as conn:
            conn.execute(
                "UPDATE documents SET encryptedData=?, contentHash=?, metadata=COALESCE(?, metadata), "
                "updatedAt=? WHERE id=?",
                (encryptedData, contentHash, json.dumps(metadata) if metadata else None,
                 self._now(), documentId),
            )
            conn.commit()

    def embeddingsByTextHash(self, documentId: str) -> Dict[str, List[float]]:
        """Existing segment embeddings of a document keyed by ``metadata.textHash``."""
        from kormem.store.sqlite_store import _unpackEmbedding

        out: Dict[str, List[float]] = {}
        with self._connectIndex() as conn:
            for r in conn.execute(
                "SELECT embedding, embeddingDim, metadata FROM segments WHERE documentId=? AND embedding IS NOT NULL",
                (documentId,),
            ):
                h = json.loads(r["metadata"] or "{}").get("textHash")
                if h:
                    out[h] = _unpackEmbedding(r["embedding"], r["embeddingDim"]).tolist()
        return out

    def replaceDocumentSegments(self, documentId: str) -> None:
        """Drop a document's segments and their keyword rows before re-chunking it."""
        with self._connectIndex() as conn:
            conn.execute(
                "DELETE FROM segments_fts WHERE rowid IN (SELECT rowid FROM segments WHERE documentId=?)",
                (documentId,),
            )
            conn.execute("DELETE FROM segments WHERE documentId=?", (documentId,))
            conn.commit()

    def documentsForNamespace(self, namespaceId: str) -> List[sqlite3.Row]:
        with self._connectIndex() as conn:
            return conn.execute(
                "SELECT id, title, canonicalUrl, platform, metadata, createdAt FROM documents "
                "WHERE namespaceId=? ORDER BY createdAt",
                (namespaceId,),
            ).fetchall()

    def documentsForCollection(self, namespaceId: str, collectionId: str) -> List[sqlite3.Row]:
        with self._connectIndex() as conn:
            return conn.execute(
                "SELECT id, title, canonicalUrl, contentHash, metadata FROM documents "
                "WHERE namespaceId=? AND collectionId=? ORDER BY canonicalUrl",
                (namespaceId, collectionId),
            ).fetchall()

    def collectionCounts(self, namespaceId: str) -> Dict[str, int]:
        with self._connectIndex() as conn:
            return {r[0]: r[1] for r in conn.execute(
                "SELECT collectionId, COUNT(*) FROM documents WHERE namespaceId=? AND collectionId IS NOT NULL "
                "GROUP BY collectionId", (namespaceId,))}

    def findDocuments(self, namespaceId: str, idPrefix: str = "", url: str = "", urlSuffix: str = "",
                      title: str = "") -> List[sqlite3.Row]:
        """Documents by id prefix, exact URL, URL suffix or exact title (first criterion given)."""
        if idPrefix:
            where, arg = "id LIKE ?", idPrefix.lower() + "%"
        elif url:
            where, arg = "canonicalUrl = ?", url
        elif urlSuffix:
            where, arg = "canonicalUrl LIKE ?", "%" + urlSuffix
        else:
            where, arg = "title = ?", title
        with self._connectIndex() as conn:
            return conn.execute(
                f"SELECT * FROM documents WHERE namespaceId=? AND {where} ORDER BY createdAt DESC LIMIT 20",
                (namespaceId, arg),
            ).fetchall()

    def renameCollection(self, namespaceId: str, old: str, new: str) -> int:
        with self._connectIndex() as conn:
            n = conn.execute("UPDATE documents SET collectionId=? WHERE namespaceId=? AND collectionId=?",
                             (new, namespaceId, old)).rowcount
            for table in ("segments", "sources"):
                conn.execute(f"UPDATE {table} SET collectionId=? WHERE namespaceId=? AND collectionId=?",
                             (new, namespaceId, old))
            conn.commit()
            return n

    def deleteDocumentFully(self, documentId: str) -> None:
        """A document with its segments and keyword rows (vault notes built from it stay)."""
        self.replaceDocumentSegments(documentId)
        with self._connectIndex() as conn:
            conn.execute("DELETE FROM documents WHERE id=?", (documentId,))
            conn.commit()

    def mergeDocumentMeta(self, documentId: str, extra: dict) -> None:
        with self._connectIndex() as conn:
            row = conn.execute("SELECT metadata FROM documents WHERE id=?", (documentId,)).fetchone()
            meta = json.loads((row["metadata"] if row else None) or "{}")
            meta.update(extra)
            conn.execute("UPDATE documents SET metadata=? WHERE id=?", (json.dumps(meta), documentId))
            conn.commit()

    def purgeNamespace(self, namespaceId: str) -> None:
        """Delete every row a namespace owns (index, notes, chats, queue). Vault files are separate."""
        with self._connectIndex() as conn:
            conn.execute(
                "DELETE FROM segments_fts WHERE rowid IN (SELECT rowid FROM segments WHERE namespaceId=?)",
                (namespaceId,),
            )
            conn.execute(
                "DELETE FROM note_links WHERE fromId IN (SELECT kormemId FROM notes WHERE namespaceId=?)",
                (namespaceId,),
            )
            for table in ("segments", "documents", "sources", "notes", "chat_sessions", "chat_turns",
                          "extraction_jobs", "extracted_sections", "vault_folders"):
                conn.execute(f"DELETE FROM {table} WHERE namespaceId=?", (namespaceId,))
            conn.commit()

    # ---------------------------------------------------------- maintenance

    def deleteSegmentsLike(self, namespaceId: str, keyPattern: str) -> int:
        """Delete segments whose key matches a LIKE pattern (e.g. a retired memory line's vector)."""
        with self._connectIndex() as conn:
            conn.execute(
                "DELETE FROM segments_fts WHERE rowid IN "
                "(SELECT rowid FROM segments WHERE namespaceId=? AND segmentKey LIKE ?)",
                (namespaceId, keyPattern),
            )
            n = conn.execute("DELETE FROM segments WHERE namespaceId=? AND segmentKey LIKE ?",
                             (namespaceId, keyPattern)).rowcount
            conn.commit()
            return n

    def deleteMemoryVectors(self, namespaceId: str, textKeys: set) -> int:
        """Drop memory-line vectors (keys ``mem:{ns}:{model}:{textKey}``) for the given text keys, any model."""
        if not textKeys:
            return 0
        with self._connectIndex() as conn:
            keys = [r[0] for r in conn.execute(
                "SELECT segmentKey FROM segments WHERE namespaceId=? AND kind='memory'", (namespaceId,))
                if r[0].rsplit(":", 1)[-1] in textKeys]
            for i in range(0, len(keys), 500):
                part = keys[i:i + 500]
                conn.execute(f"DELETE FROM segments WHERE segmentKey IN ({','.join('?' * len(part))})", part)
            conn.commit()
            return len(keys)

    def deleteNoteIndex(self, namespaceId: str, kormemId: str) -> None:
        """Forget a note: its index row, its outbound links and its vector."""
        with self._connectIndex() as conn:
            conn.execute("DELETE FROM note_links WHERE fromId=?", (kormemId,))
            conn.execute("DELETE FROM notes WHERE namespaceId=? AND kormemId=?", (namespaceId, kormemId))
            conn.execute("DELETE FROM segments_fts WHERE rowid IN (SELECT rowid FROM segments WHERE segmentKey=?)",
                         (f"note:{kormemId}",))
            conn.execute("DELETE FROM segments WHERE segmentKey=?", (f"note:{kormemId}",))
            conn.commit()

    def repointLinks(self, fromId: str, toId: str) -> None:
        """Links that resolved to a merged-away note now resolve to the note it merged into."""
        with self._connectIndex() as conn:
            conn.execute("UPDATE note_links SET toId=? WHERE toId=?", (toId, fromId))
            conn.commit()

    def segmentRows(self, namespaceId: str, kind: Optional[str] = None,
                    columns: str = "*") -> List[sqlite3.Row]:
        sql = f"SELECT {columns} FROM segments WHERE namespaceId=?"
        params: List[Any] = [namespaceId]
        if kind:
            sql += " AND kind=?"
            params.append(kind)
        with self._connectIndex() as conn:
            return conn.execute(sql, params).fetchall()

    def rewriteSegments(self, updates: List[Dict[str, Any]]) -> None:
        """Bulk update of ``embedding``/``embeddingDim``/``plainText``/``encryptedText``/``metadata`` by key."""
        cols = ("embedding", "embeddingDim", "plainText", "encryptedText", "metadata")
        with self._connectIndex() as conn:
            for u in updates:
                sets = [c for c in cols if c in u]
                if sets:
                    conn.execute(
                        f"UPDATE segments SET {', '.join(f'{c}=?' for c in sets)} WHERE segmentKey=?",
                        [json.dumps(u[c]) if c == "metadata" else u[c] for c in sets] + [u["segmentKey"]],
                    )
            conn.commit()

    def vacuum(self) -> None:
        """Give freed pages back to the file system (slow on a large index)."""
        conn = self._connectIndex()
        try:
            conn.execute("VACUUM")
        finally:
            conn.close()

    def listNamespaces(self, prefix: str = "") -> List[str]:
        with self._connectIndex() as conn:
            return [r[0] for r in conn.execute(
                "SELECT DISTINCT namespaceId FROM documents WHERE namespaceId LIKE ? ORDER BY 1", (f"{prefix}%",))]

    def foldKeywordIndex(self, conn: sqlite3.Connection) -> int:
        """One-time migration: older versions kept one keyword row per (segment, keyword) under arbitrary
        rowids, plus a JSON copy on the segment. Rebuild as one row per segment under the segment's rowid."""
        if conn.execute("PRAGMA user_version").fetchone()[0] >= 1:
            return 0
        if conn.execute("SELECT 1 FROM segments_fts LIMIT 1").fetchone():
            logger.info("folding the keyword index to one row per segment (one-time)")
            conn.executescript("""
                DROP TABLE IF EXISTS segments_fts_new;
                CREATE VIRTUAL TABLE segments_fts_new USING fts5(segmentKey UNINDEXED, keywordHash, tokenize='porter');
                INSERT INTO segments_fts_new (rowid, segmentKey, keywordHash)
                    SELECT s.rowid, f.segmentKey, group_concat(f.keywordHash, ' ')
                    FROM segments_fts f JOIN segments s ON s.segmentKey = f.segmentKey
                    GROUP BY f.segmentKey;
                DROP TABLE segments_fts;
                ALTER TABLE segments_fts_new RENAME TO segments_fts;
                UPDATE segments SET keywords = NULL WHERE keywords IS NOT NULL;
            """)
        conn.execute("PRAGMA user_version = 1")
        return 1
