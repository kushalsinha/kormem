"""SQLite-backed memory store with FTS5 keyword search and vector storage."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from kormem.core.config import getGuidancePath, getIndexPath
from kormem.core.ids import generateDocumentId, generateSegmentKey, generateSourceId
from kormem.store.memory_tables import MemoryTablesMixin

if TYPE_CHECKING:
    from kormem.vault.note import Note

NOTE_SENTINEL = "__note__"
# Rows still waiting for a vector. embeddingDim 0 marks a vector dropped on purpose (old chunks,
# see kormem.extract.maintenance): those stay keyword-searchable and are never re-embedded.
UNEMBEDDED = "embedding IS NULL AND COALESCE(embeddingDim, -1) != 0"


def _packEmbedding(vec: List[float]) -> bytes:
    """Embeddings are stored as int8 plus one float32 scale: a quarter of float32, under 1% ranking loss."""
    v = np.asarray(vec, dtype=np.float32)
    scale = float(np.abs(v).max()) / 127.0 or 1.0
    return np.float32(scale).tobytes() + np.round(v / scale).astype(np.int8).tobytes()


def _unpackEmbedding(blob: bytes, dim: Optional[int] = None) -> np.ndarray:
    """Read an int8 blob, or a float16 / float32 one written by older versions (told apart by ``dim``)."""
    if dim and len(blob) == 4 * dim:
        return np.frombuffer(blob, dtype=np.float32).astype(np.float32)
    if dim and len(blob) == 2 * dim:
        return np.frombuffer(blob, dtype=np.float16).astype(np.float32)
    scale = np.frombuffer(blob[:4], dtype=np.float32)[0]
    return np.frombuffer(blob[4:], dtype=np.int8).astype(np.float32) * scale


class SqliteStore(MemoryTablesMixin):
    """Primary DAL for segments, documents, sources, jobs, and guidance."""

    def __init__(self, indexPath: Optional[Path] = None, guidancePath: Optional[Path] = None):
        self.indexPath = indexPath or getIndexPath()
        self.guidancePath = guidancePath or getGuidancePath()
        self.indexPath.parent.mkdir(parents=True, exist_ok=True)
        self._initIndexDb()
        self._migrateIndexDb()
        self._initGuidanceDb()
        self._migrateFromGuidance()

    def _connectIndex(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.indexPath), check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        # WAL + busy_timeout lets concurrent extraction (writer) and server
        # (reader) share this file without "database is locked" errors.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _connectGuidance(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.guidancePath), check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _initIndexDb(self) -> None:
        with self._connectIndex() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS namespaces (
                    id TEXT PRIMARY KEY,
                    createdAt TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS collections (
                    id TEXT PRIMARY KEY,
                    namespaceId TEXT NOT NULL,
                    name TEXT NOT NULL,
                    path TEXT,
                    pattern TEXT,
                    createdAt TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sources (
                    id TEXT PRIMARY KEY,
                    namespaceId TEXT NOT NULL,
                    collectionId TEXT,
                    platform TEXT NOT NULL,
                    url TEXT,
                    sourceType TEXT,
                    syncStatus TEXT DEFAULT 'idle',
                    metadata TEXT,
                    createdAt TEXT NOT NULL,
                    updatedAt TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    namespaceId TEXT NOT NULL,
                    sourceId TEXT NOT NULL,
                    collectionId TEXT,
                    title TEXT,
                    canonicalUrl TEXT,
                    platform TEXT,
                    contentHash TEXT,
                    encryptedData TEXT,
                    isEncrypted INTEGER DEFAULT 0,
                    metadata TEXT,
                    createdAt TEXT NOT NULL,
                    updatedAt TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS segments (
                    segmentKey TEXT PRIMARY KEY,
                    namespaceId TEXT NOT NULL,
                    documentId TEXT NOT NULL,
                    sourceId TEXT NOT NULL,
                    collectionId TEXT,
                    segmentIndex INTEGER NOT NULL,
                    embedding BLOB,
                    embeddingDim INTEGER,
                    encryptedText TEXT,
                    plainText TEXT,
                    keywords TEXT,
                    metadata TEXT,
                    kind TEXT DEFAULT 'segment',
                    createdAt TEXT NOT NULL,
                    updatedAt TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    jobType TEXT NOT NULL,
                    status TEXT NOT NULL,
                    namespaceId TEXT NOT NULL,
                    sourceId TEXT,
                    payload TEXT,
                    lastError TEXT,
                    retryCount INTEGER DEFAULT 0,
                    createdAt TEXT NOT NULL,
                    updatedAt TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_segments_ns ON segments(namespaceId);
                CREATE INDEX IF NOT EXISTS idx_segments_doc ON segments(documentId);
                CREATE INDEX IF NOT EXISTS idx_segments_src ON segments(sourceId);
                CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
                    segmentKey UNINDEXED,
                    keywordHash,
                    tokenize='porter'
                );
                CREATE TABLE IF NOT EXISTS extraction_jobs (
                    id TEXT PRIMARY KEY,
                    namespaceId TEXT NOT NULL,
                    documentId TEXT,
                    status TEXT NOT NULL,
                    textPreview TEXT,
                    lastError TEXT,
                    createdAt TEXT NOT NULL,
                    updatedAt TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS notes (
                    kormemId TEXT PRIMARY KEY,
                    namespaceId TEXT NOT NULL,
                    title TEXT NOT NULL,
                    para TEXT NOT NULL,
                    relPath TEXT NOT NULL,
                    tags TEXT,
                    sources TEXT,
                    contentHash TEXT,
                    updatedAt TEXT
                );
                CREATE TABLE IF NOT EXISTS note_links (
                    fromId TEXT NOT NULL,
                    toTitle TEXT NOT NULL,
                    toId TEXT,
                    PRIMARY KEY (fromId, toTitle)
                );
                CREATE INDEX IF NOT EXISTS idx_notes_ns ON notes(namespaceId);
                CREATE INDEX IF NOT EXISTS idx_notes_title ON notes(namespaceId, title);
                CREATE INDEX IF NOT EXISTS idx_note_links_from ON note_links(fromId);
                CREATE INDEX IF NOT EXISTS idx_extraction_pending ON extraction_jobs(status);
                """
            )
            conn.commit()

    def _migrateIndexDb(self) -> None:
        with self._connectIndex() as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(segments)").fetchall()}
            if "kind" not in cols:
                conn.execute("ALTER TABLE segments ADD COLUMN kind TEXT DEFAULT 'segment'")
            self._initMemoryTables(conn)
            conn.commit()
            self.foldKeywordIndex(conn)
            conn.commit()

    def _migrateFromGuidance(self) -> None:
        marker = self.indexPath.parent / ".guidance_migrated"
        if marker.exists() or not self.guidancePath.exists():
            return
        try:
            from kormem.vault.vault_store import VaultStore

            with self._connectGuidance() as gconn:
                rows = gconn.execute("SELECT * FROM namespace_guidance").fetchall()
                pending = gconn.execute(
                    "SELECT * FROM extraction_jobs WHERE status='pending'"
                ).fetchall()
            for row in rows:
                ns = row["namespaceId"]
                vault = VaultStore(store=self, namespaceId=ns)
                vault.ensureMetaNotes()
                mapping = {
                    "persona": row["personaMd"],
                    "memory": row["memoryMd"],
                    "playbooks": row["playbooksMd"],
                }
                for name, content in mapping.items():
                    if not content:
                        continue
                    note = vault.readNote(f"_meta/{name}.md")
                    if note:
                        note.body = content.strip()
                        vault.writeNote(note)
            with self._connectIndex() as conn:
                for job in pending:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO extraction_jobs
                        (id, namespaceId, documentId, status, textPreview, createdAt, updatedAt)
                        VALUES (?, ?, ?, 'pending', ?, ?, ?)
                        """,
                        (
                            job["id"],
                            job["namespaceId"],
                            job["documentId"],
                            job["textPreview"],
                            job["createdAt"],
                            job["updatedAt"],
                        ),
                    )
                conn.commit()
            marker.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
        except Exception:
            pass

    def _initGuidanceDb(self) -> None:
        with self._connectGuidance() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS namespace_guidance (
                    namespaceId TEXT PRIMARY KEY,
                    personaMd TEXT,
                    memoryMd TEXT,
                    playbooksMd TEXT,
                    skillsMd TEXT,
                    updatedAt TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS extraction_jobs (
                    id TEXT PRIMARY KEY,
                    namespaceId TEXT NOT NULL,
                    documentId TEXT,
                    status TEXT NOT NULL,
                    textPreview TEXT,
                    lastError TEXT,
                    createdAt TEXT NOT NULL,
                    updatedAt TEXT NOT NULL
                );
                """
            )
            conn.commit()

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def ensureNamespace(self, namespaceId: str) -> None:
        with self._connectIndex() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO namespaces (id, createdAt) VALUES (?, ?)",
                (namespaceId, self._now()),
            )
            conn.commit()

    def upsertCollection(
        self,
        namespaceId: str,
        name: str,
        path: str,
        pattern: str = "**/*",
    ) -> str:
        self.ensureNamespace(namespaceId)
        collId = f"{namespaceId}:{name}"
        now = self._now()
        with self._connectIndex() as conn:
            conn.execute(
                """
                INSERT INTO collections (id, namespaceId, name, path, pattern, createdAt)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET path=excluded.path, pattern=excluded.pattern
                """,
                (collId, namespaceId, name, path, pattern, now),
            )
            conn.commit()
        return collId

    def createSource(
        self,
        namespaceId: str,
        platform: str,
        url: Optional[str] = None,
        collectionId: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        sourceId = generateSourceId()
        now = self._now()
        with self._connectIndex() as conn:
            conn.execute(
                """
                INSERT INTO sources (id, namespaceId, collectionId, platform, url, metadata, createdAt, updatedAt)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sourceId,
                    namespaceId,
                    collectionId,
                    platform,
                    url,
                    json.dumps(metadata or {}),
                    now,
                    now,
                ),
            )
            conn.commit()
        return sourceId

    def createDocument(
        self,
        namespaceId: str,
        sourceId: str,
        title: Optional[str] = None,
        canonicalUrl: Optional[str] = None,
        platform: str = "text",
        collectionId: Optional[str] = None,
        contentHash: Optional[str] = None,
        encryptedData: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        docId = generateDocumentId()
        now = self._now()
        with self._connectIndex() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    id, namespaceId, sourceId, collectionId, title, canonicalUrl, platform,
                    contentHash, encryptedData, isEncrypted, metadata, createdAt, updatedAt
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    docId,
                    namespaceId,
                    sourceId,
                    collectionId,
                    title,
                    canonicalUrl,
                    platform,
                    contentHash,
                    encryptedData,
                    1 if encryptedData else 0,
                    json.dumps(metadata or {}),
                    now,
                    now,
                ),
            )
            conn.commit()
        return docId

    def upsertSegments(
        self,
        namespaceId: str,
        documentId: str,
        sourceId: str,
        segments: List[dict[str, Any]],
        collectionId: Optional[str] = None,
    ) -> List[str]:
        keys: List[str] = []
        now = self._now()
        with self._connectIndex() as conn:
            for seg in segments:
                key = seg["segmentKey"]
                keys.append(key)
                kwJson = None  # keyword search reads segments_fts only; the scored list is not kept
                emb = seg.get("embedding")
                if isinstance(emb, list):
                    emb = _packEmbedding(emb)
                kind = seg.get("kind") or "segment"
                conn.execute(
                    """
                    INSERT INTO segments (
                        segmentKey, namespaceId, documentId, sourceId, collectionId,
                        segmentIndex, embedding, embeddingDim, encryptedText, plainText,
                        keywords, metadata, kind, createdAt, updatedAt
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(segmentKey) DO UPDATE SET
                        embedding=excluded.embedding,
                        embeddingDim=excluded.embeddingDim,
                        encryptedText=excluded.encryptedText,
                        plainText=excluded.plainText,
                        keywords=excluded.keywords,
                        metadata=excluded.metadata,
                        kind=excluded.kind,
                        updatedAt=excluded.updatedAt
                    """,
                    (
                        key,
                        namespaceId,
                        documentId,
                        sourceId,
                        collectionId,
                        seg.get("segmentIndex", 0),
                        emb,
                        seg.get("embeddingDim"),
                        seg.get("encryptedText"),
                        seg.get("plainText"),
                        kwJson,
                        json.dumps(seg.get("metadata") or {}),
                        kind,
                        now,
                        now,
                    ),
                )
                # One keyword row per segment, sharing the segment's rowid (hashes space-separated),
                # so a re-upsert replaces it with an indexed delete instead of a table scan.
                rowid = conn.execute("SELECT rowid FROM segments WHERE segmentKey=?", (key,)).fetchone()[0]
                conn.execute("DELETE FROM segments_fts WHERE rowid=?", (rowid,))
                hashes = list(dict.fromkeys(seg.get("keywordHashes") or []))
                if hashes:
                    conn.execute(
                        "INSERT INTO segments_fts (rowid, segmentKey, keywordHash) VALUES (?, ?, ?)",
                        (rowid, key, " ".join(hashes)),
                    )
            conn.commit()
        return keys

    def upsertNoteSegment(
        self,
        namespaceId: str,
        segment: dict[str, Any],
    ) -> str:
        key = segment["segmentKey"]
        self.upsertSegments(
            namespaceId,
            NOTE_SENTINEL,
            NOTE_SENTINEL,
            [segment],
            None,
        )
        return key

    def getSegmentsPendingEmbed(self, limit: int = 100) -> List[sqlite3.Row]:
        with self._connectIndex() as conn:
            cur = conn.execute(
                f"SELECT * FROM segments WHERE {UNEMBEDDED} LIMIT ?",
                (limit,),
            )
            return cur.fetchall()

    def getStoredEmbeddingDims(self) -> List[int]:
        with self._connectIndex() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT embeddingDim FROM segments
                WHERE embedding IS NOT NULL AND embeddingDim IS NOT NULL
                """
            ).fetchall()
        return [int(r["embeddingDim"]) for r in rows if r["embeddingDim"] is not None]

    def resetEmbeddings(self) -> int:
        """Drop every vector so the background worker re-embeds with the current model.

        Memory-line vectors are keyed by model and recomputed on use, so their rows go.
        Vectors dropped on purpose (``embeddingDim`` 0) stay dropped.
        """
        with self._connectIndex() as conn:
            conn.execute("DELETE FROM segments_fts WHERE rowid IN (SELECT rowid FROM segments WHERE kind='memory')")
            conn.execute("DELETE FROM segments WHERE kind='memory'")
            n = conn.execute(
                "UPDATE segments SET embedding=NULL, embeddingDim=NULL WHERE embedding IS NOT NULL"
            ).rowcount
            conn.execute("UPDATE vault_folders SET embedding=NULL WHERE embedding IS NOT NULL")
            conn.commit()
        return n

    def segmentsWithoutVectors(self, limit: int, namespaceId: Optional[str] = None) -> List[sqlite3.Row]:
        sql = f"SELECT * FROM segments WHERE {UNEMBEDDED}"
        params: List[Any] = []
        if namespaceId:
            sql += " AND namespaceId=?"
            params.append(namespaceId)
        with self._connectIndex() as conn:
            return conn.execute(sql + " LIMIT ?", (*params, limit)).fetchall()

    def markEmbeddingDropped(self, segmentKey: str) -> None:
        with self._connectIndex() as conn:
            conn.execute("UPDATE segments SET embeddingDim=0 WHERE segmentKey=?", (segmentKey,))
            conn.commit()

    def foldersWithoutVectors(self) -> List[sqlite3.Row]:
        with self._connectIndex() as conn:
            return conn.execute(
                "SELECT * FROM vault_folders WHERE status='active' AND embedding IS NULL"
            ).fetchall()

    def setFolderEmbedding(self, namespaceId: str, path: str, embedding: List[float]) -> None:
        with self._connectIndex() as conn:
            conn.execute("UPDATE vault_folders SET embedding=? WHERE namespaceId=? AND path=?",
                         (json.dumps(list(map(float, embedding))), namespaceId, path))
            conn.commit()

    def updateSegmentEmbedding(self, segmentKey: str, embedding: List[float]) -> None:
        blob = _packEmbedding(embedding)
        with self._connectIndex() as conn:
            conn.execute(
                """
                UPDATE segments SET embedding=?, embeddingDim=?, updatedAt=?
                WHERE segmentKey=?
                """,
                (blob, len(embedding), self._now(), segmentKey),
            )
            conn.commit()

    def getSegmentText(self, row: sqlite3.Row, encryptionKey: Optional[str] = None) -> str:
        """A segment's text: stored on the row (old rows, tiny texts), else read from where it lives.

        Chunks keep only their ``span`` in the document; notes live in the vault.
        """
        if row["plainText"]:
            return row["plainText"]
        if row["encryptedText"] and encryptionKey:
            from kormem.core.encryption import decryptText
            payload = json.loads(row["encryptedText"])
            dec = decryptText(payload, encryptionKey)
            return dec or ""
        meta = json.loads(row["metadata"] or "{}")
        if row["kind"] == "note":
            return self._noteBody(row["namespaceId"], meta.get("kormemId") or "")
        span = meta.get("span")
        if span and row["documentId"] != NOTE_SENTINEL:
            return self.documentTextCached(row["documentId"], encryptionKey)[span[0]:span[1]]
        return ""

    def documentTextCached(self, documentId: str, encryptionKey: Optional[str] = None) -> str:
        """Full document text, cached per content version (chunks of one document are read together)."""
        from kormem.ingest.chat_sessions import documentText

        doc = self.getDocument(documentId)
        if doc is None:
            return ""
        cache = self.__dict__.setdefault("_docTextCache", {})
        key = (documentId, doc["contentHash"])
        if key not in cache:
            if len(cache) >= 64:
                cache.pop(next(iter(cache)))
            cache[key] = documentText(self, doc, encryptionKey)
        return cache[key]

    def _noteBody(self, namespaceId: str, kormemId: str) -> str:
        from kormem.vault.note import Note
        from kormem.vault.paths import vaultDir

        row = self.resolveNote(namespaceId, kormemId) if kormemId else None
        path = vaultDir(namespaceId) / row["relPath"] if row else None
        if not path or not path.exists():
            return ""
        return Note.parse(path.read_text(encoding="utf-8")).body

    def vectorSearch(
        self,
        namespaceId: str,
        queryEmbedding: List[float],
        limit: int = 30,
        collectionId: Optional[str] = None,
        minScore: float = 0.0,
        kind: Optional[str] = None,
    ) -> List[Tuple[sqlite3.Row, float]]:
        q = np.array(queryEmbedding, dtype=np.float32)
        qNorm = np.linalg.norm(q)
        if qNorm == 0:
            return []
        q = q / qNorm

        # Vectors from another model size can't be compared (e.g. after switching embedders).
        sql = "SELECT * FROM segments WHERE namespaceId=? AND embedding IS NOT NULL AND embeddingDim=?"
        params: List[Any] = [namespaceId, len(q)]
        if collectionId:
            sql += " AND collectionId=?"
            params.append(collectionId)
        if kind:
            sql += " AND kind=?"
            params.append(kind)

        with self._connectIndex() as conn:
            rows = conn.execute(sql, params).fetchall()

        scored: List[Tuple[sqlite3.Row, float]] = []
        for row in rows:
            emb = _unpackEmbedding(row["embedding"], row["embeddingDim"])
            norm = np.linalg.norm(emb)
            if norm == 0:
                continue
            sim = float(np.dot(q, emb / norm))
            if sim >= minScore:
                scored.append((row, sim))
        scored.sort(key=lambda x: -x[1])
        return scored[:limit]

    def keywordSearch(
        self,
        namespaceId: str,
        query: str,
        limit: int = 30,
        collectionId: Optional[str] = None,
        kind: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        from kormem.core.keywords import queryKeywordGroups
        groups = queryKeywordGroups(query)
        if not groups:
            return []
        scores: dict[str, float] = {}
        with self._connectIndex() as conn:
            for group in groups:
                # One query word and its plural / tense forms; a chunk scores once per word.
                sql = """
                    SELECT s.segmentKey, s.kind FROM segments_fts f
                    JOIN segments s ON s.segmentKey = f.segmentKey
                    WHERE segments_fts MATCH ? AND s.namespaceId = ?
                """
                params: List[Any] = [" OR ".join(f'keywordHash:"{h}"' for h in group), namespaceId]
                if kind:
                    sql += " AND s.kind=?"
                    params.append(kind)
                for key in {row["segmentKey"] for row in conn.execute(sql, params).fetchall()}:
                    scores[key] = scores.get(key, 0.0) + 1.0
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:limit]
        return ranked

    def getSegmentsByKeys(self, keys: List[str]) -> List[sqlite3.Row]:
        if not keys:
            return []
        placeholders = ",".join("?" * len(keys))
        with self._connectIndex() as conn:
            return list(
                conn.execute(
                    f"SELECT * FROM segments WHERE segmentKey IN ({placeholders})",
                    keys,
                ).fetchall()
            )

    def listSources(self, namespaceId: str) -> List[sqlite3.Row]:
        with self._connectIndex() as conn:
            return list(
                conn.execute(
                    "SELECT * FROM sources WHERE namespaceId=? ORDER BY createdAt DESC",
                    (namespaceId,),
                ).fetchall()
            )

    def getStatus(self, namespaceId: str) -> dict[str, int]:
        with self._connectIndex() as conn:
            seg = conn.execute(
                "SELECT COUNT(*) as c FROM segments WHERE namespaceId=?",
                (namespaceId,),
            ).fetchone()["c"]
            doc = conn.execute(
                "SELECT COUNT(*) as c FROM documents WHERE namespaceId=?",
                (namespaceId,),
            ).fetchone()["c"]
            src = conn.execute(
                "SELECT COUNT(*) as c FROM sources WHERE namespaceId=?",
                (namespaceId,),
            ).fetchone()["c"]
            pending = conn.execute(
                f"SELECT COUNT(*) as c FROM segments WHERE namespaceId=? AND {UNEMBEDDED}",
                (namespaceId,),
            ).fetchone()["c"]
            extraction = conn.execute(
                "SELECT COUNT(*) as c FROM extraction_jobs WHERE namespaceId=? AND status IN ('pending','processing')",
                (namespaceId,),
            ).fetchone()["c"]
            jobs = conn.execute(
                "SELECT COUNT(*) as c FROM jobs WHERE namespaceId=? AND status='pending'",
                (namespaceId,),
            ).fetchone()["c"]
        return {
            "segmentCount": seg,
            "documentCount": doc,
            "sourceCount": src,
            "pendingEmbeds": pending,
            "pendingJobs": jobs,
            "pendingExtractions": extraction,
        }

    def enqueueJob(
        self,
        jobId: str,
        jobType: str,
        namespaceId: str,
        payload: dict,
        sourceId: Optional[str] = None,
    ) -> None:
        now = self._now()
        with self._connectIndex() as conn:
            conn.execute(
                """
                INSERT INTO jobs (id, jobType, status, namespaceId, sourceId, payload, createdAt, updatedAt)
                VALUES (?, ?, 'pending', ?, ?, ?, ?, ?)
                """,
                (jobId, jobType, namespaceId, sourceId, json.dumps(payload), now, now),
            )
            conn.commit()

    def claimJobs(self, limit: int = 5) -> List[sqlite3.Row]:
        with self._connectIndex() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE status='pending' ORDER BY createdAt LIMIT ?",
                (limit,),
            ).fetchall()
            for row in rows:
                conn.execute(
                    "UPDATE jobs SET status='processing', updatedAt=? WHERE id=?",
                    (self._now(), row["id"]),
                )
            conn.commit()
            return rows

    def completeJob(self, jobId: str) -> None:
        with self._connectIndex() as conn:
            conn.execute(
                "UPDATE jobs SET status='completed', updatedAt=? WHERE id=?",
                (self._now(), jobId),
            )
            conn.commit()

    def failJob(self, jobId: str, error: str) -> None:
        with self._connectIndex() as conn:
            conn.execute(
                "UPDATE jobs SET status='failed', lastError=?, updatedAt=? WHERE id=?",
                (error, self._now(), jobId),
            )
            conn.commit()

    def getGuidance(self, namespaceId: str) -> Optional[sqlite3.Row]:
        with self._connectGuidance() as conn:
            return conn.execute(
                "SELECT * FROM namespace_guidance WHERE namespaceId=?",
                (namespaceId,),
            ).fetchone()

    def upsertGuidance(
        self,
        namespaceId: str,
        personaMd: Optional[str] = None,
        memoryMd: Optional[str] = None,
        playbooksMd: Optional[str] = None,
        skillsMd: Optional[str] = None,
    ) -> None:
        existing = self.getGuidance(namespaceId)
        now = self._now()
        with self._connectGuidance() as conn:
            if existing:
                conn.execute(
                    """
                    UPDATE namespace_guidance SET
                        personaMd=COALESCE(?, personaMd),
                        memoryMd=COALESCE(?, memoryMd),
                        playbooksMd=COALESCE(?, playbooksMd),
                        skillsMd=COALESCE(?, skillsMd),
                        updatedAt=?
                    WHERE namespaceId=?
                    """,
                    (personaMd, memoryMd, playbooksMd, skillsMd, now, namespaceId),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO namespace_guidance
                    (namespaceId, personaMd, memoryMd, playbooksMd, skillsMd, updatedAt)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        namespaceId,
                        personaMd or "",
                        memoryMd or "",
                        playbooksMd or "",
                        skillsMd or "",
                        now,
                    ),
                )
            conn.commit()

    def claimExtractionJobs(self, limit: int = 3, namespaceId: Optional[str] = None,
                            excludeKinds: Tuple[str, ...] = ()) -> List[sqlite3.Row]:
        sql = "SELECT * FROM extraction_jobs WHERE status='pending'"
        params: List[Any] = []
        if namespaceId:
            sql += " AND namespaceId=?"
            params.append(namespaceId)
        if excludeKinds:
            sql += f" AND COALESCE(kind, 'document') NOT IN ({','.join('?' * len(excludeKinds))})"
            params += list(excludeKinds)
        with self._connectIndex() as conn:
            rows = conn.execute(sql + " ORDER BY createdAt LIMIT ?", (*params, limit)).fetchall()
            for row in rows:
                conn.execute(
                    "UPDATE extraction_jobs SET status='processing', updatedAt=? WHERE id=?",
                    (self._now(), row["id"]),
                )
            conn.commit()
            return rows

    def requeueStaleExtractions(self) -> int:
        """Jobs left 'processing' by a stopped server go back to 'pending'."""
        with self._connectIndex() as conn:
            n = conn.execute(
                "UPDATE extraction_jobs SET status='pending', updatedAt=? WHERE status='processing'",
                (self._now(),),
            ).rowcount
            conn.commit()
            return n

    def setExtractionStatus(self, fromStatus: str, toStatus: str, namespacePrefix: str = "") -> int:
        """Bulk-move extraction jobs between states (e.g. park or resume a backlog)."""
        with self._connectIndex() as conn:
            n = conn.execute(
                "UPDATE extraction_jobs SET status=?, updatedAt=? WHERE status=? AND namespaceId LIKE ?",
                (toStatus, self._now(), fromStatus, f"{namespacePrefix}%"),
            ).rowcount
            conn.commit()
            return n

    def setExtractionJobsStatus(self, jobIds: List[str], status: str) -> None:
        with self._connectIndex() as conn:
            conn.executemany("UPDATE extraction_jobs SET status=?, updatedAt=? WHERE id=?",
                             [(status, self._now(), j) for j in jobIds])
            conn.commit()

    def completeExtraction(self, jobId: str) -> None:
        with self._connectIndex() as conn:
            conn.execute(
                "UPDATE extraction_jobs SET status='completed', updatedAt=? WHERE id=?",
                (self._now(), jobId),
            )
            conn.commit()

    def failExtraction(self, jobId: str, error: str) -> None:
        with self._connectIndex() as conn:
            conn.execute(
                "UPDATE extraction_jobs SET status='failed', lastError=?, updatedAt=? WHERE id=?",
                (error, self._now(), jobId),
            )
            conn.commit()

    def getDocument(self, documentId: str) -> Optional[sqlite3.Row]:
        with self._connectIndex() as conn:
            return conn.execute(
                "SELECT * FROM documents WHERE id=?",
                (documentId,),
            ).fetchone()

    def findDocumentsByTitle(self, namespaceId: str, title: str) -> List[sqlite3.Row]:
        with self._connectIndex() as conn:
            return conn.execute(
                "SELECT * FROM documents WHERE namespaceId=? AND title=? ORDER BY createdAt DESC",
                (namespaceId, title),
            ).fetchall()

    def deleteSegmentsByDocument(self, documentId: str) -> int:
        with self._connectIndex() as conn:
            cur = conn.execute(
                "DELETE FROM segments WHERE documentId=?",
                (documentId,),
            )
            conn.commit()
            return cur.rowcount

    def deleteDocument(self, documentId: str) -> None:
        with self._connectIndex() as conn:
            conn.execute("DELETE FROM documents WHERE id=?", (documentId,))
            conn.commit()

    def docIndexPairsFor(self, segmentKeys: List[str]) -> List[Tuple[str, int]]:
        if not segmentKeys:
            return []
        placeholders = ",".join("?" * len(segmentKeys))
        with self._connectIndex() as conn:
            rows = conn.execute(
                f"""
                SELECT documentId, segmentIndex FROM segments
                WHERE segmentKey IN ({placeholders}) AND kind='segment'
                """,
                segmentKeys,
            ).fetchall()
        return [(r["documentId"], int(r["segmentIndex"])) for r in rows if r["documentId"] != NOTE_SENTINEL]

    def getAdjacentSegments(
        self,
        docIndexPairs: List[Tuple[str, int]],
        window: int = 2,
    ) -> List[dict[str, Any]]:
        if not docIndexPairs or window <= 0:
            return []
        results: List[dict[str, Any]] = []
        seen: set[str] = set()
        with self._connectIndex() as conn:
            for docId, idx in docIndexPairs:
                lo = max(0, idx - window)
                hi = idx + window
                rows = conn.execute(
                    """
                    SELECT segmentKey, documentId, segmentIndex FROM segments
                    WHERE documentId=? AND segmentIndex BETWEEN ? AND ?
                      AND segmentIndex != ? AND kind='segment'
                    """,
                    (docId, lo, hi, idx),
                ).fetchall()
                for row in rows:
                    key = row["segmentKey"]
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append({
                        "segmentKey": key,
                        "documentId": row["documentId"],
                        "segmentIndex": row["segmentIndex"],
                        "parentIndex": idx,
                    })
        return results

    def upsertNoteIndex(
        self,
        namespaceId: str,
        note: "Note",
        relPath: str,
        contentHash: str,
        outboundLinks: List[str],
    ) -> None:
        now = self._now()
        with self._connectIndex() as conn:
            conn.execute(
                """
                INSERT INTO notes (kormemId, namespaceId, title, para, relPath, tags, sources, contentHash,
                                   updatedAt, aliases, folder)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(kormemId) DO UPDATE SET
                    title=excluded.title,
                    para=excluded.para,
                    relPath=excluded.relPath,
                    tags=excluded.tags,
                    sources=excluded.sources,
                    contentHash=excluded.contentHash,
                    updatedAt=excluded.updatedAt,
                    aliases=excluded.aliases,
                    folder=excluded.folder
                """,
                (
                    note.kormemId,
                    namespaceId,
                    note.title,
                    note.para,
                    relPath,
                    json.dumps(note.tags),
                    json.dumps(note.sources),
                    contentHash,
                    note.updated or now,
                    json.dumps([a.lower() for a in getattr(note, "aliases", []) or []]),
                    getattr(note, "folder", "") or "",
                ),
            )
            # Links written before this note existed now resolve to it.
            for name in [note.title, *(getattr(note, "aliases", []) or [])]:
                conn.execute(
                    """
                    UPDATE note_links SET toId=?
                    WHERE toId IS NULL AND lower(toTitle)=lower(?)
                      AND fromId IN (SELECT kormemId FROM notes WHERE namespaceId=?)
                    """,
                    (note.kormemId, name, namespaceId),
                )
            conn.execute("DELETE FROM note_links WHERE fromId=?", (note.kormemId,))
            for title in outboundLinks:
                target = conn.execute(
                    "SELECT kormemId FROM notes WHERE namespaceId=? AND lower(title)=lower(?)",
                    (namespaceId, title),
                ).fetchone()
                toId = target["kormemId"] if target else None
                conn.execute(
                    "INSERT OR REPLACE INTO note_links (fromId, toTitle, toId) VALUES (?, ?, ?)",
                    (note.kormemId, title, toId),
                )
            conn.commit()

    def resolveNote(self, namespaceId: str, ref: str) -> Optional[sqlite3.Row]:
        with self._connectIndex() as conn:
            row = conn.execute(
                "SELECT * FROM notes WHERE namespaceId=? AND kormemId=?",
                (namespaceId, ref),
            ).fetchone()
            if row:
                return row
            row = conn.execute(
                "SELECT * FROM notes WHERE namespaceId=? AND lower(title)=lower(?)",
                (namespaceId, ref),
            ).fetchone()
            if row:
                return row
            row = conn.execute(
                "SELECT * FROM notes WHERE namespaceId=? AND relPath=?",
                (namespaceId, ref),
            ).fetchone()
            if row:
                return row
            # Aliases are stored lower-cased as a JSON array.
            return conn.execute(
                "SELECT * FROM notes WHERE namespaceId=? AND aliases LIKE ?",
                (namespaceId, f'%{json.dumps(ref.lower())}%'),
            ).fetchone()

    def listNoteIndex(
        self,
        namespaceId: str,
        para: Optional[str] = None,
    ) -> List[sqlite3.Row]:
        with self._connectIndex() as conn:
            if para:
                return list(
                    conn.execute(
                        "SELECT * FROM notes WHERE namespaceId=? AND para=? ORDER BY title",
                        (namespaceId, para),
                    ).fetchall()
                )
            return list(
                conn.execute(
                    "SELECT * FROM notes WHERE namespaceId=? ORDER BY para, title",
                    (namespaceId,),
                ).fetchall()
            )

    def getBacklinks(self, namespaceId: str, kormemId: str) -> List[sqlite3.Row]:
        note = self.resolveNote(namespaceId, kormemId)
        if not note:
            return []
        title = note["title"]
        with self._connectIndex() as conn:
            return list(
                conn.execute(
                    """
                    SELECT n.* FROM note_links l
                    JOIN notes n ON n.kormemId = l.fromId
                    WHERE n.namespaceId=? AND (l.toId=? OR lower(l.toTitle)=lower(?))
                    """,
                    (namespaceId, kormemId, title),
                ).fetchall()
            )
