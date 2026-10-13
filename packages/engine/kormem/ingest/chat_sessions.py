"""Chat sessions: index every turn now, extract once the conversation settles.

Turns are stored (encrypted when an encryption key is configured) and the
session's transcript is re-indexed on every append (unchanged chunks keep
their embeddings), so recall sees a turn immediately. The transcript itself
is not stored a second time; :func:`documentText` rebuilds it from the turns.

Extraction, the paid step, runs on the unextracted stretch of a session once
it settles: closed, idle for ``KORMEM_CHAT_IDLE_MINUTES`` (30), or holding
``KORMEM_CHAT_MAX_TURNS`` (40) unextracted turns.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from kormem.core.config import KormemConfig, loadConfig
from kormem.core.document_content import packDocumentContent, unpackDocumentContent
from kormem.ingest.service import IngestService

PLATFORM = "chat_session"
IDLE_MINUTES = float(os.environ.get("KORMEM_CHAT_IDLE_MINUTES", "30"))
MAX_TURNS = int(os.environ.get("KORMEM_CHAT_MAX_TURNS", "40"))


def normalizeRole(role: Optional[str]) -> str:
    return "assistant" if (role or "").lower() in ("assistant", "ai", "model") else "user"


def formatTranscript(turns: List[Dict[str, Any]], sessionDate: Optional[str]) -> str:
    lines = [f"[Session date: {sessionDate}]", ""] if sessionDate else []
    for t in turns:
        content = (t["content"] or "").strip()
        if content:
            lines += [f"[{t['role']}]: {content}", ""]
    return "\n".join(lines)


def chatTurns(store, namespaceId: str, sessionId: str, encryptionKey: Optional[str]) -> List[Dict[str, Any]]:
    """A session's turns with their text decrypted."""
    return [{**dict(r), "content": unpackDocumentContent(r["content"], encryptionKey)}
            for r in store.getChatTurns(namespaceId, sessionId)]


def documentText(store, doc, encryptionKey: Optional[str]) -> str:
    """Full text of any document: stored content, or a chat's transcript rebuilt from its turns."""
    if doc is None:
        return ""
    if doc["encryptedData"]:
        return unpackDocumentContent(doc["encryptedData"], encryptionKey)
    url = doc["canonicalUrl"] or ""
    if doc["platform"] == PLATFORM and url.startswith("chat://"):
        namespaceId, _, sessionId = url[len("chat://"):].partition("/")
        session = store.getChatSession(namespaceId, sessionId)
        return formatTranscript(chatTurns(store, namespaceId, sessionId, encryptionKey),
                                session["sessionDate"] if session else None)
    return ""


class ChatSessionService:
    def __init__(self, ingest: Optional[IngestService] = None, config: Optional[KormemConfig] = None):
        self.config = config or loadConfig()
        self.ingest = ingest or IngestService(config=self.config)
        self.store = self.ingest.store

    async def append(
        self,
        namespaceId: str,
        sessionId: str,
        turns: List[Dict[str, Any]],
        sessionDate: Optional[str] = None,
        collectionId: Optional[str] = None,
        project: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add turns (re-sent turns with the same ``idx`` are ignored) and re-index the transcript.

        ``project`` (e.g. the repository a coding session ran in) lets
        extraction scope the lessons it finds.
        """
        key = self.config.encryptionKey
        clean = [
            {"idx": t.get("idx"), "role": normalizeRole(t.get("role")),
             "content": packDocumentContent(str(t.get("content") or "").strip(), key), "at": t.get("at")}
            for t in turns if str(t.get("content") or "").strip()
        ]
        sessionDate = sessionDate or datetime.now(timezone.utc).strftime("%Y/%m/%d (%a) %H:%M")
        count = self.store.appendChatTurns(namespaceId, sessionId, clean, sessionDate)
        session = self.store.getChatSession(namespaceId, sessionId)
        text = formatTranscript(chatTurns(self.store, namespaceId, sessionId, key), session["sessionDate"])
        if session["documentId"]:
            await self.ingest.updateText(session["documentId"], text, storeContent=False)
            documentId = session["documentId"]
        else:
            result = await self.ingest.ingestText(
                content=text, namespaceId=namespaceId, collectionId=collectionId,
                sourceUrl=f"chat://{namespaceId}/{sessionId}", title=f"chat {sessionId}",
                platform=PLATFORM, kind="chat", sourceDate=session["sessionDate"], enqueue=False,
                storeContent=False, metadata={"project": project} if project else None,
            )
            documentId = result["documentId"]
            self.store.setChatSessionDocument(namespaceId, sessionId, documentId)
        return {"sessionId": sessionId, "documentId": documentId, "turns": count}

    def close(self, namespaceId: str, sessionId: str) -> None:
        """Mark a session ended so its last stretch is extracted on the next worker pass."""
        self.store.closeChatSession(namespaceId, sessionId)


def enqueueSettledChats(store) -> int:
    """Queue an extraction job for every settled chat stretch; returns how many."""
    settled = store.settleChatSessions(IDLE_MINUTES, MAX_TURNS)
    for s in settled:
        store.enqueueExtractionJob(
            str(uuid.uuid4()), s["namespaceId"], "chat", documentId=s["documentId"],
            spanStart=s["spanStart"], spanEnd=s["spanEnd"], payload={"sessionId": s["sessionId"]},
        )
    return len(settled)
