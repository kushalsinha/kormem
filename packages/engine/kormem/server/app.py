"""FastAPI local memory daemon."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException

from kormem import __version__
from kormem.connectors.service import ConnectorService
from kormem.core.config import apiKeysSet, ensureKormemLayout, getKormemHome, loadConfig, loadEnvFile
from kormem.core.models import snapshot as modelSnapshot
from kormem.embeddings.reembed import reembedPending, switchIfNeeded
from kormem import library
from kormem.core.schemas import (
    CollectionAddRequest,
    CollectionRequest,
    ContextEditRequest,
    ContextRequest,
    GetRequest,
    MultiGetRequest,
    UpdateRequest,
    DocumentReadRequest,
    ChatAppendRequest,
    ChatCloseRequest,
    ExtractChatRequest,
    ExtractQueueRequest,
    ExtractRunRequest,
    InitRequest,
    MaintainRequest,
    IngestPathRequest,
    IngestSessionRequest,
    IngestTextRequest,
    IngestUrlRequest,
    NoteReadRequest,
    RecallRequest,
    SearchRequest,
    SearchResponse,
    StatusResponse,
)
from kormem.extract.lessons import lessonsFor
from kormem.extract.maintenance import maintainAll
from kormem.ingest.chat_sessions import ChatSessionService, documentText
from kormem.ingest.service import IngestService
from kormem.retrieve.recall import RecallPipeline
from kormem.retrieve.rag import RagService
from kormem.embeddings.provider import describeActiveProvider
from kormem.vault.vault_store import VaultStore
from kormem.worker.ingestion_worker import IngestionWorker


logger = logging.getLogger(__name__)

# Seconds between queue polls when the extraction queue is empty.
WORKER_IDLE_SECONDS = float(os.environ.get("KORMEM_WORKER_IDLE_SECONDS", "2"))
# Hours between maintenance passes over every namespace (0 = only on request).
MAINTAIN_HOURS = float(os.environ.get("KORMEM_MAINTAIN_HOURS", "24"))


def createApp() -> FastAPI:
    loadEnvFile()
    ensureKormemLayout()
    config = loadConfig()
    ingest = IngestService(config=config)
    switchIfNeeded(ingest.store, config)
    rag = RagService(config=config)
    connectors = ConnectorService(ingest=ingest, config=config)
    worker = IngestionWorker()
    workerLock = asyncio.Lock()

    maintainedAt = getKormemHome() / "maintained_at"

    async def _maintainIfDue() -> None:
        """Run the maintenance pass when the queue is idle and the last pass is old enough."""
        import time

        if MAINTAIN_HOURS <= 0:
            return
        last = maintainedAt.stat().st_mtime if maintainedAt.exists() else 0
        if time.time() - last < MAINTAIN_HOURS * 3600:
            return
        maintainedAt.touch()
        async with workerLock:
            for report in await maintainAll(ingest.store, config, ingest.store.listNamespaces()):
                logger.info("maintenance: %s", report)

    reembedRetryAt = 0.0

    async def _reembed() -> int:
        """Give vectors to segments without one (after a provider switch, or a failed embed)."""
        nonlocal reembedRetryAt
        import time

        if time.time() < reembedRetryAt:
            return 0
        try:
            async with workerLock:
                return await reembedPending(ingest.store, config)
        except Exception:
            logger.exception("re-embedding failed; retrying in 5 minutes")
            reembedRetryAt = time.time() + 300
            return 0

    async def _workerLoop() -> None:
        """Drain ingestion, extraction and re-embedding for as long as the server runs."""
        while True:
            try:
                async with workerLock:
                    n = await worker.processBatch()
                if not n:
                    n = await _reembed()
                if not n:
                    await _maintainIfDue()
            except Exception:
                logger.exception("background worker batch failed")
                n = 0
            await asyncio.sleep(0 if n else WORKER_IDLE_SECONDS)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        ingest.store.ensureNamespace(config.defaultNamespace)
        VaultStore(store=ingest.store, config=config, namespaceId=config.defaultNamespace).ensureMetaNotes()
        requeued = ingest.store.requeueStaleExtractions()
        if requeued:
            logger.info("requeued %d extraction jobs left in progress", requeued)
        task = asyncio.create_task(_workerLoop())
        try:
            yield
        finally:
            task.cancel()

    app = FastAPI(title="Kormem Engine", version=__version__, lifespan=lifespan)
    chats = ChatSessionService(ingest=ingest, config=config)
    recallPipeline = RecallPipeline(store=ingest.store, config=config)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "keys": apiKeysSet(), "models": modelSnapshot(config)}

    @app.get("/status")
    async def status(namespaceId: Optional[str] = None) -> StatusResponse:
        ns = namespaceId or config.defaultNamespace
        stats = ingest.store.getStatus(ns)
        return StatusResponse(
            home=str(getKormemHome()),
            namespaceId=ns,
            engineVersion=__version__,
            **stats,
        )

    @app.post("/ingest/text")
    async def ingestText(req: IngestTextRequest) -> dict[str, Any]:
        return await ingest.ingestText(
            content=req.content,
            namespaceId=req.namespaceId,
            collectionId=req.collectionId,
            sourceUrl=req.sourceUrl,
            title=req.title,
            platform=req.platform,
            role=req.role,
        )

    @app.post("/ingest/url")
    async def ingestUrl(req: IngestUrlRequest) -> dict[str, Any]:
        return await connectors.ingestUrl(
            url=req.url,
            namespaceId=req.namespaceId,
            collectionId=req.collectionId,
        )

    @app.post("/ingest/path")
    async def ingestPath(req: IngestPathRequest) -> list[dict[str, Any]]:
        return await connectors.ingestPath(
            path=req.path,
            namespaceId=req.namespaceId,
            collectionId=req.collectionId,
            recursive=req.recursive,
        )

    @app.post("/chat/append")
    async def chatAppend(req: ChatAppendRequest) -> dict[str, Any]:
        """Add turns to a live chat; indexed now, extracted once the chat settles."""
        result = await chats.append(req.namespaceId, req.sessionId, req.turns, req.sessionDate,
                                    project=req.project)
        if req.close:
            chats.close(req.namespaceId, req.sessionId)
        return result

    @app.post("/chat/close")
    async def chatClose(req: ChatCloseRequest) -> dict[str, str]:
        chats.close(req.namespaceId, req.sessionId)
        return {"sessionId": req.sessionId, "status": "closed"}

    @app.post("/ingest/session")
    async def ingestSession(req: IngestSessionRequest) -> dict[str, Any]:
        """A complete past session: stored and closed in one call."""
        result = await chats.append(req.namespaceId, req.sessionId, req.turns, req.sessionDate,
                                    req.collectionId)
        chats.close(req.namespaceId, req.sessionId)
        return result

    @app.post("/extract/chat")
    async def extractChat(req: ExtractChatRequest) -> dict[str, Any]:
        """Turns go to the chat session (extracted when it settles); an agent trace is queued for procedures."""
        result = None
        if req.turns:
            result = await chats.append(req.namespaceId, req.sessionId or str(uuid.uuid4()), req.turns)
        if req.trace:
            text = json.dumps(req.trace, default=str)[:60_000]
            if req.scratchpadText:
                text += "\n\n" + req.scratchpadText[:20_000]
            ingest.store.enqueueExtractionJob(str(uuid.uuid4()), req.namespaceId, "trace",
                                              payload={"text": text})
        return {"session": result, "traceQueued": bool(req.trace)}

    @app.post("/extract/run")
    async def extractRun(req: ExtractRunRequest) -> dict[str, Any]:
        """Drain the extraction queue now (all namespaces, or one) and report what is left."""
        processed = 0
        async with workerLock:
            while processed < req.maxJobs:
                n = await worker.extraction.processPending(
                    limit=min(3, req.maxJobs - processed), namespaceId=req.namespaceId,
                )
                if not n:
                    break
                processed += n
        ns = req.namespaceId or config.defaultNamespace
        return {"processed": processed, "pending": ingest.store.getStatus(ns)["pendingExtractions"]}

    @app.post("/maintain")
    async def maintain(req: MaintainRequest) -> dict[str, Any]:
        """Shrink and tidy memory: shard, compact, archive cold notes, merge duplicates, condense."""
        namespaces = [req.namespaceId] if req.namespaceId else ingest.store.listNamespaces(req.namespacePrefix)
        async with workerLock:
            reports = await maintainAll(ingest.store, config, namespaces, condense=req.condense, vacuum=req.vacuum)
        return {"namespaces": len(reports), "reports": reports}

    @app.post("/extract/queue")
    async def extractQueue(req: ExtractQueueRequest) -> dict[str, int]:
        """Park pending extraction jobs (the worker skips them) or resume parked ones."""
        src, dst = ("pending", "parked") if req.action == "park" else ("parked", "pending")
        return {"moved": ingest.store.setExtractionStatus(src, dst, req.namespacePrefix)}

    @app.post("/query/context")
    async def queryContext(req: ContextRequest) -> dict[str, Any]:
        bundle = await rag.buildContext(req)
        brief = bundle.guidanceBrief
        return {
            "query": bundle.query,
            "queryDate": bundle.queryDate,
            "contextOrdering": bundle.contextOrdering,
            "behavioralGuidance": brief.behavioralGuidance,
            "relevantMemory": brief.relevantMemory,
            "relevantSkills": brief.relevantSkills,
            "tacticalPlaybooks": brief.tacticalPlaybooks,
            "knowledge": [k.model_dump() for k in bundle.knowledge],
            "guidance": [g.model_dump() for g in bundle.guidance],
        }

    @app.post("/recall")
    async def recall(req: RecallRequest) -> dict[str, Any]:
        """Notes, sources and findings for a question: free first pass, then the recall agent."""
        ctx = await recallPipeline.recall(req.query, req.namespaceId, req.queryDate, req.effort)
        vault = VaultStore(store=ingest.store, config=config, namespaceId=req.namespaceId)
        cols, contexts = library.collections(), library.contexts()
        for s in ctx.sources:
            doc = ingest.store.getDocument(s.get("documentId") or "")
            if doc is not None:
                s["uri"] = library.documentUri(doc, cols)
                s["docid"] = library.docid(doc["id"])
                s["context"] = library.contextFor(s["uri"], contexts)
        # Findings cite per-recall ids (S12); swap in docids, which `get` can open later.
        sidToDocid = {s["sid"]: s["docid"] for s in ctx.sources if s.get("docid")}
        findings = [re.sub(r"\bS(\d+)\b", lambda m: sidToDocid.get(m.group(0), m.group(0)), f)
                    for f in ctx.findings]
        return {
            "relevantMemory": ctx.relevantMemory,
            "lessons": lessonsFor(vault, req.project),
            "notes": ctx.notes,
            "sources": ctx.sources,
            "findings": findings,
            "debug": ctx.debug,
        }

    @app.get("/lessons")
    async def lessons(namespaceId: Optional[str] = None, project: Optional[str] = None) -> dict[str, str]:
        """Global lessons plus one project's: what to load at the start of a session."""
        vault = VaultStore(store=ingest.store, config=config, namespaceId=namespaceId or config.defaultNamespace)
        return {"lessons": lessonsFor(vault, project)}

    @app.post("/embed")
    async def embed(namespaceId: Optional[str] = None, force: bool = False) -> dict[str, int]:
        async with workerLock:
            n = await ingest.embedPending(namespaceId, force=force)
        return {"embedded": n}

    @app.get("/embeddings/status")
    async def embeddingsStatus() -> dict:
        active = describeActiveProvider(config)
        storedDims = ingest.store.getStoredEmbeddingDims()
        warnings: list[str] = []
        if storedDims and config.embeddingDim not in storedDims:
            warnings.append(
                f"Config embeddingDim={config.embeddingDim} but index has dims {storedDims}. "
                "Run `kormem embed -f` to re-embed."
            )
        if storedDims and len(storedDims) > 1:
            warnings.append(f"Mixed embedding dimensions in index: {storedDims}")
        return {
            **active,
            "storedDims": storedDims,
            "warnings": warnings,
            "rerankEnabled": config.rerankEnabled,
            "rerankModel": config.rerankModel,
        }

    @app.post("/search", response_model=SearchResponse)
    async def search(req: SearchRequest) -> SearchResponse:
        res = await rag.search(req)
        cols, ctx = library.collections(), library.contexts()
        docs: dict[str, Any] = {}
        for k in res.knowledge:
            if not k.documentId:
                continue
            if k.documentId not in docs:
                docs[k.documentId] = ingest.store.getDocument(k.documentId)
            doc = docs[k.documentId]
            if doc is None:
                continue
            uri = library.documentUri(doc, cols)
            k.metadata.update({"docid": library.docid(doc["id"]), "uri": uri,
                               "context": library.contextFor(uri, ctx)})
        return res

    @app.post("/notes/read")
    async def readNote(req: NoteReadRequest) -> dict[str, Any]:
        vault = VaultStore(store=ingest.store, config=config, namespaceId=req.namespaceId)
        note = vault.readNote(req.ref)
        if not note:
            raise HTTPException(status_code=404, detail="Note not found")
        return {
            "kormemId": note.kormemId,
            "title": note.title,
            "para": note.para,
            "tags": note.tags,
            "sources": note.sources,
            "body": note.body,
            "created": note.created,
            "updated": note.updated,
            "links": note.outboundLinks(),
        }

    @app.post("/notes/links")
    async def noteLinks(req: NoteReadRequest) -> dict[str, Any]:
        vault = VaultStore(store=ingest.store, config=config, namespaceId=req.namespaceId)
        return vault.getLinks(req.ref)

    @app.post("/notes/moc")
    async def noteMoc(namespaceId: str = "main") -> dict[str, str]:
        vault = VaultStore(store=ingest.store, config=config, namespaceId=namespaceId)
        return {"content": vault.getMoc()}

    @app.post("/notes/list")
    async def noteList(namespaceId: str = "main", para: Optional[str] = None) -> list[dict]:
        vault = VaultStore(store=ingest.store, config=config, namespaceId=namespaceId)
        return [
            {
                "kormemId": n.kormemId,
                "title": n.title,
                "para": n.para,
                "relPath": n.relPath,
                "updatedAt": n.updatedAt,
            }
            for n in vault.listNotes(para)
        ]

    @app.post("/documents/read")
    async def readDocument(req: DocumentReadRequest) -> dict[str, Any]:
        doc = ingest.store.getDocument(req.documentId)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        text = documentText(ingest.store, doc, config.encryptionKey)
        return {
            "documentId": doc["id"],
            "title": doc["title"],
            "platform": doc["platform"],
            "canonicalUrl": doc["canonicalUrl"],
            "text": text,
        }

    # --------------------------------------------- collections, contexts, addresses

    def _vault(namespaceId: str) -> VaultStore:
        return VaultStore(store=ingest.store, config=config, namespaceId=namespaceId)

    def _refError(e: library.RefError) -> HTTPException:
        return HTTPException(status_code=404, detail={"error": str(e), "suggestions": e.suggestions})

    @app.get("/collections")
    async def collectionList(namespaceId: str = "main") -> list[dict[str, Any]]:
        return library.listing(ingest.store, _vault(namespaceId), namespaceId)["collections"]

    @app.post("/collections/add")
    async def collectionAdd(req: CollectionAddRequest) -> dict[str, Any]:
        """Register a folder and index it; later `update` calls only touch changed files."""
        try:
            entry = library.addCollection(req.name, req.path, req.pattern)
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e))
        stats = await library.syncCollection(ingest, req.namespaceId, req.name, entry["path"], entry["pattern"])
        return {"name": req.name, **entry, **stats}

    @app.post("/collections/remove")
    async def collectionRemove(req: CollectionRequest) -> dict[str, Any]:
        removed = library.removeCollection(ingest.store, req.namespaceId, req.name)
        return {"name": req.name, "removedDocuments": removed}

    @app.post("/collections/rename")
    async def collectionRename(req: CollectionRequest) -> dict[str, Any]:
        try:
            n = library.renameCollection(ingest.store, req.namespaceId, req.name, req.newName or "")
        except KeyError:
            raise HTTPException(status_code=404, detail=f"No collection named {req.name!r}")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"name": req.newName, "documents": n}

    @app.post("/update")
    async def update(req: UpdateRequest) -> dict[str, Any]:
        """Re-index collections: new files are added, changed ones re-indexed, deleted ones dropped."""
        cols = library.collections()
        if req.name and req.name not in cols:
            raise HTTPException(status_code=404, detail=f"No collection named {req.name!r}")
        out = {}
        for name, c in cols.items():
            if req.name and name != req.name:
                continue
            out[name] = await library.syncCollection(ingest, req.namespaceId, name, c["path"], c["pattern"],
                                                     prune=req.prune)
        return {"collections": out}

    @app.get("/contexts")
    async def contextList() -> dict[str, str]:
        return library.contexts()

    @app.post("/contexts/add")
    async def contextAdd(req: ContextEditRequest) -> dict[str, str]:
        try:
            return {"path": library.addContext(req.path, req.text, req.cwd), "text": req.text}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/contexts/remove")
    async def contextRemove(req: ContextEditRequest) -> dict[str, Any]:
        try:
            return {"removed": library.removeContext(req.path, req.cwd)}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/get")
    async def get(req: GetRequest) -> dict[str, Any]:
        """A note or source by address, #docid, title, alias or path."""
        try:
            return library.resolve(ingest.store, _vault(req.namespaceId), req.namespaceId, req.ref,
                                   config.encryptionKey)
        except library.RefError as e:
            raise _refError(e)

    @app.post("/multi-get")
    async def multiGet(req: MultiGetRequest) -> list[dict[str, Any]]:
        return library.multiGet(ingest.store, _vault(req.namespaceId), req.namespaceId, req.pattern,
                                req.maxBytes, config.encryptionKey)

    @app.get("/ls")
    async def ls(namespaceId: str = "main", path: str = "") -> dict[str, Any]:
        return library.listing(ingest.store, _vault(namespaceId), namespaceId, path)

    @app.post("/shutdown")
    async def shutdown() -> dict[str, str]:
        """Stop the engine (the CLI restarts it on the next command, e.g. after an upgrade)."""
        import signal
        asyncio.get_running_loop().call_later(0.2, os.kill, os.getpid(), signal.SIGTERM)
        return {"status": "stopping"}

    @app.post("/worker/tick")
    async def workerTick() -> dict[str, int]:
        n = await worker.processBatch()
        return {"processed": n}

    @app.post("/init")
    async def initDb(body: InitRequest) -> dict[str, str]:
        home = ensureKormemLayout()
        ingest.store.ensureNamespace(body.namespaceId)
        vault = VaultStore(store=ingest.store, config=config, namespaceId=body.namespaceId)
        vault.ensureMetaNotes()
        return {"home": str(home), "namespaceId": body.namespaceId}

    return app


def runServer(host: str = "127.0.0.1", port: Optional[int] = None) -> None:
    import uvicorn
    config = loadConfig()
    port = port or config.enginePort
    uvicorn.run(createApp(), host=host, port=port, log_level="info")
