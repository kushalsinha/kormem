"""MCP server on stdio. Tool names follow QMD (query, get, multi_get, status) plus kormem's memory tools."""

from __future__ import annotations

import logging
import threading
from typing import List, Optional

from kormem import __version__
from kormem.cli import output as out
from kormem.cli.client import Engine, EngineError
from kormem.core.ranks import RANK_NOTE, rankIndex, rankLabel

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

INSTRUCTIONS = """kormem is the user's long-term memory: their chats with assistants, files, URLs and facts,
filed into a linked vault of notes. Use `query` before answering anything that may depend on what the
user said, decided, prefers or did before. `query` returns dated findings, topic notes and source ids;
open a source with `get` ("#abc123") when you need exact wording. Use `remember` when the user asks you
to remember something. `lessons` returns how this user wants assistants to work with them."""


def run(namespace: str = "main") -> None:
    engine = Engine(namespace)
    threading.Thread(target=lambda: _quiet(engine.ensure), daemon=True).start()
    mcp = _Server("kormem", instructions=INSTRUCTIONS)

    @mcp.tool()
    def query(query: str = "", searches: Optional[List[dict]] = None, deep: bool = False) -> str:
        """Recall memory for a question: ranked sources, the topic notes they came from, and source ids
        (#docid) to open with `get`. Fast (~2 s). Set `deep` for counts, totals, timelines, "what do I
        prefer" and anything spread over many conversations: a recall agent reads further and returns dated
        findings (~30 s). QMD-style `searches` ([{"type": "lex"|"vec"|"hyde", "query": "..."}]) are accepted
        and joined into the question."""
        q = query or " ".join(str(s.get("query", "")) for s in (searches or []))
        if not q.strip():
            return "Give a question in `query`."
        r = engine.post("/recall", {"query": q, "namespaceId": namespace, "effort": "high" if deep else "low"})
        parts = []
        if r["findings"]:
            parts.append("Findings:\n" + "\n".join(f"- {f}" for f in r["findings"]))
        for n in r["notes"]:
            parts.append(f"Note [[{n['title']}]] ({n['folder']}):\n{n['body'][:2000]}")
        index = rankIndex(r["sources"], lambda s: s.get("docid") or s["title"])
        if index:
            parts.append(f"{RANK_NOTE}\nBy rank:\n{index}")
        for s in r["sources"]:
            ctx = f"\nContext: {s['context']}" if s.get("context") else ""
            label = rankLabel(s)
            parts.append(f"Source {s.get('docid', '')} · {s['date'][:10]} · {s['title']}"
                         f"{' · ' + label if label else ''}"
                         f"{' · ' + s['uri'] if s.get('uri') else ''}{ctx}:\n{s['text'][:3000]}")
        if r.get("relevantMemory"):
            parts.append("Memory:\n" + r["relevantMemory"][:4000])
        return "\n\n---\n\n".join(parts) or "No matching memory found."

    @mcp.tool()
    def search(query: str, mode: str = "keyword", limit: int = 10, collection: Optional[str] = None) -> str:
        """Fast raw search over indexed chunks, no agent. mode: keyword (BM25) or vector (semantic)."""
        r = engine.post("/search", {"query": query, "namespaceId": namespace, "limit": limit * 3,
                                    "mode": "vector" if mode.startswith("vec") else "keyword",
                                    "collectionId": collection, "includeGuidance": False})
        seen, lines = set(), []
        for k in r.get("knowledge", []):
            md = k.get("metadata") or {}
            key = md.get("docid") or k.get("documentId")
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"{md.get('docid', '')} {md.get('uri') or k.get('sourceTitle') or ''} "
                         f"({k.get('score', 0):.3f})\n{k.get('text', '')[:600]}")
            if len(lines) >= limit:
                break
        return "\n\n".join(lines) or "No results."

    @mcp.tool()
    def get(file: str, fromLine: Optional[int] = None, maxLines: Optional[int] = None,
            lineNumbers: bool = True) -> str:
        """A note or source by #docid, kormem:// address, note title, alias or file path.
        `path:from:count` (e.g. "#abc123:120:40") reads a line range."""
        from kormem.cli.main import _splitRef
        ref, f, c = _splitRef(file)
        start = fromLine or f or 1
        try:
            doc = engine.post("/get", {"ref": ref, "namespaceId": namespace})
        except EngineError as e:
            sugg = (e.detail or {}).get("suggestions") if isinstance(e.detail, dict) else None
            return f"{e}" + (f"\nDid you mean: {', '.join(sugg)}" if sugg else "")
        text = out.numbered(out.sliceLines(doc["text"], start, maxLines or c), start, lineNumbers)
        head = f"{doc.get('uri') or doc.get('title')} {doc.get('docid', '')} {doc.get('date', '')}".strip()
        return f"{head}\n{text}"

    @mcp.tool()
    def multi_get(pattern: str, maxBytes: int = 10240, maxLines: Optional[int] = None,
                  lineNumbers: bool = True) -> str:
        """Several notes or sources: a glob ("kormem://notes/2025-05*", "vault/2-Areas/**") or a
        comma-separated list of refs and #docids."""
        docs = engine.post("/multi-get", {"pattern": pattern, "maxBytes": maxBytes, "namespaceId": namespace})
        parts = []
        for d in docs:
            if "error" in d:
                parts.append(f"{d['ref']}: {d['error']}")
            elif d.get("skipped"):
                parts.append(f"==> {out.address(d)} {d.get('docid', '')} (skipped: {d['skipped']})")
            else:
                body = out.numbered(out.sliceLines(d["text"], 1, maxLines), 1, lineNumbers)
                parts.append(f"==> {out.address(d)} {d.get('docid', '')} <==\n{body}")
        return "\n\n".join(parts) or "Nothing matched."

    @mcp.tool()
    def status() -> str:
        """Index health: sources, chunks, queues, collections and the embedding provider."""
        st = engine.get("/status", namespaceId=namespace)
        cols = engine.get("/collections", namespaceId=namespace)
        lines = [f"kormem {st['engineVersion']} · {st['documentCount']} sources · {st['segmentCount']} chunks · "
                 f"waiting: embed {st['pendingEmbeds']}, extract {st.get('pendingExtractions', 0)}"]
        lines += [f"collection {c['name']}: {c['documents']} files · {c['path']}"
                  + (f" · {c['context']}" if c.get("context") else "") for c in cols]
        return "\n".join(lines)

    @mcp.tool()
    def remember(content: str, role: Optional[str] = None) -> str:
        """Store something the user wants remembered (a fact, preference or decision)."""
        engine.post("/ingest/text", {"content": content, "namespaceId": namespace, "platform": "conversation",
                                     "role": role if role in ("user", "assistant") else "user"})
        return "Stored. It is filed into the vault in the background."

    @mcp.tool()
    def lessons(project: Optional[str] = None) -> str:
        """How this user wants assistants to work with them (style, tools, workflow): global plus a project's."""
        return engine.get("/lessons", namespaceId=namespace, project=project)["lessons"] or "No lessons yet."

    @mcp.tool()
    def ingest(target: str, collection: Optional[str] = None) -> str:
        """Add a file, folder or URL to memory. It is searchable at once and filed into the vault in the background."""
        if target.startswith(("http://", "https://")):
            r = engine.post("/ingest/url", {"url": target, "namespaceId": namespace, "collectionId": collection})
            return f"Indexed {target} ({r.get('segmentCount', 0)} chunks)."
        rows = engine.post("/ingest/path", {"path": target, "namespaceId": namespace,
                                            "collectionId": collection, "recursive": True})
        return f"Indexed {len(rows)} file(s)."

    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per engine call is noise in MCP logs
    mcp.run()


def _quiet(fn) -> None:
    try:
        fn()
    except Exception:
        pass


__all__ = ["run", "__version__"]
