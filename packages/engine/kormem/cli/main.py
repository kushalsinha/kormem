"""``kormem`` commands. Names and flags follow QMD wherever the job is the same.

QMD-compatible: collection, context, embed, update, search, vsearch, query,
get, multi-get, ls, status, doctor, cleanup, init, mcp.
Memory: remember, lessons, chat, hook, extract, maintain, notes, ingest, model.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
from typing import Any, Dict, List, Optional

from kormem import __version__
from kormem.cli import output as out
from kormem.cli.client import Engine, EngineError, baseUrl, kormemHome, logPath
from kormem.core.config import API_KEYS, getEnvPath, loadEnvFile, writeEnvTemplate
from kormem.core.ranks import RANK_NOTE, rankIndex, rankLabel

QUICK_HELP = """\
Quick start:
  kormem init                                     set up keys, the engine and Claude Code (run once)
  kormem collection add ~/notes --name notes      index a folder (only changed files on `kormem update`)
  kormem context add kormem://notes "Personal notes and ideas"
  kormem search "project timeline"                fast keyword search
  kormem vsearch "how to deploy"                  semantic search
  kormem query "what did we decide about X?"      ranked notes + sources; --deep runs the recall agent
  kormem get "#abc123"                             a source by docid, or a note / file by name or path
  kormem remember "I prefer pnpm over npm"        a fact, filed into the vault in the background
  kormem mcp                                      MCP server for Claude Code, Claude Desktop, Cursor
  kormem model                                    the model behind each job, and the recommended setup

Run `kormem <command> -h` for options.
"""


def _engine(args) -> Engine:
    return Engine(args.namespace)


def _fmt(args) -> str:
    for f in ("json", "csv", "md", "xml", "files"):
        if getattr(args, f"as_{f}", False):
            return f
    return getattr(args, "format", None) or "cli"


def _limit(args, fmt: str) -> int:
    if getattr(args, "all", False):
        return 1000
    return args.n or (20 if fmt in ("json", "files") else 5)


# ------------------------------------------------------------------ setup


MARKETPLACE = "kushalsinha/kormem"


def _claudeCode() -> str:
    """Install the Claude Code plugin (MCP server, session hooks, skill); fall back to the MCP server alone."""
    import json
    import shutil
    import subprocess

    claude = shutil.which("claude")
    if not claude:
        return "not found, skipped"

    def run(*cmd: str) -> subprocess.CompletedProcess:
        return subprocess.run([claude, *cmd], capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL)

    try:
        listed = run("plugin", "list", "--json")
        if listed.returncode == 0 and any(str(p.get("id", "")).startswith("kormem@")
                                          for p in json.loads(listed.stdout or "[]")):
            return "plugin already installed"
        run("plugin", "marketplace", "add", MARKETPLACE, "--sparse", ".claude-plugin", "plugin")
        if run("plugin", "install", "kormem@kormem").returncode == 0:
            return "plugin installed: MCP server, session hooks, recall skill (restart Claude Code)"
        cmd = ["kormem", "mcp"] if shutil.which("kormem") else [sys.executable, "-m", "kormem.cli", "mcp"]
        if run("mcp", "add", "--scope", "user", "kormem", "--", *cmd).returncode == 0:
            return "plugin install failed; added the MCP server only (sessions are not remembered without the hooks)"
        return "setup failed; see README > Claude Code"
    except (OSError, ValueError, subprocess.SubprocessError) as err:
        return f"setup failed ({err}); see README > Claude Code"


def cmdInit(args) -> None:
    """Set up everything a new user needs: data folder, key file, engine and Claude Code."""
    envPath = getEnvPath()
    if not os.environ.get("KORMEM_ENV_FILE"):
        envPath, created = writeEnvTemplate()
        if created:
            print(f"keys          created {envPath}: add your API keys there")
    e = _engine(args)
    r = e.post("/init", {"namespaceId": args.namespace})
    print(f"home          {r['home']} (namespace {r['namespaceId']})")
    keys = [k for k in API_KEYS if os.environ.get(k)]
    print(f"api keys      {', '.join(keys) if keys else 'none yet'} (from the environment or {envPath})")
    emb = e.get("/embeddings/status")
    print(f"engine        running at {baseUrl()}, log {logPath()}")
    print(f"embeddings    {emb['provider']} / {emb['model']}")
    from kormem.core import models
    from kormem.core.config import loadConfig
    config = loadConfig()
    for role, label, what in (("extraction", "extraction", " (files chats, notes and facts into the vault "
                                                            "in the background)"),
                              ("recall", "recall agent", "")):
        model = models.llmModel(role, config)[0]
        key = models.providerKey(models.llmProvider(model) or "openai")
        print(f"{label:13} {model if os.environ.get(key) else f'waiting for {key}'}{what}")
    decision = models.decisionModel(config)[0]
    print(f"{'decisions':13} " + (f"{decision} (re-ranks and trims the sources recall returns)"
                                   if models.decisionKey(config) else
                                   "local ranking (add TYPESAFE_API_KEY for Jev: sharper context, ~$0.001 a query)"))
    hours = float(os.environ.get("KORMEM_MAINTAIN_HOURS", "24"))
    print(f"maintenance   {f'on (merges, condenses and archives notes every {hours:g} h)' if hours > 0 else 'only on request'}")
    if not args.skip_agents:
        print(f"claude code   {_claudeCode()}")
    if not keys:
        print(f"\nAdd OPENAI_API_KEY (extraction, recall and embeddings) to {envPath};\n"
              "the next kormem command restarts the engine with them. `kormem model` explains the setup.")
    print("\nNext: kormem collection add <folder> --name <name>, then kormem query \"...\"")


def cmdCollection(args) -> None:
    e = _engine(args)
    sub = args.sub
    if sub == "add":
        name = args.name or os.path.basename(os.path.abspath(os.path.expanduser(args.path))) or "files"
        print(f"Indexing {args.path} as '{name}'…", file=sys.stderr)
        r = e.post("/collections/add", {"name": name, "path": args.path, "pattern": args.mask,
                                        "namespaceId": args.namespace})
        print(f"Collection '{name}' → {r['path']} ({r['pattern']}): {r['added']} added, "
              f"{r['updated']} updated, {r['unchanged']} unchanged.")
        print("Extraction into the vault runs in the background (kormem status shows the queue).")
    elif sub in ("list", "ls"):
        rows = e.get("/collections", namespaceId=args.namespace)
        if args.json:
            return out.printJson(rows)
        if not rows:
            print("No collections. Add one: kormem collection add <folder> --name <name>")
        for c in rows:
            print(f"{out.bold(c['name'])}  {c['path']}  {out.dim(c['pattern'])}  {c['documents']} files")
            if c.get("context"):
                print(out.dim(f"  Context: {c['context']}"))
    elif sub in ("remove", "rm"):
        r = e.post("/collections/remove", {"name": args.name, "namespaceId": args.namespace})
        print(f"Removed '{args.name}' ({r['removedDocuments']} files). Vault notes built from it stay.")
    elif sub in ("rename", "mv"):
        r = e.post("/collections/rename", {"name": args.old, "newName": args.new, "namespaceId": args.namespace})
        print(f"Renamed '{args.old}' → '{args.new}' ({r['documents']} files).")
    elif sub == "show":
        rows = {c["name"]: c for c in e.get("/collections", namespaceId=args.namespace)}
        c = rows.get(args.name)
        if not c:
            raise EngineError(f"No collection named {args.name!r}")
        if args.json:
            return out.printJson(c)
        for k in ("name", "uri", "path", "pattern", "documents", "context"):
            print(f"{k:10} {c.get(k) or ''}")


def cmdContext(args) -> None:
    e = _engine(args)
    if args.sub == "add":
        path, text = (args.items[0], args.items[1]) if len(args.items) >= 2 else ("", args.items[0])
        r = e.post("/contexts/add", {"path": path, "text": text, "cwd": os.getcwd()})
        print(f"Context for {r['path']}: {r['text']}")
    elif args.sub in ("list", "ls"):
        ctx = e.get("/contexts")
        if args.json:
            return out.printJson(ctx)
        if not ctx:
            print('No contexts. Add one: kormem context add kormem://notes "Personal notes and ideas"')
        for k, v in ctx.items():
            print(f"{out.bold(k)}  {v}")
    elif args.sub in ("rm", "remove"):
        r = e.post("/contexts/remove", {"path": args.path, "cwd": os.getcwd()})
        print("Removed." if r["removed"] else "No context there.")


def cmdIngest(args) -> None:
    e = _engine(args)
    t = args.target
    if t.startswith(("http://", "https://")):
        r = e.post("/ingest/url", {"url": t, "namespaceId": args.namespace, "collectionId": args.collection})
        print(f"Indexed {t} as {r.get('documentId', '')[:8]} ({r.get('segmentCount', 0)} chunks).")
    else:
        rows = e.post("/ingest/path", {"path": t, "namespaceId": args.namespace,
                                       "collectionId": args.collection, "recursive": True})
        print(f"Indexed {len(rows)} file(s). For a folder you keep editing, `kormem collection add` "
              "re-indexes only what changed on `kormem update`.")
    print("Extraction into the vault runs in the background.")


def cmdUpdate(args) -> None:
    r = _engine(args).post("/update", {"namespaceId": args.namespace, "name": args.name, "prune": not args.no_prune})
    if args.json:
        return out.printJson(r)
    if not r["collections"]:
        print("No collections to update.")
    for name, s in r["collections"].items():
        print(f"{name}: {s['added']} added, {s['updated']} updated, {s['removed']} removed, "
              f"{s['unchanged']} unchanged" + (f", {s['failed']} unreadable" if s["failed"] else ""))


def cmdEmbed(args) -> None:
    r = _engine(args).post("/embed", None, namespaceId=args.namespace, force=str(bool(args.force)).lower())
    print(f"Embedded {r['embedded']} chunks{' (forced re-embed)' if args.force else ''}.")


# ----------------------------------------------------------------- search


def _searchItems(r: Dict[str, Any], limit: int, minScore: float) -> List[Dict[str, Any]]:
    """One result per source (its best chunk), like QMD's per-document results."""
    seen, items = set(), []
    for k in r.get("knowledge", []):
        md = k.get("metadata") or {}
        key = md.get("docid") or k.get("documentId") or k["segmentKey"]
        if key in seen or k.get("score", 0) < minScore:
            continue
        seen.add(key)
        items.append({"docid": md.get("docid", ""), "uri": md.get("uri"),
                      "title": k.get("sourceTitle") or md.get("sourceTitle") or "",
                      "score": k.get("score", 0.0), "context": md.get("context", ""),
                      "date": md.get("sourceDate", ""), "text": k.get("text", "")})
        if len(items) >= limit:
            break
    return items


def cmdSearch(args) -> None:
    fmt = _fmt(args)
    limit = _limit(args, fmt)
    e = _engine(args)
    mode = "keyword" if args.command == "search" else "vector"
    r = e.post("/search", {"query": args.query, "namespaceId": args.namespace, "mode": mode,
                           "limit": limit * 3, "collectionId": args.collection, "minScore": args.min_score,
                           "includeGuidance": False, "rerank": False if args.no_rerank else None})
    items = _searchItems(r, limit, args.min_score)
    if args.full:
        for it in items:
            if it["docid"]:
                it["text"] = e.post("/get", {"ref": it["docid"], "namespaceId": args.namespace})["text"]
    out.results(items, fmt, full=args.full)


def cmdQuery(args) -> None:
    fmt = _fmt(args)
    r = _engine(args).post("/recall", {"query": args.query, "namespaceId": args.namespace,
                                        "effort": "high" if args.deep else "low", "queryDate": args.date})
    if fmt == "json":
        return out.printJson(r)
    if fmt == "files":
        for n in r["notes"]:
            print(f"[[{n['title']}]]")
        for s in r["sources"]:
            print(s.get("uri") or s.get("docid") or s["title"])
        return
    if fmt in ("md", "xml"):
        parts = []
        if r["findings"]:
            parts.append("## Findings\n\n" + "\n".join(f"- {f}" for f in r["findings"]))
        for n in r["notes"]:
            parts.append(f"## Note: {n['title']} ({n['folder']})\n\n{n['body']}")
        index = rankIndex(r["sources"], lambda s: s.get("docid") or s["title"])
        if index:
            parts.append(f"## Sources by rank\n\n{RANK_NOTE}\n\n{index}")
        for s in r["sources"]:
            label = rankLabel(s)
            parts.append(f"## Source {s.get('docid', '')} · {s['date'][:10]} · {s['title']}"
                         f"{' · ' + label if label else ''}\n\n{s['text']}")
        if r.get("relevantMemory"):
            parts.append("## Memory\n\n" + r["relevantMemory"])
        print("\n\n".join(parts) or "No matching memory found.")
        return
    if not (r["findings"] or r["notes"] or r["sources"]):
        print("No matching memory found.")
        return
    if r["findings"]:
        print(out.bold("Findings"))
        for f in r["findings"]:
            print(f"  - {f}")
    if r["notes"]:
        print(out.bold("\nNotes"))
        for n in r["notes"]:
            print(f"  [[{n['title']}]]  {out.dim(n['folder'])}")
            if args.full:
                print("    " + n["body"].replace("\n", "\n    "))
    if r["sources"]:
        print(out.bold("\nSources"))
        for s in r["sources"]:
            where = s.get("uri") or ""
            label = rankLabel(s)
            print(f"  {s.get('docid', ''):8} {s['date'][:10]}  {s['kind']:8} {s['title']}  "
                  f"{out.dim(label + '  ' if label else '')}{out.dim(where)}")
            if s.get("context"):
                print(out.dim(f"           Context: {s['context']}"))
            if args.full:
                print("    " + s["text"].replace("\n", "\n    "))
            elif not r["findings"]:  # no agent ran (default effort, or a simple lookup)
                text = " ".join(s["text"].split())
                print(f"           {text[:400]}{'…' if len(text) > 400 else ''}")
    print(out.dim("\nOpen any of these: kormem get <docid | note title>"))


def _splitRef(ref: str):
    """``path:from:count`` → (path, from, count)."""
    import re
    m = re.match(r"^(.*?)(?::(\d+))?(?::(\d+))?$", ref)
    return m.group(1), (int(m.group(2)) if m.group(2) else None), (int(m.group(3)) if m.group(3) else None)


def cmdGet(args) -> None:
    ref, fromLine, count = _splitRef(args.ref)
    fromLine = args.from_line or fromLine or 1
    maxLines = args.l or count
    try:
        doc = _engine(args).post("/get", {"ref": ref, "namespaceId": args.namespace})
    except EngineError as e:
        print(f"{e}", file=sys.stderr)
        sugg = (e.detail or {}).get("suggestions") if isinstance(e.detail, dict) else None
        if sugg:
            print("Did you mean:\n  " + "\n  ".join(sugg), file=sys.stderr)
        sys.exit(1)
    doc["text"] = out.sliceLines(doc["text"], fromLine, maxLines)
    doc["fromLine"] = fromLine
    out.documents([doc], _fmt(args), lineNumbers=not args.no_line_numbers)


def cmdMultiGet(args) -> None:
    docs = _engine(args).post("/multi-get", {"pattern": args.pattern, "maxBytes": args.max_bytes,
                                             "namespaceId": args.namespace})
    for d in docs:
        if "text" in d and args.l:
            d["text"] = out.sliceLines(d["text"], 1, args.l)
    out.documents(docs, _fmt(args), lineNumbers=not args.no_line_numbers)


def cmdLs(args) -> None:
    r = _engine(args).get("/ls", namespaceId=args.namespace, path=args.path or "")
    if args.json:
        return out.printJson(r)
    if "entries" not in r:
        print(out.bold("Collections"))
        for c in r["collections"] or []:
            print(f"  {c['uri']:32} {c['documents']:>5} files  {out.dim(c['path'])}")
        if not r["collections"]:
            print(out.dim("  (none: kormem collection add <folder> --name <name>)"))
        print(out.bold("\nVault"))
        for f in r["vault"]:
            print(f"  {f['uri']:48} {f['notes']:>5} notes")
        return
    for e in r["entries"]:
        print(f"{e.get('docid', ''):8} {e['date']:10}  {e['uri']}" + (f"  {out.dim(e['title'])}" if e["kind"] != "note" else ""))
    if not r["entries"]:
        print(f"Nothing in {r['path']}.")


# ----------------------------------------------------------------- memory


def cmdRemember(args) -> None:
    text = " ".join(args.text) if args.text else args.content
    if not text:
        raise EngineError('Usage: kormem remember "text" [--role user|assistant]')
    _engine(args).post("/ingest/text", {"content": text, "namespaceId": args.namespace,
                                        "platform": "conversation", "role": args.role})
    print("Remembered. It is filed into the vault in the background (kormem extract --wait to do it now).")


def cmdLessons(args) -> None:
    r = _engine(args).get("/lessons", namespaceId=args.namespace, project=args.project)
    if args.json:
        return out.printJson(r)
    print(r["lessons"] or "No lessons yet. They come from things like \"always use pnpm\" said in your sessions.")


def cmdChat(args) -> None:
    e = _engine(args)
    if args.sub == "close":
        out.printJson(e.post("/chat/close", {"namespaceId": args.namespace, "sessionId": args.session}))
        return
    turn = {"role": args.role, "content": args.content}
    out.printJson(e.post("/chat/append", {"namespaceId": args.namespace, "sessionId": args.session,
                                          "turns": [turn], "project": args.project}))


def cmdExtract(args) -> None:
    e = _engine(args)
    if args.park or args.resume:
        r = e.post("/extract/queue", {"action": "park" if args.park else "resume", "namespacePrefix": args.prefix})
        print(f"{'Parked' if args.park else 'Resumed'} {r['moved']} extraction job(s).")
        return
    if not args.wait:
        st = e.get("/status", namespaceId=args.namespace)
        print(f"{st.get('pendingExtractions', 0)} source(s) waiting for extraction. "
              "It runs in the background; --wait processes them now.")
        return
    r = e.post("/extract/run", {"namespaceId": args.namespace, "maxJobs": args.max})
    print(f"Extracted {r['processed']} source(s); {r['pending']} still waiting.")


def _maintain(args, condense: bool) -> None:
    allNs = getattr(args, "all", False) or bool(getattr(args, "prefix", ""))
    r = _engine(args).post("/maintain", {
        "namespaceId": None if allNs else args.namespace, "namespacePrefix": getattr(args, "prefix", "") or "",
        "condense": condense, "vacuum": args.vacuum})
    if args.json:
        return out.printJson(r)
    keys = ("memoryLinesMoved", "segmentsCompacted", "vectorsRepacked", "notesArchived", "notesMerged",
            "notesCondensed", "memoryVectorsDropped", "chunkVectorsDropped")
    total = {k: sum(int(rep.get(k) or 0) for rep in r["reports"]) for k in keys}
    done = ", ".join(f"{k} {v}" for k, v in total.items() if v)
    print(f"Maintained {r['namespaces']} namespace(s): {done or 'nothing to do'}.")


def cmdMaintain(args) -> None:
    _maintain(args, condense=not args.no_condense)


def cmdCleanup(args) -> None:
    _maintain(args, condense=False)


def cmdNotes(args) -> None:
    e = _engine(args)
    if args.sub in ("ls", "list"):
        rows = e.post("/notes/list", None, namespaceId=args.namespace, para=args.para)
        if args.json:
            return out.printJson(rows)
        for n in rows:
            print(f"{n['relPath']}")
    elif args.sub == "read":
        print(e.post("/get", {"ref": args.ref, "namespaceId": args.namespace})["text"])
    elif args.sub == "links":
        out.printJson(e.post("/notes/links", {"ref": args.ref, "namespaceId": args.namespace}))
    elif args.sub == "moc":
        print(e.post("/notes/moc", None, namespaceId=args.namespace)["content"])


# ------------------------------------------------------------------ admin


def cmdStatus(args) -> None:
    e = _engine(args)
    st = e.get("/status", namespaceId=args.namespace)
    cols = e.get("/collections", namespaceId=args.namespace)
    ctx = e.get("/contexts")
    emb = e.get("/embeddings/status")
    if args.json:
        return out.printJson({**st, "collections": cols, "contexts": ctx, "embeddings": emb})
    print(f"{out.bold('kormem')} {st['engineVersion']}  home {st['home']}  namespace {st['namespaceId']}")
    print(f"Sources {st['documentCount']}  chunks {st['segmentCount']}  "
          f"waiting: embed {st['pendingEmbeds']}, extract {st.get('pendingExtractions', 0)}")
    print(f"Embeddings {emb['provider']} / {emb['model']} / {emb['dimension']}-d")
    for w in emb.get("warnings") or []:
        print(f"  warning: {w}")
    print(out.bold("Collections"))
    for c in cols:
        print(f"  {c['name']:16} {c['documents']:>5} files  {c['path']}")
        if c.get("context"):
            print(out.dim(f"                   Context: {c['context']}"))
    if not cols:
        print(out.dim("  (none)"))
    if ctx.get("/"):
        print(out.dim(f"Global context: {ctx['/']}"))


def cmdDoctor(args) -> None:
    ok = lambda b: "ok" if b else "missing"  # noqa: E731
    print(f"kormem        {__version__}")
    print(f"python        {platform.python_version()} ({sys.executable})")
    print(f"home          {kormemHome()}")
    print(f"engine url    {baseUrl()}")
    print(f"key file      {getEnvPath()}{'' if getEnvPath().exists() else ' (missing: run kormem init)'}")
    for key in API_KEYS:
        print(f"{key:13} {'set' if os.environ.get(key) else 'not set'}")
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        print("              embeddings run locally (BAAI/bge-base-en-v1.5)")
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        print("              extraction and the recall agent need OPENAI_API_KEY; "
              "query returns its first pass without it")
    if not os.environ.get("TYPESAFE_API_KEY"):
        print("              recall ranks locally; TYPESAFE_API_KEY adds Jev ranking (recommended)")
    print("models        see `kormem model`")
    e = _engine(args)
    try:
        e.ensure()
        emb = e.get("/embeddings/status")
        print(f"engine        {ok(True)} ({(e.health() or {}).get('version', 'version unknown')})")
        print(f"embeddings    {emb['provider']} / {emb['model']} / {emb['dimension']}-d")
        if emb.get("storedDims"):
            print(f"index dims    {', '.join(map(str, emb['storedDims']))}")
        print(f"rerank        {emb.get('rerankModel') if emb.get('rerankEnabled') else 'off'}")
        for w in emb.get("warnings") or []:
            print(f"warning       {w}")
    except EngineError as err:
        print(f"engine        not running: {err}")
        print(f"              log: {logPath()}")


MODEL_GUIDE = """\
Answers come from your assistant's own model (Claude Code, Cursor, ...); kormem finds the context.

Recommended setup:
  OPENAI_API_KEY  the one key you need: gpt-6-luna for extraction and recall, and
                  text-embedding-3-small for embeddings (768-d, $0.02 per million tokens).
  GEMINI_API_KEY  optional: gemini-embedding-2 ($0.20 per million tokens) when it is the only key,
                  or with `kormem model set embedding gemini-embedding-2`.
  TYPESAFE_API_KEY  optional: Jev re-ranks and trims the sources `kormem query` returns (~0.2 s,
                  ~$0.001 a query). Short previews of matching sources are sent to TypeSafe.
  No key: embeddings run locally (free and private; the first index is slower).
  Pick the embedding model once. Changing it re-embeds everything you have indexed.

Change: kormem model set llm|extraction|recall <model>   (gpt-* or gemini-*, or "default")
        kormem model set decision <model>                (jev-*, or "default")
        kormem model set embedding <model>               (gemini-embedding-2, text-embedding-3-small,
                                                          local or default; re-embeds the whole index)"""


def _vectorCount() -> int:
    """Vectors a re-embed would rebuild (memory-line vectors are recomputed on use)."""
    import sqlite3
    from kormem.core.config import getIndexPath
    path = getIndexPath()
    if not path.exists():
        return 0
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            return conn.execute("SELECT COUNT(*) FROM segments WHERE embedding IS NOT NULL "
                                "AND kind != 'memory'").fetchone()[0]
        finally:
            conn.close()
    except sqlite3.Error:
        return 0


def _showModels(args) -> None:
    from kormem.core import models
    from kormem.core.config import loadConfig
    config = loadConfig()
    rows = []
    for role in models.LLM_ROLES:
        model, source = models.llmModel(role, config)
        provider = models.llmProvider(model)
        key = models.providerKey(provider) if provider else None
        rows.append({"role": role, "model": model, "source": source, "key": key,
                     "keySet": bool(key and os.environ.get(key))})
    decision, source = models.decisionModel(config)
    rows.append({"role": "decision", "model": decision, "source": source, "key": "TYPESAFE_API_KEY",
                 "keySet": bool(models.decisionKey(config))})
    emb = models.embeddingModel(config)
    rows.append({"role": "embedding", "model": emb, "dimension": config.embeddingDim,
                 "key": None if emb.startswith(("local", "hash")) else
                 "GEMINI_API_KEY" if emb.startswith("gemini") else "OPENAI_API_KEY",
                 "keySet": not emb.startswith(("local", "hash")), "vectors": _vectorCount()})
    if args.json:
        return out.printJson(rows)
    for r in rows:
        if r["role"] == "embedding":
            note = f"{r['dimension']}-d, {r['vectors']:,} vectors"
        else:
            missing = "not set: local ranking" if r["role"] == "decision" else "MISSING"
            keyNote = f"{r['key']} {'set' if r['keySet'] else missing}" if r["key"] else "unknown provider"
            note = f"{r['source']}, {keyNote}"
        print(f"{r['role']:11} {r['model']:28} {out.dim(note)}")
    wanted = (config.embeddingProvider or "auto").lower()
    if wanted == "auto" and not (os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        print(out.dim("            (no OPENAI_API_KEY: embeddings run locally)"))
    elif wanted in models.EMBED_KEYS and not os.environ.get(models.EMBED_KEYS[wanted]):
        print(out.dim(f"            (no {models.EMBED_KEYS[wanted]}: embeddings run locally)"))
    llms = {r["model"] for r in rows[:2]}
    if os.environ.get("OPENAI_API_KEY") and any(m.startswith("gemini") for m in llms):
        print(f"\nTip: {models.DEFAULT_OPENAI_MODEL} costs much less than Gemini for extraction and recall: "
              f"kormem model set llm {models.DEFAULT_OPENAI_MODEL}")
    print("\n" + MODEL_GUIDE)


def _setLlm(roles, value: str, raw: dict) -> None:
    from kormem.core import models
    fields = {"extraction": "extractionModel", "recall": "queryModel"}
    if value != "default":
        provider = models.llmProvider(value)
        if not provider:
            raise EngineError(f"Unknown model {value!r}: use a gpt-* (OpenAI) or gemini-* (Gemini) model, "
                              "or 'default'.")
        key = models.providerKey(provider)
        if not os.environ.get(key):
            print(f"warning: {value} needs {key}; add it to {getEnvPath()}.", file=sys.stderr)
    for role in roles:
        if value == "default":
            raw.pop(fields[role], None)
        else:
            raw[fields[role]] = value
        for var in models.LLM_SETTINGS[role][1]:
            if os.environ.get(var):
                print(f"warning: {var}={os.environ[var]} is set in the environment and overrides this.",
                      file=sys.stderr)
        print(f"{role:11} {value}")


def _setDecision(value: str, raw: dict) -> None:
    from kormem.core import models
    if value == "default":
        raw.pop("decisionModel", None)
    elif value.lower().startswith("jev"):
        raw["decisionModel"] = value
    else:
        raise EngineError(f"Unknown decision model {value!r}: use a jev-* model (e.g. jev-1.13.0) or 'default'.")
    if not os.environ.get("TYPESAFE_API_KEY"):
        print(f"warning: Jev needs TYPESAFE_API_KEY; add it to {getEnvPath()}.", file=sys.stderr)
    if os.environ.get(models.DECISION_ENV):
        print(f"warning: {models.DECISION_ENV}={os.environ[models.DECISION_ENV]} is set in the environment "
              "and overrides this.", file=sys.stderr)
    print(f"{'decision':11} {value}")


def _setEmbedding(value: str, raw: dict, yes: bool) -> bool:
    """Write the embedding choice after a warning and a confirmation. False when nothing changed."""
    from kormem.core import models
    from kormem.core.config import KormemConfig
    new = dict(raw)
    for k in ("embeddingProvider", "embeddingModel", "localEmbedModel"):
        new.pop(k, None)
    if value.startswith(("gemini", "text-embedding")):
        provider = "gemini" if value.startswith("gemini") else "openai"
        key = models.EMBED_KEYS[provider]
        if not os.environ.get(key):
            raise EngineError(f"{value} needs {key}: add it to {getEnvPath()} first.")
        new.update(embeddingProvider=provider, embeddingModel=value)
    elif value == "local" or value.startswith("local:"):
        new["embeddingProvider"] = "local"
        if value.startswith("local:"):
            new["localEmbedModel"] = value.split(":", 1)[1]
    elif value != "default":
        raise EngineError(f"Unknown embedding model {value!r}: use gemini-embedding-2, text-embedding-3-small, "
                          "local, local:<huggingface model> or default.")
    current = models.embeddingIdentity(KormemConfig.fromDict(raw))
    target = models.embeddingIdentity(KormemConfig.fromDict(new))
    if current == target:
        print(f"embedding   {models.embeddingModel(KormemConfig.fromDict(new))} is already in use.")
        return False
    n = _vectorCount()
    tokens = n * models.TOKENS_PER_CHUNK
    newModel = models.embeddingModel(KormemConfig.fromDict(new))
    price = models.EMBED_PRICE_PER_M.get(newModel)
    cost = (", free (on this machine)" if newModel.startswith("local") else
            f", about ${tokens / 1e6 * price:.2f} at list price" if price is not None else "")
    print(f"Changing the embedding model re-embeds everything: {n:,} vectors, ~{tokens:,} tokens{cost}.\n"
          "Keyword search keeps working; semantic search covers more of the index as the re-embed\n"
          "runs in the background. Vectors from different models can't be mixed, so change this rarely.",
          file=sys.stderr)
    if not yes:
        if not sys.stdin.isatty():
            raise EngineError("Re-run with --yes to confirm the embedding change.")
        if input(f"Switch embeddings to {value}? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Unchanged.")
            return False
    raw.clear()
    raw.update(new)
    print(f"embedding   {models.embeddingModel(KormemConfig.fromDict(new))}")
    return True


def cmdModel(args) -> None:
    """Show or change the model behind each job: extraction, recall, decisions and embeddings."""
    from kormem.core.config import readRawConfig, writeRawConfig
    if args.sub in (None, "show"):
        return _showModels(args)
    raw = readRawConfig()
    role, value = args.role, args.model
    if role == "embedding":
        if not _setEmbedding(value, raw, args.yes):
            return
    elif role == "decision":
        _setDecision(value, raw)
    else:
        _setLlm(("extraction", "recall") if role == "llm" else (role,), value, raw)
    writeRawConfig(raw)
    e = _engine(args)
    if e.health():
        e.ensure()  # restarts the engine on the new models; an embedding change re-embeds from there
        print("Engine restarted with the new models.")


def cmdStop(args) -> None:
    print("Engine stopped." if Engine(args.namespace).stop() else "Engine was not running.")


def cmdServe(args) -> None:
    from kormem.server.app import runServer
    runServer(host=args.host, port=args.port)


def cmdMcp(args) -> None:
    from kormem.cli.mcp_server import run
    run(args.namespace)


def cmdHook(args) -> None:
    from kormem.cli.hook import run
    run(args.namespace)


# ----------------------------------------------------------------- parser


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--namespace", default=os.environ.get("KORMEM_NAMESPACE", "main"),
                        help="memory namespace (default: main, or $KORMEM_NAMESPACE)")
    common.add_argument("--json", action="store_true", dest="as_json", help="JSON output")

    fmt = argparse.ArgumentParser(add_help=False)
    fmt.add_argument("--format", choices=out.FORMATS, help="cli (default) | json | csv | md | xml | files")
    for f in ("csv", "md", "xml", "files"):
        fmt.add_argument(f"--{f}", action="store_true", dest=f"as_{f}", help=argparse.SUPPRESS)

    p = argparse.ArgumentParser(prog="kormem", description="Local memory for AI agents: index, vault, recall.",
                                epilog=QUICK_HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"kormem {__version__}")
    sub = p.add_subparsers(dest="command", metavar="<command>")

    def add(name, fn, help_, parents=(common,), aliases=()):
        sp = sub.add_parser(name, help=help_, parents=list(parents), aliases=list(aliases))
        sp.set_defaults(fn=fn)
        return sp

    s = add("init", cmdInit, "set up ~/.kormem, the key file, the engine and Claude Code")
    s.add_argument("--skip-agents", action="store_true", help="don't install the Claude Code plugin")

    c = add("collection", cmdCollection, "add, list, remove, rename or show indexed folders")
    cs = c.add_subparsers(dest="sub", required=True)
    a = cs.add_parser("add", parents=[common], help="index a folder (or file) as a collection")
    a.add_argument("path")
    a.add_argument("--name", help="collection name (default: folder name)")
    a.add_argument("--mask", default="**/*", help='glob(s), comma-separated, e.g. "**/*.md,docs/**"')
    cs.add_parser("list", aliases=["ls"], parents=[common])
    r = cs.add_parser("remove", aliases=["rm"], parents=[common])
    r.add_argument("name")
    r = cs.add_parser("rename", aliases=["mv"], parents=[common])
    r.add_argument("old")
    r.add_argument("new")
    r = cs.add_parser("show", parents=[common])
    r.add_argument("name")

    c = add("context", cmdContext, "describe a collection, folder or everything, to guide filing and recall")
    cs = c.add_subparsers(dest="sub", required=True)
    a = cs.add_parser("add", parents=[common], help='context add [path] "description"')
    a.add_argument("items", nargs="+", metavar="[path] text")
    cs.add_parser("list", aliases=["ls"], parents=[common])
    r = cs.add_parser("rm", aliases=["remove"], parents=[common])
    r.add_argument("path")

    s = add("ingest", cmdIngest, "index a file, folder or URL once")
    s.add_argument("target")
    s.add_argument("--collection", "-c")

    s = add("update", cmdUpdate, "re-index collections (only changed files)")
    s.add_argument("name", nargs="?")
    s.add_argument("--no-prune", action="store_true", help="keep sources whose files were deleted")

    s = add("embed", cmdEmbed, "embed chunks still waiting (-f re-embeds everything)")
    s.add_argument("-f", "--force", action="store_true")

    for name, help_ in (("search", "keyword (BM25) search, fast"),
                        ("vsearch", "semantic vector search")):
        s = add(name, cmdSearch, help_, parents=(common, fmt),
                aliases=("vector-search",) if name == "vsearch" else ())
        s.add_argument("query")
        s.add_argument("-n", type=int, help="results (default 5, or 20 for --json/--files)")
        s.add_argument("-c", "--collection")
        s.add_argument("--all", action="store_true")
        s.add_argument("--min-score", type=float, default=0.0)
        s.add_argument("--full", action="store_true", help="show whole sources")
        s.add_argument("--no-rerank", action="store_true")

    s = add("query", cmdQuery, "recall: ranked notes and sources; --deep runs the recall agent",
            parents=(common, fmt), aliases=("deep-search", "recall"))
    s.add_argument("query")
    s.add_argument("--deep", action="store_true",
                   help="run the recall agent (~30 s): counts, timelines, preferences, dated findings")
    s.add_argument("--full", action="store_true", help="show note bodies and source text")
    s.add_argument("--date", help="answer as of this date (YYYY-MM-DD)")

    s = add("get", cmdGet, "a note or source: #docid, kormem:// address, title or path[:from[:count]]",
            parents=(common, fmt))
    s.add_argument("ref")
    s.add_argument("-l", type=int, help="max lines")
    s.add_argument("--from", type=int, dest="from_line", help="start line")
    s.add_argument("--no-line-numbers", action="store_true")
    s.add_argument("--full", action="store_true", help=argparse.SUPPRESS)

    s = add("multi-get", cmdMultiGet, "several notes or sources by glob or comma-separated list",
            parents=(common, fmt))
    s.add_argument("pattern")
    s.add_argument("-l", type=int, help="max lines per item")
    s.add_argument("--max-bytes", type=int, default=65_536)
    s.add_argument("--no-line-numbers", action="store_true")

    s = add("ls", cmdLs, "collections and vault folders, or what is inside one")
    s.add_argument("path", nargs="?")

    s = add("remember", cmdRemember, "store a fact; filed into the vault in the background")
    s.add_argument("text", nargs="*")
    s.add_argument("--content")
    s.add_argument("--role", choices=("user", "assistant"))

    s = add("lessons", cmdLessons, "what you told assistants to remember about working with you")
    s.add_argument("--project")

    c = add("chat", cmdChat, "append turns to a live chat, or close it")
    cs = c.add_subparsers(dest="sub", required=True)
    a = cs.add_parser("append", parents=[common])
    a.add_argument("--session", required=True)
    a.add_argument("--role", choices=("user", "assistant"), default="user")
    a.add_argument("--content", required=True)
    a.add_argument("--project")
    a = cs.add_parser("close", parents=[common])
    a.add_argument("--session", required=True)

    add("hook", cmdHook, "Claude Code hook: SessionStart / Stop / SessionEnd event on stdin")

    s = add("extract", cmdExtract, "extraction queue: show it, or --wait to process it now")
    s.add_argument("--wait", action="store_true")
    s.add_argument("--max", type=int, default=50)
    s.add_argument("--park", action="store_true", help="hold pending jobs")
    s.add_argument("--resume", action="store_true", help="release held jobs")
    s.add_argument("--prefix", default="", help="namespace prefix for --park / --resume")

    for name, fn, help_ in (("maintain", cmdMaintain, "shrink and tidy memory: archive, merge, condense (runs daily)"),
                            ("cleanup", cmdCleanup, "free part of maintain: compact the index, no model calls")):
        s = add(name, fn, help_)
        s.add_argument("--all", action="store_true", help="every namespace")
        s.add_argument("--prefix", default="", help="namespaces starting with this")
        s.add_argument("--vacuum", action="store_true", help="give freed space back to the disk")
        if name == "maintain":
            s.add_argument("--no-condense", action="store_true", help="skip the one paid step")

    c = add("notes", cmdNotes, "vault notes: ls, read, links, moc")
    cs = c.add_subparsers(dest="sub", required=True)
    a = cs.add_parser("ls", aliases=["list"], parents=[common])
    a.add_argument("--para", choices=("projects", "areas", "resources", "archive", "source", "meta"))
    for n in ("read", "links"):
        a = cs.add_parser(n, parents=[common])
        a.add_argument("ref")
    cs.add_parser("moc", parents=[common])

    add("status", cmdStatus, "index, queue, collections and embedding provider")
    add("doctor", cmdDoctor, "check the install, keys and engine")
    m = add("model", cmdModel, "show or change the models for extraction, recall, decisions and embeddings")
    ms = m.add_subparsers(dest="sub")
    ms.add_parser("show", parents=[common], help="each job's model, its key and the recommended setup")
    a = ms.add_parser("set", parents=[common], help="model set llm|extraction|recall|decision|embedding <model>")
    a.add_argument("role", choices=("llm", "extraction", "recall", "decision", "embedding"),
                   help="llm sets extraction and recall together")
    a.add_argument("model", help='e.g. gpt-6-luna, gemini-3.8-flash, jev-1.13.0, gemini-embedding-2, local, '
                                 'or "default"')
    a.add_argument("--yes", action="store_true", help="confirm an embedding change without asking")
    add("mcp", cmdMcp, "MCP server on stdio")
    add("stop", cmdStop, "stop the background engine")
    s = add("serve", cmdServe, "run the engine in the foreground")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=None)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    loadEnvFile()
    parser = _parser()
    args = parser.parse_args(argv)
    if not getattr(args, "fn", None):
        parser.print_help()
        return
    args.json = getattr(args, "as_json", False)
    try:
        args.fn(args)
    except EngineError as e:
        print(f"kormem: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
    except BrokenPipeError:
        pass


if __name__ == "__main__":
    main()
