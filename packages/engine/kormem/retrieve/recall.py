"""Recall: question -> the notes, sources and findings needed to answer it.

1. First pass (free): rank memory lines and topic notes by embedding,
   search source chunks (hybrid), and score candidate sources with the local
   cross-encoder. A source is any chat, document or batch of remembered
   facts; memory lines and note bullets link straight to their source.
2. Jev (when ``TYPESAFE_API_KEY`` is set): scores the previews of the top
   30 candidates in parallel, and its ranking is fused with the first pass
   (:mod:`kormem.retrieve.jev`).
3. ``effort="low"`` (the default): the top 8 sources, with ranks 6-8 dropped
   when Jev scores them below 0.3. ``effort="high"``: the recursive agent
   starts from a 12-source shortlist, walks notes, links and sources, and
   returns the sources the answer needs plus dated findings
   (:mod:`kormem.retrieve.recursive_agent`).
4. Temporal resolution (always): relative dates in returned source text are
   resolved against each source's date; memory lines carry absolute dates.

Every returned source carries its ``rank`` (1 = most relevant) and Jev's
``relevance``, so the answering model knows where to look first; sources stay
in date order.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from kormem.core.config import KormemConfig, loadConfig
from kormem.ingest.chat_sessions import documentText
from kormem.embeddings.provider import getInternalEmbedder
from kormem.extract.memory_index import cleanLine, lineEmbeddings, readMemory, trailingLinks
from kormem.retrieve import jev
from kormem.retrieve.knowledge import KnowledgeSearch, extractSearchKeywords
from kormem.retrieve.memory_search import _parse_memory_lines
from kormem.retrieve.recursive_agent import recursiveRecall
from kormem.retrieve.reranker import CrossEncoderReranker, sigmoidScore
from kormem.retrieve.temporal import classifyQuestion, describeOffset, parseQueryDate, resolveRelativeDates
from kormem.store.sqlite_store import SqliteStore
from kormem.vault.note_sections import parseSections
from kormem.vault.vault_store import VaultStore

logger = logging.getLogger(__name__)

PROFILE = {
    # effort: (sources (low) or agent shortlist (high), source chars in the final context, memory lines)
    "low": (8, 100_000, 40),
    "high": (12, 100_000, 40),
}
EFFORT_ALIASES = {"medium": "high"}  # 0.1.x clients sent "medium" for the agent
JEV_CANDIDATES = 30             # first-pass candidates Jev scores
TRIM_KEEP = 5                   # low effort always keeps the top 5 ...
TRIM_BELOW = 0.3                # ... and drops lower ranks Jev scores below this
SCORE_ALL_SOURCES = 60          # below this many sources, cross-encode every one
EXCERPT_WINDOW = 1000           # read_source excerpts: window size ...
EXCERPT_CHARS = 4000            # ... and total excerpt length
SKIP_TOP_PROB = 0.9             # skip the agent when one source clearly answers a simple lookup
SKIP_SECOND_PROB = 0.5
# Memory lines are scored for the last N months only (0 = all); older knowledge is reached through
# topic notes' current facts and source search, so recall cost stays flat as memory grows.
MEMORY_SCAN_MONTHS = int(os.environ.get("KORMEM_MEMORY_SCAN_MONTHS", "12"))
MIN_CURRENT_NOTES = 2           # archived notes join the candidates only when fewer current notes match
RRF_K = 60                      # reciprocal rank fusion constant (the standard value, not tuned)
_AGGREGATION_RE = re.compile(
    r"\b(how many|how much|in total|total|altogether|combined|order of|from earliest|from first|"
    r"list|all the|each|every|sum|average|first|last|latest|before|after|since|ago|when)\b", re.I)


@dataclass
class Source:
    sid: str
    documentId: str
    title: str
    kind: str
    date: Optional[datetime]
    dateStr: str
    location: str
    sourceNote: str = ""
    _text: Optional[str] = None


@dataclass
class RecallContext:
    relevantMemory: str
    notes: List[Dict[str, str]]
    sources: List[Dict[str, str]]
    findings: List[str]
    debug: Dict = field(default_factory=dict)


class RecallPipeline:
    def __init__(self, store: Optional[SqliteStore] = None, config: Optional[KormemConfig] = None):
        self.store = store or SqliteStore()
        self.config = config or loadConfig()
        self.embedder = getInternalEmbedder(self.config)  # notes + memory lines rank locally, free
        self.knowledge = KnowledgeSearch(self.store, self.config)
        self.reranker = CrossEncoderReranker(self.config)

    # ------------------------------------------------------------ helpers

    def _sources(self, namespaceId: str) -> List[Source]:
        out: List[Source] = []
        for r in self.store.documentsForNamespace(namespaceId):
            meta = json.loads(r["metadata"] or "{}")
            dateStr = str(meta.get("date") or r["createdAt"] or "")
            out.append(Source(
                sid="", documentId=r["id"], title=r["title"] or "Untitled", kind=meta.get("kind") or "document",
                date=parseQueryDate(dateStr), dateStr=dateStr[:22], location=r["canonicalUrl"] or "",
                sourceNote=meta.get("sourceNote") or "",
            ))
        out.sort(key=lambda s: (s.date or datetime.min.replace(tzinfo=timezone.utc)))
        for i, s in enumerate(out):
            s.sid = f"S{i + 1}"
        return out

    def _text(self, s: Source) -> str:
        if s._text is None:
            s._text = documentText(self.store, self.store.getDocument(s.documentId), self.config.encryptionKey)
        return s._text

    async def _lineEmbeddings(self, namespaceId: str, texts: List[str]) -> np.ndarray:
        return await lineEmbeddings(self.store, self.embedder, namespaceId, texts)

    async def _crossScores(self, query: str, previews: List[str]) -> List[float]:
        if not previews:
            return []
        try:
            ce = await asyncio.to_thread(self.reranker._load)
            return await asyncio.to_thread(lambda: [float(x) for x in ce.rerank(query, previews)])
        except Exception:
            logger.warning("cross-encoder unavailable; ranking sources by first-pass score", exc_info=True)
            return [0.0] * len(previews)

    async def _excerpts(self, text: str, query: str) -> str:
        """The windows of ``text`` that best match ``query`` (local cross-encoder), in reading order."""
        step = EXCERPT_WINDOW // 2
        windows = [(i, text[i:i + EXCERPT_WINDOW]) for i in range(0, len(text), step)]
        scores = await self._crossScores(query, [w for _, w in windows])
        if not any(scores):  # cross-encoder unavailable: fall back to keyword overlap
            kw = set(re.findall(r"\w{4,}", query.lower()))
            scores = [len(kw & set(re.findall(r"\w{4,}", w.lower()))) for _, w in windows]
        best = sorted(range(len(windows)), key=lambda i: -scores[i])[:EXCERPT_CHARS // EXCERPT_WINDOW]
        out, lastEnd = [], -1
        for i in sorted(best):
            start, w = windows[i]
            if start < lastEnd:
                w = text[lastEnd:start + EXCERPT_WINDOW]
            out.append(("" if start <= lastEnd else "…\n") + w)
            lastEnd = start + EXCERPT_WINDOW
        return "".join(out)

    @staticmethod
    def _skipReason(query: str, topProbs: List[float]) -> Optional[str]:
        """Skip the agent only for a simple lookup whose answer the first pass plainly found.

        Counting, ordering, time and "current value" questions always get the agent.
        """
        if classifyQuestion(query) != "none" or _AGGREGATION_RE.search(query):
            return None
        if len(topProbs) >= 1 and topProbs[0] >= SKIP_TOP_PROB and (len(topProbs) < 2 or topProbs[1] < SKIP_SECOND_PROB):
            return f"simple lookup; top source scored {topProbs[0]:.2f}"
        return None

    # -------------------------------------------------------------- recall

    async def recall(self, query: str, namespaceId: str, queryDate: Optional[str] = None,
                     effort: str = "low") -> RecallContext:
        question = query
        effort = EFFORT_ALIASES.get(effort, effort)
        effort = effort if effort in PROFILE else "low"
        shortlistN, budget, memoryN = PROFILE.get(effort, PROFILE["low"])
        qDate = parseQueryDate(queryDate) if queryDate else datetime.now(timezone.utc)
        vault = VaultStore(store=self.store, config=self.config, namespaceId=namespaceId)
        sources = self._sources(namespaceId)
        bySid = {s.sid: s for s in sources}
        byDoc = {s.documentId: s for s in sources}
        byNote = {s.sourceNote.lower(): s for s in sources if s.sourceNote}

        # Memory lines, each tied to its source through its trailing [[source]] link.
        since = None
        if MEMORY_SCAN_MONTHS > 0:
            y, m = qDate.year, qDate.month - MEMORY_SCAN_MONTHS
            while m < 1:
                y, m = y - 1, m + 12
            since = f"{y:04d}-{m:02d}"
        parsed = [ml for ml in _parse_memory_lines(readMemory(vault, since=since))
                  if not ml.is_structural and ml.text.strip() and ml.date_str]
        lineSrc = []
        for ml in parsed:
            hit = next((byNote[l.lower()] for l in trailingLinks(ml.text) if l.lower() in byNote), None)
            lineSrc.append(hit.sid if hit else None)
        clean = [cleanLine(ml.text) for ml in parsed]
        qEmb = np.asarray(await self.embedder.encodeQuery(query), dtype=np.float32)
        qEmb /= max(float(np.linalg.norm(qEmb)), 1e-9)
        lineScores = np.zeros(len(parsed), dtype=np.float32)
        lineMatrix = await self._lineEmbeddings(namespaceId, clean) if parsed else None
        if parsed:
            lineScores = lineMatrix @ qEmb
            kw = set(extractSearchKeywords(query).split())
            for i, t in enumerate(clean):
                lineScores[i] += 0.02 * min(len(kw & set(re.findall(r"\w+", t.lower()))), 3)
        lineOrder = [int(i) for i in np.argsort(-lineScores)]

        segItems, _ = await self.knowledge.search(query=query, namespaceId=namespaceId, limit=30,
                                                  rerank=False, queryDate=queryDate, stage1Limit=300)
        noteHits = [(json.loads(r["metadata"] or "{}"), sim) for r, sim in
                    self.store.vectorSearch(namespaceId, list(map(float, qEmb)), limit=12, kind="note")]
        current = [m["title"] for m, _ in noteHits if m.get("para") not in ("meta", "source", "archive")]
        archived = [m["title"] for m, _ in noteHits if m.get("para") == "archive"]
        topNotes = (current + (archived if len(current) < MIN_CURRENT_NOTES else []))[:6]

        # First-pass signals, kept apart so each ranks the candidates on its own.
        memSig: Dict[str, float] = {}   # best matching memory line
        segSig: Dict[str, float] = {}   # chunk search, rank-weighted
        noteSig: Dict[str, float] = {}  # links from the best-matching notes
        excerpt: Dict[str, str] = {}
        for i in lineOrder[:25]:
            if lineSrc[i]:
                memSig[lineSrc[i]] = max(memSig.get(lineSrc[i], 0.0), float(lineScores[i]))
        for rank, it in enumerate(segItems):
            s = byDoc.get(it.documentId or "")
            if s:
                segSig[s.sid] = segSig.get(s.sid, 0.0) + 1.0 / (rank + 3)
                excerpt.setdefault(s.sid, it.text[:500])
        for title in topNotes:
            note = vault.readNote(title)
            for link in re.findall(r"\[\[([^\]]+)\]\]", note.body if note else ""):
                s = byNote.get(link.lower())
                if s:
                    noteSig[s.sid] = noteSig.get(s.sid, 0.0) + 0.05
        prior: Dict[str, float] = {}
        for sig in (memSig, segSig, noteSig):
            for sid, v in sig.items():
                prior[sid] = prior.get(sid, 0.0) + v

        candidates = sources if len(sources) <= SCORE_ALL_SOURCES else [bySid[k] for k in prior]
        linesBySrc: Dict[str, List[int]] = {}
        for i in lineOrder:
            if lineSrc[i]:
                linesBySrc.setdefault(lineSrc[i], []).append(i)

        def preview(s: Source) -> str:
            mem = " ".join(clean[i] for i in linesBySrc.get(s.sid, [])[:4])
            ex = excerpt.get(s.sid) or self._text(s)[:400]
            return re.sub(r"\s+", " ", f"{s.dateStr} {s.title}. {mem} {ex}")[:1500]

        previews = [preview(s) for s in candidates]
        ce = await self._crossScores(query, previews)
        # Reciprocal rank fusion of the cross-encoder with the first-pass signals: the cross-encoder
        # alone favours the one session that states the answer and drops earlier values and partial
        # instances, which counts and "previous value" questions need.
        ceOf = {s.sid: c for s, c in zip(candidates, ce)}
        fused: Dict[str, float] = {s.sid: 0.0 for s in candidates}
        for sig in (ceOf, memSig, segSig, noteSig):
            order = sorted((sid for sid in fused if sid in sig), key=lambda sid: -sig[sid])
            for rank, sid in enumerate(order):
                fused[sid] += 1.0 / (RRF_K + rank + 1)
        ranked = sorted(zip(candidates, ce, previews), key=lambda x: (-fused[x[0].sid], -x[1]))

        # Jev re-ranks the top candidates; its ranking is fused with the first pass's.
        jevOf: Dict[str, float] = {}
        jevMs = None
        if jev.available(self.config):
            head = ranked[:JEV_CANDIDATES]
            t0 = time.monotonic()
            scores = await jev.scoreSources(question, queryDate or qDate.date().isoformat(),
                                            [p for _, _, p in head], self.config)
            jevMs = int(1000 * (time.monotonic() - t0))
            if scores:
                jevOf = {s.sid: sc for (s, _, _), sc in zip(head, scores) if sc is not None}
                fused2 = {s.sid: 1.0 / (RRF_K + i + 1) for i, (s, _, _) in enumerate(head)}
                for i, sid in enumerate(sorted(jevOf, key=lambda k: -jevOf[k])):
                    fused2[sid] += 1.0 / (RRF_K + i + 1)
                ranked = sorted(head, key=lambda x: -fused2[x[0].sid]) + ranked[JEV_CANDIDATES:]
        shortlist = [s for s, _, _ in ranked[:shortlistN]]
        trimmed: List[str] = []
        if effort == "low":
            trimmed = [s.sid for s in shortlist[TRIM_KEEP:] if jevOf.get(s.sid, 1.0) < TRIM_BELOW]
            shortlist = [s for s in shortlist if s.sid not in trimmed]
        previewOf = {s.sid: p for s, _, p in ranked}
        rankOf = {s.sid: i + 1 for i, (s, _, _) in enumerate(ranked)}

        # ----- recursive agent over the vault
        async def search(query: str) -> str:
            items, _ = await self.knowledge.search(query=query, namespaceId=namespaceId, limit=8,
                                                   rerank=False, queryDate=queryDate, stage1Limit=300)
            out = []
            for it in items:
                s = byDoc.get(it.documentId or "")
                if s:
                    out.append(f"[{s.sid} {s.dateStr}] " + re.sub(r"\s+", " ", it.text)[:500])
            e = np.asarray(await self.embedder.encodeQuery(query), dtype=np.float32)
            e /= max(float(np.linalg.norm(e)), 1e-9)
            for r, _ in self.store.vectorSearch(namespaceId, list(map(float, e)), limit=6, kind="note"):
                m = json.loads(r["metadata"] or "{}")
                if m.get("para") not in ("meta", "source"):
                    tag = " archived" if m.get("para") == "archive" else ""
                    out.append(f"note{tag}: [[{m.get('title')}]] ({m.get('relPath')})")
            if parsed:
                sc = lineMatrix @ e
                for i in np.argsort(-sc)[:8]:
                    out.append(f"memory [{lineSrc[int(i)] or '?'}]: {clean[int(i)]}")
            return "\n".join(out) or "(no matches)"

        async def read_note(title: str) -> str:
            note = vault.readNote(title)
            if not note:
                return f"(no note titled {title})"
            body = note.body
            for link in set(re.findall(r"\[\[([^\]]+)\]\]", body)):
                s = byNote.get(link.lower())
                if s:
                    body = body.replace(f"[[{link}]]", f"[[{link}]]({s.sid})")
            return f"# {note.title}  ({note.folder or note.para})\n{body[:8000]}"

        async def note_links(title: str) -> str:
            links = vault.getLinks(title)
            def tag(t):
                s = byNote.get((t or "").lower())
                return f"{t} ({s.sid})" if s else t
            return ("Links to: " + ", ".join(tag(l["title"]) for l in links["outbound"]) + "\n"
                    + "Linked from: " + ", ".join(tag(b["title"]) for b in links["backlinks"]))

        async def list_folder(path: str = "") -> str:
            notes = [n for n in vault.listNotes() if n.para not in ("meta", "source")]
            if not path:
                counts: Dict[str, int] = {}
                for n in notes:
                    counts[n.relPath.rsplit("/", 1)[0]] = counts.get(n.relPath.rsplit("/", 1)[0], 0) + 1
                return "\n".join(f"{f}/ ({c} notes)" for f, c in sorted(counts.items())) or "(empty vault)"
            inside = [n.title for n in notes if n.relPath.startswith(path.rstrip("/") + "/")]
            return "\n".join(f"[[{t}]]" for t in sorted(inside)) or f"(no notes in {path})"

        async def list_sources() -> str:
            mark = {s.sid for s in shortlist}
            return "\n".join(
                f"{s.sid} | {s.dateStr} | {s.kind} | {describeOffset(s.date, qDate)}"
                f"{' [shortlisted]' if s.sid in mark else ''} | {previewOf.get(s.sid, s.title)[:140]}"
                for s in sources[:400]
            )

        async def read_source(source_id: str, query: str = "", full: bool = False) -> str:
            s = bySid.get(source_id.strip().strip("[]"))
            if not s:
                return f"(no source {source_id}; ids look like S12)"
            text = resolveRelativeDates(self._text(s), s.date)
            head = f"[{s.sid}] {s.title} · {s.dateStr} ({describeOffset(s.date, qDate)})"
            if full or len(text) <= EXCERPT_CHARS:
                return f"{head}\n{text[:24000]}"
            return f"{head} — excerpts for: {query or question}\n{await self._excerpts(text, query or question)}"

        def noteDigest(title: str) -> str:
            """A note's facts and timeline, with [[source]] links tagged by source id."""
            note = vault.readNote(title)
            if not note:
                return ""
            _, secs = parseSections(note.body)
            lines = (secs.get("Current facts") or [])[:8] + (secs.get("Timeline") or [])[-10:]
            body = "\n".join(lines)
            for link in set(re.findall(r"\[\[([^\]]+)\]\]", body)):
                src = byNote.get(link.lower())
                body = body.replace(f"[[{link}]]", f"({src.sid})" if src else "")
            return f"[[{note.title}]] ({note.folder or note.para})\n{body[:1500]}"

        briefing = "\n".join([
            f"Question date (today): {queryDate or qDate.date().isoformat()}",
            f"{len(sources)} sources in total.",
            "Shortlisted sources:",
            *[f"- {s.sid} {s.dateStr} {s.kind} ({describeOffset(s.date, qDate)}): {previewOf[s.sid][:300]}"
              for s in shortlist],
            "Best-matching notes (facts and timeline):",
            *[d for d in (noteDigest(t) for t in topNotes[:3]) if d],
            "Other matching notes: " + (", ".join(f"[[{t}]]" for t in topNotes[3:]) or "(none)"),
        ])
        agentDebug: Dict = {}
        findings: List[str] = []
        chosen: List[Source] = []
        notesRead: List[str] = []
        skip = ("low effort" if effort == "low" else
                self._skipReason(query, [sigmoidScore(c) for c in sorted(ce, reverse=True)[:2]]))
        if skip:
            agentDebug = {"skipped": skip}
        else:
            try:
                result = await recursiveRecall(
                    query, briefing,
                    {"search": search, "read_note": read_note, "note_links": note_links,
                     "list_folder": list_folder, "list_sources": list_sources, "read_source": read_source},
                    set(bySid), effort=effort, config=self.config,
                )
                findings = result.findings
                chosen = [bySid[s] for s in result.sources]
                notesRead = result.notesRead
                agentDebug = {"rounds": result.rounds, "sources": result.sources, "toolCalls": result.toolCalls}
            except Exception as e:
                logger.warning("recall agent failed; using the first-pass shortlist: %s", e)
                agentDebug = {"error": str(e)[:300]}

        # ----- assemble: agent's sources first, then the shortlist, within budget
        picked: List[Source] = []
        used = 0
        for s in {x.sid: x for x in chosen + shortlist}.values():
            n = len(self._text(s))
            if picked and used + n > budget:
                continue
            picked.append(s)
            used += n
        # Rank: the agent's sources first (in its order), then the ranking; 1 = most relevant.
        chosenIds = {s.sid for s in chosen}
        order = sorted(picked, key=lambda s: (s.sid not in chosenIds,
                                              [c.sid for c in chosen].index(s.sid) if s.sid in chosenIds
                                              else rankOf.get(s.sid, len(rankOf) + 1)))
        rankIn = {s.sid: i + 1 for i, s in enumerate(order)}
        picked.sort(key=lambda s: s.date or datetime.min.replace(tzinfo=timezone.utc))
        pickedIds = {s.sid for s in picked}
        keep = set(lineOrder[:memoryN]) | {i for i, sid in enumerate(lineSrc) if sid in pickedIds}
        byDate: Dict[str, List[str]] = {}
        for i in sorted(keep):
            byDate.setdefault(parsed[i].date_str, []).append(clean[i])
        relevantMemory = "\n\n".join(f"Date: {d}\n" + "\n".join(v) for d, v in sorted(byDate.items()))
        noteTitles = list(dict.fromkeys(notesRead + topNotes[:2]))[:5]
        notes = []
        for t in noteTitles:
            n = vault.readNote(t)
            if n:
                notes.append({"title": n.title, "folder": n.folder or n.para, "body": n.body[:4000]})

        return RecallContext(
            relevantMemory=relevantMemory,
            notes=notes,
            sources=[{"sid": s.sid, "documentId": s.documentId, "title": s.title, "kind": s.kind, "date": s.dateStr,
                      "rank": rankIn[s.sid], "relevance": round(jevOf[s.sid], 2) if s.sid in jevOf else None,
                      "agentChose": s.sid in chosenIds,
                      "text": resolveRelativeDates(self._text(s), s.date)} for s in picked],
            findings=findings,
            debug={"effort": effort, "nSources": len(sources), "shortlist": [s.sid for s in shortlist],
                   "candidates": [{"sid": s.sid, "title": s.title, "date": s.dateStr, "crossEncoder": round(c, 3),
                                   "prior": round(prior.get(s.sid, 0.0), 3), "fused": round(fused[s.sid], 5),
                                   "jev": jevOf.get(s.sid)}
                                  for s, c, _ in ranked[:30]],
                   "jevMs": jevMs, "trimmed": trimmed,
                   "selected": [s.sid for s in picked], "agent": agentDebug, "sourceChars": used},
        )
