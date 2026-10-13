"""Semantic memory search for avatarMemory.

Implements Mastra-style line-level semantic retrieval on the avatarMemory blob:
1. Parse memory lines preserving Date: headers and emoji priority
2. Embed lines with Gemini (disk cache → in-process cache → Gemini API)
3. Expand query: main + quoted text + proper nouns + sentences + preference boost
4. Cosine-similarity search top-K lines with per-line score adjustments:
   - Recency boost: +0.05/year closer to reference_date
   - Emoji priority boost: 🔴 +0.03, 🟡 +0.01
   - Preference boost queries run separately at stricter threshold (0.8) and merged
5. Expand context around matches (neighboring lines / full date block)
6. Rebuild output with Date: headers intact

This replaces the current approach of sending the whole avatarMemory blob to a
guidance-brief LLM and hoping it picks the right entries.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TOP_K = 60
MIN_SIMILARITY = 0.25           # lower than Mastra's 0.4; our memories are denser
HIGH_SCORE_THRESHOLD = 0.75     # include full date block
MEDIUM_SCORE_THRESHOLD = 0.40   # include ±2 neighbor lines
# Preference boost: separate retrieval pass at stricter threshold (Mastra: 0.9, we use 0.8)
PREFERENCE_BOOST_MIN_SIM = 0.80
PREFERENCE_BOOST_QUERIES = [
    "likes prefers wants favorites interested in enjoys",
    "user preference hobby routine",
]
# Recency boost: +RECENCY_BOOST_PER_YEAR per year closer to the reference date.
# Helps knowledge-update questions find the most recent value.
RECENCY_BOOST_PER_YEAR = 0.05
# Emoji priority boosts (additive, applied after similarity scoring)
EMOJI_BOOST = {"🔴": 0.03, "🟡": 0.01}

# Lines that are purely structural (Date: headers, empty, gap markers) -- never embed alone
_STRUCTURAL_RE = re.compile(r"^\s*(Date:\s*\d{4}|\[---|\s*)$", re.I)
_DATE_HEADER_RE = re.compile(r"^(Date:\s*\d{4}-\d{2}-\d{2}.*)", re.I)
_EMOJI_RE = re.compile(r"^([🔴🟡🟢✅⚪])\s*")

# Simple noun-phrase / token extraction for query expansion
_QUOTED_RE = re.compile(r'"([^"]+)"')
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b")

# Per-process embedding cache: { memory_hash -> list of embeddings (same order as parsed lines) }
_embedding_cache: Dict[str, List[List[float]]] = {}
_MAX_CACHE_ENTRIES = 50  # evict oldest when exceeded


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MemoryLine:
    """A single observation line with its structural context."""
    text: str                        # the raw line text
    date_header: str = ""            # the Date: header of the block this line belongs to
    date_str: str = ""               # ISO date string, e.g. "2023-05-26"
    line_index: int = 0              # position in the parsed list
    emoji: str = ""                  # "🔴", "🟡", "🟢", "✅", or ""
    is_structural: bool = False      # True for Date: headers, gap markers, empty lines
    block_index: int = 0             # which date block this belongs to


@dataclass
class SearchResult:
    """Result from semantic memory search."""
    filtered_memory: str             # rebuilt memory string with only relevant lines
    matched_line_count: int = 0
    total_line_count: int = 0
    used_semantic_search: bool = False


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_memory_lines(memory_text: str) -> List[MemoryLine]:
    """Parse avatarMemory into MemoryLine objects preserving structure."""
    lines = memory_text.splitlines()
    parsed: List[MemoryLine] = []
    current_date_header = ""
    current_date_str = ""
    current_block_index = -1

    for idx, raw_line in enumerate(lines):
        is_structural = False

        if _DATE_HEADER_RE.match(raw_line.strip()):
            current_date_header = raw_line.strip()
            # Extract ISO date
            m = re.search(r"(\d{4}-\d{2}-\d{2})", current_date_header)
            current_date_str = m.group(1) if m else ""
            current_block_index += 1
            is_structural = True
        elif not raw_line.strip() or _STRUCTURAL_RE.match(raw_line.strip()):
            is_structural = True

        emoji_match = _EMOJI_RE.match(raw_line.strip())
        emoji = emoji_match.group(1) if emoji_match else ""

        parsed.append(MemoryLine(
            text=raw_line,
            date_header=current_date_header,
            date_str=current_date_str,
            line_index=idx,
            emoji=emoji,
            is_structural=is_structural,
            block_index=current_block_index,
        ))

    return parsed


def _lines_for_embedding(parsed: List[MemoryLine]) -> List[Tuple[int, str]]:
    """Return (original_index, text) for non-structural lines worth embedding."""
    result = []
    for ml in parsed:
        if ml.is_structural or not ml.text.strip():
            continue
        result.append((ml.line_index, ml.text.strip()))
    return result


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def _cosine_sim(a: List[float], b: List[float]) -> float:
    av = np.array(a, dtype=np.float32)
    bv = np.array(b, dtype=np.float32)
    na = np.linalg.norm(av)
    nb = np.linalg.norm(bv)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------

def _expand_queries(query: str) -> List[str]:
    """Build a list of main sub-queries from the query (no preference boost).

    Preference boost queries are handled in a separate retrieval pass with a
    stricter similarity threshold so they don't flood results with noise.
    """
    queries = [query]

    # Quoted phrases (highest priority — user is asking about a specific phrase)
    quoted = _QUOTED_RE.findall(query)
    queries.extend(quoted)

    # Individual sentences within the query (Mastra: length >= 20 chars).
    # Helps multi-part questions like "What did I say about X and what is my Y?"
    sentences = [s.strip() for s in re.split(r"[.?!]", query) if len(s.strip()) >= 20]
    # Only add sentences that are meaningfully different from the full query
    for s in sentences:
        if s != query.strip():
            queries.append(s)

    # Proper nouns / named entities
    proper_nouns = _PROPER_NOUN_RE.findall(query)
    for pn in proper_nouns:
        if pn.lower() not in {"what", "when", "where", "why", "who", "how", "i"}:
            queries.append(pn)

    # Deduplicate while preserving order
    seen: set = set()
    unique = []
    for q in queries:
        if q and q not in seen:
            seen.add(q)
            unique.append(q)
    return unique


_PREFERENCE_QUESTION_RE = re.compile(
    r"\b(recommend|suggest|prefer|like|enjoy|favorite|favourite|interested in|"
    r"what should i|help me (choose|decide|pick)|any tips)\b",
    re.I,
)


def _is_preference_query(query: str) -> bool:
    return bool(_PREFERENCE_QUESTION_RE.search(query))


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

def _memory_hash(memory_text: str) -> str:
    return hashlib.md5(memory_text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _cache_put(key: str, embeddings: List[List[float]]) -> None:
    global _embedding_cache
    if len(_embedding_cache) >= _MAX_CACHE_ENTRIES:
        # Evict the first (oldest) entry
        oldest = next(iter(_embedding_cache))
        del _embedding_cache[oldest]
    _embedding_cache[key] = embeddings


# ---------------------------------------------------------------------------
# Core search
# ---------------------------------------------------------------------------

async def semantic_memory_search(
    query: str,
    memory_text: str,
    top_k: int = TOP_K,
    min_similarity: float = MIN_SIMILARITY,
    high_score_threshold: float = HIGH_SCORE_THRESHOLD,
    medium_score_threshold: float = MEDIUM_SCORE_THRESHOLD,
    reference_date: Optional[str] = None,
) -> SearchResult:
    """
    Perform semantic search on avatarMemory and return a filtered memory string.

    Args:
        query: The user's question.
        memory_text: Raw avatarMemory string (markdown with Date: headers and emoji lines).
        top_k: Maximum number of matching lines to retrieve.
        min_similarity: Minimum cosine similarity to include a line.
        high_score_threshold: Similarity >= this pulls in the whole date block.
        medium_score_threshold: Similarity >= this pulls in ±2 neighboring lines.
        reference_date: ISO date string used as "today" for temporal context.

    Returns:
        SearchResult with filtered_memory string and stats.
    """
    if not memory_text or not memory_text.strip():
        return SearchResult(filtered_memory=memory_text or "", used_semantic_search=False)

    try:
        from kormem.core.config import loadConfig
        from kormem.embeddings.provider import getEmbeddingProvider

        config = loadConfig()
        embedSvc = getEmbeddingProvider(config)
    except Exception as e:
        logger.debug("Could not load embedding provider: %s", e)
        return SearchResult(filtered_memory=memory_text, used_semantic_search=False)

    async def _encodeBatch(texts: List[str], queryMode: bool = False) -> List[List[float]]:
        if queryMode:
            return [await embedSvc.encodeQuery(t) for t in texts]
        return await embedSvc.encodeBatch(texts)

    # 1. Parse memory into lines
    parsed = _parse_memory_lines(memory_text)
    embeddable = _lines_for_embedding(parsed)

    if len(embeddable) < 3:
        # Too few lines to justify embedding overhead
        return SearchResult(filtered_memory=memory_text, used_semantic_search=False)

    # 2. Embed memory lines (with two-layer caching: in-process hash, then disk)
    mem_hash = _memory_hash(memory_text)
    if mem_hash in _embedding_cache:
        line_embeddings = _embedding_cache[mem_hash]
    else:
        textsToEmbed = [text for _, text in embeddable]
        try:
            line_embeddings = await _encodeBatch(textsToEmbed, queryMode=False)
            _cache_put(mem_hash, line_embeddings)
        except Exception as e:
            logger.warning("Failed to embed memory lines: %s", e)
            return SearchResult(filtered_memory=memory_text, used_semantic_search=False)

    if len(line_embeddings) != len(embeddable):
        logger.warning("Embedding count mismatch; skipping semantic memory search")
        return SearchResult(filtered_memory=memory_text, used_semantic_search=False)

    # Build lookup: original_line_index -> embedding
    idx_to_embedding: Dict[int, List[float]] = {
        orig_idx: line_embeddings[i]
        for i, (orig_idx, _) in enumerate(embeddable)
    }

    # 3. Embed queries (main + sentence expansions + proper nouns + quoted text)
    is_pref = _is_preference_query(query)
    sub_queries = _expand_queries(query)

    queryEmbeddings: List[List[float]] = []
    try:
        queryEmbeddings = await _encodeBatch(sub_queries, queryMode=True)
    except Exception as e:
        logger.warning("Failed to embed queries for memory search: %s", e)
        return SearchResult(filtered_memory=memory_text, used_semantic_search=False)

    # 4. Score each embeddable line: max similarity across main sub-queries
    line_scores: Dict[int, float] = {}
    for orig_idx, emb in idx_to_embedding.items():
        best = 0.0
        for qEmb in queryEmbeddings:
            s = _cosine_sim(qEmb, emb)
            if s > best:
                best = s
        line_scores[orig_idx] = best

    # 4b. Preference boost pass (Mastra pattern): separate retrieval with stricter
    # threshold (0.8) so preference queries don't flood results with noise.
    # Results are merged into line_scores keeping max per line.
    if is_pref and PREFERENCE_BOOST_QUERIES:
        try:
            prefEmbeddings = await _encodeBatch(PREFERENCE_BOOST_QUERIES, queryMode=True)
            for orig_idx, emb in idx_to_embedding.items():
                prefBest = 0.0
                for qEmb in prefEmbeddings:
                    s = _cosine_sim(qEmb, emb)
                    if s > prefBest:
                        prefBest = s
                if prefBest >= PREFERENCE_BOOST_MIN_SIM:
                    line_scores[orig_idx] = max(line_scores.get(orig_idx, 0.0), prefBest)
        except Exception as e:
            logger.debug("Preference boost embedding failed (non-fatal): %s", e)

    # 4c. Apply per-line score adjustments:
    #   - Recency boost: +RECENCY_BOOST_PER_YEAR per year closer to reference_date
    #   - Emoji priority boost: 🔴 +0.03, 🟡 +0.01
    idx_to_ml: Dict[int, MemoryLine] = {ml.line_index: ml for ml in parsed}
    if reference_date or EMOJI_BOOST:
        ref_year: Optional[int] = None
        if reference_date:
            try:
                ref_year = int(reference_date[:4])
            except (ValueError, TypeError):
                pass

        for orig_idx in list(line_scores.keys()):
            ml = idx_to_ml.get(orig_idx)
            if ml is None:
                continue
            adj = 0.0
            # Recency boost
            if ref_year and ml.date_str:
                try:
                    line_year = int(ml.date_str[:4])
                    years_diff = ref_year - line_year  # positive = older
                    if years_diff >= 0:
                        # Closer to reference_date gets a larger boost
                        adj += max(0.0, RECENCY_BOOST_PER_YEAR * (1.0 - years_diff * 0.5))
                except (ValueError, TypeError):
                    pass
            # Emoji priority boost
            adj += EMOJI_BOOST.get(ml.emoji, 0.0)
            if adj:
                line_scores[orig_idx] = line_scores[orig_idx] + adj

    # 5. Pick top-K lines above min_similarity
    scored_sorted = sorted(
        [(orig_idx, score) for orig_idx, score in line_scores.items() if score >= min_similarity],
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    if not scored_sorted:
        logger.debug("No memory lines above min_similarity=%.2f; returning full memory", min_similarity)
        return SearchResult(
            filtered_memory=memory_text,
            matched_line_count=0,
            total_line_count=len(embeddable),
            used_semantic_search=True,
        )

    # 6. Expand context around matched lines
    included_line_indices: set = set()

    # Group parsed by block_index for easy full-block inclusion
    block_to_lines: Dict[int, List[MemoryLine]] = {}
    for ml in parsed:
        block_to_lines.setdefault(ml.block_index, []).append(ml)

    # idx_to_ml was already built in step 4c; reuse it here.

    for orig_idx, score in scored_sorted:
        ml = idx_to_ml.get(orig_idx)
        if ml is None:
            continue

        if score >= high_score_threshold:
            # Include the entire date block this line belongs to
            for block_ml in block_to_lines.get(ml.block_index, []):
                included_line_indices.add(block_ml.line_index)
        elif score >= medium_score_threshold:
            # Include ±2 neighbors
            for offset in range(-2, 3):
                neighbor_idx = orig_idx + offset
                if neighbor_idx in idx_to_ml:
                    included_line_indices.add(neighbor_idx)
        else:
            included_line_indices.add(orig_idx)

    # Always include structural lines (Date: headers, blank lines) that are part
    # of included date blocks so the rebuilt string is properly formatted.
    included_blocks: set = set()
    for li in included_line_indices:
        ml = idx_to_ml.get(li)
        if ml:
            included_blocks.add(ml.block_index)

    for ml in parsed:
        if ml.is_structural and ml.block_index in included_blocks:
            included_line_indices.add(ml.line_index)

    # 7. Rebuild filtered memory, ordering by block max-score DESC so that
    # downstream char-limit truncation never drops the top semantic matches
    # when they happen to live in later date blocks. Within a block, keep
    # chronological line order.
    block_score: Dict[int, float] = {}
    for orig_idx, score in scored_sorted:
        ml = idx_to_ml.get(orig_idx)
        if ml is None:
            continue
        block_score[ml.block_index] = max(block_score.get(ml.block_index, 0.0), score)

    sorted_blocks = sorted(
        included_blocks,
        key=lambda b: -block_score.get(b, 0.0),
    )

    filtered_lines: List[str] = []
    for b in sorted_blocks:
        block_line_idxs = sorted(
            i for i in included_line_indices
            if idx_to_ml.get(i) is not None and idx_to_ml[i].block_index == b
        )
        for i in block_line_idxs:
            if i < len(parsed):
                filtered_lines.append(parsed[i].text)
        # blank separator between blocks for readability
        if filtered_lines and filtered_lines[-1].strip():
            filtered_lines.append("")

    while filtered_lines and not filtered_lines[0].strip():
        filtered_lines.pop(0)
    while filtered_lines and not filtered_lines[-1].strip():
        filtered_lines.pop()

    filtered_memory = "\n".join(filtered_lines) if filtered_lines else ""

    matched_count = len(scored_sorted)
    logger.info(
        "[memory_search] query=%r | matched=%d/%d lines | blocks=%d | output=%d chars",
        query[:80],
        matched_count,
        len(embeddable),
        len(included_blocks),
        len(filtered_memory),
    )

    return SearchResult(
        filtered_memory=filtered_memory,
        matched_line_count=matched_count,
        total_line_count=len(embeddable),
        used_semantic_search=True,
    )


# Sentinel returned when semantic search successfully ran but found zero matching lines.
# run_controller uses this to emit an explicit "no memory found" note that encourages
# the LLM to abstain rather than hallucinate an answer.
ABSTENTION_SIGNAL = "__NO_RELEVANT_MEMORY__"


async def search_memory_for_query(
    query: str,
    memory_text: str,
    memory_char_limit: int = 2000,
    reference_date: Optional[str] = None,
) -> str:
    """
    High-level entry point: returns a memory string suitable for injection
    into the LLM context.

    Special return values:
      - ``ABSTENTION_SIGNAL`` ("__NO_RELEVANT_MEMORY__"): semantic search ran
        successfully but found zero lines above min_similarity.  The caller
        should treat this as "pass empty memory" to encourage abstention.
      - ``""``  (empty string): memory_text is blank; nothing to search.
      - Any other string: filtered (and possibly truncated) memory content.

    Falls back to the raw memory text (truncated) if semantic search throws an
    exception or the memory is small enough to fit within the char limit.

    Args:
        query: User question.
        memory_text: Raw avatarMemory.
        memory_char_limit: Max chars for the returned memory string.
        reference_date: ISO date used as "today".

    Returns:
        Memory string (possibly filtered and/or truncated), or ABSTENTION_SIGNAL.
    """
    if not memory_text or not memory_text.strip():
        return ""

    # If memory fits comfortably within the limit, no need for semantic search
    if len(memory_text) <= memory_char_limit:
        return memory_text

    # Attempt semantic search
    result = await semantic_memory_search(
        query=query,
        memory_text=memory_text,
        reference_date=reference_date,
    )

    if result.used_semantic_search:
        if not result.filtered_memory and result.matched_line_count == 0:
            # Semantic search ran and found nothing relevant; signal abstention.
            logger.info(
                "[memory_search] 0 matches found for query=%r; returning abstention signal",
                query[:80],
            )
            return ABSTENTION_SIGNAL
        candidate = result.filtered_memory or memory_text
    else:
        candidate = memory_text

    # Apply char limit to the candidate (preserve whole lines)
    if len(candidate) <= memory_char_limit:
        return candidate

    # Truncate on line boundary to preserve readability
    lines = candidate.splitlines()
    truncated_lines: List[str] = []
    char_count = 0
    for line in lines:
        if char_count + len(line) + 1 > memory_char_limit:
            break
        truncated_lines.append(line)
        char_count += len(line) + 1

    return "\n".join(truncated_lines)
