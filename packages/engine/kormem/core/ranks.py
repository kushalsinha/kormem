"""How recall's ranks are shown to the model that answers.

``/recall`` returns sources in date order, each with ``rank`` (1 = most
relevant), ``relevance`` (Jev's probability that it helps answer, when Jev
ran) and ``agentChose``. MCP, ``kormem query`` and the benchmark show them
the same way.
"""

from __future__ import annotations

from typing import Callable, Dict, List

RANK_NOTE = ("Sources are in date order. Rank 1 is the most relevant; relevance is the probability that a "
             "source helps answer the question. Start from the highest-ranked sources, but lower-ranked ones can "
             "still hold one item of a count, an earlier value or a date.")


def rankLabel(source: Dict) -> str:
    """``rank 2 · relevance 0.94`` for a returned source (empty for contexts without ranks)."""
    if not source.get("rank"):
        return ""
    parts = [f"rank {source['rank']}"]
    if source.get("relevance") is not None:
        parts.append(f"relevance {source['relevance']:.2f}")
    if source.get("agentChose"):
        parts.append("chosen by the recall agent")
    return " · ".join(parts)


def rankIndex(sources: List[Dict], ref: Callable[[Dict], str]) -> str:
    """The returned sources by rank, e.g. ``1. #a1b2c3 (0.94)``; ``ref`` names a source."""
    ranked = sorted((s for s in sources if s.get("rank")), key=lambda s: s["rank"])
    return "\n".join(f"{s['rank']}. {ref(s)}" + (f" ({s['relevance']:.2f})" if s.get("relevance") is not None else "")
                     for s in ranked)
