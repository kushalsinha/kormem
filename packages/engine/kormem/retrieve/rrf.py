"""Reciprocal Rank Fusion."""

from __future__ import annotations

from typing import Dict, List, Tuple


def reciprocalRankFusion(
    rankedLists: List[List[Tuple[str, float]]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    bestRanks: Dict[str, int] = {}
    for ranked in rankedLists:
        for rank, (key, _) in enumerate(ranked):
            if key not in bestRanks or rank < bestRanks[key]:
                bestRanks[key] = rank
    scores: Dict[str, float] = {}
    for key, rank in bestRanks.items():
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])
