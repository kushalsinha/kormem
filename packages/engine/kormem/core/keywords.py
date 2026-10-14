"""Keyword hashing for privacy-preserving keyword search."""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List


STOP_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their",
    "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
}


def computeHash(word: str) -> str:
    return hashlib.sha256(word.encode()).hexdigest()[:16]


def hashKeywords(text: str, maxKeywords: int = 25) -> List[str]:
    if not text:
        return []
    words = re.findall(r"\b[a-zA-Z0-9]+\b", text)
    keywords = [
        w.lower()
        for w in words
        if w.lower() not in STOP_WORDS and len(w) > 2
    ]
    seen: set[str] = set()
    unique: List[str] = []
    for w in keywords:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return [computeHash(kw) for kw in unique[:maxKeywords]]


def wordVariants(word: str) -> List[str]:
    """The word plus simple plural / tense forms, so "rollback" also finds "rollbacks"."""
    w = word.lower()
    out = {w, w + "s", w + "es", w + "ed", w + "ing"}
    for suffix in ("ies", "es", "s", "ed", "ing"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            stem = w[: -len(suffix)]
            out |= {stem, stem + "s", stem + "e", stem + "y" if suffix == "ies" else stem}
    return sorted(out)


def queryKeywordGroups(text: str, maxKeywords: int = 25) -> List[List[str]]:
    """Hashes per query word: each group is one word and its variants (a match counts once)."""
    words = [w.lower() for w in re.findall(r"\b[a-zA-Z0-9]+\b", text or "")]
    unique = list(dict.fromkeys(w for w in words if w not in STOP_WORDS and len(w) > 2))
    return [[computeHash(v) for v in wordVariants(w)] for w in unique[:maxKeywords]]


def extractTfidfKeywords(text: str, maxKeywords: int = 25) -> Dict[str, float]:
    words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    freq: Dict[str, int] = {}
    for w in words:
        if w not in STOP_WORDS and len(w) > 2:
            freq[w] = freq.get(w, 0) + 1
    if not freq:
        return {}
    total = sum(freq.values())
    scored = {w: c / total for w, c in freq.items()}
    top = sorted(scored.items(), key=lambda x: -x[1])[:maxKeywords]
    return dict(top)
