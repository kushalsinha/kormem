import pytest

from kormem.core.config import KormemConfig
from kormem.core.schemas import KnowledgeItem
from kormem.retrieve.reranker import CrossEncoderReranker


def testCeFiltersTopScoreGuard():
    reranker = CrossEncoderReranker(KormemConfig(rerankEnabled=True))
    items = [
        KnowledgeItem(segmentKey="a", text="a", ceScore=0.05),
        KnowledgeItem(segmentKey="b", text="b", ceScore=0.04),
    ]
    assert reranker._applyCeFilters(items) == []


def testCeFiltersNoiseFloor():
    reranker = CrossEncoderReranker(KormemConfig(rerankEnabled=True))
    items = [
        KnowledgeItem(segmentKey="a", text="a", ceScore=0.9),
        KnowledgeItem(segmentKey="b", text="b", ceScore=0.04),
        KnowledgeItem(segmentKey="c", text="c", ceScore=0.8),
    ]
    out = reranker._applyCeFilters(items)
    assert len(out) == 1
    assert out[0].segmentKey == "a"


def testCeFiltersGapCutoff():
    reranker = CrossEncoderReranker(KormemConfig(rerankEnabled=True))
    items = [
        KnowledgeItem(segmentKey="a", text="a", ceScore=0.9),
        KnowledgeItem(segmentKey="b", text="b", ceScore=0.05),
    ]
    out = reranker._applyCeFilters(items)
    assert len(out) == 1


@pytest.mark.asyncio
async def testRerankGracefulFallback():
    reranker = CrossEncoderReranker(KormemConfig(rerankEnabled=True, rerankModel="nonexistent-model"))
    items = [
        KnowledgeItem(segmentKey="a", text="hello", score=0.5),
        KnowledgeItem(segmentKey="b", text="world", score=0.4),
    ]
    out = await reranker.rerank("query", items)
    assert len(out) == 2
