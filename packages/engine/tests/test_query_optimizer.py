import pytest

from kormem.core.config import KormemConfig
from kormem.retrieve.query_optimizer import QueryOptimizer, QueryComplexity


@pytest.fixture
def optimizer():
    return QueryOptimizer(KormemConfig(openaiApiKey=None))


def testClassifySimple(optimizer):
    assert optimizer.classifyQuery("what is kubernetes") == QueryComplexity.SIMPLE


def testClassifyComplex(optimizer):
    q = "Compare our billing system and deployment pipeline and explain differences"
    assert optimizer.classifyQuery(q) == QueryComplexity.COMPLEX


def testExpandQueryFallback(optimizer):
    variants = optimizer._expandQuery("What is my favorite color?")
    assert variants[0] == "What is my favorite color?"
    assert len(variants) >= 2


@pytest.mark.asyncio
async def testOptimizeWithoutLlm(optimizer):
    result = await optimizer.optimize("kubernetes helm deploy")
    assert len(result.queryVariants) >= 1
    assert len(result.queryVariants) <= 3
