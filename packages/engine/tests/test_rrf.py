from kormem.retrieve.rrf import reciprocalRankFusion


def testRrfMergesLists():
    listA = [("a", 1.0), ("b", 0.9)]
    listB = [("b", 1.0), ("c", 0.8)]
    fused = reciprocalRankFusion([listA, listB], k=60)
    keys = [k for k, _ in fused]
    assert "b" in keys
    assert "a" in keys
    assert "c" in keys
    scores = {k: s for k, s in fused}
    assert scores["b"] >= scores["c"]
