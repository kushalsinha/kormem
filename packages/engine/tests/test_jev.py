import asyncio
import json

import httpx
import pytest

from kormem.core import models
from kormem.core.config import KormemConfig
from kormem.core.ranks import rankIndex, rankLabel
from kormem.retrieve import jev

_AsyncClient = httpx.AsyncClient
_sleep = asyncio.sleep


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("KORMEM_HOME", str(tmp_path))
    monkeypatch.delenv(models.DECISION_ENV, raising=False)
    return tmp_path


def _fakeHttp(monkeypatch, handler):
    """Route jev's httpx client through ``handler(request) -> httpx.Response``."""
    sent = []

    async def record(request):
        sent.append(json.loads(request.content))
        return await handler(request)
    monkeypatch.setattr(jev.httpx, "AsyncClient",
                        lambda **kw: _AsyncClient(transport=httpx.MockTransport(record), **kw))
    return sent


def _answer(p: float) -> httpx.Response:
    return httpx.Response(200, json={"model": "jev-1.13.0", "answers": {"needed": {"noul": p}},
                                     "usage": {"input_tokens": 120, "output_tokens": 3}})


async def testNoKeyMeansNoCall(home, monkeypatch):
    sent = _fakeHttp(monkeypatch, lambda r: _answer(0.9))
    assert await jev.scoreSources("q", "2023-05-30", ["a"], KormemConfig()) is None
    assert sent == []


async def testScoresEachPreviewWithTheTestedQuestion(home, monkeypatch):
    async def handler(request):
        state = json.loads(request.content)["state"]
        if state["source"] == "broken":
            return httpx.Response(503)
        return _answer(0.94 if "Target" in state["source"] else 0.05)
    sent = _fakeHttp(monkeypatch, handler)
    monkeypatch.setattr(jev.asyncio, "sleep", lambda s: _sleep(0))
    config = KormemConfig(typesafeApiKey="k")
    scores = await jev.scoreSources("Where did I redeem the coupon?", "2023-05-30",
                                    ["redeemed at Target", "weather chat", "broken"], config)
    assert scores == [0.94, 0.05, None]  # a source that keeps failing is unscored, not fatal
    body = sent[0]
    assert body["model"] == "jev-1.13.0"
    assert body["state"]["question_date"] == "2023-05-30"
    assert body["questions"]["needed"]["type"] == "noul" and "even partly" in body["questions"]["needed"]["instructions"]
    assert sum(b["state"]["source"] == "broken" for b in sent) == jev.RETRIES


async def testSlowOrFailingJevFallsBack(home, monkeypatch):
    async def slow(request):
        await asyncio.sleep(1)
        return _answer(0.9)
    _fakeHttp(monkeypatch, slow)
    monkeypatch.setattr(jev, "TIMEOUT_SEC", 0.05)
    assert await jev.scoreSources("q", "d", ["a", "b"], KormemConfig(typesafeApiKey="k")) is None

    async def unauthorized(request):
        return httpx.Response(401)
    _fakeHttp(monkeypatch, unauthorized)
    monkeypatch.setattr(jev, "TIMEOUT_SEC", 3.0)
    assert await jev.scoreSources("q", "d", ["a"], KormemConfig(typesafeApiKey="bad")) is None


def testDecisionModelSettings(home, monkeypatch):
    from kormem.cli.main import main
    from kormem.core import config as cfg
    assert models.decisionModel(cfg.loadConfig()) == ("jev-1.13.0", "default")
    main(["model", "set", "decision", "jev-1.14.0"])
    assert models.decisionModel(cfg.loadConfig()) == ("jev-1.14.0", "config")
    monkeypatch.setenv(models.DECISION_ENV, "jev-latest")
    assert models.decisionModel(cfg.loadConfig()) == ("jev-latest", "env:KORMEM_DECISION_MODEL")
    monkeypatch.delenv(models.DECISION_ENV)
    main(["model", "set", "decision", "default"])
    assert "decisionModel" not in cfg.readRawConfig()
    with pytest.raises(SystemExit):
        main(["model", "set", "decision", "gpt-6-luna"])


def testQueryRewriteIsOffEvenForOldConfigs():
    assert KormemConfig.fromDict({"queryOptimize": True}).queryRewrite is False  # 0.1.x wrote this default
    assert KormemConfig.fromDict({"queryRewrite": True}).queryRewrite is True


def testRankLabels():
    s = {"rank": 2, "relevance": 0.937, "agentChose": True}
    assert rankLabel(s) == "rank 2 · relevance 0.94 · chosen by the recall agent"
    assert rankLabel({"rank": 1, "relevance": None}) == "rank 1"
    assert rankLabel({"text": "old cached context"}) == ""
    assert rankIndex([{"rank": 2, "relevance": 0.1, "id": "b"}, {"rank": 1, "relevance": 0.9, "id": "a"}],
                     lambda s: s["id"]) == "1. a (0.90)\n2. b (0.10)"


# ------------------------------------------------------------ recall


@pytest.fixture
async def tenChats(tmp_path, monkeypatch):
    from kormem.ingest.chat_sessions import ChatSessionService
    from kormem.ingest.service import IngestService
    from kormem.store.sqlite_store import SqliteStore
    monkeypatch.setenv("KORMEM_HOME", str(tmp_path))
    config = KormemConfig(defaultNamespace="t", embeddingDim=64, embeddingProvider="hash", rerankEnabled=False)
    store = SqliteStore(indexPath=tmp_path / "i.sqlite", guidancePath=tmp_path / "g.sqlite")
    chats = ChatSessionService(ingest=IngestService(store=store, config=config), config=config)
    for i in range(10):
        topic = "coupon redeemed at Target" if i in (2, 5, 8) else f"small talk number {i}"
        await chats.append("t", f"c{i}", [{"role": "user", "content": f"Chat {i}: {topic}."}],
                           sessionDate=f"2023/05/{10 + i:02d} (Wed) 10:00")
    return store, config


def _pipeline(store, config, monkeypatch, scores):
    from kormem.retrieve import recall as recallMod
    calls = {"agent": 0, "jev": 0}

    async def fakeScores(question, questionDate, previews, cfg):
        calls["jev"] += 1
        return scores(previews)

    async def fakeAgent(question, briefing, tools, valid, **kw):
        calls["agent"] += 1
        from kormem.retrieve.recursive_agent import RecallResult
        return RecallResult(sources=["S6"], findings=["Redeemed at Target (S6)"], rounds=1)
    monkeypatch.setattr(recallMod.jev, "scoreSources", fakeScores)
    monkeypatch.setattr(recallMod, "recursiveRecall", fakeAgent)
    pipe = recallMod.RecallPipeline(store=store, config=config)
    monkeypatch.setattr(pipe.reranker, "_load", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    return pipe, calls


async def testLowEffortRanksWithJevAndTrims(tenChats, monkeypatch):
    store, config = tenChats
    config.typesafeApiKey = "k"
    pipe, calls = _pipeline(store, config, monkeypatch,
                            lambda previews: [0.95 if "Target" in p else 0.05 for p in previews])
    ctx = await pipe.recall("Where did I redeem the coupon?", "t", "2023/05/30 (Tue) 20:42")
    assert calls == {"agent": 0, "jev": 1}  # low effort: no agent
    assert not ctx.findings and ctx.debug["agent"] == {"skipped": "low effort"}
    cand = {c["sid"]: c for c in ctx.debug["candidates"]}
    trimmed = ctx.debug["trimmed"]
    assert trimmed and all(cand[sid]["jev"] < 0.3 for sid in trimmed)  # only low scorers ...
    assert all(list(cand).index(sid) >= 5 for sid in trimmed)           # ... below the top 5
    assert len(ctx.sources) == 8 - len(trimmed)
    target = [s for s in ctx.sources if s["relevance"] == 0.95]
    assert len(target) == 3  # every source Jev rates as needed stays
    assert sorted(s["rank"] for s in ctx.sources) == list(range(1, len(ctx.sources) + 1))
    assert [s["date"] for s in ctx.sources] == sorted(s["date"] for s in ctx.sources)  # still date order


async def testWithoutJevNothingIsTrimmed(tenChats, monkeypatch):
    store, config = tenChats
    pipe, calls = _pipeline(store, config, monkeypatch, lambda previews: None)
    ctx = await pipe.recall("Where did I redeem the coupon?", "t", "2023/05/30 (Tue) 20:42")
    assert calls == {"agent": 0, "jev": 0}  # no key: Jev is not asked
    assert len(ctx.sources) == 8 and ctx.debug["trimmed"] == []
    assert all(s["relevance"] is None for s in ctx.sources)
    assert sorted(s["rank"] for s in ctx.sources) == list(range(1, 9))


@pytest.mark.parametrize("effort", ["high", "medium"])  # medium: what 0.1.x clients send
async def testHighEffortRunsTheAgent(tenChats, monkeypatch, effort):
    store, config = tenChats
    config.typesafeApiKey = "k"
    pipe, calls = _pipeline(store, config, monkeypatch,
                            lambda previews: [0.95 if "Target" in p else 0.05 for p in previews])
    ctx = await pipe.recall("Where did I redeem the coupon?", "t", "2023/05/30 (Tue) 20:42", effort=effort)
    assert calls == {"agent": 1, "jev": 1}
    assert ctx.findings and ctx.debug["effort"] == "high" and ctx.debug["trimmed"] == []
    chosen = [s for s in ctx.sources if s["agentChose"]]
    assert [s["sid"] for s in chosen] == ["S6"] and chosen[0]["rank"] == 1
