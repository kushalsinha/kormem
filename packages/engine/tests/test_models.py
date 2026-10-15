import pytest

from kormem.core import config as cfg
from kormem.core import models


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("KORMEM_HOME", str(tmp_path))
    monkeypatch.setenv("KORMEM_ENGINE_URL", "http://127.0.0.1:9")  # never a running engine
    for var in ("OPENAI_API_KEY", "GEMINI_API_KEY", "KORMEM_EXTRACTION_MODEL", "KORMEM_RECALL_MODEL",
                "KORMEM_QUERY_MODEL", "KORMEM_EMBEDDING_PROVIDER", "GEMINI_EMBEDDING_MODEL"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def test_luna_is_the_default_whenever_openai_is_available(home, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    assert models.llmModel("extraction", cfg.loadConfig()) == ("gemini-3.8-flash", "default")
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    assert models.llmModel("extraction", cfg.loadConfig()) == ("gpt-6-luna", "default")
    assert models.llmModel("recall", cfg.loadConfig()) == ("gpt-6-luna", "default")


def test_environment_overrides_config(home, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    cfg.writeRawConfig({"queryModel": "gpt-6.1-sol"})
    assert models.llmModel("recall", cfg.loadConfig()) == ("gpt-6.1-sol", "config")
    monkeypatch.setenv("KORMEM_RECALL_MODEL", "gpt-6-astra")
    assert models.llmModel("recall", cfg.loadConfig()) == ("gpt-6-astra", "env:KORMEM_RECALL_MODEL")


def test_model_set_llm(home, monkeypatch):
    from kormem.cli.main import main

    monkeypatch.setenv("OPENAI_API_KEY", "o")
    main(["model", "set", "llm", "gpt-6.1-sol"])
    raw = cfg.readRawConfig()
    assert raw["extractionModel"] == raw["queryModel"] == "gpt-6.1-sol"
    main(["model", "set", "recall", "default"])
    assert "queryModel" not in cfg.readRawConfig()
    assert models.llmModel("recall", cfg.loadConfig()) == ("gpt-6-luna", "default")
    with pytest.raises(SystemExit):
        main(["model", "set", "llm", "llama-4"])  # unknown provider


def test_embedding_change_needs_confirmation(home, monkeypatch):
    from kormem.cli.main import main

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    monkeypatch.setenv("GEMINI_API_KEY", "g")  # Gemini embeddings in use
    with pytest.raises(SystemExit):
        main(["model", "set", "embedding", "local"])  # not a terminal and no --yes
    assert cfg.readRawConfig().get("embeddingProvider") != "local"
    main(["model", "set", "embedding", "local", "--yes"])
    assert cfg.readRawConfig()["embeddingProvider"] == "local"
    main(["model", "set", "embedding", "local"])  # already in use: no prompt, no change
    monkeypatch.delenv("GEMINI_API_KEY")
    with pytest.raises(SystemExit):
        main(["model", "set", "embedding", "gemini-embedding-2", "--yes"])  # no GEMINI_API_KEY


def test_model_show(home, monkeypatch, capsys):
    from kormem.cli.main import main

    monkeypatch.setenv("OPENAI_API_KEY", "o")
    main(["model"])
    shown = capsys.readouterr().out
    assert "extraction  gpt-6-luna" in shown and "Recommended setup" in shown
    assert "embedding   text-embedding-3-small" in shown  # one OpenAI key runs everything


async def test_openai_embeddings_batch_in_order(monkeypatch):
    from kormem.embeddings import openai as oe

    sent = []

    async def fakePost(self, payload):
        sent.append(payload)
        n = len(payload["input"])
        return {"data": [{"index": i, "embedding": [float(len(payload["input"][i]))] * payload["dimensions"]}
                         for i in reversed(range(n))], "usage": {"prompt_tokens": n}}

    monkeypatch.setattr(oe.OpenAIEmbeddingProvider, "_post", fakePost)
    monkeypatch.setattr(oe, "_BATCH", 2)
    p = oe.OpenAIEmbeddingProvider("k", dimension=8)
    vecs = await p.encodeBatch(["a", "", "bbb", "cc", "dddd"])
    assert [v[0] for v in vecs] == [1.0, 0.0, 3.0, 2.0, 4.0]  # blanks get a zero vector, order kept
    assert all(len(v) == 8 for v in vecs)
    assert [len(s["input"]) for s in sent] == [2, 2]
    assert sent[0]["dimensions"] == 8 and sent[0]["model"] == "text-embedding-3-small"


def test_openai_embeddings_selected_only_with_key(home, monkeypatch):
    from kormem.embeddings.provider import getEmbeddingProvider

    cfg.writeRawConfig({"embeddingProvider": "openai"})
    assert models.embeddingModel(cfg.loadConfig()) == "local:BAAI/bge-base-en-v1.5"  # no key: local
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    config = cfg.loadConfig()
    assert models.embeddingIdentity(config) == "openai:text-embedding-3-small:768"
    p = getEmbeddingProvider(config)
    assert (p.providerName, p.modelName, p.dimension) == ("openai", "text-embedding-3-small", 768)


def test_model_set_embedding_openai(home, monkeypatch):
    from kormem.cli.main import main

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    cfg.writeRawConfig({"embeddingProvider": "gemini"})  # an index built with Gemini
    with pytest.raises(SystemExit):
        main(["model", "set", "embedding", "text-embedding-3-small", "--yes"])  # no OPENAI_API_KEY
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    main(["model", "set", "embedding", "text-embedding-3-small", "--yes"])
    raw = cfg.readRawConfig()
    assert (raw["embeddingProvider"], raw["embeddingModel"]) == ("openai", "text-embedding-3-small")


def test_auto_embedding_prefers_openai(home, monkeypatch):
    pick = lambda: models.embeddingIdentity(cfg.loadConfig())  # noqa: E731
    assert pick() == "local:BAAI/bge-base-en-v1.5:768"  # no key
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    assert pick() == "gemini:gemini-embedding-2:768"  # Gemini is the only key
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    assert pick() == "openai:text-embedding-3-small:768"  # OpenAI wins when both are set
    cfg.writeRawConfig({"embeddingProvider": "gemini", "embeddingModel": "gemini-embedding-2"})
    assert pick() == "gemini:gemini-embedding-2:768"  # an explicit choice (0.1.x configs) stays
