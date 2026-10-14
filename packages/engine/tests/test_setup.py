import os

from kormem.core import config as cfg


def test_env_file_parsing_and_precedence(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n"
        "GEMINI_API_KEY = abc\n"
        "export OPENAI_API_KEY='sk-1'\n"
        "FIRECRAWL_API_KEY=\n"
        "KORMEM_MAINTAIN_HOURS=6 # every 6 hours\n"
    )
    monkeypatch.setenv("KORMEM_ENV_FILE", str(env))
    monkeypatch.setenv("OPENAI_API_KEY", "from-shell")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.delenv("KORMEM_MAINTAIN_HOURS", raising=False)
    loaded = cfg.loadEnvFile()
    assert os.environ["GEMINI_API_KEY"] == "abc"
    assert os.environ["OPENAI_API_KEY"] == "from-shell"  # the environment wins
    assert os.environ["KORMEM_MAINTAIN_HOURS"] == "6"
    assert "FIRECRAWL_API_KEY" not in os.environ  # empty values are skipped
    assert set(loaded) == {"GEMINI_API_KEY", "KORMEM_MAINTAIN_HOURS"}


def test_template_is_private_and_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.delenv("KORMEM_ENV_FILE", raising=False)
    path, created = cfg.writeEnvTemplate(tmp_path)
    assert created and path == tmp_path / ".env"
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    assert "GEMINI_API_KEY=" in path.read_text()
    path.write_text("GEMINI_API_KEY=x\n")
    assert cfg.writeEnvTemplate(tmp_path) == (path, False)
    assert path.read_text() == "GEMINI_API_KEY=x\n"


class _Store:
    def __init__(self):
        self.resets = 0

    def resetEmbeddings(self):
        self.resets += 1
        return 0


def test_provider_switch_drops_vectors_once(tmp_path, monkeypatch):
    from kormem.embeddings.reembed import switchIfNeeded

    monkeypatch.setenv("KORMEM_HOME", str(tmp_path))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    store = _Store()
    assert switchIfNeeded(store, cfg.loadConfig()) is None  # first start: just records "local"
    assert cfg.getEmbeddingPinPath().read_text().strip() == "local"
    assert switchIfNeeded(store, cfg.loadConfig()) is None and store.resets == 0

    monkeypatch.setenv("GEMINI_API_KEY", "k")  # key added: vectors are dropped and re-made
    assert switchIfNeeded(store, cfg.loadConfig()) == "local"
    assert store.resets == 1
    assert cfg.getEmbeddingPinPath().read_text().strip() == "gemini"
    assert switchIfNeeded(store, cfg.loadConfig()) is None and store.resets == 1


async def test_reset_then_reembed_restores_every_vector(tmp_path, monkeypatch):
    from kormem.embeddings.provider import HashEmbeddingProvider
    from kormem.embeddings.reembed import reembedPending
    from kormem.ingest.service import IngestService
    from kormem.store.sqlite_store import SqliteStore

    monkeypatch.setenv("KORMEM_HOME", str(tmp_path / "home"))
    config = cfg.KormemConfig(defaultNamespace="t", embeddingDim=64, embeddingProvider="hash")
    store = SqliteStore(indexPath=tmp_path / "i.sqlite", guidancePath=tmp_path / "g.sqlite")
    ingest = IngestService(store=store, config=config)
    ingest.embedder = HashEmbeddingProvider(64)
    await ingest.ingestText("Blue-green deploys on Fly.io. " * 200, namespaceId="t", title="Deploy", enqueue=False)
    before = len(store.vectorSearch("t", [1.0] * 64, limit=1000))
    assert before > 1

    store.resetEmbeddings()
    assert store.vectorSearch("t", [1.0] * 64) == []
    assert store.getStatus("t")["pendingEmbeds"] == before

    total = 0
    while n := await reembedPending(store, config, limit=2):
        total += n
    assert total == before
    assert len(store.vectorSearch("t", [1.0] * 64, limit=1000)) == before
