import os
import sqlite3

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


def test_embedding_pin_follows_keys_until_the_index_has_vectors(tmp_path, monkeypatch):
    monkeypatch.setenv("KORMEM_HOME", str(tmp_path))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert cfg.pinEmbeddingProvider() == ("local", "local")

    monkeypatch.setenv("GEMINI_API_KEY", "k")  # key added before anything was indexed
    assert cfg.pinEmbeddingProvider() == ("gemini", "gemini")

    with sqlite3.connect(cfg.getIndexPath()) as conn:
        conn.execute("CREATE TABLE segments (embedding BLOB)")
        conn.execute("INSERT INTO segments VALUES (x'00')")
    monkeypatch.delenv("GEMINI_API_KEY")  # key removed after indexing: stay on gemini
    assert cfg.pinEmbeddingProvider() == ("gemini", "local")
    assert cfg.loadConfig().embeddingProvider == "gemini"
