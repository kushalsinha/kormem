import sqlite3

from kormem.core.config import KormemConfig
from kormem.store.sqlite_store import SqliteStore
from kormem.vault.vault_store import VaultStore


def testGuidanceMigration(tmp_path, monkeypatch):
    monkeypatch.setenv("KORMEM_HOME", str(tmp_path))
    guidance = tmp_path / "guidance.sqlite"
    index = tmp_path / "index.sqlite"
    conn = sqlite3.connect(str(guidance))
    conn.executescript(
        """
        CREATE TABLE namespace_guidance (
            namespaceId TEXT PRIMARY KEY,
            personaMd TEXT, memoryMd TEXT, playbooksMd TEXT, skillsMd TEXT, updatedAt TEXT
        );
        CREATE TABLE extraction_jobs (
            id TEXT PRIMARY KEY, namespaceId TEXT, documentId TEXT, status TEXT,
            textPreview TEXT, lastError TEXT, createdAt TEXT, updatedAt TEXT
        );
        INSERT INTO namespace_guidance VALUES (
            'main', '- Friendly tone', '- Lives in SF', '', '', '2026-01-01T00:00:00+00:00'
        );
        INSERT INTO extraction_jobs VALUES (
            'j1', 'main', 'd1', 'pending', 'preview', NULL, '2026-01-01', '2026-01-01'
        );
        """
    )
    conn.commit()
    conn.close()

    config = KormemConfig(defaultNamespace="main")
    store = SqliteStore(indexPath=index, guidancePath=guidance)
    vault = VaultStore(store=store, config=config, namespaceId="main")
    memory = vault.readNote("_meta/memory.md")
    assert memory is not None
    assert "SF" in memory.body or "Lives" in memory.body
    with store._connectIndex() as conn:
        row = conn.execute(
            "SELECT id FROM extraction_jobs WHERE id='j1' AND status='pending'"
        ).fetchone()
    assert row is not None
