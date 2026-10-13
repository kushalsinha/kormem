import pytest
from pathlib import Path

from kormem.connectors.file_connector import FileConnector


@pytest.mark.asyncio
async def testReadMarkdown(tmp_path):
    f = tmp_path / "note.md"
    f.write_text("# Hello\n\nDeployment uses Docker.", encoding="utf-8")
    conn = FileConnector()
    text = await conn.fetchContent(str(f))
    assert "Docker" in text
