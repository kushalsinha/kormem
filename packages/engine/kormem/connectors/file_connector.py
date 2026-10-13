"""Local file connector — PDF, DOCX, text, markdown, code."""

from __future__ import annotations

import glob as globModule
from pathlib import Path
from typing import List, Optional

from kormem.connectors.base import BaseConnector, SourceFile

SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".json", ".xml", ".html", ".htm",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".c", ".cpp",
    ".pdf", ".docx",
}


class FileConnector(BaseConnector):
    sourceType = "file"

    async def listFiles(self, target: str) -> List[SourceFile]:
        path = Path(target).expanduser()
        if path.is_file():
            return [SourceFile(path=str(path), name=path.name)]
        if "*" in target or "?" in target:
            files = []
            for p in globModule.glob(str(path), recursive=True):
                fp = Path(p)
                if fp.is_file() and fp.suffix.lower() in SUPPORTED_EXTENSIONS:
                    files.append(SourceFile(path=str(fp), name=fp.name))
            return files
        if path.is_dir():
            files = []
            for fp in path.rglob("*"):
                if fp.is_file() and fp.suffix.lower() in SUPPORTED_EXTENSIONS:
                    files.append(SourceFile(path=str(fp), name=fp.name))
            return files
        return []

    async def fetchContent(self, target: str) -> str:
        path = Path(target).expanduser()
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._readPdf(path)
        if suffix == ".docx":
            return self._readDocx(path)
        return path.read_text(encoding="utf-8", errors="replace")

    def _readPdf(self, path: Path) -> str:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        parts = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
        return "\n\n".join(parts)

    def _readDocx(self, path: Path) -> str:
        from docx import Document
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text)
