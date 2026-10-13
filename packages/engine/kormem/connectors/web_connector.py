"""Web URL connector — HTTP fetch with optional Firecrawl."""

from __future__ import annotations

import os
import re
from typing import List, Optional

import httpx
from bs4 import BeautifulSoup

from kormem.connectors.base import BaseConnector, SourceFile


class WebConnector(BaseConnector):
    sourceType = "web"

    def __init__(self, firecrawlApiKey: Optional[str] = None):
        self.firecrawlApiKey = firecrawlApiKey or os.environ.get("FIRECRAWL_API_KEY")

    async def listFiles(self, target: str) -> List[SourceFile]:
        return [SourceFile(path=target, name=target)]

    async def fetchContent(self, target: str) -> str:
        if self.firecrawlApiKey:
            text = await self._fetchFirecrawl(target)
            if text:
                return text
        return await self._fetchHttp(target)

    async def _fetchHttp(self, url: str) -> str:
        async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
            resp = await client.get(url, headers={"User-Agent": "Kormem/0.1"})
            resp.raise_for_status()
            contentType = resp.headers.get("content-type", "")
            if "html" in contentType:
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer"]):
                    tag.decompose()
                text = soup.get_text(separator="\n")
                return re.sub(r"\n{3,}", "\n\n", text).strip()
            return resp.text

    async def _fetchFirecrawl(self, url: str) -> Optional[str]:
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    "https://api.firecrawl.dev/v1/scrape",
                    headers={
                        "Authorization": f"Bearer {self.firecrawlApiKey}",
                        "Content-Type": "application/json",
                    },
                    json={"url": url, "formats": ["markdown"]},
                )
                if resp.status_code != 200:
                    return None
                data = resp.json()
                md = (data.get("data") or {}).get("markdown")
                return md or None
        except Exception:
            return None
