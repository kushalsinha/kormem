"""Shared document content packing helpers."""

from __future__ import annotations

import json
from typing import Optional

from kormem.core.encryption import decryptText, encryptText


def packDocumentContent(content: str, encryptionKey: Optional[str]) -> Optional[str]:
    enc = encryptText(content, encryptionKey)
    if enc:
        return json.dumps(enc)
    if content:
        return json.dumps({"plain": content})
    return None


def unpackDocumentContent(blob: str, encryptionKey: Optional[str]) -> str:
    if not blob:
        return ""
    try:
        payload = json.loads(blob)
        if isinstance(payload, dict) and "plain" in payload:
            return str(payload["plain"])
        return decryptText(payload, encryptionKey) or ""
    except Exception:
        return ""
