"""Optional AES-256-GCM encryption at rest."""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Optional


def encryptText(plaintext: str, keyB64: Optional[str]) -> Optional[dict[str, str]]:
    if not keyB64 or not plaintext:
        return None
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        return None
    key = base64.b64decode(keyB64)
    if len(key) != 32:
        return None
    iv = os.urandom(12)
    aesgcm = AESGCM(key)
    ct = aesgcm.encrypt(iv, plaintext.encode("utf-8"), None)
    return {
        "algorithm": "aes-256-gcm",
        "encryptedData": base64.b64encode(ct).decode("ascii"),
        "iv": base64.b64encode(iv).decode("ascii"),
        "authTag": "",
    }


def decryptText(payload: dict[str, Any], keyB64: Optional[str]) -> Optional[str]:
    if not keyB64 or not payload:
        return None
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        return None
    key = base64.b64decode(keyB64)
    if len(key) != 32:
        return None
    iv = base64.b64decode(payload["iv"])
    data = base64.b64decode(payload["encryptedData"])
    aesgcm = AESGCM(key)
    pt = aesgcm.decrypt(iv, data, None)
    return pt.decode("utf-8")


def encryptJson(obj: Any, keyB64: Optional[str]) -> Optional[str]:
    enc = encryptText(json.dumps(obj), keyB64)
    return json.dumps(enc) if enc else None


def decryptJson(blob: str, keyB64: Optional[str]) -> Any:
    payload = json.loads(blob)
    text = decryptText(payload, keyB64)
    return json.loads(text) if text else None
