"""Which model does each job.

- ``extraction``: files chats, documents and facts into the vault (background).
- ``recall``: query rewriting and the recall agent behind ``kormem query``.
- ``embedding``: the vectors behind semantic search. Every stored vector comes
  from one model, so changing it re-embeds the whole index.

The answer itself comes from the harness's own model (Claude Code, Cursor, ...).

``kormem model set`` writes the choice to ``config.yaml``; an environment
variable (``KORMEM_EXTRACTION_MODEL``, ``KORMEM_RECALL_MODEL``) overrides it.
Unset, the LLM jobs use ``gpt-6-luna`` when ``OPENAI_API_KEY`` is set, and
Gemini only when its key is the only one.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

from kormem.core.config import KormemConfig, apiEmbeddingModel, selectedEmbeddingProvider

DEFAULT_OPENAI_MODEL = "gpt-6-luna"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
LLM_ROLES = ("extraction", "recall")
# For the re-embed estimate in `kormem model set embedding`: chunks are ~1,000 characters.
TOKENS_PER_CHUNK = 250
# List prices per million input tokens, Oct 2026, for the estimate only.
EMBED_PRICE_PER_M = {"gemini-embedding-2": 0.20, "text-embedding-3-small": 0.02, "text-embedding-3-large": 0.13}
ROLES = (*LLM_ROLES, "embedding")

# role -> (config field, environment variables that override it, first wins)
LLM_SETTINGS = {
    "extraction": ("extractionModel", ("KORMEM_EXTRACTION_MODEL",)),
    "recall": ("queryModel", ("KORMEM_RECALL_MODEL", "KORMEM_QUERY_MODEL")),
}
_PROVIDER_KEYS = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY"}
EMBED_KEYS = _PROVIDER_KEYS  # embedding provider -> its key


def _key(name: str, config: KormemConfig) -> bool:
    attr = {"OPENAI_API_KEY": "openaiApiKey", "GEMINI_API_KEY": "geminiApiKey"}[name]
    return bool(os.environ.get(name) or getattr(config, attr, None))


def defaultLlm(config: KormemConfig) -> str:
    """gpt-6-luna when an OpenAI key is set (cheapest strong option), else Gemini."""
    if _key("OPENAI_API_KEY", config):
        return DEFAULT_OPENAI_MODEL
    if _key("GEMINI_API_KEY", config):
        return DEFAULT_GEMINI_MODEL
    return DEFAULT_OPENAI_MODEL


def llmProvider(model: str) -> Optional[str]:
    m = model.lower()
    if m.startswith("gemini"):
        return "gemini"
    if m.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    return None


def providerKey(provider: str) -> str:
    return _PROVIDER_KEYS[provider]


def llmModel(role: str, config: KormemConfig) -> Tuple[str, str]:
    """``(model, source)`` for an LLM role; source is ``env:<VAR>``, ``config`` or ``default``."""
    field, envVars = LLM_SETTINGS[role]
    for var in envVars:
        if os.environ.get(var):
            return os.environ[var], f"env:{var}"
    if getattr(config, field):
        return getattr(config, field), "config"
    return defaultLlm(config), "default"


def embeddingModel(config: KormemConfig) -> str:
    """The embedding model in use, as ``kormem model`` names it."""
    provider = selectedEmbeddingProvider(config)
    if provider in ("gemini", "openai"):
        return apiEmbeddingModel(config)
    if provider == "hash":
        return "hash"
    return f"local:{config.localEmbedModel}"


def embeddingIdentity(config: KormemConfig) -> str:
    """What the stored vectors depend on: provider, model and dimension."""
    provider = selectedEmbeddingProvider(config)
    model = config.localEmbedModel if provider == "local" else apiEmbeddingModel(config) if provider != "hash" else ""
    return f"{provider}:{model}:{config.embeddingDim}"


def snapshot(config: KormemConfig) -> Dict[str, str]:
    """Every role's model, for /health: the CLI restarts an engine whose models differ."""
    out = {role: llmModel(role, config)[0] for role in LLM_ROLES}
    out["embedding"] = embeddingIdentity(config)
    return out
