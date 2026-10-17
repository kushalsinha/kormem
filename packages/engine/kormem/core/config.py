"""Kormem configuration and paths."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

DEFAULT_NAMESPACE = "main"
DEFAULT_EMBEDDING_DIM = 768
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"
DEFAULT_OPENAI_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_LOCAL_EMBED_MODEL = "BAAI/bge-base-en-v1.5"
# Embeddings for kormem's own ranking (notes, memory lines, folders). "auto": the
# configured API provider (OpenAI or Gemini) when it is available, else this local model.
DEFAULT_INTERNAL_EMBED_MODEL = "auto"
LOCAL_INTERNAL_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
INTERNAL_EMBED_DIMS = {"BAAI/bge-small-en-v1.5": 384, "BAAI/bge-base-en-v1.5": 768}
# "auto": OpenAI when its key is set, else Gemini when its key is set, else local.
DEFAULT_EMBEDDING_PROVIDER = "auto"
DEFAULT_ENGINE_PORT = 8787
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_QUERY_MODEL = ""  # empty: pick by available key (see extract.llm_client)
DEFAULT_EXTRACTION_MODEL = ""  # empty: pick by available key (see extract.llm_client)
DEFAULT_DECISION_MODEL = ""  # empty: jev-1.13.0 (see core.models)


def getKormemHome() -> Path:
    raw = os.environ.get("KORMEM_HOME", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path.home() / ".kormem"


def getIndexPath(home: Optional[Path] = None) -> Path:
    return (home or getKormemHome()) / "index.sqlite"


def getGuidancePath(home: Optional[Path] = None) -> Path:
    return (home or getKormemHome()) / "guidance.sqlite"


def getConfigPath(home: Optional[Path] = None) -> Path:
    return (home or getKormemHome()) / "config.yaml"


def getEnvPath(home: Optional[Path] = None) -> Path:
    raw = os.environ.get("KORMEM_ENV_FILE", "").strip()
    if raw and home is None:
        return Path(raw).expanduser()
    return (home or getKormemHome()) / ".env"


# API keys kormem uses. Their presence decides which services run.
API_KEYS = ("GEMINI_API_KEY", "OPENAI_API_KEY", "TYPESAFE_API_KEY", "FIRECRAWL_API_KEY")

ENV_TEMPLATE = """\
# OPENAI_API_KEY is the only key kormem needs: gpt-6-luna runs extraction into
# the vault and the recall agent, and text-embedding-3-small runs embeddings.
# https://platform.openai.com/api-keys
OPENAI_API_KEY=

# Optional. GEMINI_API_KEY runs embeddings with gemini-embedding-2 when it is the
# only key, or after `kormem model set embedding gemini-embedding-2`.
# Without any key, embeddings run locally (free and private, slower to build).
# https://aistudio.google.com/apikey
GEMINI_API_KEY=

# Optional. TYPESAFE_API_KEY lets Jev (jev-1.13.0) re-rank and trim the sources
# recall returns: better context in about 0.2 s, ~$0.001 per question. Short
# previews of the matching sources are sent to TypeSafe. Without it, recall
# ranks locally. https://typesafe.ai
TYPESAFE_API_KEY=

# `kormem model` shows and changes the model behind each job.
# Variables already set in the environment win over this file.
"""


def parseEnvFile(path: Path) -> dict[str, str]:
    """``KEY=value`` lines (spaces around ``=``, quotes and ``export`` allowed); empty values skipped."""
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key and value:
            values[key] = value
    return values


def loadEnvFile(home: Optional[Path] = None) -> list[str]:
    """Load ``~/.kormem/.env`` (or ``KORMEM_ENV_FILE``) into ``os.environ``; set variables win.

    Returns the names it set.
    """
    loaded = []
    for key, value in parseEnvFile(getEnvPath(home)).items():
        if not os.environ.get(key):
            os.environ[key] = value
            loaded.append(key)
    return loaded


def writeEnvTemplate(home: Optional[Path] = None) -> tuple[Path, bool]:
    """Create the env file with empty keys (owner-only) unless it exists. Returns (path, created)."""
    path = getEnvPath(home)
    if path.exists():
        return path, False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ENV_TEMPLATE, encoding="utf-8")
    path.chmod(0o600)
    return path, True


def apiKeysSet() -> list[str]:
    return [k for k in API_KEYS if os.environ.get(k)]


@dataclass
class CollectionConfig:
    name: str
    path: str
    pattern: str = "**/*"


@dataclass
class KormemConfig:
    defaultNamespace: str = DEFAULT_NAMESPACE
    embeddingDim: int = DEFAULT_EMBEDDING_DIM
    embeddingProvider: str = DEFAULT_EMBEDDING_PROVIDER
    embeddingModel: str = DEFAULT_EMBEDDING_MODEL
    localEmbedModel: str = DEFAULT_LOCAL_EMBED_MODEL
    internalEmbedModel: str = DEFAULT_INTERNAL_EMBED_MODEL
    openaiApiKey: Optional[str] = None
    geminiApiKey: Optional[str] = None
    firecrawlApiKey: Optional[str] = None
    encryptionKey: Optional[str] = None
    enginePort: int = DEFAULT_ENGINE_PORT
    collections: dict[str, CollectionConfig] = field(default_factory=dict)
    # Rewrite the question with the recall LLM (rewrite, HyDE, sub-questions) before chunk search.
    # Off: on LongMemEval it added ~5 s per query and did not change recall.
    queryRewrite: bool = False
    rerankEnabled: bool = True
    rerankModel: str = DEFAULT_RERANK_MODEL
    queryModel: str = DEFAULT_QUERY_MODEL
    extractionModel: str = DEFAULT_EXTRACTION_MODEL
    decisionModel: str = DEFAULT_DECISION_MODEL
    typesafeApiKey: Optional[str] = None
    memoryExtractionEnabled: bool = True

    @classmethod
    def fromDict(cls, data: dict[str, Any]) -> "KormemConfig":
        collections: dict[str, CollectionConfig] = {}
        for name, col in (data.get("collections") or {}).items():
            if isinstance(col, dict):
                collections[name] = CollectionConfig(
                    name=name,
                    path=col.get("path", ""),
                    pattern=col.get("pattern", "**/*"),
                )
        embeddingProvider = (
            data.get("embeddingProvider")
            or os.environ.get("KORMEM_EMBEDDING_PROVIDER")
            or DEFAULT_EMBEDDING_PROVIDER
        )
        embeddingDim = int(
            data.get("embeddingDim")
            or os.environ.get("GEMINI_EMBEDDING_DIMENSIONS")
            or DEFAULT_EMBEDDING_DIM
        )
        embeddingModel = (
            data.get("embeddingModel")
            or os.environ.get("GEMINI_EMBEDDING_MODEL")
            or DEFAULT_EMBEDDING_MODEL
        )
        localEmbedModel = (
            data.get("localEmbedModel")
            or os.environ.get("KORMEM_LOCAL_EMBED_MODEL")
            or DEFAULT_LOCAL_EMBED_MODEL
        )
        internalEmbedModel = (
            data.get("internalEmbedModel")
            or os.environ.get("KORMEM_INTERNAL_EMBED_MODEL")
            or DEFAULT_INTERNAL_EMBED_MODEL
        )
        rerankModel = (
            data.get("rerankModel")
            or os.environ.get("KORMEM_RERANK_MODEL")
            or DEFAULT_RERANK_MODEL
        )
        rerankEnabledRaw = data.get("rerankEnabled")
        if rerankEnabledRaw is None:
            envRerank = os.environ.get("KORMEM_RERANK_ENABLED")
            rerankEnabled = envRerank.lower() not in ("0", "false", "no") if envRerank else True
        else:
            rerankEnabled = bool(rerankEnabledRaw)
        queryModel = (
            data.get("queryModel")
            or os.environ.get("KORMEM_QUERY_MODEL")
            or DEFAULT_QUERY_MODEL
        )
        extractionModel = (
            data.get("extractionModel")
            or os.environ.get("KORMEM_EXTRACTION_MODEL")
            or DEFAULT_EXTRACTION_MODEL
        )
        memoryExtractionEnabled = bool(
            data.get("memoryExtractionEnabled", True)
            if data.get("memoryExtractionEnabled") is not None
            else os.environ.get("KORMEM_MEMORY_EXTRACTION_ENABLED", "true").lower()
            in ("1", "true", "yes")
        )
        return cls(
            defaultNamespace=data.get("defaultNamespace", DEFAULT_NAMESPACE),
            embeddingDim=embeddingDim,
            embeddingProvider=embeddingProvider,
            embeddingModel=embeddingModel,
            localEmbedModel=localEmbedModel,
            internalEmbedModel=internalEmbedModel,
            openaiApiKey=data.get("openaiApiKey") or os.environ.get("OPENAI_API_KEY"),
            geminiApiKey=data.get("geminiApiKey") or os.environ.get("GEMINI_API_KEY"),
            firecrawlApiKey=data.get("firecrawlApiKey") or os.environ.get("FIRECRAWL_API_KEY"),
            encryptionKey=data.get("encryptionKey") or os.environ.get("KORMEM_ENCRYPTION_KEY"),
            enginePort=int(data.get("enginePort", DEFAULT_ENGINE_PORT)),
            collections=collections,
            # 0.1.x wrote `queryOptimize: true` into every config, so that key is not read.
            queryRewrite=bool(data.get("queryRewrite", False)),
            rerankEnabled=rerankEnabled,
            rerankModel=rerankModel,
            queryModel=queryModel,
            extractionModel=extractionModel,
            decisionModel=data.get("decisionModel") or DEFAULT_DECISION_MODEL,
            typesafeApiKey=data.get("typesafeApiKey") or os.environ.get("TYPESAFE_API_KEY"),
            memoryExtractionEnabled=memoryExtractionEnabled,
        )

    def toDict(self) -> dict[str, Any]:
        return {
            "defaultNamespace": self.defaultNamespace,
            "embeddingDim": self.embeddingDim,
            "embeddingProvider": self.embeddingProvider,
            "embeddingModel": self.embeddingModel,
            "localEmbedModel": self.localEmbedModel,
            "internalEmbedModel": self.internalEmbedModel,
            "enginePort": self.enginePort,
            "queryRewrite": self.queryRewrite,
            "rerankEnabled": self.rerankEnabled,
            "rerankModel": self.rerankModel,
            "queryModel": self.queryModel,
            "extractionModel": self.extractionModel,
            "decisionModel": self.decisionModel,
            "memoryExtractionEnabled": self.memoryExtractionEnabled,
            "collections": {
                name: {"path": c.path, "pattern": c.pattern}
                for name, c in self.collections.items()
            },
        }


def loadConfig(home: Optional[Path] = None) -> KormemConfig:
    path = getConfigPath(home)
    if not path.exists():
        return KormemConfig.fromDict({})  # still reads API keys and settings from the environment
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return KormemConfig.fromDict(data)


def getEmbeddingPinPath(home: Optional[Path] = None) -> Path:
    """The embedding provider the stored vectors were made with (see kormem.embeddings.reembed)."""
    return (home or getKormemHome()) / "embedding-provider"


def selectedEmbeddingProvider(config: KormemConfig) -> str:
    """The provider the config and keys select: the configured one, an API without its key falling back
    to local. ``auto`` picks OpenAI, then Gemini, by which key is set."""
    name = (config.embeddingProvider or DEFAULT_EMBEDDING_PROVIDER).lower()
    if name == "auto":
        name = "openai" if config.openaiApiKey else "gemini"
    if name in ("local", "hash"):
        return name
    if name == "openai":
        return "openai" if config.openaiApiKey else "local"
    return "gemini" if config.geminiApiKey else "local"


def apiEmbeddingModel(config: KormemConfig) -> str:
    """The API embedding model for the selected provider (``embeddingModel`` names it)."""
    model = config.embeddingModel or ""
    if selectedEmbeddingProvider(config) == "openai":
        return model if model.startswith("text-embedding") else DEFAULT_OPENAI_EMBED_MODEL
    return model if model.startswith("gemini") else DEFAULT_EMBEDDING_MODEL




def saveConfig(config: KormemConfig, home: Optional[Path] = None) -> Path:
    home = home or getKormemHome()
    home.mkdir(parents=True, exist_ok=True)
    path = getConfigPath(home)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(config.toDict(), f, default_flow_style=False)
    return path


def readRawConfig(home: Optional[Path] = None) -> dict[str, Any]:
    """config.yaml as written, keeping keys :class:`KormemConfig` does not model."""
    path = getConfigPath(home)
    if not path.exists():
        return KormemConfig().toDict()
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def writeRawConfig(data: dict[str, Any], home: Optional[Path] = None) -> Path:
    home = home or getKormemHome()
    home.mkdir(parents=True, exist_ok=True)
    path = getConfigPath(home)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return path


def ensureKormemLayout(home: Optional[Path] = None) -> Path:
    home = home or getKormemHome()
    for sub in ("queue", "models", "logs", "vault"):
        (home / sub).mkdir(parents=True, exist_ok=True)
    if not getConfigPath(home).exists():
        saveConfig(KormemConfig(), home)
    if not os.environ.get("KORMEM_ENV_FILE"):
        writeEnvTemplate(home)
    return home
