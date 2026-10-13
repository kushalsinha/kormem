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
DEFAULT_LOCAL_EMBED_MODEL = "BAAI/bge-base-en-v1.5"
# Embeddings for kormem's own ranking (notes, memory lines, folders). "auto": the
# configured provider (Gemini) when it is available, else this local model.
DEFAULT_INTERNAL_EMBED_MODEL = "auto"
LOCAL_INTERNAL_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
INTERNAL_EMBED_DIMS = {"BAAI/bge-small-en-v1.5": 384, "BAAI/bge-base-en-v1.5": 768}
DEFAULT_EMBEDDING_PROVIDER = "gemini"
DEFAULT_ENGINE_PORT = 8787
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_QUERY_MODEL = "gpt-6-luna"
DEFAULT_EXTRACTION_MODEL = ""  # empty: pick by available key (see extract.llm_client)


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
    queryOptimize: bool = True
    rerankEnabled: bool = True
    rerankModel: str = DEFAULT_RERANK_MODEL
    queryModel: str = DEFAULT_QUERY_MODEL
    extractionModel: str = DEFAULT_EXTRACTION_MODEL
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
            queryOptimize=bool(data.get("queryOptimize", True)),
            rerankEnabled=rerankEnabled,
            rerankModel=rerankModel,
            queryModel=queryModel,
            extractionModel=extractionModel,
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
            "queryOptimize": self.queryOptimize,
            "rerankEnabled": self.rerankEnabled,
            "rerankModel": self.rerankModel,
            "queryModel": self.queryModel,
            "extractionModel": self.extractionModel,
            "memoryExtractionEnabled": self.memoryExtractionEnabled,
            "collections": {
                name: {"path": c.path, "pattern": c.pattern}
                for name, c in self.collections.items()
            },
        }


def loadConfig(home: Optional[Path] = None) -> KormemConfig:
    path = getConfigPath(home)
    if not path.exists():
        return KormemConfig()
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return KormemConfig.fromDict(data)


def saveConfig(config: KormemConfig, home: Optional[Path] = None) -> Path:
    home = home or getKormemHome()
    home.mkdir(parents=True, exist_ok=True)
    path = getConfigPath(home)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(config.toDict(), f, default_flow_style=False)
    return path


def ensureKormemLayout(home: Optional[Path] = None) -> Path:
    home = home or getKormemHome()
    for sub in ("queue", "models", "logs", "vault"):
        (home / sub).mkdir(parents=True, exist_ok=True)
    if not getConfigPath(home).exists():
        saveConfig(KormemConfig(), home)
    return home
