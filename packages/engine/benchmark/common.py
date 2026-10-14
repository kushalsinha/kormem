"""Shared helpers for the LongMemEval harness."""

from __future__ import annotations

import logging
import os
from pathlib import Path


logger = logging.getLogger(__name__)

# Each LongMemEval question gets its own namespace: ``lme_<question_id>``.
NAMESPACE_PREFIX = "lme_"


def _loadDotEnv() -> None:
    """Load the first ``.env`` found in packages/engine, packages/ or the repo root."""
    for candidate in (
        Path(__file__).resolve().parents[1] / ".env",
        Path(__file__).resolve().parents[2] / ".env",
        Path(__file__).resolve().parents[3] / ".env",
    ):
        if candidate.exists():
            for raw in candidate.read_text(encoding="utf-8").splitlines():
                raw = raw.strip()
                if not raw or raw.startswith("#") or "=" not in raw:
                    continue
                k, _, v = raw.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            logger.info("loaded env from %s", candidate)
            break
    # The published runs used OpenAI for extraction, query rewriting and recall; kormem's own
    # default is Gemini (one key), so pin them here to keep runs comparable.
    for var in ("KORMEM_EXTRACTION_MODEL", "KORMEM_QUERY_MODEL", "KORMEM_RECALL_MODEL"):
        os.environ.setdefault(var, "gpt-6-luna")


def datasetPath(explicit: str | None = None) -> str:
    """Resolve the LongMemEval dataset: explicit path, else ``LONGMEMEVAL_DATASET``."""
    path = explicit or os.environ.get("LONGMEMEVAL_DATASET")
    if not path:
        raise SystemExit(
            "LongMemEval dataset not set: pass --dataset or set LONGMEMEVAL_DATASET "
            "(download longmemeval_s_cleaned.json from "
            "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)"
        )
    return str(Path(path).expanduser())
