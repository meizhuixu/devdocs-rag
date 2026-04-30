"""Reranker dispatch.

`get_reranker()` returns a process-level singleton, picked by
`settings.reranker_type`:

- `cross_encoder` (production default) → `CrossEncoderReranker` with the
  ms-marco-MiniLM model. Loaded lazily on first call; ~3 s cold load on M3
  MPS, then ~50-100 ms per top-50 rerank pass.
- `identity` → `IdentityReranker`. No model load. Used by tests (via
  `tests/conftest.py`) and by environments without local accelerators.

Caching matters: without the singleton, every API call would re-load the
cross-encoder model from disk, paying the cold-start cost per query.

Phase 4 adds `cohere` once we wire `CohereReranker` against the same Protocol.
"""

from __future__ import annotations

import logging
from typing import Protocol

from devdocs_rag.config import get_settings
from devdocs_rag.retrieval.hybrid import RetrievedChunk

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]: ...


class IdentityReranker:
    """No-op reranker — returns the input unchanged, truncated to top_k."""

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return chunks[:top_k]


_RERANKER_SINGLETON: Reranker | None = None


def get_reranker() -> Reranker:
    """Return the active reranker, cached per-process after first call."""
    global _RERANKER_SINGLETON
    if _RERANKER_SINGLETON is not None:
        return _RERANKER_SINGLETON

    settings = get_settings()
    if settings.reranker_type == "identity":
        _RERANKER_SINGLETON = IdentityReranker()
        return _RERANKER_SINGLETON

    if settings.reranker_type == "cross_encoder":
        # Lazy import — Phase 1 / mock-mode test paths never load
        # sentence_transformers.CrossEncoder.
        from devdocs_rag.retrieval.cross_encoder_reranker import CrossEncoderReranker

        _RERANKER_SINGLETON = CrossEncoderReranker(model_name=settings.cross_encoder_model)
        return _RERANKER_SINGLETON

    # TODO(Phase 4): elif settings.reranker_type == "cohere": return CohereReranker(...)
    raise ValueError(f"unknown reranker_type={settings.reranker_type!r}")


def _reset_reranker_for_tests() -> None:
    """Drop the cached reranker. Tests use this between cases."""
    global _RERANKER_SINGLETON
    _RERANKER_SINGLETON = None
