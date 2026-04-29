"""Reranker: Cohere Rerank API (prod) or cross-encoder fallback (local)."""

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
    """No-op reranker. Phase 1 default — returns the input unchanged (top_k cut)."""

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return chunks[:top_k]


def get_reranker() -> Reranker:
    """Choose reranker based on env. Phase 1 always returns Identity."""
    settings = get_settings()
    if settings.cohere_api_key:
        # TODO(Phase 3): return CohereReranker(settings.cohere_api_key)
        logger.info("cohere key present but Phase 1 uses IdentityReranker")
    # TODO(Phase 3): return CrossEncoderReranker(settings.cross_encoder_model)
    return IdentityReranker()
