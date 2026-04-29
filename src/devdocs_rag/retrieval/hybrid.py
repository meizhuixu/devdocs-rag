"""Hybrid retrieval: BM25 + dense, fused via reciprocal rank fusion."""

from __future__ import annotations

import logging
from collections.abc import Iterable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

RRF_K = 60


class RetrievedChunk(BaseModel):
    """A single retrieved chunk plus the score that placed it here."""

    namespace: str
    path: str
    symbol: str
    chunk_type: str = Field(description="code | doc")
    text: str
    score: float
    metadata: dict[str, str] = Field(default_factory=dict)


def reciprocal_rank_fusion(
    *ranked_lists: Iterable[tuple[str, float]], k: int = RRF_K
) -> list[tuple[str, float]]:
    """Standard RRF. Inputs are iterables of (doc_id, score) sorted by score desc.

    Returns a unified list of (doc_id, fused_score) sorted desc.
    """
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda pair: pair[1], reverse=True)


def search(
    query: str,
    namespaces: list[str],
    top_k: int = 20,
) -> list[RetrievedChunk]:
    """Run hybrid retrieval across the given namespaces.

    Phase 1 returns an empty list — no data indexed yet. The shape and fusion
    logic are real, so Phase 2 only fills in the BM25/dense calls.
    """
    logger.info(
        "hybrid search",
        extra={"namespaces": namespaces, "top_k": top_k, "query_len": len(query)},
    )
    # TODO(Phase 2): per-namespace BM25 + dense, then RRF, then return top_k.
    return []
