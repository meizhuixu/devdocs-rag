"""Hybrid retrieval: BM25 + dense, fused via reciprocal rank fusion."""

from __future__ import annotations

import logging
from collections.abc import Iterable

from pydantic import BaseModel, Field

from devdocs_rag.config import get_settings
from devdocs_rag.retrieval._bm25_registry import get_bm25_index
from devdocs_rag.retrieval.dense import dense_search

logger = logging.getLogger(__name__)

RRF_K = 60

# Sentinel for "this doc didn't appear in this retriever". Used in tie-break
# so doc-not-in-retriever ranks worse than any real rank without crashing.
_RANK_INF = 10**9


class RetrievedChunk(BaseModel):
    """A single retrieved chunk plus the score that placed it here."""

    namespace: str
    path: str
    symbol: str
    chunk_type: str = Field(description="code | doc")
    text: str
    score: float
    metadata: dict[str, str] = Field(default_factory=dict)


class RetrievalDebug(BaseModel):
    """Intermediate retrieval stages, exposed when settings.expose_retrieval_debug is on.

    Each list is `(doc_id, score)` pairs in their respective ranking order:
    - `bm25_top` and `dense_top` are the raw retriever outputs (truncated to
      `retriever_top_k`).
    - `rrf_top` is the rank-aware RRF fusion before any truncation/rerank.
    """

    bm25_top: list[tuple[str, float]] = Field(default_factory=list)
    dense_top: list[tuple[str, float]] = Field(default_factory=list)
    rrf_top: list[tuple[str, float]] = Field(default_factory=list)


def reciprocal_rank_fusion(
    *ranked_lists: Iterable[tuple[str, float]], k: int = RRF_K
) -> list[tuple[str, float]]:
    """Standard RRF. Inputs are iterables of (doc_id, score) sorted by score desc.

    Returns a unified list of (doc_id, fused_score) sorted desc by fused score.
    Tie-break is non-deterministic — use `hybrid_fuse()` for the rank-aware
    deterministic version.
    """
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda pair: pair[1], reverse=True)


def hybrid_fuse(
    bm25_hits: list[tuple[str, float]],
    dense_hits: list[tuple[str, float]],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """RRF fuse BM25 + dense with rank-aware deterministic tie-break.

    Tie-break key: `(-fused_score, min(bm25_rank, dense_rank), doc_id)`.

    The min-of-ranks term keeps the tiebreaker on the same axis as RRF itself
    (rank, not raw score). BM25 scores and cosine similarity live in different
    scales, so a raw-score tiebreaker would be a subtle bug. `doc_id` is the
    final deterministic anchor.

    Empty inputs are valid — fusion of one list is monotonic in rank.
    """
    bm25_rank = {doc_id: rank for rank, (doc_id, _) in enumerate(bm25_hits)}
    dense_rank = {doc_id: rank for rank, (doc_id, _) in enumerate(dense_hits)}

    fused: dict[str, float] = {}
    for ranked in (bm25_hits, dense_hits):
        for rank, (doc_id, _score) in enumerate(ranked):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    def sort_key(pair: tuple[str, float]) -> tuple[float, int, str]:
        doc_id, fused_score = pair
        br = bm25_rank.get(doc_id, _RANK_INF)
        dr = dense_rank.get(doc_id, _RANK_INF)
        return (-fused_score, min(br, dr), doc_id)

    return sorted(fused.items(), key=sort_key)


def search(
    query: str,
    namespaces: list[str],
    top_k: int = 10,
) -> list[RetrievedChunk]:
    """Run hybrid retrieval and hydrate the top_k payloads.

    Wrapper around `search_with_debug` that drops the debug payload — kept as
    a stable narrow API for tests / probe scripts that don't need debug.
    """
    chunks, _ = search_with_debug(query, namespaces, top_k)
    return chunks


def search_with_debug(
    query: str,
    namespaces: list[str],
    top_k: int = 10,
) -> tuple[list[RetrievedChunk], RetrievalDebug]:
    """Run hybrid retrieval and return both the hydrated chunks and the
    intermediate BM25 / dense / RRF stages.

    Used by /query/stream when `expose_retrieval_debug=true` so the UI can
    show the four-layer (BM25 / dense / RRF / reranked) comparison side by
    side.

    Phase 3 supports a single namespace. Multi-namespace cross-RRF + per-NS
    hydration is Phase 4 work — surfaced explicitly so callers don't silently
    get a single-namespace result when they expected multi.
    """
    debug = RetrievalDebug()
    if not query.strip() or not namespaces:
        return [], debug
    if len(namespaces) > 1:
        # TODO(Phase 4): per-namespace BM25/dense → cross-namespace RRF →
        # per-pair namespace tracking for hydration.
        raise NotImplementedError(
            "Phase 3 supports a single namespace; multi-namespace cross-RRF lands in Phase 4"
        )

    namespace = namespaces[0]
    settings = get_settings()
    pool_size = settings.retriever_top_k

    bm25_hits = get_bm25_index(namespace).search(query, top_k=pool_size)
    dense_hits = dense_search(query, namespace, top_k=pool_size)
    rrf_full = hybrid_fuse(bm25_hits, dense_hits)
    fused = rrf_full[:top_k]

    debug = RetrievalDebug(
        bm25_top=bm25_hits,
        dense_top=dense_hits,
        rrf_top=rrf_full[:top_k],
    )

    logger.info(
        "hybrid: namespace=%s bm25=%d dense=%d fused=%d returning=%d",
        namespace,
        len(bm25_hits),
        len(dense_hits),
        len(fused),
        min(len(fused), top_k),
    )

    if not fused:
        return [], debug

    return _hydrate_chunks(fused, namespace), debug


def _hydrate_chunks(pairs: list[tuple[str, float]], namespace: str) -> list[RetrievedChunk]:
    """Fetch payloads for the fused top_k by point ID, in fused order."""
    from qdrant_client import QdrantClient

    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    ids = [pid for pid, _ in pairs]
    points = client.retrieve(
        collection_name=namespace,
        ids=ids,
        with_payload=True,
        with_vectors=False,
    )
    by_id = {str(p.id): p for p in points}

    out: list[RetrievedChunk] = []
    for pid, score in pairs:
        p = by_id.get(pid)
        if p is None:
            continue
        payload = p.payload or {}
        out.append(
            RetrievedChunk(
                namespace=str(payload.get("namespace") or namespace),
                path=str(payload.get("file_path") or ""),
                symbol=str(payload.get("symbol") or ""),
                chunk_type=str(payload.get("chunk_type") or "doc"),
                text=str(payload.get("text") or ""),
                score=score,
                metadata={
                    "heading_path": str(payload.get("heading_path") or ""),
                    "language": str(payload.get("language") or ""),
                    "commit_sha": str(payload.get("commit_sha") or ""),
                    "start_line": str(payload.get("start_line") or ""),
                    "end_line": str(payload.get("end_line") or ""),
                },
            )
        )
    return out
