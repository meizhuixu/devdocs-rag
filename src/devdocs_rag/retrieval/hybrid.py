"""Hybrid retrieval: BM25 + dense, fused via reciprocal rank fusion.

Phase 4 lifts the single-namespace assertion. Multi-namespace search runs
per-namespace BM25+dense, fuses each via `hybrid_fuse` (rank-aware RRF),
then RRF-of-RRFs the per-namespace lists via `cross_namespace_fuse`. Doc
IDs are namespace-unique by construction (uuid5 of `<ns>|<path>|...`), so
a flat `dict[doc_id, namespace]` is sufficient to track which Qdrant
collection to hydrate from.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

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

    Each list is `(namespace, doc_id, score)` triples. Phase 4 added the
    namespace tag so multi-NS debug breakdowns can show which namespace
    contributed which hit. Single-NS callers see one namespace value
    repeated, which the UI handles fine.
    """

    bm25_top: list[tuple[str, str, float]] = Field(default_factory=list)
    dense_top: list[tuple[str, str, float]] = Field(default_factory=list)
    rrf_top: list[tuple[str, str, float]] = Field(default_factory=list)


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


def cross_namespace_fuse(
    per_ns_fused: list[list[tuple[str, float]]],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """RRF-of-RRF across namespaces. Each input list is one namespace's
    already-fused BM25+dense ranking.

    Tie-break: `(-cross_fused_score, min_rank_across_ns, doc_id)`. Same
    rank-axis philosophy as `hybrid_fuse` — RRF is rank-based so the
    tiebreaker stays on rank, not raw score (which would mix incompatible
    scales across namespaces with different BM25 token distributions).

    Empty inputs are valid: a single non-empty list passes through, and an
    all-empty input list set returns `[]`.
    """
    rank_in: list[dict[str, int]] = [
        {doc_id: rk for rk, (doc_id, _) in enumerate(lst)} for lst in per_ns_fused
    ]
    fused: dict[str, float] = {}
    for lst in per_ns_fused:
        for rank, (doc_id, _score) in enumerate(lst):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    def sort_key(pair: tuple[str, float]) -> tuple[float, int, str]:
        doc_id, score = pair
        ranks = [r.get(doc_id, _RANK_INF) for r in rank_in]
        return (-score, min(ranks) if ranks else _RANK_INF, doc_id)

    return sorted(fused.items(), key=sort_key)


def search(
    query: str,
    namespaces: list[str],
    top_k: int = 10,
) -> list[RetrievedChunk]:
    """Run hybrid retrieval and hydrate the top_k payloads.

    Wrapper around `search_with_debug` that drops the debug payload — kept
    as a stable narrow API for tests / probe scripts that don't need debug.
    """
    chunks, _ = search_with_debug(query, namespaces, top_k)
    return chunks


def search_with_debug(
    query: str,
    namespaces: list[str],
    top_k: int = 10,
) -> tuple[list[RetrievedChunk], RetrievalDebug]:
    """Run hybrid retrieval across one or more namespaces.

    Per-namespace flow:
        BM25 (top retriever_top_k) ─┐
                                    ├─ hybrid_fuse → per-NS fused list
        Dense (top retriever_top_k) ─┘

    Cross-namespace flow (only when len(namespaces) > 1):
        per_ns_fused lists → cross_namespace_fuse → flat fused list

    Hydration:
        Top top_k pairs → group by namespace → per-collection batch
        retrieve → emit RetrievedChunks in fused order.

    `RetrievalDebug` is populated regardless of whether the API exposes it
    (the API's `expose_retrieval_debug` flag controls only whether it ships
    over the wire); cost is modest since we already have the data in hand.
    """
    debug = RetrievalDebug()
    if not query.strip() or not namespaces:
        return [], debug

    settings = get_settings()
    pool_size = settings.retriever_top_k

    per_ns_fused: list[list[tuple[str, float]]] = []
    ns_of: dict[str, str] = {}
    bm25_all: list[tuple[str, str, float]] = []
    dense_all: list[tuple[str, str, float]] = []

    for ns in namespaces:
        bm25_hits = get_bm25_index(ns).search(query, top_k=pool_size)
        dense_hits = dense_search(query, ns, top_k=pool_size)
        per_ns = hybrid_fuse(bm25_hits, dense_hits)

        per_ns_fused.append(per_ns)
        for doc_id, _ in per_ns:
            ns_of.setdefault(doc_id, ns)
        bm25_all.extend((ns, did, s) for did, s in bm25_hits)
        dense_all.extend((ns, did, s) for did, s in dense_hits)

        logger.info(
            "hybrid: ns=%s bm25=%d dense=%d fused=%d",
            ns,
            len(bm25_hits),
            len(dense_hits),
            len(per_ns),
        )

    cross_fused = per_ns_fused[0] if len(per_ns_fused) == 1 else cross_namespace_fuse(per_ns_fused)
    final_pairs = cross_fused[:top_k]

    debug = RetrievalDebug(
        bm25_top=bm25_all,
        dense_top=dense_all,
        rrf_top=[(ns_of.get(did, ""), did, score) for did, score in final_pairs],
    )

    logger.info(
        "cross-NS: namespaces=%s candidates=%d returning=%d",
        namespaces,
        len(cross_fused),
        len(final_pairs),
    )

    if not final_pairs:
        return [], debug

    return _hydrate_chunks_multi_ns(final_pairs, ns_of), debug


def _hydrate_chunks_multi_ns(
    pairs: list[tuple[str, float]],
    ns_of: dict[str, str],
) -> list[RetrievedChunk]:
    """Fetch payloads for `pairs` across whichever namespaces they live in.

    Groups pairs by namespace, batches one Qdrant `retrieve` per
    collection, then re-emits results in the original fused order.
    Single-namespace case takes a single retrieve call (same cost as the
    Phase 3 single-NS path).
    """
    from qdrant_client import QdrantClient

    settings = get_settings()

    by_ns: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for doc_id, score in pairs:
        ns = ns_of.get(doc_id)
        if ns:
            by_ns[ns].append((doc_id, score))

    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    payload_by_id: dict[str, dict[str, Any]] = {}
    for ns, ns_pairs in by_ns.items():
        ids = [pid for pid, _ in ns_pairs]
        if not ids:
            continue
        points = client.retrieve(
            collection_name=ns,
            ids=ids,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            payload_by_id[str(p.id)] = p.payload or {}

    out: list[RetrievedChunk] = []
    for doc_id, score in pairs:
        payload = payload_by_id.get(doc_id)
        if payload is None:
            continue
        ns = ns_of.get(doc_id, "")
        out.append(
            RetrievedChunk(
                namespace=str(payload.get("namespace") or ns),
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
