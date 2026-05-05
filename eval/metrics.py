"""Deterministic retrieval metrics — no LLM judge required.

Each function accepts a ranked list of RetrievedChunk (highest-score first)
and a golden list of expected chunks (`relevant`), and returns a float in [0, 1].

`relevant` items are dicts with string keys "namespace", "path", "symbol" —
the same triple stored in every Qdrant chunk payload.
"""

from __future__ import annotations

from devdocs_rag.retrieval.hybrid import RetrievedChunk


def _chunk_key(chunk: RetrievedChunk) -> tuple[str, str, str]:
    return (chunk.namespace, chunk.path, chunk.symbol)


def _rel_keys(relevant: list[dict[str, str]]) -> set[tuple[str, str, str]]:
    return {(r["namespace"], r["path"], r["symbol"]) for r in relevant}


def recall_at_k(
    retrieved: list[RetrievedChunk],
    relevant: list[dict[str, str]],
    k: int,
) -> float:
    """Fraction of the relevant set found in the top-k retrieved results.

    Returns 1.0 when `relevant` is empty (vacuously satisfied).
    """
    if not relevant:
        return 1.0
    rel = _rel_keys(relevant)
    hits = sum(1 for c in retrieved[:k] if _chunk_key(c) in rel)
    return hits / len(rel)


def precision_at_k(
    retrieved: list[RetrievedChunk],
    relevant: list[dict[str, str]],
    k: int,
) -> float:
    """Fraction of the top-k retrieved chunks that appear in the relevant set.

    Returns 0.0 when the retrieved list is empty.
    """
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    rel = _rel_keys(relevant)
    return sum(1 for c in top_k if _chunk_key(c) in rel) / len(top_k)


def mrr_at_k(
    retrieved: list[RetrievedChunk],
    relevant: list[dict[str, str]],
    k: int,
) -> float:
    """Reciprocal rank of the first relevant chunk in the top-k retrieved results.

    Returns 1.0 when `relevant` is empty; 0.0 when no relevant chunk is found.
    """
    if not relevant:
        return 1.0
    rel = _rel_keys(relevant)
    for rank, chunk in enumerate(retrieved[:k], start=1):
        if _chunk_key(chunk) in rel:
            return 1.0 / rank
    return 0.0
