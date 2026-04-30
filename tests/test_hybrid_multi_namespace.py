"""Unit tests for cross_namespace_fuse — the RRF-of-RRF used when
hybrid.search() spans multiple namespaces.

Pure-function tests: no Qdrant, no embedder, no registry. Mirrors the
shape of test_hybrid.py for hybrid_fuse.
"""

from __future__ import annotations

from devdocs_rag.retrieval.hybrid import RRF_K, cross_namespace_fuse


def _expected_rrf(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank + 1)


# ---------- basic behavior ----------


def test_single_input_passes_through_in_order() -> None:
    """One namespace's fused list, fed to cross-NS fuse, should yield the
    same doc_id ordering — RRF of one list is monotonic in rank.
    """
    per_ns = [[("a", 0.05), ("b", 0.04), ("c", 0.03)]]
    out = cross_namespace_fuse(per_ns)
    assert [d for d, _ in out] == ["a", "b", "c"]


def test_two_disjoint_namespaces_interleave_by_rank() -> None:
    """Two NS with no overlapping doc_ids — RRF of two single-rank lists
    interleaves them. doc_id is the deterministic anchor when ranks tie.
    """
    ns1 = [("a", 0.05), ("b", 0.04)]
    ns2 = [("c", 0.05), ("d", 0.04)]
    out = cross_namespace_fuse([ns1, ns2])
    ids = [d for d, _ in out]
    # Each rank-0 of either NS → fused = 1/61, each rank-1 → 1/62.
    # Within tied scores, doc_id breaks ties: "a" < "c", "b" < "d".
    assert ids == ["a", "c", "b", "d"]


def test_doc_in_multiple_namespaces_gets_summed_score() -> None:
    """Same doc_id (vanishingly rare in practice — UUIDs collide with
    probability ~0 — but worth pinning the math)."""
    ns1 = [("x", 0.05)]
    ns2 = [("x", 0.05)]
    out = cross_namespace_fuse([ns1, ns2])
    assert out == [("x", _expected_rrf(0) * 2)]


# ---------- tie-break ----------


def test_tiebreak_prefers_lower_min_rank_across_ns() -> None:
    """When fused scores tie, the doc with lower rank in some NS wins."""
    ns1 = [("a", 0.05), ("b", 0.04), ("c", 0.03)]
    ns2 = [("b", 0.05), ("a", 0.04), ("c", 0.03)]
    out = cross_namespace_fuse([ns1, ns2])
    # a: ranks (0, 1) → fused = 1/61 + 1/62; min_rank = 0
    # b: ranks (1, 0) → fused = 1/62 + 1/61; min_rank = 0  -- same fused, same min_rank
    # c: ranks (2, 2) → fused = 2/63; min_rank = 2
    # a vs b: same fused, same min_rank, doc_id breaks → a first
    ids = [d for d, _ in out]
    assert ids == ["a", "b", "c"]


def test_tiebreak_min_rank_resolves_when_ranks_differ() -> None:
    """One doc only in NS1 at rank 0; another only in NS2 at rank 1."""
    ns1 = [("p", 0.05)]
    ns2 = [("dummy", 0.05), ("q", 0.04)]
    out = cross_namespace_fuse([ns1, ns2])
    # p: rank 0 in ns1, absent in ns2 → fused = 1/61, min_rank = 0
    # q: rank 1 in ns2 → fused = 1/62, min_rank = 1
    # dummy: rank 0 in ns2 → fused = 1/61 (tied with p), min_rank = 0
    # p vs dummy: tied fused + min_rank → doc_id ("dummy" < "p")
    ids = [d for d, _ in out]
    assert ids == ["dummy", "p", "q"]


# ---------- empty inputs ----------


def test_one_empty_namespace_is_ignored() -> None:
    ns1: list[tuple[str, float]] = []
    ns2 = [("a", 0.05), ("b", 0.04)]
    out = cross_namespace_fuse([ns1, ns2])
    assert [d for d, _ in out] == ["a", "b"]


def test_all_empty_returns_empty() -> None:
    assert cross_namespace_fuse([[], [], []]) == []


def test_zero_namespaces_returns_empty() -> None:
    assert cross_namespace_fuse([]) == []


# ---------- determinism ----------


def test_deterministic_across_runs() -> None:
    """Same input → same output. RRF + tie-break must be stable."""
    ns1 = [("a", 0.05), ("b", 0.04), ("c", 0.03)]
    ns2 = [("d", 0.05), ("e", 0.04)]
    first = cross_namespace_fuse([ns1, ns2])
    second = cross_namespace_fuse([ns1, ns2])
    assert first == second
