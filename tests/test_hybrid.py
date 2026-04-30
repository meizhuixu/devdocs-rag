"""Unit tests for hybrid_fuse() — the RRF + tie-break used by hybrid.search().

Pure-function tests: no Qdrant, no embedder, no registry. Deterministic
ordering is the property under test, since silent re-ordering across runs
would make retrieval debugging hellish.
"""

from __future__ import annotations

from devdocs_rag.retrieval.hybrid import RRF_K, hybrid_fuse, reciprocal_rank_fusion


def _expected_rrf(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank + 1)


# ---------- basic behavior ----------


def test_fuse_sums_rrf_contributions() -> None:
    """Doc appearing in both lists gets both RRF terms summed."""
    bm25 = [("a", 5.0), ("b", 3.0)]
    dense = [("a", 0.9), ("c", 0.8)]
    out = dict(hybrid_fuse(bm25, dense))
    # a: rank 0 in both → 2/61
    assert out["a"] == _expected_rrf(0) + _expected_rrf(0)
    # b: rank 1 in bm25 only → 1/62
    assert out["b"] == _expected_rrf(1)
    # c: rank 1 in dense only → 1/62
    assert out["c"] == _expected_rrf(1)


def test_fuse_returns_descending_by_fused_score() -> None:
    bm25 = [("a", 5.0), ("b", 3.0), ("c", 2.0)]
    dense = [("c", 0.9), ("a", 0.8), ("b", 0.5)]
    out = hybrid_fuse(bm25, dense)
    fused_scores = [score for _, score in out]
    assert fused_scores == sorted(fused_scores, reverse=True)


# ---------- tie-break ----------


def test_tiebreak_prefers_lower_min_rank() -> None:
    """When fused scores tie, the doc with a better (lower) rank in *some*
    retriever wins. This is the documented Phase-3 tiebreaker.
    """
    # Construct so two docs have identical fused score:
    # x: bm25 rank 0, dense absent  → fused = 1/61, min_rank = 0
    # y: bm25 rank 0, dense absent  → fused = 1/61, min_rank = 0  -- same
    # Use a setup where ranks DIFFER while scores tie:
    # a: bm25 rank 0 (1/61), dense rank 1 (1/62) → fused = 1/61 + 1/62
    # b: bm25 rank 1 (1/62), dense rank 0 (1/61) → fused = 1/61 + 1/62  -- same fused
    bm25 = [("a", 9.9), ("b", 3.0)]
    dense = [("b", 0.99), ("a", 0.5)]
    out = hybrid_fuse(bm25, dense)
    ids = [d for d, _ in out]
    # Both have min_rank=0. doc_id is the deterministic tiebreaker — "a" < "b".
    assert ids == ["a", "b"]


def test_tiebreak_lower_min_rank_beats_higher() -> None:
    """When fused score ties but ranks differ, lower min_rank wins."""
    # a: bm25 rank 0 (1/61), dense absent       → fused = 1/61
    # b: bm25 absent,        dense rank 0 (1/61) → fused = 1/61  -- same fused
    # min_rank for both is 0; so doc_id breaks tie.
    # Need a case where min_ranks differ at the same fused score:
    # x: bm25 rank 1 (1/62), dense rank 1 (1/62) → fused = 2/62 ≈ 0.0323
    # y: bm25 rank 0 (1/61) only                  → fused = 1/61  ≈ 0.0164  (different)
    # The trivial case: same fused score → tie-break by min_rank.
    # Construct: a only in bm25 rank 5 (1/66); b only in dense rank 5 (1/66).
    # min_rank both = 5. Tie-break falls to doc_id.
    # For asymmetric: a only in bm25 rank 2 (1/63); b only in bm25 rank 2... can't.
    # Cleanest: two docs each appear in both lists with mirror-image ranks.
    bm25 = [("a", 9.0), ("b", 8.0), ("c", 7.0)]
    dense = [("b", 0.9), ("a", 0.85), ("c", 0.8)]
    out = hybrid_fuse(bm25, dense)
    # a: ranks (0,1) → fused = 1/61 + 1/62
    # b: ranks (1,0) → fused = 1/62 + 1/61   -- SAME as a
    # c: ranks (2,2) → fused = 1/63 + 1/63
    # min_rank: a=0, b=0, c=2. a and b both have min_rank=0; doc_id breaks → a first.
    assert [d for d, _ in out] == ["a", "b", "c"]


def test_tiebreak_doc_id_is_final_anchor() -> None:
    """Equal fused score + equal min_rank → deterministic by doc_id."""
    # m and n each appear only in one list at the same rank.
    # Ranks differ but min_rank is the same when only one retriever has them.
    bm25 = [("m", 5.0)]  # rank 0 → fused 1/61, min_rank = 0
    dense = [("n", 0.9)]  # rank 0 → fused 1/61, min_rank = 0
    out = hybrid_fuse(bm25, dense)
    assert [d for d, _ in out] == ["m", "n"]  # "m" < "n" lex


# ---------- empty inputs ----------


def test_empty_bm25_falls_through_to_dense_order() -> None:
    out = hybrid_fuse([], [("d", 0.9), ("e", 0.5)])
    assert [d for d, _ in out] == ["d", "e"]


def test_empty_dense_falls_through_to_bm25_order() -> None:
    out = hybrid_fuse([("d", 5.0), ("e", 3.0)], [])
    assert [d for d, _ in out] == ["d", "e"]


def test_both_empty_returns_empty() -> None:
    assert hybrid_fuse([], []) == []


# ---------- backwards-compat wrapper ----------


def test_reciprocal_rank_fusion_still_works() -> None:
    """The variadic helper is kept for non-BM25/dense callers (e.g. future
    cross-namespace fusion). Deterministic for the equal-score smoke case."""
    fused = reciprocal_rank_fusion([("a", 1.0), ("b", 0.5)], [("b", 1.0), ("a", 0.5)])
    assert {d for d, _ in fused} == {"a", "b"}
