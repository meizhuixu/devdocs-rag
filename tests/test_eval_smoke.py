"""Unit tests for eval/metrics.py.

All tests are deterministic — no Qdrant, no embedder, no network.
RetrievedChunk objects are constructed in-process; relevant dicts match
the (namespace, path, symbol) triple stored in every Qdrant payload.
"""

from __future__ import annotations

import sys
from pathlib import Path

from devdocs_rag.retrieval.hybrid import RetrievedChunk

# eval/ is not an installed package; add it to sys.path so `metrics` resolves.
_EVAL_DIR = Path(__file__).parent.parent / "eval"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))
from metrics import mrr_at_k, precision_at_k, recall_at_k  # noqa: E402, I001


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def chunk(namespace: str, path: str, symbol: str, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        namespace=namespace,
        path=path,
        symbol=symbol,
        chunk_type="code",
        text="stub",
        score=score,
    )


def rel(namespace: str, path: str, symbol: str) -> dict[str, str]:
    return {"namespace": namespace, "path": path, "symbol": symbol}


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------


def test_recall_perfect():
    retrieved = [
        chunk("ns", "a.py", "foo"),
        chunk("ns", "b.py", "bar"),
    ]
    relevant = [rel("ns", "a.py", "foo"), rel("ns", "b.py", "bar")]
    assert recall_at_k(retrieved, relevant, k=5) == 1.0


def test_recall_partial():
    retrieved = [chunk("ns", "a.py", "foo"), chunk("ns", "other.py", "baz")]
    relevant = [rel("ns", "a.py", "foo"), rel("ns", "b.py", "bar")]
    assert recall_at_k(retrieved, relevant, k=5) == 0.5


def test_recall_zero():
    retrieved = [chunk("ns", "other.py", "baz")]
    relevant = [rel("ns", "a.py", "foo")]
    assert recall_at_k(retrieved, relevant, k=5) == 0.0


def test_recall_empty_relevant():
    assert recall_at_k([chunk("ns", "a.py", "foo")], [], k=5) == 1.0


def test_recall_respects_k_cutoff():
    # relevant chunk is at position 3 (0-indexed 2), k=2 should miss it
    retrieved = [
        chunk("ns", "x.py", "x"),
        chunk("ns", "y.py", "y"),
        chunk("ns", "a.py", "foo"),
    ]
    relevant = [rel("ns", "a.py", "foo")]
    assert recall_at_k(retrieved, relevant, k=2) == 0.0
    assert recall_at_k(retrieved, relevant, k=3) == 1.0


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------


def test_precision_all_relevant():
    retrieved = [chunk("ns", "a.py", "foo"), chunk("ns", "b.py", "bar")]
    relevant = [rel("ns", "a.py", "foo"), rel("ns", "b.py", "bar")]
    assert precision_at_k(retrieved, relevant, k=5) == 1.0


def test_precision_half():
    retrieved = [
        chunk("ns", "a.py", "foo"),
        chunk("ns", "noise.py", "noise"),
    ]
    relevant = [rel("ns", "a.py", "foo")]
    assert precision_at_k(retrieved, relevant, k=2) == 0.5


def test_precision_zero():
    retrieved = [chunk("ns", "noise.py", "noise")]
    relevant = [rel("ns", "a.py", "foo")]
    assert precision_at_k(retrieved, relevant, k=5) == 0.0


def test_precision_empty_retrieved():
    assert precision_at_k([], [rel("ns", "a.py", "foo")], k=5) == 0.0


def test_precision_k_limits_denominator():
    # 4 retrieved but k=2: denominator is 2, not 4
    retrieved = [
        chunk("ns", "a.py", "foo"),
        chunk("ns", "b.py", "bar"),
        chunk("ns", "c.py", "baz"),
        chunk("ns", "d.py", "qux"),
    ]
    relevant = [rel("ns", "a.py", "foo"), rel("ns", "b.py", "bar")]
    assert precision_at_k(retrieved, relevant, k=2) == 1.0
    assert precision_at_k(retrieved, relevant, k=4) == 0.5


# ---------------------------------------------------------------------------
# mrr_at_k
# ---------------------------------------------------------------------------


def test_mrr_hit_at_rank1():
    retrieved = [chunk("ns", "a.py", "foo")]
    relevant = [rel("ns", "a.py", "foo")]
    assert mrr_at_k(retrieved, relevant, k=10) == 1.0


def test_mrr_hit_at_rank3():
    retrieved = [
        chunk("ns", "x.py", "x"),
        chunk("ns", "y.py", "y"),
        chunk("ns", "a.py", "foo"),
    ]
    relevant = [rel("ns", "a.py", "foo")]
    assert abs(mrr_at_k(retrieved, relevant, k=10) - 1.0 / 3) < 1e-9


def test_mrr_no_hit():
    retrieved = [chunk("ns", "x.py", "x")]
    relevant = [rel("ns", "a.py", "foo")]
    assert mrr_at_k(retrieved, relevant, k=10) == 0.0


def test_mrr_empty_relevant():
    assert mrr_at_k([chunk("ns", "a.py", "foo")], [], k=10) == 1.0


def test_mrr_first_relevant_wins():
    # two relevant chunks; mrr should be 1/rank of the first one encountered
    retrieved = [
        chunk("ns", "b.py", "bar"),
        chunk("ns", "a.py", "foo"),
    ]
    relevant = [rel("ns", "a.py", "foo"), rel("ns", "b.py", "bar")]
    # b.py/bar is rank 1 → MRR = 1.0
    assert mrr_at_k(retrieved, relevant, k=10) == 1.0


def test_mrr_respects_k_cutoff():
    # relevant chunk is rank 4; k=3 should miss it
    retrieved = [
        chunk("ns", "x.py", "x"),
        chunk("ns", "y.py", "y"),
        chunk("ns", "z.py", "z"),
        chunk("ns", "a.py", "foo"),
    ]
    relevant = [rel("ns", "a.py", "foo")]
    assert mrr_at_k(retrieved, relevant, k=3) == 0.0
    assert mrr_at_k(retrieved, relevant, k=4) == 0.25
