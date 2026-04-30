"""Unit tests for CrossEncoderReranker.

Stub model + injection — no torch import beyond what numpy already needs,
no real CrossEncoder weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from devdocs_rag.retrieval.cross_encoder_reranker import CrossEncoderReranker
from devdocs_rag.retrieval.hybrid import RetrievedChunk


class StubCrossEncoder:
    """Returns scores in user-supplied order; records call args for assertions."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.predict_calls: list[list[tuple[str, str]]] = []

    def predict(
        self,
        sentences: list[tuple[str, str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        self.predict_calls.append(list(sentences))
        return np.array(self._scores[: len(sentences)], dtype=np.float32)


def _chunk(text: str, score: float = 0.0, **meta: str) -> RetrievedChunk:
    return RetrievedChunk(
        namespace="pytorch_docs",
        path=f"docs/{text[:8]}.md",
        symbol=text[:16],
        chunk_type="doc",
        text=text,
        score=score,
        metadata=meta,
    )


def _make(scores: list[float]) -> tuple[CrossEncoderReranker, StubCrossEncoder]:
    stub = StubCrossEncoder(scores)
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="cpu",
        batch_size=4,
        model=stub,
    )
    return reranker, stub


def test_rerank_empty_chunks_short_circuits() -> None:
    reranker, stub = _make([])
    out = reranker.rerank("anything", [], top_k=5)
    assert out == []
    assert stub.predict_calls == []


def test_rerank_reorders_by_predicted_score() -> None:
    chunks = [_chunk("alpha", score=0.0), _chunk("bravo", score=0.0), _chunk("charlie", score=0.0)]
    # Higher logit = more relevant. Stub assigns: alpha=1, bravo=3, charlie=2.
    # Expected order: bravo > charlie > alpha.
    reranker, _ = _make([1.0, 3.0, 2.0])
    out = reranker.rerank("test query", chunks, top_k=3)
    assert [c.text for c in out] == ["bravo", "charlie", "alpha"]


def test_rerank_replaces_score_with_logit() -> None:
    chunks = [_chunk("a", score=0.99), _chunk("b", score=0.01)]
    reranker, _ = _make([2.5, -1.5])
    out = reranker.rerank("q", chunks, top_k=2)
    # Score field is the cross-encoder logit, not the upstream RRF score.
    # Stub returns float32 → expect approx, not exact.
    assert out[0].score == pytest.approx(2.5)
    assert out[1].score == pytest.approx(-1.5)


def test_rerank_truncates_to_top_k() -> None:
    chunks = [_chunk(f"c{i}", score=0.0) for i in range(5)]
    reranker, _ = _make([0.1, 0.2, 0.3, 0.4, 0.5])
    out = reranker.rerank("q", chunks, top_k=2)
    assert len(out) == 2
    # Top 2 by logit: c4 (0.5), c3 (0.4)
    assert [c.text for c in out] == ["c4", "c3"]


def test_rerank_top_k_larger_than_input_returns_all() -> None:
    chunks = [_chunk("x", 0.0), _chunk("y", 0.0)]
    reranker, _ = _make([1.0, 2.0])
    out = reranker.rerank("q", chunks, top_k=10)
    assert len(out) == 2


def test_rerank_preserves_chunk_metadata() -> None:
    """Path / symbol / heading_path / chunk_type must survive the rerank."""
    src = _chunk(
        "extending pytorch with autograd",
        score=0.0,
        heading_path="Extending PyTorch > autograd.Function",
    )
    reranker, _ = _make([4.2])
    out = reranker.rerank("custom autograd", [src], top_k=1)
    assert len(out) == 1
    assert out[0].path == src.path
    assert out[0].symbol == src.symbol
    assert out[0].chunk_type == src.chunk_type
    assert out[0].metadata == src.metadata
    assert out[0].score == pytest.approx(4.2)  # score was replaced (float32)


def test_rerank_calls_predict_once_with_all_pairs() -> None:
    chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
    reranker, stub = _make([1.0, 2.0, 3.0])
    reranker.rerank("query text", chunks, top_k=3)
    assert len(stub.predict_calls) == 1
    pairs = stub.predict_calls[0]
    assert pairs == [("query text", "a"), ("query text", "b"), ("query text", "c")]
