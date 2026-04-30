"""Local cross-encoder reranker for Phase 3.

Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~90 MB, ~150 MB resident on
MPS). One forward pass per (query, chunk) pair; ~50-100 ms for top-50 on
M3 MPS. Mirrors the lazy-import + injectable-model pattern from
SentenceTransformerEmbedder so unit tests don't load real weights.

Phase 4 will add a CohereReranker behind the same Protocol; the dispatcher
in `reranker.py` swaps based on `settings.reranker_type`.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from devdocs_rag.retrieval.hybrid import RetrievedChunk

logger = logging.getLogger(__name__)


class _CrossEncoderLike(Protocol):
    """Subset of `sentence_transformers.CrossEncoder` we depend on.

    Declared as a Protocol so tests inject a stub without importing torch.
    """

    def predict(
        self,
        sentences: list[tuple[str, str]],
        batch_size: int = ...,
        show_progress_bar: bool = ...,
    ) -> np.ndarray: ...


def _auto_device() -> str:
    """Return 'mps' on Apple Silicon, else 'cpu'.

    Duplicated from sentence_transformer_embedder._auto_device() — two callsites
    isn't worth a shared util module yet (see Phase 3 plan §1.6).
    """
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        logger.warning("torch MPS check failed — using CPU", exc_info=True)
    return "cpu"


class CrossEncoderReranker:
    """Reranks RetrievedChunks by cross-encoder relevance score (logits).

    `rerank()` replaces each chunk's `score` field with the cross-encoder
    logit (higher = more relevant) — a different scale from the upstream
    RRF fused score. Callers comparing pre/post rerank should keep this
    in mind; the probe script labels which scoring system is in use.
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        batch_size: int = 32,
        model: _CrossEncoderLike | None = None,
    ) -> None:
        resolved_device = device if device is not None else _auto_device()

        resolved_model: _CrossEncoderLike
        if model is None:
            from sentence_transformers import CrossEncoder

            resolved_model = CrossEncoder(model_name, device=resolved_device)
        else:
            resolved_model = model

        self._model: _CrossEncoderLike = resolved_model
        self._device = resolved_device
        self._batch_size = batch_size
        self._model_name = model_name

        logger.info(
            "[reranker] device=%s, model=%s, batch=%d",
            resolved_device,
            model_name,
            batch_size,
        )

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []

        pairs: list[tuple[str, str]] = [(query, c.text) for c in chunks]
        scores = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        ranked = sorted(
            zip(chunks, scores, strict=True),
            key=lambda pair: float(pair[1]),
            reverse=True,
        )

        out: list[RetrievedChunk] = []
        for chunk, score in ranked[:top_k]:
            out.append(chunk.model_copy(update={"score": float(score)}))
        return out
