"""BM25 sparse retrieval.

Phase 1: in-memory rank-bm25 over a list of documents the caller hands us.
Phase 2 will persist the index alongside Qdrant payloads.
"""

from __future__ import annotations

import logging
import re

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Index:
    """Thin wrapper over rank_bm25 with our tokenizer."""

    def __init__(self, corpus: list[str]) -> None:
        self._corpus = corpus
        self._tokenized = [tokenize(doc) for doc in corpus]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """Return [(doc_index, score), ...] sorted by score desc."""
        if self._bm25 is None or not self._corpus:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
        return [(idx, float(score)) for idx, score in ranked[:top_k]]
