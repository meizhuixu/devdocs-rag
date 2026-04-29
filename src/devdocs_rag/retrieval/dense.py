"""Dense embedding interface.

Phase 1: returns deterministic pseudo-random vectors so tests are reproducible.
Phase 2 swaps in `sentence-transformers` with `bge-large-en-v1.5`.
"""

from __future__ import annotations

import hashlib
import logging
import struct
from typing import Protocol

from devdocs_rag.config import get_settings

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """The interface every dense embedder must satisfy."""

    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class MockEmbedder:
    """Deterministic pseudo-random vectors derived from sha256(text).

    Stable across runs (so tests don't flake) and the right shape so downstream
    code (Qdrant upsert, cosine math) doesn't care that it's mocked.
    """

    def __init__(self, dim: int | None = None) -> None:
        self._dim = dim or get_settings().dense_dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        # Hash → bytes → floats in [-1, 1]. Repeat to fill `dim`.
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        floats: list[float] = []
        i = 0
        while len(floats) < self._dim:
            chunk = digest[(i * 4) % len(digest) : (i * 4) % len(digest) + 4]
            if len(chunk) < 4:
                chunk = (chunk + digest)[:4]
            (raw,) = struct.unpack("i", chunk)
            floats.append(raw / 2**31)
            i += 1
        return floats[: self._dim]


def get_embedder() -> Embedder:
    """Phase 1: always returns MockEmbedder. Phase 2 wires the real one."""
    settings = get_settings()
    if settings.use_mock_embeddings:
        return MockEmbedder(settings.dense_dim)
    # TODO(Phase 2): return SentenceTransformerEmbedder(settings.dense_model_name)
    return MockEmbedder(settings.dense_dim)
