"""Dense embedding interface.

Phase 1: returns deterministic pseudo-random vectors so tests are reproducible.
Phase 2: swaps in `sentence-transformers` with `bge-base-en-v1.5` when the
`USE_MOCK_EMBEDDINGS` flag is False.
"""

from __future__ import annotations

import hashlib
import logging
import struct
from typing import TYPE_CHECKING, Any, Protocol

from devdocs_rag.config import get_settings

if TYPE_CHECKING:
    from redis import Redis

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
    """Return the active embedder.

    `USE_MOCK_EMBEDDINGS=true` (Phase 1 default) → MockEmbedder.
    `USE_MOCK_EMBEDDINGS=false` → SentenceTransformerEmbedder with Redis cache.
    Redis unavailable degrades to no-cache (logs a warning, doesn't crash).
    """
    settings = get_settings()
    if settings.use_mock_embeddings:
        return MockEmbedder(settings.dense_dim)

    # Lazy imports keep Phase 1 / mock-mode test paths free of torch + redis.
    from devdocs_rag.retrieval.sentence_transformer_embedder import (
        SentenceTransformerEmbedder,
    )

    cache = _connect_cache(settings.redis_url)
    return SentenceTransformerEmbedder(
        model_name=settings.dense_model_name,
        cache=cache,
        batch_size=settings.embed_batch_size,
        max_seq_length=settings.max_seq_length,
    )


def _connect_cache(redis_url: str) -> Redis[Any] | None:
    """Try to connect to Redis. None if unreachable (caller embeds without cache)."""
    try:
        import redis

        client: Redis[Any] = redis.from_url(redis_url)
        client.ping()
        return client
    except Exception:
        logger.warning("redis cache unreachable at %s — embedding without cache", redis_url)
        return None
