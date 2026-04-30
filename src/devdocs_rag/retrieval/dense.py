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


_EMBEDDER_SINGLETON: Embedder | None = None


def get_embedder() -> Embedder:
    """Return the active embedder, cached per-process after first call.

    `USE_MOCK_EMBEDDINGS=true` (Phase 1 default) → MockEmbedder.
    `USE_MOCK_EMBEDDINGS=false` → SentenceTransformerEmbedder with Redis cache.
    Redis unavailable degrades to no-cache (logs a warning, doesn't crash).

    Process-level caching matters for the interactive retrieval path: every
    call into `dense_search()` would otherwise reload bge-base from disk
    (~2.5s on M3 MPS) and burn the cold-start cost on every query.
    """
    global _EMBEDDER_SINGLETON
    if _EMBEDDER_SINGLETON is not None:
        return _EMBEDDER_SINGLETON

    settings = get_settings()
    if settings.use_mock_embeddings:
        _EMBEDDER_SINGLETON = MockEmbedder(settings.dense_dim)
        return _EMBEDDER_SINGLETON

    # Lazy imports keep Phase 1 / mock-mode test paths free of torch + redis.
    from devdocs_rag.retrieval.sentence_transformer_embedder import (
        SentenceTransformerEmbedder,
    )

    cache = _connect_cache(settings.redis_url)
    _EMBEDDER_SINGLETON = SentenceTransformerEmbedder(
        model_name=settings.dense_model_name,
        cache=cache,
        batch_size=settings.embed_batch_size,
        max_seq_length=settings.max_seq_length,
    )
    return _EMBEDDER_SINGLETON


def _reset_embedder_for_tests() -> None:
    """Drop the cached embedder. Tests use this between cases."""
    global _EMBEDDER_SINGLETON
    _EMBEDDER_SINGLETON = None


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


def dense_search(
    query: str,
    namespace: str,
    top_k: int,
    embedder: Embedder | None = None,
) -> list[tuple[str, float]]:
    """Embed `query` and run a Qdrant nearest-neighbour search in `namespace`.

    Returns `[(point_id, cosine_score), ...]` sorted desc. The collection name
    is the namespace itself (one collection per namespace by convention), so
    no payload filter is needed.
    """
    settings = get_settings()
    if embedder is None:
        embedder = get_embedder()
    vector = embedder.embed([query])[0]

    from qdrant_client import QdrantClient

    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    response = client.query_points(
        collection_name=namespace,
        query=vector,
        limit=top_k,
        with_payload=False,
        with_vectors=False,
    )
    return [(str(p.id), float(p.score)) for p in response.points]
