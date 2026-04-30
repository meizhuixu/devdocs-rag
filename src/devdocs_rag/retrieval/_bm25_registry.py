"""In-memory BM25 cache, one index per namespace.

Built lazily on first call (`get_bm25_index`); the FastAPI lifespan hook
calls `prime_namespace()` so the cost is paid once at startup rather than
on the first user query.

Source-of-truth for the indexed corpus is Qdrant — we just rebuild from a
scroll. After re-ingestion, restart the API to refresh the index. Phase 4
adds an admin reload endpoint when multi-namespace incremental updates
need it.
"""

from __future__ import annotations

import logging
import threading

from devdocs_rag.config import get_settings
from devdocs_rag.ingestion.qdrant_writer import QdrantWriter
from devdocs_rag.retrieval.bm25 import BM25Index

logger = logging.getLogger(__name__)

_INDEXES: dict[str, BM25Index] = {}
_LOCK = threading.Lock()


def get_bm25_index(namespace: str) -> BM25Index:
    """Return the BM25 index for `namespace`, building lazily on first call.

    Double-checked locking — concurrent first-callers don't double-build.
    """
    if namespace in _INDEXES:
        return _INDEXES[namespace]
    with _LOCK:
        if namespace not in _INDEXES:
            _INDEXES[namespace] = _build(namespace)
    return _INDEXES[namespace]


def prime_namespace(namespace: str) -> None:
    """Eagerly build the index. FastAPI lifespan startup calls this."""
    get_bm25_index(namespace)


def clear() -> None:
    """Drop all cached indexes. Tests use this to reset between cases."""
    with _LOCK:
        _INDEXES.clear()


def _build(namespace: str) -> BM25Index:
    settings = get_settings()
    # vector_dim is unused for read-only ops, but the writer constructor wants it.
    writer = QdrantWriter(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection=namespace,
        vector_dim=settings.dense_dim,
    )
    return BM25Index.from_qdrant(writer)
