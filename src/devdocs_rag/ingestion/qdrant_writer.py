"""Qdrant collection bootstrap + writes.

Idempotent: `ensure_collection()` is safe to call on every run. Payload
indexes are created on first creation; subsequent calls are no-ops.
"""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    HnswConfigDiff,
    MatchValue,
    OptimizersConfigDiff,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

logger = logging.getLogger(__name__)

_INDEXED_FIELDS: tuple[str, ...] = (
    "namespace",
    "file_path",
    "commit_sha",
    "chunk_type",
)


class QdrantWriter:
    """Thin wrapper around qdrant_client for one collection (= one namespace)."""

    def __init__(
        self,
        url: str,
        api_key: str | None,
        collection: str,
        vector_dim: int,
    ) -> None:
        self._client = QdrantClient(url=url, api_key=api_key, prefer_grpc=False)
        self._collection = collection
        self._vector_dim = vector_dim

    @property
    def collection(self) -> str:
        return self._collection

    def ensure_collection(self) -> None:
        """Create the collection + payload indexes if missing. Idempotent."""
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection in existing:
            logger.info("collection exists: %s", self._collection)
            return

        logger.info("creating collection: %s (dim=%d)", self._collection, self._vector_dim)
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=self._vector_dim, distance=Distance.COSINE),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20000,
                memmap_threshold=10000,
            ),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
        )
        for field in _INDEXED_FIELDS:
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        logger.info("payload indexes created: %s", _INDEXED_FIELDS)

    def upsert_points(self, points: list[PointStruct]) -> None:
        if not points:
            return
        self._client.upsert(collection_name=self._collection, points=points, wait=True)

    def delete_by_file_path(self, file_path: str) -> None:
        self._client.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="file_path", match=MatchValue(value=file_path))]
                )
            ),
            wait=True,
        )

    def scroll_file_shas(self) -> dict[str, str]:
        """Return `{file_path: commit_sha}` for every file represented in the collection.

        Uses Qdrant scroll over payload-only fetches; `with_vectors=False` keeps
        memory bounded.
        """
        result: dict[str, str] = {}
        offset: Any = None
        while True:
            points, next_offset = self._client.scroll(
                collection_name=self._collection,
                limit=1024,
                with_payload=["file_path", "commit_sha"],
                with_vectors=False,
                offset=offset,
            )
            for p in points:
                payload = p.payload or {}
                fp = payload.get("file_path")
                cs = payload.get("commit_sha")
                if isinstance(fp, str) and isinstance(cs, str):
                    result.setdefault(fp, cs)
            if next_offset is None:
                break
            offset = next_offset
        return result

    def count(self) -> int:
        return self._client.count(collection_name=self._collection, exact=True).count
