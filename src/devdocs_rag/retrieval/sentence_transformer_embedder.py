"""Real dense embedder backed by sentence-transformers + Redis cache.

Selected via `USE_MOCK_EMBEDDINGS=false`. Cache key is content-addressable
(`embed:{model_tag}:{sha256(text)}`) so swapping models never returns stale
vectors. Vectors are stored as raw float32 bytes — ~4x smaller and ~10x faster
to (de)serialize than JSON.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from redis import Redis

logger = logging.getLogger(__name__)


class _ModelLike(Protocol):
    """Subset of sentence_transformers.SentenceTransformer we depend on.

    Declared as a Protocol so tests can pass a lightweight stand-in without
    importing torch / loading real weights.
    """

    def encode(
        self,
        sentences: list[str],
        batch_size: int = ...,
        normalize_embeddings: bool = ...,
        show_progress_bar: bool = ...,
        convert_to_numpy: bool = ...,
    ) -> np.ndarray: ...

    def get_sentence_embedding_dimension(self) -> int: ...


def _auto_device() -> str:
    """Return 'mps' on Apple Silicon, else 'cpu'. Lazy-imports torch."""
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        logger.warning("torch MPS check failed — using CPU", exc_info=True)
    return "cpu"


def _model_tag(model_name: str) -> str:
    """Compact, cache-key-safe tag for a HF model name. e.g. 'bge-base-en-v1.5'."""
    return model_name.replace("/", "_").replace("BAAI_", "")


class SentenceTransformerEmbedder:
    """Dense embedder with Redis-backed content-addressable cache."""

    def __init__(
        self,
        model_name: str,
        cache: Redis[Any] | None,
        batch_size: int = 32,
        max_seq_length: int = 512,
        device: str | None = None,
        model: _ModelLike | None = None,
    ) -> None:
        resolved_device = device if device is not None else _auto_device()

        resolved_model: _ModelLike
        if model is None:
            from sentence_transformers import SentenceTransformer

            real_model = SentenceTransformer(model_name, device=resolved_device)
            real_model.max_seq_length = max_seq_length
            resolved_model = real_model
        else:
            resolved_model = model

        self._model: _ModelLike = resolved_model
        self._device = resolved_device
        self._batch_size = batch_size
        self._max_seq_length = max_seq_length
        self._cache = cache
        self._model_tag = _model_tag(model_name)
        self._dim = resolved_model.get_sentence_embedding_dimension()

        logger.info(
            "[embedder] device=%s, model=%s, batch=%d, max_seq=%d, dim=%d, cache=%s",
            resolved_device,
            model_name,
            batch_size,
            max_seq_length,
            self._dim,
            "redis" if cache is not None else "off",
        )

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        keys = [self._key(t) for t in texts]
        cached: list[list[float] | None] = (
            self._cache_mget(keys) if self._cache is not None else [None] * len(texts)
        )
        missing_idx = [i for i, v in enumerate(cached) if v is None]

        if missing_idx:
            missing_texts = [texts[i] for i in missing_idx]
            new_vectors = self._encode(missing_texts)
            if self._cache is not None:
                self._cache_mset({keys[i]: new_vectors[j] for j, i in enumerate(missing_idx)})
            for j, i in enumerate(missing_idx):
                cached[i] = new_vectors[j]

        # All slots are filled now.
        return [vec for vec in cached if vec is not None]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        arr = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [vec.astype(np.float32).tolist() for vec in arr]

    def _key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"embed:{self._model_tag}:{digest}"

    def _cache_mget(self, keys: list[str]) -> list[list[float] | None]:
        assert self._cache is not None
        raw_list = self._cache.mget(keys)
        out: list[list[float] | None] = []
        for raw in raw_list:
            if raw is None:
                out.append(None)
            else:
                arr = np.frombuffer(raw, dtype=np.float32)
                out.append(arr.tolist())
        return out

    def _cache_mset(self, mapping: dict[str, list[float]]) -> None:
        assert self._cache is not None
        pipe = self._cache.pipeline()
        for k, vec in mapping.items():
            pipe.set(k, np.asarray(vec, dtype=np.float32).tobytes())
        pipe.execute()
