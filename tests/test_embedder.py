"""Unit tests for SentenceTransformerEmbedder.

These tests do NOT require a real Redis or real model weights — they inject
a tiny in-memory FakeRedis and a stub model. CI-safe.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from devdocs_rag.retrieval.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
    _model_tag,
)


class FakeRedis:
    """Dict-backed Redis subset: mget / pipeline.set / execute / ping."""

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.calls = {"mget": 0, "set": 0}

    def mget(self, keys: list[str]) -> list[bytes | None]:
        self.calls["mget"] += 1
        return [self.store.get(k) for k in keys]

    def pipeline(self) -> FakeRedis:
        return self

    def set(self, k: str, v: bytes) -> FakeRedis:
        self.calls["set"] += 1
        self.store[k] = v
        return self

    def execute(self) -> None:
        pass

    def ping(self) -> bool:
        return True


class StubModel:
    """Deterministic stand-in for sentence_transformers.SentenceTransformer.

    encode() returns vectors derived from each text's hash so output is stable
    and we can verify cache contents.
    """

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim
        self.encode_calls: list[list[str]] = []

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        self.encode_calls.append(list(sentences))
        out = np.zeros((len(sentences), self._dim), dtype=np.float32)
        for i, s in enumerate(sentences):
            seed = abs(hash(s)) % (2**31)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self._dim).astype(np.float32)
            v /= np.linalg.norm(v) or 1.0
            out[i] = v
        return out

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim


@pytest.fixture
def embedder_factory() -> Any:
    def _make(cache: FakeRedis | None = None, dim: int = 8) -> Any:
        model = StubModel(dim=dim)
        emb = SentenceTransformerEmbedder(
            model_name="BAAI/bge-base-en-v1.5",
            cache=cache,
            batch_size=4,
            max_seq_length=128,
            device="cpu",
            model=model,
        )
        return emb, model

    return _make


def test_model_tag_strips_namespace() -> None:
    assert _model_tag("BAAI/bge-base-en-v1.5") == "bge-base-en-v1.5"
    assert _model_tag("sentence-transformers/all-MiniLM-L6-v2") == (
        "sentence-transformers_all-MiniLM-L6-v2"
    )


def test_dim_matches_model(embedder_factory: Any) -> None:
    emb, _ = embedder_factory(cache=None, dim=8)
    assert emb.dim == 8


def test_empty_input_short_circuits(embedder_factory: Any) -> None:
    emb, model = embedder_factory(cache=FakeRedis())
    assert emb.embed([]) == []
    assert model.encode_calls == []


def test_cache_miss_calls_encode_then_caches(embedder_factory: Any) -> None:
    cache = FakeRedis()
    emb, model = embedder_factory(cache=cache)

    out = emb.embed(["a", "b", "c"])

    assert len(out) == 3
    assert all(len(v) == 8 for v in out)
    # one encode call for all three (single batch)
    assert len(model.encode_calls) == 1
    assert model.encode_calls[0] == ["a", "b", "c"]
    # all three got cached
    assert cache.calls["set"] == 3
    assert len(cache.store) == 3


def test_cache_hit_skips_encode(embedder_factory: Any) -> None:
    cache = FakeRedis()
    emb, model = embedder_factory(cache=cache)

    # warm the cache
    first = emb.embed(["x", "y"])
    assert len(model.encode_calls) == 1

    # now everything's cached — encode must NOT be called again
    second = emb.embed(["x", "y"])
    assert len(model.encode_calls) == 1
    assert second == first


def test_partial_cache_only_encodes_missing(embedder_factory: Any) -> None:
    cache = FakeRedis()
    emb, model = embedder_factory(cache=cache)

    # warm cache for "a" only
    emb.embed(["a"])
    assert model.encode_calls[-1] == ["a"]

    # now ask for ["a", "b", "c"] — only b and c hit the model
    emb.embed(["a", "b", "c"])
    assert model.encode_calls[-1] == ["b", "c"]


def test_no_cache_always_encodes(embedder_factory: Any) -> None:
    emb, model = embedder_factory(cache=None)
    emb.embed(["foo"])
    emb.embed(["foo"])
    assert len(model.encode_calls) == 2


def test_cache_preserves_order(embedder_factory: Any) -> None:
    cache = FakeRedis()
    emb, _model = embedder_factory(cache=cache)

    # warm "b" only
    out_b = emb.embed(["b"])

    # ask for ["a", "b", "c"] — output order must match input
    out_full = emb.embed(["a", "b", "c"])
    assert out_full[1] == out_b[0]  # "b" came back from cache, same vector
    # "a" and "c" are fresh
    assert out_full[0] != out_full[1]
    assert out_full[2] != out_full[1]


def test_cache_key_format(embedder_factory: Any) -> None:
    cache = FakeRedis()
    emb, _ = embedder_factory(cache=cache)
    emb.embed(["hello"])
    keys = list(cache.store.keys())
    assert len(keys) == 1
    assert keys[0].startswith("embed:bge-base-en-v1.5:")
    # sha256 hex = 64 chars
    assert len(keys[0].split(":")[-1]) == 64
