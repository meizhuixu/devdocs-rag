"""Runtime configuration loaded from environment via pydantic-settings."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime config. Read once, injected everywhere."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM providers
    anthropic_api_key: str | None = Field(default=None)
    openai_api_key: str | None = Field(default=None)

    # Reranker
    cohere_api_key: str | None = Field(default=None)

    # Vector DB
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str | None = Field(default=None)

    # Cache / queue
    redis_url: str = Field(default="redis://localhost:6379/0")

    # Runtime
    log_level: str = Field(default="INFO")
    environment: str = Field(default="development")

    # Feature flags — Phase 1 ships with mocks ON.
    use_mock_llm: bool = Field(default=True)
    use_mock_embeddings: bool = Field(default=True)

    # Embedding model — bge-base (768d) chosen over bge-large (1024d) to fit
    # M3 Air thermal/RAM budget; MTEB delta is ~0.7. See ARCHITECTURE.md D6.
    dense_model_name: str = Field(default="BAAI/bge-base-en-v1.5")
    dense_dim: int = Field(default=768)
    embed_batch_size: int = Field(default=32)
    max_seq_length: int = Field(default=512)

    # Reranker model (fallback / local)
    cross_encoder_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Retrieval defaults — pre-rerank pool size (BM25 + dense each return this many).
    retriever_top_k: int = Field(default=50)
    rerank_top_k: int = Field(default=10)

    # Reranker dispatch. Production default: cross_encoder (local, ~150MB MPS).
    # Tests override to "identity" via conftest.py so they don't load the model.
    # TODO(Phase 4): add "cohere" once we wire CohereReranker.
    reranker_type: Literal["cross_encoder", "identity"] = Field(default="cross_encoder")

    # Ingestion flush boundaries — flush whichever fires first.
    qdrant_flush_chunks: int = Field(default=256)
    qdrant_flush_files: int = Field(default=32)

    # Per-namespace gitignore-style globs applied during the source walk.
    # Patterns are repo-root-relative; `**` matches zero or more path segments.
    # Per-namespace (not global) because user-repo namespaces will need
    # different patterns than docs namespaces (e.g. node_modules, __pycache__).
    namespace_ignore_globs: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "pytorch_docs": [
                "docs/source/scripts/**",
                "docs/source/_static/**",
            ],
        }
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance."""
    return Settings()


def configure_logging(level: str | None = None) -> None:
    """Configure root logger. Call once from app entry point."""
    settings = get_settings()
    logging.basicConfig(
        level=level or settings.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
