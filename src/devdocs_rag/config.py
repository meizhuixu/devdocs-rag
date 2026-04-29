"""Runtime configuration loaded from environment via pydantic-settings."""

from __future__ import annotations

import logging
from functools import lru_cache

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

    # Embedding model
    dense_model_name: str = Field(default="BAAI/bge-large-en-v1.5")
    dense_dim: int = Field(default=1024)

    # Reranker model (fallback / local)
    cross_encoder_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Retrieval defaults
    retrieval_top_k: int = Field(default=20)
    rerank_top_k: int = Field(default=5)


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
