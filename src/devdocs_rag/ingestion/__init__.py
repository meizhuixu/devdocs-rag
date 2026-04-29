"""Ingestion module: load → chunk → embed → write to Qdrant."""

from __future__ import annotations

from devdocs_rag.ingestion.pipeline import IngestionReport, run
from devdocs_rag.ingestion.state import IngestionState

__all__ = ["IngestionReport", "IngestionState", "run"]
