"""Ingestion module: load → chunk → embed → write to Qdrant."""

from __future__ import annotations

from devdocs_rag.ingestion.pipeline import IngestionReport, run

__all__ = ["IngestionReport", "run"]
