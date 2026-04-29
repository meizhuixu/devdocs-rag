"""Top-level ingestion pipeline.

Responsibilities:
- Decide which loader to use per file (code vs doc).
- Compute per-file commit_sha.
- Skip files whose commit_sha already exists in Qdrant for the namespace.
- Chunk → embed → upsert.

Phase 1 implements only the orchestration shape; the actual write is a no-op.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, Field

from devdocs_rag.ingestion.loaders.code_loader import load_code_file
from devdocs_rag.ingestion.loaders.doc_loader import load_doc_file

logger = logging.getLogger(__name__)


class IngestionReport(BaseModel):
    """What happened during a single `run()`."""

    namespace: str
    files_seen: int = 0
    files_indexed: int = 0
    files_skipped_unchanged: int = 0
    chunks_written: int = 0
    errors: list[str] = Field(default_factory=list)


CODE_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".go", ".rs", ".java"}
DOC_SUFFIXES = {".md", ".rst", ".txt"}


def _classify(path: Path) -> str | None:
    if path.suffix in CODE_SUFFIXES:
        return "code"
    if path.suffix in DOC_SUFFIXES:
        return "doc"
    return None


def run(namespace: str, source_path: Path) -> IngestionReport:
    """Run the ingestion pipeline for one namespace.

    Phase 1: walks the directory, classifies files, calls the right loader,
    counts chunks. Does not write to Qdrant.
    """
    report = IngestionReport(namespace=namespace)

    if not source_path.exists():
        logger.warning("source_path missing — skipping", extra={"path": str(source_path)})
        return report

    for path in source_path.rglob("*"):
        if not path.is_file():
            continue
        kind = _classify(path)
        if kind is None:
            continue

        report.files_seen += 1
        try:
            chunks = load_code_file(path) if kind == "code" else load_doc_file(path)
        except Exception as exc:
            logger.exception("loader failed", extra={"path": str(path)})
            report.errors.append(f"{path}: {exc}")
            continue

        report.files_indexed += 1
        report.chunks_written += len(chunks)

    logger.info(
        "ingestion complete",
        extra={
            "namespace": namespace,
            "files_seen": report.files_seen,
            "chunks_written": report.chunks_written,
        },
    )
    return report
