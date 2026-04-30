"""Top-level ingestion pipeline: walk → chunk → embed → upsert.

Incremental: per-file `commit_sha` is the change-detection key. Files whose
sha matches Qdrant's stored sha are skipped. Source-of-truth for "what's
already indexed" is Qdrant itself.

Flush boundary: whichever fires first — chunk buffer ≥ `qdrant_flush_chunks`
or every `qdrant_flush_files` files.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from qdrant_client.http.models import PointStruct

from devdocs_rag.config import get_settings
from devdocs_rag.ingestion.loaders.code_loader import CodeChunk, load_code_file
from devdocs_rag.ingestion.loaders.doc_loader import DocChunk, load_doc_file
from devdocs_rag.ingestion.qdrant_writer import QdrantWriter
from devdocs_rag.ingestion.state import _CODE_SUFFIXES, _DOC_SUFFIXES, IngestionState
from devdocs_rag.retrieval.dense import get_embedder

logger = logging.getLogger(__name__)


class IngestionReport(BaseModel):
    """Summary of one `run()` invocation."""

    namespace: str
    files_seen: int = 0
    files_indexed: int = 0
    files_skipped_unchanged: int = 0
    files_deleted: int = 0
    chunks_written: int = 0
    duration_s: float = 0.0
    errors: list[str] = Field(default_factory=list)


def _make_point_id(
    namespace: str, file_path: str, start_line: int, end_line: int, symbol: str
) -> str:
    name = f"{namespace}|{file_path}|{symbol}|{start_line}-{end_line}"
    return str(uuid.uuid5(uuid.NAMESPACE_OID, name))


def _chunk_to_payload(
    namespace: str,
    file_path: str,
    commit_sha: str,
    indexed_at: str,
    chunk: CodeChunk | DocChunk,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "namespace": namespace,
        "file_path": file_path,
        "commit_sha": commit_sha,
        "indexed_at": indexed_at,
        "chunk_type": chunk.chunk_type,
        "symbol": chunk.symbol,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "text": chunk.text,
    }
    if isinstance(chunk, DocChunk):
        payload["heading_path"] = chunk.heading_path
        payload["level"] = chunk.level
    elif isinstance(chunk, CodeChunk):
        payload["language"] = chunk.language
    return payload


def _load_chunks(file_abs: Path) -> list[CodeChunk | DocChunk]:
    chunks: list[CodeChunk | DocChunk]
    if file_abs.suffix in _CODE_SUFFIXES:
        chunks = list(load_code_file(file_abs))
    else:
        chunks = list(load_doc_file(file_abs))
    return chunks


def run(
    namespace: str,
    source_path: Path,
    repo_root: Path | None = None,
    smoke_file: str | None = None,
    dry_run: bool = False,
    force_reindex: bool = False,
) -> IngestionReport:
    """Run ingestion for one namespace.

    Args:
        namespace: Qdrant collection name.
        source_path: directory to scan (e.g. `data/raw/pytorch/docs/source`).
        repo_root: git repo root used to compute per-file commit_sha. Defaults
            to `source_path`.
        smoke_file: if set, restrict to this single repo-relative path. Useful
            for the Phase 2 first-run validation.
        dry_run: walk + chunk only; no Qdrant connection, no writes. Used by
            unit tests so they don't need a live Qdrant.
        force_reindex: bypass commit_sha skip — every file already in Qdrant
            goes through delete-then-reingest. Use after chunker logic
            changes (e.g. RST cleaning) so existing chunks reflect the
            new transform. Redis embedding cache by content-sha256 still
            saves work where chunk text didn't actually change.
    """
    settings = get_settings()
    started = time.monotonic()
    report = IngestionReport(namespace=namespace)

    if not source_path.exists():
        logger.warning("source_path missing — %s", source_path)
        return report
    if repo_root is None:
        repo_root = source_path

    suffixes = _DOC_SUFFIXES | _CODE_SUFFIXES

    if dry_run:
        # Walk + chunk only, no Qdrant. Useful for unit tests.
        for path in source_path.rglob("*"):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            report.files_seen += 1
            try:
                chunks = _load_chunks(path)
            except Exception as exc:
                logger.exception("loader failed: %s", path)
                report.errors.append(f"{path}: {exc}")
                continue
            report.files_indexed += 1
            report.chunks_written += len(chunks)
        report.duration_s = time.monotonic() - started
        return report

    # Real run: connect Qdrant, embed, upsert.
    embedder = get_embedder()
    writer = QdrantWriter(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection=namespace,
        vector_dim=embedder.dim,
    )
    writer.ensure_collection()

    existing = writer.scroll_file_shas()
    ignore_globs = settings.namespace_ignore_globs.get(namespace, [])
    if force_reindex:
        logger.info("force_reindex=True: every existing file will be re-processed")
    state = IngestionState.from_dirs(
        source_path=source_path,
        repo_root=repo_root,
        existing=existing,
        suffixes=suffixes,
        ignore_globs=ignore_globs,
        force_reindex=force_reindex,
    )

    if smoke_file is not None:
        state = state.restrict_to({smoke_file})
        logger.info("smoke mode: restricted to %s", smoke_file)

    report.files_seen = len(state.current)
    report.files_skipped_unchanged = len(state.to_skip)

    # Sweep deleted files first.
    for fp in state.to_delete:
        writer.delete_by_file_path(fp)
        report.files_deleted += 1
        logger.info("swept stale file: %s", fp)

    # For files that changed, drop the old chunks before re-ingesting.
    for fp in state.to_update:
        writer.delete_by_file_path(fp)

    files_to_index = sorted(state.to_add | state.to_update)
    total = len(files_to_index)
    indexed_at = datetime.now(UTC).isoformat()

    point_buffer: list[PointStruct] = []
    files_done_in_buffer = 0

    def flush() -> None:
        nonlocal point_buffer, files_done_in_buffer
        if not point_buffer:
            return
        writer.upsert_points(point_buffer)
        report.chunks_written += len(point_buffer)
        logger.info(
            "flushed %d points (total chunks_written=%d)",
            len(point_buffer),
            report.chunks_written,
        )
        point_buffer = []
        files_done_in_buffer = 0

    for i, fp in enumerate(files_to_index, start=1):
        file_abs = repo_root / fp
        try:
            chunks = _load_chunks(file_abs)
        except Exception as exc:
            logger.exception("loader failed: %s", fp)
            report.errors.append(f"{fp}: {exc}")
            continue

        if not chunks:
            report.files_indexed += 1
            files_done_in_buffer += 1
            logger.info("[%d/%d] %s (no chunks)", i, total, fp)
            continue

        texts = [c.text for c in chunks]
        vectors = embedder.embed(texts)
        commit_sha = state.commit_sha(fp)

        for chunk, vec in zip(chunks, vectors, strict=True):
            payload = _chunk_to_payload(namespace, fp, commit_sha, indexed_at, chunk)
            point_buffer.append(
                PointStruct(
                    id=_make_point_id(
                        namespace, fp, chunk.start_line, chunk.end_line, chunk.symbol
                    ),
                    vector=vec,
                    payload=payload,
                )
            )
        report.files_indexed += 1
        files_done_in_buffer += 1
        logger.info("[%d/%d] %s (%d chunks)", i, total, fp, len(chunks))

        if (
            len(point_buffer) >= settings.qdrant_flush_chunks
            or files_done_in_buffer >= settings.qdrant_flush_files
        ):
            flush()

    flush()
    report.duration_s = time.monotonic() - started
    logger.info(
        "ingestion complete: %s indexed=%d skipped=%d deleted=%d chunks=%d in %.1fs",
        namespace,
        report.files_indexed,
        report.files_skipped_unchanged,
        report.files_deleted,
        report.chunks_written,
        report.duration_s,
    )
    return report
