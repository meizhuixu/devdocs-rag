"""Phase 2 integration test — gated on RUN_INTEGRATION=1.

Requires:
    - `docker compose up -d` (Qdrant on :6333)
    - `./scripts/fetch_pytorch_docs.sh` (data/raw/pytorch/docs/source)
    - real bge-base model weights cached locally (HF Hub or already downloaded)

Default-skipped so CI doesn't try to download torch + 440MB of model weights
or talk to a Qdrant that isn't there.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REQUIRES_INTEGRATION = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1",
    reason="set RUN_INTEGRATION=1 (and have Qdrant + PyTorch docs available)",
)


@REQUIRES_INTEGRATION
def test_smoke_extending_rst_writes_to_qdrant() -> None:
    """End-to-end smoke: ingest one .rst file, verify Qdrant has chunks for it."""
    # Force the real path even if .env says mock.
    os.environ["USE_MOCK_EMBEDDINGS"] = "false"

    # Reset the cached settings so the override above takes effect.
    from devdocs_rag.config import get_settings

    get_settings.cache_clear()  # type: ignore[attr-defined]

    from devdocs_rag.ingestion.pipeline import run
    from devdocs_rag.ingestion.qdrant_writer import QdrantWriter

    repo_root = Path("data/raw/pytorch")
    source_path = repo_root / "docs" / "source"
    smoke_file = "docs/source/notes/extending.rst"

    assert (
        repo_root / smoke_file
    ).exists(), f"missing {smoke_file} — run scripts/fetch_pytorch_docs.sh first"

    namespace = "pytorch_docs_smoke_test"
    report = run(
        namespace=namespace,
        source_path=source_path,
        repo_root=repo_root,
        smoke_file=smoke_file,
        dry_run=False,
    )

    assert report.errors == []
    assert report.files_indexed == 1
    assert (
        report.chunks_written >= 5
    ), f"expected ≥5 chunks for extending.rst, got {report.chunks_written}"

    settings = get_settings()
    writer = QdrantWriter(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection=namespace,
        vector_dim=settings.dense_dim,
    )
    assert writer.count() == report.chunks_written
