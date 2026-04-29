"""Ingestion CLI.

Examples:
    # Smoke run on a single file (Phase 2 first-run validation):
    python -m devdocs_rag.ingestion \\
        --namespace pytorch_docs \\
        --source data/raw/pytorch/docs/source \\
        --repo-root data/raw/pytorch \\
        --smoke docs/source/notes/extending.rst

    # Full run:
    python -m devdocs_rag.ingestion \\
        --namespace pytorch_docs \\
        --source data/raw/pytorch/docs/source \\
        --repo-root data/raw/pytorch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from devdocs_rag.config import configure_logging
from devdocs_rag.ingestion.pipeline import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="devdocs_rag.ingestion")
    parser.add_argument("--namespace", required=True, help="Qdrant collection name")
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Directory to scan (e.g. data/raw/pytorch/docs/source)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Git repo root for commit_sha computation (defaults to --source)",
    )
    parser.add_argument(
        "--smoke",
        type=str,
        default=None,
        help="Restrict to this single repo-relative path (smoke run)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Walk + chunk only; no Qdrant writes",
    )
    args = parser.parse_args(argv)

    configure_logging()
    report = run(
        namespace=args.namespace,
        source_path=args.source,
        repo_root=args.repo_root,
        smoke_file=args.smoke,
        dry_run=args.dry_run,
    )
    print(report.model_dump_json(indent=2))
    return 0 if not report.errors else 1


if __name__ == "__main__":
    sys.exit(main())
