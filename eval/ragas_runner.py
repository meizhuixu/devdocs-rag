"""Retrieval regression runner — Phase 5 M3.

Loads golden JSONL, calls hybrid.search() per item, computes deterministic
metrics (recall@k, mrr@k, precision@k) and returns an EvalReport.

Usage:
    python -m eval.ragas_runner [--dataset PATH] [--top-k N] [--validate-only]
    python eval/ragas_runner.py  --dataset eval/datasets/golden_qa.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from devdocs_rag.retrieval.hybrid import RetrievedChunk, search

# eval/ is not an installed package — insert its directory so `metrics` resolves
# whether this file is run as a script (`python eval/ragas_runner.py`) or as a
# module (`python -m eval.ragas_runner`).
_EVAL_DIR = Path(__file__).parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))
from metrics import mrr_at_k, precision_at_k, recall_at_k  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class RelevantChunkSpec(BaseModel):
    namespace: str
    path: str
    symbol: str


class GoldenItem(BaseModel):
    id: str
    query: str
    namespaces: list[str]
    relevant_chunks: list[RelevantChunkSpec]
    answer_keywords: list[str] = Field(default_factory=list)


class MetricSummary(BaseModel):
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr_at_10: float = 0.0
    precision_at_5: float = 0.0


class QueryResult(BaseModel):
    id: str
    query: str
    retrieved_count: int
    metrics: MetricSummary
    skipped: bool = False
    skip_reason: str = ""


class EvalReport(BaseModel):
    items: int
    evaluated: int = 0
    skipped: int = 0
    aggregate: MetricSummary = Field(default_factory=MetricSummary)
    per_query: list[QueryResult] = Field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_dataset(path: Path) -> list[GoldenItem]:
    if not path.exists():
        return []
    items: list[GoldenItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(GoldenItem.model_validate(json.loads(line)))
    return items


# ---------------------------------------------------------------------------
# Per-query runner
# ---------------------------------------------------------------------------


def _run_query(item: GoldenItem, top_k: int) -> QueryResult:
    relevant = [r.model_dump() for r in item.relevant_chunks]
    try:
        retrieved: list[RetrievedChunk] = search(item.query, item.namespaces, top_k=top_k)
    except Exception as exc:
        logger.warning("retrieval failed for %s: %s", item.id, exc)
        return QueryResult(
            id=item.id,
            query=item.query,
            retrieved_count=0,
            metrics=MetricSummary(),
            skipped=True,
            skip_reason=str(exc),
        )

    metrics = MetricSummary(
        recall_at_5=recall_at_k(retrieved, relevant, k=5),
        recall_at_10=recall_at_k(retrieved, relevant, k=10),
        mrr_at_10=mrr_at_k(retrieved, relevant, k=10),
        precision_at_5=precision_at_k(retrieved, relevant, k=5),
    )
    return QueryResult(
        id=item.id,
        query=item.query,
        retrieved_count=len(retrieved),
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def run(dataset_path: Path, top_k: int = 10) -> EvalReport:
    """Load golden JSONL, run hybrid.search() per item, compute aggregate metrics."""
    items = load_dataset(dataset_path)
    if not items:
        return EvalReport(items=0, error="dataset empty or not found")

    results: list[QueryResult] = [_run_query(item, top_k=top_k) for item in items]

    evaluated = [r for r in results if not r.skipped]
    n_skipped = len(results) - len(evaluated)

    if evaluated:
        n = len(evaluated)
        agg = MetricSummary(
            recall_at_5=sum(r.metrics.recall_at_5 for r in evaluated) / n,
            recall_at_10=sum(r.metrics.recall_at_10 for r in evaluated) / n,
            mrr_at_10=sum(r.metrics.mrr_at_10 for r in evaluated) / n,
            precision_at_5=sum(r.metrics.precision_at_5 for r in evaluated) / n,
        )
    else:
        agg = MetricSummary()

    return EvalReport(
        items=len(items),
        evaluated=len(evaluated),
        skipped=n_skipped,
        aggregate=agg,
        per_query=results,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run retrieval eval on a golden set.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("eval/datasets/golden_qa.jsonl"),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate dataset structure only — no retrieval calls.",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=None,
        metavar="THRESHOLD",
        help=(
            "Fail (exit 1) if recall@10 on evaluated items is below THRESHOLD. "
            "Skipped items (Qdrant unavailable) do not count as failures."
        ),
    )
    args = parser.parse_args(argv)

    if args.validate_only:
        items = load_dataset(args.dataset)
        print(f"validate-only: {len(items)} items loaded from {args.dataset}")
        return 0

    report = run(args.dataset, top_k=args.top_k)
    print(f"items={report.items}  evaluated={report.evaluated}  skipped={report.skipped}")
    if report.evaluated > 0:
        a = report.aggregate
        print(
            f"recall@5={a.recall_at_5:.3f}  recall@10={a.recall_at_10:.3f}"
            f"  mrr@10={a.mrr_at_10:.3f}  precision@5={a.precision_at_5:.3f}"
        )
        if args.min_recall is not None and a.recall_at_10 < args.min_recall:
            print(
                f"FAIL: recall@10={a.recall_at_10:.4f} < threshold={args.min_recall}",
                file=sys.stderr,
            )
            return 1
    if report.error:
        print(f"error: {report.error}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
