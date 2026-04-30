"""TEMP: M1/M2 baseline probe — delete at Phase 3 close.

Runs three hand-picked queries against the live `pytorch_docs` collection
and prints the top-5 results. Used as the pre-rerank baseline at M1; rerun
at M2 with `USE_RERANKER=true` (default) to quantify reranker contribution
before any golden set exists.

Usage:
    # M1 baseline (no rerank, RRF score truncation):
    USE_MOCK_EMBEDDINGS=false USE_RERANKER=false uv run python scripts/probe_retrieval.py

    # M2 (cross-encoder rerank, top-50 fused -> top-5 reranked):
    USE_MOCK_EMBEDDINGS=false RERANKER_TYPE=cross_encoder \\
        uv run python scripts/probe_retrieval.py

Note: scores in the output come from different scales depending on mode:
  - rerank=false: RRF fused score (~0.01-0.05)
  - rerank=true:  cross-encoder logit (typically -10..+10)
"""

from __future__ import annotations

import os
import time

from devdocs_rag.config import configure_logging, get_settings
from devdocs_rag.retrieval._bm25_registry import prime_namespace
from devdocs_rag.retrieval.hybrid import RetrievedChunk, search
from devdocs_rag.retrieval.reranker import get_reranker

NAMESPACE = "pytorch_docs"
FINAL_TOP_K = 5

QUERIES: list[str] = [
    "how do I write a custom autograd function",
    "CUDA stream synchronization",
    "register_buffer",
]


def _format_hit(rank: int, chunk: RetrievedChunk) -> str:
    heading = chunk.metadata.get("heading_path") or "<no heading>"
    return (
        f"  {rank}. score={chunk.score:.4f}  "
        f"file={chunk.path}\n"
        f"     heading={heading}\n"
        f"     symbol={chunk.symbol!r}"
    )


def main() -> int:
    configure_logging()
    settings = get_settings()
    use_reranker = os.environ.get("USE_RERANKER", "true").lower() == "true"

    pool_size = settings.retriever_top_k if use_reranker else FINAL_TOP_K
    score_label = "cross-encoder logit" if use_reranker else "RRF fused"
    print(
        f"=== probe config: rerank={use_reranker} reranker_type={settings.reranker_type} "
        f"pool={pool_size} final_top_k={FINAL_TOP_K} score={score_label} ==="
    )

    print(f"\n=== priming BM25 for namespace={NAMESPACE} ===")
    t0 = time.monotonic()
    prime_namespace(NAMESPACE)
    print(f"BM25 ready in {time.monotonic() - t0:.2f}s")

    reranker = get_reranker() if use_reranker else None
    if reranker is not None:
        # Trigger lazy model load now so the per-query timing reflects
        # reranking only, not first-call cold start.
        t0 = time.monotonic()
        _ = reranker.rerank("warmup", [], top_k=1)
        print(f"reranker singleton ready in {time.monotonic() - t0:.2f}s")

    for q in QUERIES:
        print(f"\n=== query: {q!r} ===")
        t = time.monotonic()
        candidates = search(q, namespaces=[NAMESPACE], top_k=pool_size)
        if reranker is not None:
            final = reranker.rerank(q, candidates, top_k=FINAL_TOP_K)
        else:
            final = candidates[:FINAL_TOP_K]
        elapsed = time.monotonic() - t
        if not final:
            print("  (no hits)")
        for i, c in enumerate(final, start=1):
            print(_format_hit(i, c))
        print(f"  -- {len(final)} of {len(candidates)} candidates in {elapsed * 1000:.0f}ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
