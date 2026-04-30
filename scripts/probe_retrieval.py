"""TEMP: M1/M2 baseline probe — delete at Phase 3 close.

Runs three hand-picked queries against the live `pytorch_docs` collection
and prints the top-5 hybrid retrieval results. Used as the pre-rerank
baseline at M1; re-run at M2 with cross_encoder reranker enabled to
quantify reranker contribution before any golden set exists.

Usage:
    USE_MOCK_EMBEDDINGS=false uv run python scripts/probe_retrieval.py

Requires Qdrant up (docker compose up -d) and pytorch_docs collection
populated (data/raw/pytorch fetched + ingested).
"""

from __future__ import annotations

import time

from devdocs_rag.config import configure_logging
from devdocs_rag.retrieval._bm25_registry import prime_namespace
from devdocs_rag.retrieval.hybrid import RetrievedChunk, search

NAMESPACE = "pytorch_docs"
TOP_K = 5

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
    print(f"=== priming BM25 for namespace={NAMESPACE} ===")
    t0 = time.monotonic()
    prime_namespace(NAMESPACE)
    print(f"BM25 ready in {time.monotonic() - t0:.2f}s\n")

    for q in QUERIES:
        print(f"=== query: {q!r} ===")
        t = time.monotonic()
        chunks = search(q, namespaces=[NAMESPACE], top_k=TOP_K)
        elapsed = time.monotonic() - t
        if not chunks:
            print("  (no hits)")
        for i, c in enumerate(chunks, start=1):
            print(_format_hit(i, c))
        print(f"  -- {len(chunks)} hits in {elapsed * 1000:.0f}ms\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
