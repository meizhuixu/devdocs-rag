"""3-way dense-recall comparison: bge-base (prod) vs bge-small base vs bge-small fine-tuned.

All three models are evaluated using in-memory cosine similarity (no BM25, no
reranker, no Qdrant vector search). This isolates the embedding model's contribution
so models with different vector dimensions (768d vs 384d) can be compared fairly.

Corpus texts are scrolled from Qdrant for all namespaces referenced in the golden
set. The fine-tuned model is loaded from eval/finetune/bge-small-finetuned/ — if
that directory does not exist the column is skipped with a warning.

Usage:
    USE_MOCK_EMBEDDINGS=false python eval/finetune/eval_comparison.py
    USE_MOCK_EMBEDDINGS=false python eval/finetune/eval_comparison.py --top-k 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from devdocs_rag.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_GOLDEN = Path(__file__).parent.parent / "datasets" / "golden_qa.jsonl"
_FINETUNED = Path(__file__).parent / "bge-small-finetuned"

_MODELS: list[tuple[str, str]] = [
    ("bge-base-en-v1.5 (prod, 768d)", "BAAI/bge-base-en-v1.5"),
    ("bge-small-en-v1.5 (base, 384d)", "BAAI/bge-small-en-v1.5"),
    ("bge-small fine-tuned (384d)", str(_FINETUNED)),
]


# ---------------------------------------------------------------------------
# Corpus loader
# ---------------------------------------------------------------------------


def _load_corpus(namespaces: list[str]) -> list[dict[str, str]]:
    """Scroll all chunk texts from Qdrant for the given namespaces."""
    from qdrant_client import QdrantClient

    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    chunks: list[dict[str, str]] = []

    for ns in namespaces:
        offset: Any = None
        while True:
            points, next_offset = client.scroll(
                collection_name=ns,
                offset=offset,
                limit=500,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                payload = p.payload or {}
                chunks.append(
                    {
                        "namespace": ns,
                        "path": str(payload.get("file_path") or ""),
                        "symbol": str(payload.get("symbol") or ""),
                        "text": str(payload.get("text") or ""),
                    }
                )
            if next_offset is None:
                break
            offset = next_offset
        logger.info(
            "loaded %d chunks from namespace=%s", sum(1 for c in chunks if c["namespace"] == ns), ns
        )

    return chunks


# ---------------------------------------------------------------------------
# Dense-only recall evaluation
# ---------------------------------------------------------------------------


def _recall_at_k(
    retrieved_keys: list[tuple[str, str, str]],
    relevant: list[dict[str, str]],
    k: int,
) -> float:
    rel = {(r["namespace"], r["path"], r["symbol"]) for r in relevant}
    hits = sum(1 for key in retrieved_keys[:k] if key in rel)
    return hits / len(rel) if rel else 1.0


def _eval_model(
    label: str,
    model_path: str,
    all_chunks: list[dict[str, str]],
    golden_items: list[dict[str, Any]],
    top_k: int,
) -> float | None:
    """Return mean recall@top_k for one model across all golden queries.

    Returns None if the model can't be loaded (e.g. fine-tuned path missing).
    """
    if model_path == str(_FINETUNED) and not _FINETUNED.exists():
        logger.warning("fine-tuned model not found at %s — skipping", _FINETUNED)
        return None

    from sentence_transformers import SentenceTransformer

    try:
        model = SentenceTransformer(model_path)
    except Exception as exc:
        logger.error("failed to load %s: %s", label, exc)
        return None

    # Embed all corpus texts once
    chunk_texts = [c["text"] for c in all_chunks]
    logger.info("[%s] embedding %d corpus chunks …", label, len(chunk_texts))
    corpus_matrix = np.array(
        model.encode(chunk_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    )

    recalls: list[float] = []
    for item in golden_items:
        query: str = item["query"]
        ns_set: set[str] = set(item["namespaces"])
        relevant: list[dict[str, str]] = item["relevant_chunks"]

        # Embed query
        qvec = np.array(model.encode([query], normalize_embeddings=True)[0])

        # Filter corpus to query's namespaces
        ns_mask = np.array([c["namespace"] in ns_set for c in all_chunks])
        if not ns_mask.any():
            continue

        # Cosine similarity (embeddings already normalised → dot product = cosine)
        scores = corpus_matrix[ns_mask] @ qvec
        top_local = np.argsort(scores)[::-1][:top_k]

        # Resolve to (namespace, path, symbol) keys
        ns_indices = np.where(ns_mask)[0]
        top_keys = [
            (
                all_chunks[ns_indices[i]]["namespace"],
                all_chunks[ns_indices[i]]["path"],
                all_chunks[ns_indices[i]]["symbol"],
            )
            for i in top_local
        ]
        recalls.append(_recall_at_k(top_keys, relevant, k=top_k))

    return sum(recalls) / len(recalls) if recalls else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="3-way dense-recall comparison.")
    parser.add_argument("--golden", type=Path, default=_GOLDEN)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)

    settings = get_settings()
    if settings.use_mock_embeddings:
        print(
            "ERROR: USE_MOCK_EMBEDDINGS must be false.\n"
            "Run: USE_MOCK_EMBEDDINGS=false python eval/finetune/eval_comparison.py",
            file=sys.stderr,
        )
        return 1

    golden_items = [
        json.loads(line)
        for line in args.golden.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    logger.info("loaded %d golden items", len(golden_items))

    # Collect all namespaces referenced in the golden set
    all_namespaces: set[str] = set()
    for item in golden_items:
        all_namespaces.update(item["namespaces"])
    logger.info("namespaces: %s", sorted(all_namespaces))

    all_chunks = _load_corpus(sorted(all_namespaces))
    logger.info("total corpus: %d chunks", len(all_chunks))

    print(f"\nDense-only recall@{args.top_k} (in-memory cosine, no BM25/reranker)")
    print(f"Golden set: {len(golden_items)} queries  •  Corpus: {len(all_chunks)} chunks\n")
    print(f"{'Model':<35}  {'recall@' + str(args.top_k):>12}")
    print("-" * 50)

    results: list[tuple[str, float | None]] = []
    for label, model_path in _MODELS:
        recall = _eval_model(label, model_path, all_chunks, golden_items, top_k=args.top_k)
        results.append((label, recall))
        val = f"{recall:.4f}" if recall is not None else "N/A (model not found)"
        print(f"  {label:<33}  {val:>12}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
