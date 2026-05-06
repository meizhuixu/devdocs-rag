"""Mine hard-negative training triples from the 50-item golden-QA set.

Requires Qdrant running with the indexed corpus. Set USE_MOCK_EMBEDDINGS=false
before running so hybrid.search() calls the real embedder.

For each golden item:
  1. Fetches positive chunk texts directly from Qdrant (by file_path + symbol).
  2. Runs hybrid.search(top_k=20) to surface semantically-close non-relevant
     chunks as hard negatives.
  3. Emits (query, positive, hard_negative) triples to triples.jsonl.

With 50 items x avg 1.5 positives x 3 hard negatives ~ 225 training triples.

Usage:
    USE_MOCK_EMBEDDINGS=false python eval/finetune/mine_triples.py
    USE_MOCK_EMBEDDINGS=false python eval/finetune/mine_triples.py --top-k 20 --hard-negs 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from devdocs_rag.config import get_settings
from devdocs_rag.retrieval.hybrid import RetrievedChunk, search

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_GOLDEN = Path(__file__).parent.parent / "datasets" / "golden_qa.jsonl"
_OUTPUT = Path(__file__).parent / "triples.jsonl"


def _fetch_chunk_text(namespace: str, path: str, symbol: str) -> str | None:
    """Return the stored text for one chunk by (namespace, file_path, symbol).

    file_path has a Qdrant payload index (D11); symbol does not, so the filter
    does a linear scan — fine for the 2905-chunk corpus.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    points, _ = client.scroll(
        collection_name=namespace,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="file_path", match=MatchValue(value=path)),
                FieldCondition(key="symbol", match=MatchValue(value=symbol)),
            ]
        ),
        with_payload=True,
        with_vectors=False,
        limit=1,
    )
    if points:
        text = str((points[0].payload or {}).get("text") or "").strip()
        return text or None
    return None


def mine_item(
    item: dict[str, Any],
    top_k: int,
    hard_negs_per_item: int,
) -> list[dict[str, str]]:
    """Return triples for one golden item. Returns [] if data is insufficient."""
    query: str = item["query"]
    namespaces: list[str] = item["namespaces"]
    rel_keys = {(r["namespace"], r["path"], r["symbol"]) for r in item["relevant_chunks"]}

    # Fetch positive texts from Qdrant (source of truth, independent of retrieval rank)
    pos_texts: list[str] = []
    for r in item["relevant_chunks"]:
        text = _fetch_chunk_text(r["namespace"], r["path"], r["symbol"])
        if text:
            pos_texts.append(text)
        else:
            logger.warning(
                "positive not found: ns=%s  path=%s  symbol=%s",
                r["namespace"],
                r["path"],
                r["symbol"],
            )

    if not pos_texts:
        logger.warning("item %s: no positives found — skipped", item.get("id"))
        return []

    # Fetch hard negatives from hybrid search results
    try:
        results: list[RetrievedChunk] = search(query, namespaces, top_k=top_k)
    except Exception as exc:
        logger.warning("item %s: search failed — %s", item.get("id"), exc)
        return []

    neg_texts: list[str] = []
    for chunk in results:
        if len(neg_texts) >= hard_negs_per_item:
            break
        key = (chunk.namespace, chunk.path, chunk.symbol)
        if key not in rel_keys and chunk.text.strip():
            neg_texts.append(chunk.text)

    if not neg_texts:
        logger.warning("item %s: no hard negatives found — skipped", item.get("id"))
        return []

    triples: list[dict[str, str]] = [
        {"query": query, "positive": pos, "hard_negative": neg}
        for pos in pos_texts
        for neg in neg_texts
    ]
    return triples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mine hard-negative triples from golden-QA set.")
    parser.add_argument("--golden", type=Path, default=_GOLDEN)
    parser.add_argument("--output", type=Path, default=_OUTPUT)
    parser.add_argument("--top-k", type=int, default=20, help="Retrieval pool size for negatives")
    parser.add_argument("--hard-negs", type=int, default=3, help="Hard negatives per golden item")
    args = parser.parse_args(argv)

    settings = get_settings()
    if settings.use_mock_embeddings:
        print(
            "ERROR: USE_MOCK_EMBEDDINGS must be false. "
            "Run: USE_MOCK_EMBEDDINGS=false python eval/finetune/mine_triples.py",
            file=sys.stderr,
        )
        return 1

    rows = [
        json.loads(line)
        for line in args.golden.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    logger.info("loaded %d golden items from %s", len(rows), args.golden)

    all_triples: list[dict[str, str]] = []
    for item in rows:
        triples = mine_item(item, top_k=args.top_k, hard_negs_per_item=args.hard_negs)
        all_triples.extend(triples)
        logger.info("item %s → %d triples", item.get("id"), len(triples))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for triple in all_triples:
            fh.write(json.dumps(triple, ensure_ascii=False) + "\n")

    logger.info("wrote %d triples to %s", len(all_triples), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
