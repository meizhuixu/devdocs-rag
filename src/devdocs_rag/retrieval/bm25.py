"""BM25 sparse retrieval.

In-memory `rank_bm25.BM25Okapi` over a corpus keyed by Qdrant point UUIDs.
The doc-id keying matters: hybrid fusion (BM25 + dense) needs identifiers that
match what `dense_search()` returns from Qdrant, otherwise RRF silently fuses
nothing.

Lifecycle: `BM25Index.from_qdrant(writer)` rebuilds from a full scroll. The
FastAPI lifespan hook calls this once at startup. There is no on-disk
persistence — the math (10MB / ~1s for 2k chunks) doesn't justify it. See
ARCHITECTURE.md D16.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from rank_bm25 import BM25Okapi

if TYPE_CHECKING:
    from devdocs_rag.ingestion.qdrant_writer import QdrantWriter

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lowercase + `\\w+` split.

    Notes for hybrid retrieval correctness:
    - `register_buffer` stays whole (`_` is a word char).
    - `nn.Linear` splits on `.` → `["nn", "linear"]`. Symmetric: queries get
      tokenized the same way, so this is consistent rather than buggy.
    """
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Index:
    """Doc-id-keyed BM25Okapi index.

    `search()` returns `(doc_id, score)` tuples sorted desc, filtered to
    score > 0 — RRF gains nothing from zero-score hits and they'd be noise
    in the rank list.
    """

    def __init__(self, doc_ids: list[str], texts: list[str]) -> None:
        if len(doc_ids) != len(texts):
            raise ValueError(
                f"doc_ids ({len(doc_ids)}) and texts ({len(texts)}) must be the same length"
            )
        self._doc_ids = doc_ids
        self._tokenized = [tokenize(t) for t in texts]
        self._bm25: BM25Okapi | None = BM25Okapi(self._tokenized) if self._tokenized else None

    @property
    def size(self) -> int:
        return len(self._doc_ids)

    @classmethod
    def from_documents(cls, doc_ids: list[str], texts: list[str]) -> BM25Index:
        return cls(doc_ids, texts)

    @classmethod
    def from_qdrant(cls, writer: QdrantWriter) -> BM25Index:
        """Rebuild from a full Qdrant scroll. Called once per namespace at API startup."""
        doc_ids: list[str] = []
        texts: list[str] = []
        for pid, text in writer.scroll_text_payloads():
            doc_ids.append(pid)
            texts.append(text)
        logger.info(
            "BM25Index built from qdrant: collection=%s size=%d",
            writer.collection,
            len(doc_ids),
        )
        return cls(doc_ids, texts)

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        """Return `[(doc_id, score), ...]` for the top_k matches, score > 0 only."""
        if self._bm25 is None:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
        out: list[tuple[str, float]] = []
        for idx, score in ranked[:top_k]:
            if score <= 0.0:
                continue
            out.append((self._doc_ids[idx], float(score)))
        return out
