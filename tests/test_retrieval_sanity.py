"""Phase 3 sanity probes — gated on RUN_INTEGRATION=1.

Five hand-crafted lower-bound checks. **Not** a regression gate (Phase 5
ships the 50-item golden set + Ragas). The point: if hybrid retrieval is
so broken these don't pass, *something* is fundamentally wrong, and we
catch it before staring at silent quality drift.

Coverage spread (per Phase 3 plan §1.10):
- 1 strong dense recall   (conceptual prose query)
- 1 strong BM25 recall    (exact-symbol query, autodoc-gap aware)
- 1 hybrid agreement      (both retrievers should converge)
- 2 broader scenarios     (cross-section topical queries)

Each probe asserts two lower-bounds on `hybrid.search` top-5:
  - at least one chunk's text contains `expected_keyword_in_top5_text`
  - at least one chunk's file_path contains `expected_file_path_substring`

These are deliberately loose. Reranker isn't applied — that's a separate
quality dimension covered by the M2 probe comparison and (eventually)
Phase 5's golden set.

Requires:
    docker compose up -d                                 # Qdrant
    ./scripts/fetch_pytorch_docs.sh                       # corpus
    USE_MOCK_EMBEDDINGS=false python -m devdocs_rag.ingestion ...
    RUN_INTEGRATION=1 USE_MOCK_EMBEDDINGS=false pytest tests/test_retrieval_sanity.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

REQUIRES_INTEGRATION = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1",
    reason="set RUN_INTEGRATION=1 (and have Qdrant + pytorch_docs corpus available)",
)


@dataclass(frozen=True)
class SanityProbe:
    name: str
    query: str
    expected_keyword_in_top5_text: str
    expected_file_path_substring: str
    description: str
    namespaces: tuple[str, ...] = ("pytorch_docs",)
    expected_namespace: str | None = None  # if set, top-5 must include ≥1 chunk in this ns


PROBES: list[SanityProbe] = [
    SanityProbe(
        name="dense_recall_custom_autograd",
        query="how do I write a custom autograd function",
        expected_keyword_in_top5_text="autograd",
        expected_file_path_substring="extending",
        description=(
            "Conceptual prose — dense retriever should surface notes/extending* "
            "files. Tests that bge-base understands the question semantically."
        ),
    ),
    SanityProbe(
        name="bm25_recall_register_buffer",
        query="register_buffer",
        expected_keyword_in_top5_text="buffer",
        expected_file_path_substring="extending",
        description=(
            "Exact-symbol — BM25 should pull the chunk that mentions "
            "register_buffer (Adding a Module section in extending.rst). "
            "Autodoc gap acknowledged: top-1 may be the only strong match."
        ),
    ),
    SanityProbe(
        name="hybrid_cuda_streams",
        query="CUDA stream synchronization",
        expected_keyword_in_top5_text="stream",
        expected_file_path_substring="cuda",
        description=(
            "Both retrievers should converge — CUDA tutorials live in notes/cuda.rst and cuda.md."
        ),
    ),
    SanityProbe(
        name="broad_torch_compile_dynamic_shapes",
        query="torch.compile dynamic shapes",
        expected_keyword_in_top5_text="dynamic",
        expected_file_path_substring="compiler",
        description=(
            "Broader scenario crossing user_guide/torch_compiler hierarchy. "
            "Tests deeper-nested file paths surface."
        ),
    ),
    SanityProbe(
        name="broad_distributed_ddp",
        query="distributed data parallel gradient synchronization",
        expected_keyword_in_top5_text="ddp",
        expected_file_path_substring="ddp",
        description=(
            "Multi-token technical query — should hit ddp.rst. Tests "
            "that broad distributed-training docs are reachable."
        ),
    ),
    # ----- Phase 4 cross-namespace probes -----
    SanityProbe(
        name="cross_ns_devdocs_self_code",
        query="BM25Index from_qdrant scroll method",
        expected_keyword_in_top5_text="bm25",
        expected_file_path_substring="bm25",
        description=(
            "Cross-NS probe that should route to repo_devdocs_rag. The query "
            "names internal code (BM25Index.from_qdrant); pytorch_docs has "
            "no such symbol. Verifies cross-NS RRF doesn't misroute."
        ),
        namespaces=("pytorch_docs", "repo_devdocs_rag"),
        expected_namespace="repo_devdocs_rag",
    ),
    SanityProbe(
        name="cross_ns_pytorch_routing_under_multi",
        query="CUDA stream synchronization",
        expected_keyword_in_top5_text="cuda",
        expected_file_path_substring="cuda",
        description=(
            "Cross-NS probe that must stay in pytorch_docs even when "
            "repo_devdocs_rag is also in scope. Validates the cross-NS "
            "RRF doesn't dilute the right-namespace result."
        ),
        namespaces=("pytorch_docs", "repo_devdocs_rag"),
        expected_namespace="pytorch_docs",
    ),
]


@REQUIRES_INTEGRATION
@pytest.mark.parametrize("probe", PROBES, ids=lambda p: p.name)
def test_sanity_probe(probe: SanityProbe) -> None:
    """Lower-bound: top-5 must contain at least one chunk matching both
    keyword and file_path substring criteria for the given query.
    """
    # Force real path even if .env says mock; reset settings cache so the
    # override takes effect in this process.
    os.environ["USE_MOCK_EMBEDDINGS"] = "false"
    from devdocs_rag.config import get_settings

    get_settings.cache_clear()  # type: ignore[attr-defined]

    from devdocs_rag.retrieval.hybrid import search

    chunks = search(probe.query, namespaces=list(probe.namespaces), top_k=5)
    assert chunks, f"no chunks returned for {probe.query!r}"

    keyword = probe.expected_keyword_in_top5_text.lower()
    matched_text = [c for c in chunks if keyword in c.text.lower()]
    assert matched_text, (
        f"expected at least one chunk in top-5 to contain {keyword!r}\n"
        f"  query:  {probe.query!r}\n"
        f"  desc:   {probe.description}\n"
        f"  top-5 paths: {[c.path for c in chunks]}"
    )

    path_sub = probe.expected_file_path_substring.lower()
    matched_path = [c for c in chunks if path_sub in c.path.lower()]
    assert matched_path, (
        f"expected at least one chunk in top-5 to have file_path containing "
        f"{path_sub!r}\n"
        f"  query:  {probe.query!r}\n"
        f"  desc:   {probe.description}\n"
        f"  top-5 paths: {[c.path for c in chunks]}"
    )

    if probe.expected_namespace is not None:
        matched_ns = [c for c in chunks if c.namespace == probe.expected_namespace]
        assert matched_ns, (
            f"expected at least one chunk in top-5 from namespace "
            f"{probe.expected_namespace!r}\n"
            f"  query:  {probe.query!r}\n"
            f"  desc:   {probe.description}\n"
            f"  namespaces in top-5: {[c.namespace for c in chunks]}"
        )
