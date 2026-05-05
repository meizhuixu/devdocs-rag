"""Sanity probes — gated on RUN_INTEGRATION=1.

Ten hand-crafted lower-bound checks across 4 namespaces. **Not** a
regression gate (Phase 5 ships the 50-item golden set + deterministic
recall@k eval). The point: if hybrid retrieval is so broken these don't
pass, *something* is fundamentally wrong, and we catch it before staring
at silent quality drift.

Coverage spread:
- Phase 3 (5 probes): pytorch_docs — dense / BM25 / hybrid / topical
- Phase 4 (2 probes): cross-NS routing (repo_devdocs_rag vs pytorch_docs)
- Phase 5 (3 probes): repo_auto_sentinel, repo_devcontext_mcp, 4-NS cross

Each probe asserts two lower-bounds on `hybrid.search` top-5:
  - at least one chunk's text contains `expected_keyword_in_top5_text`
  - at least one chunk's file_path contains `expected_file_path_substring`

These are deliberately loose. Reranker isn't applied — that's a separate
quality dimension covered by the golden set + Ragas eval (Phase 5 M3/M4).

Requires:
    docker compose up -d
    USE_MOCK_EMBEDDINGS=false python -m devdocs_rag.ingestion ...  (all 4 ns)
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
    # ----- Phase 5 probes: new namespaces -----
    SanityProbe(
        name="auto_sentinel_parse_log_node",
        query="parse JSON error log file validate fields",
        expected_keyword_in_top5_text="parse",
        expected_file_path_substring="parse_log",
        description=(
            "Targets repo_auto_sentinel nodes/parse_log.py — the parse_log "
            "node reads and validates a JSON error log. Verifies the new "
            "namespace is indexed and BM25 catches 'parse' + 'log' tokens."
        ),
        namespaces=("repo_auto_sentinel",),
        expected_namespace="repo_auto_sentinel",
    ),
    SanityProbe(
        name="devcontext_mcp_search_codebase_tool",
        query="semantic search codebase MCP tool implementation",
        # CLAUDE.md chunk (sym: "4. `search_codebase`") ranks #1; its text
        # explicitly names the tool. The .py impl file is short (30 lines)
        # and loses to the richer markdown docs in semantic ranking.
        expected_keyword_in_top5_text="search_codebase",
        expected_file_path_substring="CLAUDE",
        description=(
            "Targets repo_devcontext_mcp — the CLAUDE.md tool-description "
            "chunk for search_codebase ranks #1 over the .py file because "
            "markdown docs are semantically richer. Verifies the namespace "
            "is indexed and routing to repo_devcontext_mcp is correct."
        ),
        namespaces=("repo_devcontext_mcp",),
        expected_namespace="repo_devcontext_mcp",
    ),
    SanityProbe(
        name="cross_ns_all_four_code_routing",
        query="BM25 hybrid retrieval implementation",
        # retrieval/__init__.py docstring: "Retrieval module: hybrid (BM25 + dense)
        # + reranker." — BM25 appears in text; path contains "retrieval".
        expected_keyword_in_top5_text="bm25",
        expected_file_path_substring="retrieval",
        description=(
            "4-NS cross-namespace probe: devdocs_rag's retrieval/__init__.py "
            "routes to repo_devdocs_rag even with all 4 namespaces in scope. "
            "Validates cross-NS RRF-of-RRF at full namespace width."
        ),
        namespaces=(
            "pytorch_docs",
            "repo_devdocs_rag",
            "repo_auto_sentinel",
            "repo_devcontext_mcp",
        ),
        expected_namespace="repo_devdocs_rag",
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
