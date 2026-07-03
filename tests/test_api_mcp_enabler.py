"""M4 MCP-enabler API contract tests.

devcontext-mcp Phase 2 consumes /query/stream over HTTP and only needs the
`retrieved` event. Two additive, wire-compatible enablers are locked here:

1. `_chunk_summary` carries `start_line` / `end_line` (int | None, from the
   string metadata Qdrant payloads store) and `chunk_type` ("code" | "doc").
2. `QueryRequest.retrieval_only: bool = False` — when true the SSE stream is
   `retrieved` → `done` with no LLM call at all (no tokens, no tracer span).

Chunks are injected through a fake reranker patched into the api.main module
namespace (mock mode short-circuits retrieval to [], so the reranker is the
seam that feeds real-looking chunks to the wire payload).
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import devdocs_rag.api.main as api_main
from devdocs_rag.api.main import QueryRequest, _chunk_summary, create_app
from devdocs_rag.retrieval.hybrid import RetrievedChunk


def _code_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        namespace="repo_devdocs_rag",
        path="src/devdocs_rag/retrieval/hybrid.py",
        symbol="hybrid_fuse",
        chunk_type="code",
        text="def hybrid_fuse(bm25_hits, dense_hits, k=RRF_K):\n    ...",
        score=0.91,
        metadata={
            "heading_path": "",
            "language": "python",
            "commit_sha": "abc123",
            "start_line": "75",
            "end_line": "104",
        },
    )


def _doc_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        namespace="pytorch_docs",
        path="docs/source/notes/extending.rst",
        symbol="Extending PyTorch",
        chunk_type="doc",
        text="You can extend autograd by defining a Function subclass.",
        score=0.84,
        metadata={
            "heading_path": "Extending PyTorch > Extending autograd",
            "language": "",
            "commit_sha": "def456",
            "start_line": "",
            "end_line": "",
        },
    )


class _FakeReranker:
    """Reranker stand-in that returns canned chunks regardless of input."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return self._chunks[:top_k]


def _collect_sse_events(client: TestClient, payload: dict[str, object]) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    with client.stream(
        "POST",
        "/query/stream",
        json=payload,
        headers={"Accept": "text/event-stream"},
    ) as resp:
        assert resp.status_code == 200
        current_event: str | None = None
        for line in resp.iter_lines():
            if not line:
                current_event = None
                continue
            if line.startswith("event:"):
                current_event = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and current_event is not None:
                events.append((current_event, line.split(":", 1)[1].lstrip()))
    return events


# ---------------------------------------------------------------------------
# _chunk_summary enrichment
# ---------------------------------------------------------------------------


def test_chunk_summary_code_chunk_has_int_lines_and_chunk_type() -> None:
    summary = _chunk_summary(_code_chunk())
    assert summary["start_line"] == 75
    assert isinstance(summary["start_line"], int)
    assert summary["end_line"] == 104
    assert isinstance(summary["end_line"], int)
    assert summary["chunk_type"] == "code"
    # Existing fields unchanged (wire compatibility for current consumers).
    assert summary["namespace"] == "repo_devdocs_rag"
    assert summary["file_path"] == "src/devdocs_rag/retrieval/hybrid.py"
    assert summary["symbol"] == "hybrid_fuse"
    assert summary["heading_path"] == ""
    assert summary["score"] == 0.91
    assert summary["snippet"].startswith("def hybrid_fuse")


def test_chunk_summary_doc_chunk_lines_are_null() -> None:
    summary = _chunk_summary(_doc_chunk())
    assert summary["start_line"] is None
    assert summary["end_line"] is None
    assert summary["chunk_type"] == "doc"


def test_chunk_summary_missing_line_metadata_is_null() -> None:
    chunk = _doc_chunk()
    chunk.metadata.pop("start_line")
    chunk.metadata.pop("end_line")
    summary = _chunk_summary(chunk)
    assert summary["start_line"] is None
    assert summary["end_line"] is None


# ---------------------------------------------------------------------------
# retrieval_only=true: retrieved → done, no tokens, no LLM client call
# ---------------------------------------------------------------------------


def test_retrieval_only_emits_retrieved_then_done_no_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm_calls: list[str] = []

    def _fail_if_called() -> object:
        llm_calls.append("get_llm_client")
        raise AssertionError("retrieval_only must not touch the LLM client")

    monkeypatch.setattr(api_main, "get_llm_client", _fail_if_called)
    monkeypatch.setattr(
        api_main, "get_reranker", lambda: _FakeReranker([_code_chunk(), _doc_chunk()])
    )

    app = create_app()
    with TestClient(app) as client:
        events = _collect_sse_events(
            client, {"question": "how does hybrid fusion work?", "retrieval_only": True}
        )

    event_types = [e for e, _ in events]
    assert event_types == ["retrieved", "done"]
    assert llm_calls == []

    chunks = json.loads(events[0][1])["chunks"]
    assert len(chunks) == 2
    code, doc = chunks[0], chunks[1]
    assert code["chunk_type"] == "code"
    assert code["start_line"] == 75
    assert code["end_line"] == 104
    assert doc["chunk_type"] == "doc"
    assert doc["start_line"] is None
    assert doc["end_line"] is None


# ---------------------------------------------------------------------------
# default retrieval_only=false: existing streaming behavior unchanged
# ---------------------------------------------------------------------------


def test_retrieval_only_defaults_to_false() -> None:
    assert QueryRequest(question="q").retrieval_only is False


def test_default_request_still_streams_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_main, "get_reranker", lambda: _FakeReranker([_code_chunk()]))

    app = create_app()
    with TestClient(app) as client:
        events = _collect_sse_events(client, {"question": "anything", "top_k": 5})

    event_types = [e for e, _ in events]
    assert event_types[0] == "retrieved"
    assert "token" in event_types
    assert event_types[-1] == "done"
