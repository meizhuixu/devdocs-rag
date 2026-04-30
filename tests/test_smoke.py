"""Smoke tests — every module imports, mock pipeline runs end-to-end."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from devdocs_rag import __version__
from devdocs_rag.api.main import create_app
from devdocs_rag.config import get_settings
from devdocs_rag.generation.llm_client import LLMMessage, MockLLMClient, get_llm_client
from devdocs_rag.generation.prompts import build_rag_messages
from devdocs_rag.ingestion import run as ingestion_run
from devdocs_rag.ingestion.loaders.code_loader import load_code_file
from devdocs_rag.ingestion.loaders.doc_loader import load_doc_file
from devdocs_rag.retrieval.dense import MockEmbedder
from devdocs_rag.retrieval.hybrid import reciprocal_rank_fusion
from devdocs_rag.retrieval.reranker import IdentityReranker, get_reranker


def test_version_is_set() -> None:
    assert __version__ == "0.1.0"


def test_settings_load_with_defaults() -> None:
    s = get_settings()
    assert s.use_mock_llm is True
    assert s.dense_dim == 768  # bge-base
    assert "bge-base" in s.dense_model_name


def test_code_loader_chunks_python_functions(tmp_path: Path) -> None:
    f = tmp_path / "sample.py"
    f.write_text("def foo():\n    return 1\n\nclass Bar:\n    def baz(self):\n        return 2\n")
    chunks = load_code_file(f)
    symbols = {c.symbol for c in chunks}
    assert "foo" in symbols
    assert "Bar" in symbols


def test_doc_loader_chunks_markdown_headings(tmp_path: Path) -> None:
    f = tmp_path / "sample.md"
    f.write_text("# Top\nA\n\n## Sub\nB\n")
    chunks = load_doc_file(f)
    symbols = {c.symbol for c in chunks}
    assert "Top" in symbols
    assert "Sub" in symbols


def test_mock_embedder_shape_and_determinism() -> None:
    e = MockEmbedder(dim=16)
    a = e.embed(["hello"])
    b = e.embed(["hello"])
    assert len(a[0]) == 16
    assert a == b


def test_rrf_fuses_two_lists() -> None:
    fused = reciprocal_rank_fusion(
        [("a", 1.0), ("b", 0.5)],
        [("b", 1.0), ("a", 0.5)],
    )
    ids = [doc_id for doc_id, _ in fused]
    assert set(ids) == {"a", "b"}


def test_identity_reranker_truncates() -> None:
    from devdocs_rag.retrieval.hybrid import RetrievedChunk

    chunks = [
        RetrievedChunk(
            namespace="x",
            path="p",
            symbol=str(i),
            chunk_type="code",
            text="t",
            score=float(i),
        )
        for i in range(5)
    ]
    out = IdentityReranker().rerank("q", chunks, top_k=2)
    assert len(out) == 2


def test_get_reranker_returns_identity_in_phase1() -> None:
    assert isinstance(get_reranker(), IdentityReranker)


def test_build_rag_messages_with_no_chunks() -> None:
    msgs = build_rag_messages("what is X?", chunks=[])
    assert len(msgs) == 2
    assert msgs[0].role == "system"
    assert msgs[1].role == "user"


@pytest.mark.asyncio
async def test_mock_llm_client_streams_tokens() -> None:
    client = MockLLMClient(response="hello world", token_delay_s=0.0)
    msgs = [LLMMessage(role="user", content="hi")]
    tokens: list[str] = []
    async for tok in client.stream(msgs):
        tokens.append(tok)
    assert "".join(tokens) == "hello world"


@pytest.mark.asyncio
async def test_mock_llm_client_complete() -> None:
    client = MockLLMClient(response="abc def", token_delay_s=0.0)
    full = await client.complete([LLMMessage(role="user", content="x")])
    assert full == "abc def"


def test_get_llm_client_returns_mock() -> None:
    assert isinstance(get_llm_client(), MockLLMClient)


def test_ingestion_pipeline_walks_files(tmp_path: Path) -> None:
    (tmp_path / "code.py").write_text("def f():\n    return 1\n")
    (tmp_path / "readme.md").write_text("# Title\nbody\n")
    (tmp_path / "skip.bin").write_bytes(b"\x00\x01")

    # dry_run=True: walk + chunk only, no Qdrant connection.
    report = ingestion_run(namespace="test_ns", source_path=tmp_path, dry_run=True)
    assert report.namespace == "test_ns"
    assert report.files_seen == 2
    assert report.files_indexed == 2
    assert report.chunks_written >= 2


def test_ingestion_pipeline_missing_path(tmp_path: Path) -> None:
    report = ingestion_run(namespace="test_ns", source_path=tmp_path / "nope", dry_run=True)
    assert report.files_seen == 0


def test_health_endpoint() -> None:
    app = create_app()
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["mock_llm"] is True


def test_query_stream_emits_retrieved_then_tokens_then_done() -> None:
    """Mock-mode SSE: retrieved (empty chunks) → token... → done.

    Locks the event-order contract Streamlit and any other client will rely on.
    Mock mode means retrieval short-circuits to empty chunks; the test still
    verifies the structural envelope.
    """
    app = create_app()
    events: list[tuple[str, str]] = []
    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/query/stream",
            json={"question": "anything", "top_k": 5},
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

    event_types = [e for e, _ in events]
    assert event_types[0] == "retrieved"
    assert "token" in event_types
    assert event_types[-1] == "done"
