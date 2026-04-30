"""FastAPI app: hybrid retrieval + cross-encoder rerank + mock LLM streaming.

Phase 3 wires real retrieval (BM25 + bge-base + RRF) and a local cross-encoder
reranker behind /query/stream. The LLM stays mocked — Phase 4 swaps in
Anthropic Claude behind the same `LLMClient` Protocol.

Lifespan: at startup, BM25 is primed from Qdrant scroll and (in production
mode) the cross-encoder is loaded onto MPS. Both are O(seconds), paid once.
Mock-mode tests skip both and stay fast.

SSE event order:
    event: retrieved   data: {"chunks": [...], "debug": {...}?}
    event: token       data: <token-fragment>
    ...
    event: done        data:
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from devdocs_rag.config import configure_logging, get_settings
from devdocs_rag.generation.llm_client import LLMMessage, MockLLMClient, get_llm_client
from devdocs_rag.generation.prompts import build_rag_messages
from devdocs_rag.retrieval._bm25_registry import prime_namespaces
from devdocs_rag.retrieval.hybrid import RetrievalDebug, RetrievedChunk, search_with_debug
from devdocs_rag.retrieval.reranker import get_reranker

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    """Inbound query. Namespaces optional — empty falls back to the default."""

    question: str = Field(min_length=1, max_length=2000)
    namespaces: list[str] = Field(default_factory=list)
    top_k: int = Field(default=10, ge=1, le=50)


class HealthResponse(BaseModel):
    status: str
    version: str
    mock_llm: bool
    mock_embeddings: bool
    reranker_type: str
    expose_retrieval_debug: bool


def _chunk_summary(chunk: RetrievedChunk, snippet_len: int = 300) -> dict[str, Any]:
    return {
        "namespace": chunk.namespace,
        "file_path": chunk.path,
        "symbol": chunk.symbol,
        "heading_path": chunk.metadata.get("heading_path", ""),
        "score": chunk.score,
        "snippet": chunk.text[:snippet_len],
    }


def _build_chunk_summary_text(chunks: list[RetrievedChunk]) -> str:
    """Mock-LLM response text that enumerates per-namespace chunk counts.

    Phase 4 multi-namespace format: the demo's visible proof that
    retrieval crossed namespaces — namespace selection on the UI →
    namespace breakdown in the answer → matches the chunks above.
    """
    if not chunks:
        return (
            "[mock-Phase-4] No chunks retrieved. "
            "Real LLM integration: Phase 6 (Anthropic Claude / DeepSeek-V3 via Volcano)."
        )
    ns_counts = Counter(c.namespace for c in chunks)
    lines = [f"[mock-Phase-4] Multi-namespace search across {len(ns_counts)} namespaces:"]
    for ns, n in ns_counts.most_common():
        lines.append(f"  {ns}: {n} chunks")
    lines.append("Real LLM integration: Phase 6 (Anthropic Claude / DeepSeek-V3 via Volcano).")
    return "\n".join(lines)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup: prime BM25 + warm reranker singleton.

    Mock-mode (USE_MOCK_EMBEDDINGS=true, the test/CI default) skips both —
    Qdrant isn't expected to be reachable and the cross-encoder model
    shouldn't be downloaded just to run unit tests.
    """
    configure_logging()
    settings = get_settings()
    if settings.use_mock_embeddings:
        logger.info("API startup: mock mode — skipping BM25 prime + reranker warmup")
        yield
        return

    logger.info("API startup: priming namespaces=%s", settings.default_namespaces)
    prime_namespaces(settings.default_namespaces)
    if settings.reranker_type != "identity":
        logger.info("API startup: warming reranker (%s)", settings.reranker_type)
        get_reranker()
    logger.info("API ready")
    yield


def create_app() -> FastAPI:
    """App factory. Used by uvicorn and by tests."""
    configure_logging()
    settings = get_settings()
    app = FastAPI(
        title="devdocs-rag",
        version="0.1.0",
        description="Code-aware RAG over PyTorch docs and personal repos.",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version="0.1.0",
            mock_llm=settings.use_mock_llm,
            mock_embeddings=settings.use_mock_embeddings,
            reranker_type=settings.reranker_type,
            expose_retrieval_debug=settings.expose_retrieval_debug,
        )

    @app.post("/query/stream")
    async def query_stream(req: QueryRequest) -> EventSourceResponse:
        ns = req.namespaces or settings.default_namespaces
        logger.info(
            "query received: namespaces=%s top_k=%d question_len=%d",
            ns,
            req.top_k,
            len(req.question),
        )

        if settings.use_mock_embeddings:
            # Mock mode short-circuits retrieval entirely — keeps the API
            # runnable without Qdrant for plumbing demos.
            candidates: list[RetrievedChunk] = []
            debug = RetrievalDebug()
        else:
            candidates, debug = search_with_debug(req.question, ns, top_k=settings.retriever_top_k)
        reranker = get_reranker()
        reranked = reranker.rerank(req.question, candidates, top_k=req.top_k)

        retrieved_payload: dict[str, Any] = {
            "chunks": [_chunk_summary(c) for c in reranked],
        }
        if settings.expose_retrieval_debug:
            retrieved_payload["debug"] = {
                "bm25_top": [
                    {"namespace": namespace, "doc_id": did, "score": s}
                    for namespace, did, s in debug.bm25_top
                ],
                "dense_top": [
                    {"namespace": namespace, "doc_id": did, "score": s}
                    for namespace, did, s in debug.dense_top
                ],
                "rrf_top": [
                    {"namespace": namespace, "doc_id": did, "score": s}
                    for namespace, did, s in debug.rrf_top
                ],
                "reranked_top": [
                    {"namespace": c.namespace, "file_path": c.path, "score": c.score}
                    for c in reranked
                ],
            }

        # Real chunks → mock LLM response that enumerates the file_path
        # distribution, so the demo visibly proves retrieval is wired.
        client = get_llm_client()
        if isinstance(client, MockLLMClient):
            client = MockLLMClient(response=_build_chunk_summary_text(reranked))
        messages = build_rag_messages(req.question, reranked)

        async def event_stream() -> AsyncIterator[dict[str, str]]:
            yield {"event": "retrieved", "data": json.dumps(retrieved_payload)}
            async for token in client.stream(messages):
                yield {"event": "token", "data": token}
            yield {"event": "done", "data": ""}

        return EventSourceResponse(event_stream())

    return app


app = create_app()


__all__ = [
    "HealthResponse",
    "LLMMessage",
    "MockLLMClient",
    "QueryRequest",
    "app",
    "create_app",
]
