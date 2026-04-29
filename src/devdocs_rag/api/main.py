"""FastAPI app with an SSE streaming endpoint.

Phase 1: returns a fixed mocked stream. The plumbing (request validation,
SSE generator, error handling) is real so Phase 3 only swaps the LLM call.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from fastapi import FastAPI
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from devdocs_rag.config import configure_logging, get_settings
from devdocs_rag.generation.llm_client import LLMMessage, MockLLMClient, get_llm_client
from devdocs_rag.generation.prompts import build_rag_messages
from devdocs_rag.retrieval.hybrid import RetrievedChunk

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    """Inbound query. Namespaces is optional; empty = search all."""

    question: str = Field(min_length=1, max_length=2000)
    namespaces: list[str] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=50)


class HealthResponse(BaseModel):
    status: str
    version: str
    mock_llm: bool


def create_app() -> FastAPI:
    """App factory. Used by uvicorn and by tests."""
    configure_logging()
    settings = get_settings()
    app = FastAPI(
        title="devdocs-rag",
        version="0.1.0",
        description="Code-aware RAG over PyTorch docs and personal repos.",
    )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version="0.1.0",
            mock_llm=settings.use_mock_llm,
        )

    @app.post("/query/stream")
    async def query_stream(req: QueryRequest) -> EventSourceResponse:
        logger.info(
            "query received",
            extra={"namespaces": req.namespaces, "top_k": req.top_k},
        )

        # Phase 1: skip retrieval (no data indexed). Hand the query straight to
        # the mock LLM with empty context. The plumbing is what we're testing.
        chunks: list[RetrievedChunk] = []
        messages = build_rag_messages(req.question, chunks)
        client = get_llm_client()

        async def event_stream() -> AsyncIterator[dict[str, str]]:
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
