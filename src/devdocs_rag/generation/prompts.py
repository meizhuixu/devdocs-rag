"""Prompt templates.

Single source of truth for system prompts and the RAG message format.
"""

from __future__ import annotations

from devdocs_rag.generation.llm_client import LLMMessage
from devdocs_rag.retrieval.hybrid import RetrievedChunk

SYSTEM_PROMPT = """You are devdocs-rag, a code-aware retrieval assistant for the user's \
private knowledge base (PyTorch docs and the user's own GitHub repos). \
Answer using only the supplied context. \
Cite sources by `namespace:path:symbol`. \
If the context is insufficient, say so explicitly — do not invent symbols or APIs."""


def _format_chunk(chunk: RetrievedChunk) -> str:
    header = f"[{chunk.namespace}:{chunk.path}:{chunk.symbol}]"
    return f"{header}\n{chunk.text}"


def build_rag_messages(question: str, chunks: list[RetrievedChunk]) -> list[LLMMessage]:
    """Compose the system + user messages for a RAG turn."""
    if chunks:
        context = "\n\n---\n\n".join(_format_chunk(c) for c in chunks)
        user_content = f"Context:\n\n{context}\n\nQuestion: {question}"
    else:
        user_content = f"(No retrieved context — Phase 1 mock mode.)\n\nQuestion: {question}"

    return [
        LLMMessage(role="system", content=SYSTEM_PROMPT),
        LLMMessage(role="user", content=user_content),
    ]
