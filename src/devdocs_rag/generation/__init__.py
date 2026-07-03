"""Generation module: prompt templating + LLM client (mock or real Ark)."""

from __future__ import annotations

from devdocs_rag.generation.ark_client import ArkLLMClient
from devdocs_rag.generation.errors import GenerationError
from devdocs_rag.generation.llm_client import LLMClient, LLMMessage, MockLLMClient, get_llm_client
from devdocs_rag.generation.prompts import build_rag_messages

__all__ = [
    "ArkLLMClient",
    "GenerationError",
    "LLMClient",
    "LLMMessage",
    "MockLLMClient",
    "build_rag_messages",
    "get_llm_client",
]
