"""Phase 6 — end-to-end real streaming trace into Langfuse (mirrors auto-sentinel T068).

Verifies the streaming/tracing contract that DEBT.md flags as unverified at the
real-run level: Ark's OpenAI-compatible stream delivers usage in a final chunk
(`stream_options={"include_usage": True}`), and LLMTracer ships the span with
correct completion_tokens + CNY cost on context exit.

Makes ONE real Ark streaming call (doubao-seed-2.0-pro, tiny prompt, ~¥0.0005).

Requirements (.env in repo root):
  - ARK_API_KEY            — Volcano Ark key (shared portfolio key)
  - LANGFUSE_HOST          — e.g. http://localhost:3000
  - LANGFUSE_PUBLIC_KEY    — pk-lf-...
  - LANGFUSE_SECRET_KEY    — sk-lf-...
The `tracing` extra must be installed:  uv sync --extra tracing

Usage:
    uv run python scripts/send_real_trace.py
    # then check Langfuse UI → Traces, filter tag project:devdocs-rag

Exit code 0 only if tokens streamed AND the real LLMTracer was active.
Deliberately NOT part of the hermetic pytest suite (real network + real spend).
"""

from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv

from devdocs_rag.generation import ark_client as ark_module
from devdocs_rag.generation.ark_client import ArkLLMClient
from devdocs_rag.generation.llm_client import LLMMessage

PING_MAX_TOKENS = 24


async def main() -> int:
    load_dotenv()

    if ark_module.LLMTracer is None:
        print(
            "FAIL: llmops_dashboard.LLMTracer is not installed "
            "(run: uv sync --extra tracing). No span would be shipped.",
            file=sys.stderr,
        )
        return 1
    print(f"Real LLMTracer active: {ark_module.LLMTracer}")

    # Fresh Settings AFTER load_dotenv — bypass the mock flag deliberately;
    # this script exists to exercise the real path.
    from devdocs_rag.config import Settings

    settings = Settings()
    if not settings.ark_api_key:
        print("FAIL: ARK_API_KEY is not set in .env", file=sys.stderr)
        return 1

    client = ArkLLMClient(
        api_key=settings.ark_api_key,
        model=settings.ark_model,
        base_url=settings.ark_base_url,
        temperature=0.0,
        max_tokens=PING_MAX_TOKENS,
    )

    fragments: list[str] = []
    async for fragment in client.stream([LLMMessage(role="user", content="ping")]):
        fragments.append(fragment)

    if not fragments:
        print("FAIL: stream yielded no fragments", file=sys.stderr)
        return 1

    print(f"streamed {len(fragments)} fragments: {''.join(fragments)!r}")
    print("Span shipped to Langfuse on tracer __exit__. Filter Traces by tag project:devdocs-rag.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
