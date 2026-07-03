"""Concrete LLMClient pointed at Volcano-Engine Ark (doubao series).

Phase 6 real-LLM integration. Satisfies the same `LLMClient` Protocol as
MockLLMClient, so the API layer is untouched (interface-first design).

Provider decision (see docs/PROJECT.md): reuse the portfolio's single Ark
gateway + ARK_API_KEY with doubao-seed-2.0-pro — zero new billing surface,
native-CNY cost accounting consistent with auto-sentinel. Swapping in an
Anthropic/OpenAI client later means adding another Protocol implementation,
nothing else.

LLMTracer is imported at module level under the symbol
`devdocs_rag.generation.ark_client.LLMTracer`. Unit tests patch THAT symbol
(auto-sentinel's seam pattern), so the hermetic suite runs without the
llmops-dashboard extra or a Langfuse backend.

Streaming/tracing contract (the previously-unverified DEBT.md item): Ark's
OpenAI-compatible stream reports usage in a final chunk when
`stream_options={"include_usage": True}` is set, so `set_tokens` /
`set_cost_breakdown` land AFTER the stream is exhausted but BEFORE the tracer
context exits — the span ships on `__exit__` with correct token counts.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import nullcontext
from typing import Any

import openai

from devdocs_rag.generation.errors import GenerationError
from devdocs_rag.generation.llm_client import LLMMessage

logger = logging.getLogger(__name__)

# Module-level patch target for unit tests; None when the optional `tracing`
# extra (llmops-dashboard) isn't installed — streaming must still work.
try:
    from llmops_dashboard.instrumentation.tracer import LLMTracer
except ImportError:
    LLMTracer = None


# Ark pricing in **CNY per 1M tokens** — the native billing currency (no
# exchange-rate conversion anywhere; portfolio-wide convention shared with
# auto-sentinel). Keyed by endpoint id; friendly alias kept for readability.
_ARK_PRICING_CNY_PER_M: dict[str, dict[str, float]] = {
    "ep-20260508052420-fwq5q": {"input": 3.20, "output": 16.00},  # doubao-seed-2.0-pro
    "doubao-seed-2.0-pro": {"input": 3.20, "output": 16.00},
}

_TRACER_PROJECT = "devdocs-rag"
_TRACER_COMPONENT = "rag-api"


class ArkLLMClient:
    """OpenAI-SDK-backed streaming client for the Volcano Ark gateway."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        sdk_client: Any | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        # `Any`: tests inject a fake SDK (hermetic, no real key); production
        # constructs the real AsyncOpenAI here.
        self._sdk: Any = sdk_client or openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=1,
        )

    async def stream(self, messages: list[LLMMessage]) -> AsyncIterator[str]:
        """Yield token fragments; ship one traced generation span per call.

        The tracer is opened WITHOUT a trace_id: LLMTracer then self-generates
        one and owns the parent trace (trace + generation + tags — the standard
        single-call pattern). Injecting a trace_id flips it to
        owns_trace=False, which emits an orphan generation unless the caller
        creates the parent trace itself (auto-sentinel's open_parent_trace
        pattern). External trace_id injection (e.g. from devcontext-mcp) is a
        deliberate M4 extension point — it needs both a Protocol signature
        change and a parent-trace owner.
        """
        tracer_cm = (
            LLMTracer(
                project=_TRACER_PROJECT,
                component=_TRACER_COMPONENT,
                model=self._model,
            )
            if LLMTracer is not None
            else nullcontext()
        )

        prompt_tokens: int | None = None
        completion_tokens: int | None = None

        with tracer_cm as tracer:
            try:
                sdk_stream = await self._sdk.chat.completions.create(
                    model=self._model,
                    messages=[m.model_dump() for m in messages],
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                async for chunk in sdk_stream:
                    if chunk.choices:
                        fragment = chunk.choices[0].delta.content
                        if fragment:
                            yield fragment
                    if chunk.usage is not None:
                        prompt_tokens = chunk.usage.prompt_tokens
                        completion_tokens = chunk.usage.completion_tokens
            except openai.APIError as e:
                raise GenerationError(f"Ark API error: {e}") from e

            trace_id = getattr(tracer, "trace_id", None)

            # Post-stream, pre-exit: usage is only known once the stream is
            # exhausted; the span ships with these values on __exit__.
            if prompt_tokens is None or completion_tokens is None:
                logger.warning(
                    "Ark stream ended without a usage chunk — span ships without tokens",
                    extra={"trace_id": trace_id, "model": self._model},
                )
            elif tracer is not None:
                input_cost, output_cost = self._compute_cost(
                    self._model, prompt_tokens, completion_tokens
                )
                tracer.set_tokens(prompt=prompt_tokens, completion=completion_tokens)
                tracer.set_cost_breakdown(
                    input_cost=input_cost, output_cost=output_cost, currency="CNY"
                )

        logger.info(
            "ark stream complete",
            extra={
                "trace_id": trace_id,
                "model": self._model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )

    async def complete(self, messages: list[LLMMessage]) -> str:
        """Non-streaming convenience: collect the full string (still traced)."""
        parts: list[str] = []
        async for fragment in self.stream(messages):
            parts.append(fragment)
        return "".join(parts)

    @staticmethod
    def _compute_cost(
        model: str, prompt_tokens: int, completion_tokens: int
    ) -> tuple[float, float]:
        prices = _ARK_PRICING_CNY_PER_M.get(model, {"input": 0.0, "output": 0.0})
        input_cost = (prompt_tokens / 1_000_000) * prices["input"]
        output_cost = (completion_tokens / 1_000_000) * prices["output"]
        return input_cost, output_cost
