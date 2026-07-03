"""Phase 6 — ArkLLMClient (real LLM behind the LLMClient Protocol).

Hermetic: the OpenAI SDK layer is replaced by an injected fake, so the suite
runs without any real API key (CLAUDE.md DO #4). The LLMTracer symbol is
patched on the ark_client module, mirroring auto-sentinel's test seam.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from devdocs_rag.generation import ark_client as ark_module
from devdocs_rag.generation.ark_client import ArkLLMClient
from devdocs_rag.generation.errors import GenerationError
from devdocs_rag.generation.llm_client import (
    LLMMessage,
    MockLLMClient,
    get_llm_client,
)

DOUBAO_PRO = "ep-20260508052420-fwq5q"  # doubao-seed-2.0-pro; ¥3.2 in / ¥16 out per 1M


def _content_chunk(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text))],
        usage=None,
    )


def _usage_chunk(prompt: int, completion: int) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
    )


class _FakeStream:
    def __init__(self, chunks: list[SimpleNamespace]) -> None:
        self._chunks = chunks

    def __aiter__(self) -> _FakeStream:
        self._it = iter(self._chunks)
        return self

    async def __anext__(self) -> SimpleNamespace:
        try:
            return next(self._it)
        except StopIteration as e:
            raise StopAsyncIteration from e


class _FakeSDK:
    """Stands in for openai.AsyncOpenAI. Records create() kwargs."""

    def __init__(
        self, chunks: list[SimpleNamespace] | None = None, error: Exception | None = None
    ) -> None:
        self.create_kwargs: dict[str, Any] | None = None
        self._chunks = chunks or []
        self._error = error

        async def _create(**kwargs: Any) -> _FakeStream:
            self.create_kwargs = kwargs
            if self._error is not None:
                raise self._error
            return _FakeStream(self._chunks)

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))


class _RecordingTracer:
    """Fake LLMTracer context manager recording call order."""

    instances: list[_RecordingTracer] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.events: list[tuple[Any, ...]] = []
        _RecordingTracer.instances.append(self)

    def __enter__(self) -> _RecordingTracer:
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.events.append(("exit",))
        return False

    def set_tokens(self, *, prompt: int, completion: int) -> None:
        self.events.append(("tokens", prompt, completion))

    def set_cost_breakdown(
        self, *, input_cost: float, output_cost: float, currency: str
    ) -> None:
        self.events.append(("cost", input_cost, output_cost, currency))


@pytest.fixture(autouse=True)
def _reset_tracer_instances() -> None:
    _RecordingTracer.instances = []


_MESSAGES = [LLMMessage(role="user", content="ping")]

_TOKEN_CHUNKS = [
    _content_chunk("Hello"),
    _content_chunk(" "),
    _content_chunk("world"),
    _usage_chunk(prompt=9, completion=7),
]


def _client(sdk: _FakeSDK) -> ArkLLMClient:
    return ArkLLMClient(api_key="test-key", model=DOUBAO_PRO, sdk_client=sdk)


async def test_stream_yields_tokens_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ark_module, "LLMTracer", None)
    client = _client(_FakeSDK(chunks=_TOKEN_CHUNKS))
    tokens = [t async for t in client.stream(_MESSAGES)]
    assert tokens == ["Hello", " ", "world"]


async def test_stream_requests_usage_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ark_module, "LLMTracer", None)
    sdk = _FakeSDK(chunks=_TOKEN_CHUNKS)
    client = _client(sdk)
    _ = [t async for t in client.stream(_MESSAGES)]
    assert sdk.create_kwargs is not None
    assert sdk.create_kwargs["stream"] is True
    assert sdk.create_kwargs["stream_options"] == {"include_usage": True}
    assert sdk.create_kwargs["model"] == DOUBAO_PRO
    assert sdk.create_kwargs["messages"] == [{"role": "user", "content": "ping"}]


async def test_complete_concatenates_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ark_module, "LLMTracer", None)
    client = _client(_FakeSDK(chunks=_TOKEN_CHUNKS))
    assert await client.complete(_MESSAGES) == "Hello world"


async def test_tracer_gets_tokens_and_cost_after_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The DEBT.md streaming contract: completion_tokens is only known after the
    stream closes, so set_tokens/set_cost_breakdown must land post-stream but
    before the tracer context exits (span ships on __exit__)."""
    monkeypatch.setattr(ark_module, "LLMTracer", _RecordingTracer)
    client = _client(_FakeSDK(chunks=_TOKEN_CHUNKS))
    tokens = [t async for t in client.stream(_MESSAGES)]
    assert tokens == ["Hello", " ", "world"]

    assert len(_RecordingTracer.instances) == 1
    tracer = _RecordingTracer.instances[0]
    assert tracer.kwargs["project"] == "devdocs-rag"
    assert tracer.kwargs["model"] == DOUBAO_PRO
    # 32-char lowercase hex, OTel-compatible (portfolio-wide convention).
    trace_id = tracer.kwargs["trace_id"]
    assert len(trace_id) == 32 and all(c in "0123456789abcdef" for c in trace_id)

    kinds = [e[0] for e in tracer.events]
    assert kinds == ["tokens", "cost", "exit"]
    assert tracer.events[0] == ("tokens", 9, 7)
    _, input_cost, output_cost, currency = tracer.events[1]
    assert currency == "CNY"
    assert input_cost == pytest.approx(9 / 1_000_000 * 3.20)
    assert output_cost == pytest.approx(7 / 1_000_000 * 16.00)


async def test_streams_without_tracer_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """llmops_dashboard is an optional extra — absence must not break streaming."""
    monkeypatch.setattr(ark_module, "LLMTracer", None)
    client = _client(_FakeSDK(chunks=_TOKEN_CHUNKS))
    assert [t async for t in client.stream(_MESSAGES)] == ["Hello", " ", "world"]


async def test_api_error_raises_generation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ark_module, "LLMTracer", None)
    sdk = _FakeSDK(error=openai.APIConnectionError(request=httpx.Request("POST", "http://ark")))
    client = _client(sdk)
    with pytest.raises(GenerationError):
        _ = [t async for t in client.stream(_MESSAGES)]


# --- get_llm_client dispatch -------------------------------------------------


def _patch_settings(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> None:
    from devdocs_rag import config
    from devdocs_rag.generation import llm_client as llm_client_module

    settings = config.Settings(_env_file=None, **overrides)
    monkeypatch.setattr(llm_client_module, "get_settings", lambda: settings)


def test_dispatch_mock_flag_returns_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch, use_mock_llm=True)
    assert isinstance(get_llm_client(), MockLLMClient)


def test_dispatch_real_returns_ark(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch, use_mock_llm=False, ark_api_key="test-key")
    assert isinstance(get_llm_client(), ArkLLMClient)


def test_dispatch_real_without_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch, use_mock_llm=False, ark_api_key=None)
    with pytest.raises(GenerationError):
        get_llm_client()
