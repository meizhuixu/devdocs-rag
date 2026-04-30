"""Streamlit demo for devdocs-rag.

Run:
    USE_MOCK_EMBEDDINGS=false RERANKER_TYPE=cross_encoder \\
        uvicorn devdocs_rag.api.main:app --reload   # backend
    streamlit run src/devdocs_rag/ui/streamlit_app.py    # frontend

Two-phase SSE consumption (plan §1.9 / §8 risk #1):

    `event: retrieved`  → buffered into a dict, chunks rendered as expanders
    `event: token` ...  → fed into st.write_stream as a text generator
    `event: done`       → terminates the token generator

The SSE parser handles multi-line data fields per spec — a single token
event whose data contains `\\n` is serialized as multiple `data:` lines on
the wire and must be re-joined with `\\n` on the client side. Without that,
mock-LLM responses with embedded newlines (the per-file chunk distribution)
render as run-on text.

The retrieval-debug expander (BM25 / dense / RRF / reranked side-by-side)
only renders when the API set `expose_retrieval_debug=true` and included
the `debug` key in the `retrieved` payload. Default-collapsed.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("DEVDOCS_API_URL", "http://localhost:8000")
DEFAULT_NAMESPACES = ["pytorch_docs"]
PHASE_4_NAMESPACES = [
    "repo_devdocs_rag",
    "repo_auto_sentinel",
    "repo_devcontext_mcp",
    "repo_llmops_dashboard",
    "repo_csye6225",
]


# --------------------------------------------------------------------------- #
# SSE plumbing
# --------------------------------------------------------------------------- #


def _parse_sse_field(line: str) -> tuple[str, str] | None:
    """Return (field, value) for one SSE line, or None for comments/blanks.

    Strips exactly one leading space after the colon (per SSE spec) so token
    fragments like " world" are preserved as-is.
    """
    if not line or line.startswith(":"):
        return None
    if ":" not in line:
        return (line, "")
    field, _, value = line.partition(":")
    if value.startswith(" "):
        value = value[1:]
    return (field, value)


def _consume_until_retrieved(
    line_iter: Iterator[str],
) -> tuple[dict[str, Any], Iterator[str]]:
    """Read the SSE stream until the `retrieved` event, return its payload
    plus a generator that yields the rest of the `token` data fragments
    until `done`.

    Multi-line data fields are joined with `\\n` per SSE spec — this matters
    for tokens whose data contains real newlines.
    """
    event: str | None = None
    data_buf: list[str] = []
    retrieved: dict[str, Any] = {}

    def _flush_phase1() -> bool:
        nonlocal retrieved
        if event == "retrieved":
            retrieved = json.loads("\n".join(data_buf))
            return True
        return False

    for line in line_iter:
        if line == "":
            if _flush_phase1():
                break
            event = None
            data_buf = []
            continue
        parsed = _parse_sse_field(line)
        if parsed is None:
            continue
        field, value = parsed
        if field == "event":
            event = value.strip()
        elif field == "data":
            data_buf.append(value)

    def _token_gen() -> Iterator[str]:
        ev: str | None = None
        buf: list[str] = []
        for ln in line_iter:
            if ln == "":
                if ev == "token" and buf:
                    yield "\n".join(buf)
                elif ev == "done":
                    return
                ev = None
                buf = []
                continue
            parsed = _parse_sse_field(ln)
            if parsed is None:
                continue
            field, value = parsed
            if field == "event":
                ev = value.strip()
            elif field == "data":
                buf.append(value)

    return retrieved, _token_gen()


# --------------------------------------------------------------------------- #
# Health probe (cached so we don't re-hit /health on every Streamlit rerun)
# --------------------------------------------------------------------------- #


@st.cache_data(ttl=10)
def _check_health(api_url: str) -> dict[str, Any] | None:
    try:
        resp = httpx.get(f"{api_url}/health", timeout=2.0)
        if resp.status_code == 200:
            return resp.json()  # type: ignore[no-any-return]
    except httpx.RequestError:
        return None
    return None


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #


def _render_chunk(chunk: dict[str, Any]) -> None:
    path = chunk.get("file_path", "")
    heading = chunk.get("heading_path") or ""
    title = f"{path} > {heading}" if heading else path
    with st.expander(title):
        st.caption(
            f"score = {chunk.get('score', 0.0):.4f}  ·  "
            f"namespace = {chunk.get('namespace', '?')}  ·  "
            f"symbol = {chunk.get('symbol', '?')}"
        )
        snippet = chunk.get("snippet", "")
        st.text(snippet)


def _render_debug(debug: dict[str, Any]) -> None:
    """Side-by-side BM25 / dense / RRF / reranked tables. Default-collapsed."""
    with st.expander("Debug: retrieval breakdown (raw → fused → reranked)", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("**BM25 raw**")
            st.dataframe(_doc_score_df(debug.get("bm25_top", [])), height=320)
        with c2:
            st.markdown("**Dense raw**")
            st.dataframe(_doc_score_df(debug.get("dense_top", [])), height=320)
        with c3:
            st.markdown("**RRF fused**")
            st.dataframe(_doc_score_df(debug.get("rrf_top", [])), height=320)
        with c4:
            st.markdown("**Reranked**")
            reranked = debug.get("reranked_top", [])
            if reranked:
                df = pd.DataFrame(
                    [
                        {"file_path": d.get("file_path", ""), "score": d.get("score", 0.0)}
                        for d in reranked[:10]
                    ]
                )
                st.dataframe(df, height=320)
            else:
                st.caption("(empty)")


def _doc_score_df(items: list[dict[str, Any]]) -> pd.DataFrame:
    """First-8-chars doc_id + score; top-10 only."""
    return pd.DataFrame(
        [
            {"doc_id": str(d.get("doc_id", ""))[:8], "score": float(d.get("score", 0.0))}
            for d in items[:10]
        ]
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def _render_sidebar(health: dict[str, Any] | None) -> tuple[list[str], int]:
    with st.sidebar:
        st.subheader("Namespaces")
        namespaces = st.multiselect(
            "Search in",
            options=DEFAULT_NAMESPACES + PHASE_4_NAMESPACES,
            default=DEFAULT_NAMESPACES,
            help="Phase 3 supports a single namespace; Phase 4 lifts that.",
        )
        top_k = st.slider("top_k", 1, 20, 10)

        with st.expander("Connection info", expanded=False):
            if health is None:
                st.error(f"API unreachable: {API_URL}")
                st.caption("Did you start uvicorn? See the run command in the docstring.")
            else:
                st.success(f"API healthy: {API_URL}")
                st.caption(f"reranker: `{health.get('reranker_type', '?')}`")
                st.caption(f"mock_llm: {health.get('mock_llm')}")
                st.caption(f"mock_embeddings: {health.get('mock_embeddings')}")

        if health and health.get("expose_retrieval_debug"):
            with st.expander("Debug mode active", expanded=False):
                st.info(
                    "Server has `EXPOSE_RETRIEVAL_DEBUG=true`. The retrieved "
                    "event will include BM25/dense/RRF/reranked breakdowns; "
                    "they appear in a collapsed expander below the answer."
                )

    return namespaces, top_k


def main() -> None:
    st.set_page_config(page_title="devdocs-rag", layout="wide")
    st.title("devdocs-rag")
    st.caption(
        "Phase 3 — hybrid retrieval (BM25 + bge-base + RRF) + cross-encoder rerank · mock LLM"
    )

    health = _check_health(API_URL)
    namespaces, top_k = _render_sidebar(health)

    question = st.text_input(
        "Ask a question",
        value="",
        placeholder="e.g. how do I write a custom autograd function",
    )
    submit = st.button("Ask", type="primary", disabled=health is None)

    if not (submit and question.strip()):
        return

    payload = {"question": question, "namespaces": namespaces, "top_k": top_k}
    try:
        with httpx.stream(
            "POST",
            f"{API_URL}/query/stream",
            json=payload,
            timeout=httpx.Timeout(120.0, read=120.0),
            headers={"Accept": "text/event-stream"},
        ) as resp:
            resp.raise_for_status()
            line_iter = resp.iter_lines()
            with st.spinner("retrieving + reranking..."):
                retrieved, token_gen = _consume_until_retrieved(line_iter)

            chunks = retrieved.get("chunks", [])
            st.subheader(f"Retrieved {len(chunks)} chunks")
            if not chunks:
                st.info(
                    "No chunks returned. If the API is in mock mode "
                    "(`USE_MOCK_EMBEDDINGS=true`), retrieval is intentionally "
                    "short-circuited."
                )
            for chunk in chunks:
                _render_chunk(chunk)

            st.subheader("Answer")
            # st.write_stream batches deltas inside the click handler — tokens
            # collapse into a single render at the end. Explicit placeholder +
            # per-yield .markdown() forces a WebSocket frame per token, the
            # Streamlit-canonical typewriter pattern. Confirmed via timing probe:
            # tokens arrive evenly spaced ~10ms apart from the SSE generator.
            answer_placeholder = st.empty()
            accumulated = ""
            for token in token_gen:
                accumulated += token
                answer_placeholder.markdown(accumulated)

            if "debug" in retrieved:
                _render_debug(retrieved["debug"])

    except httpx.HTTPStatusError as exc:
        st.error(f"API returned {exc.response.status_code}: {exc.response.text[:300]}")
    except httpx.RequestError as exc:
        st.error(f"API connection failed: {exc}")
    except json.JSONDecodeError as exc:
        st.error(f"Failed to parse retrieved event JSON: {exc}")


if __name__ == "__main__":
    main()
