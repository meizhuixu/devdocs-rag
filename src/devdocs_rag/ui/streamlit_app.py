"""Single-page Streamlit demo wired to the FastAPI SSE endpoint.

Run: `streamlit run src/devdocs_rag/ui/streamlit_app.py`
Phase 1: connects to the mock SSE stream — proves the full token loop works.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

import httpx
import streamlit as st

API_URL = os.environ.get("DEVDOCS_API_URL", "http://localhost:8000")


def stream_tokens(question: str, namespaces: list[str], top_k: int) -> Iterator[str]:
    payload = {"question": question, "namespaces": namespaces, "top_k": top_k}
    with httpx.stream(
        "POST",
        f"{API_URL}/query/stream",
        json=payload,
        timeout=60.0,
        headers={"Accept": "text/event-stream"},
    ) as resp:
        resp.raise_for_status()
        event: str | None = None
        for line in resp.iter_lines():
            if not line:
                event = None
                continue
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and event == "token":
                yield line.split(":", 1)[1].lstrip()
            elif line.startswith("data:") and event == "done":
                return


def main() -> None:
    st.set_page_config(page_title="devdocs-rag", page_icon=None, layout="wide")
    st.title("devdocs-rag")
    st.caption("Phase 1 — mock pipeline. No real LLM yet.")

    with st.sidebar:
        st.subheader("Namespaces")
        namespaces = st.multiselect(
            "Search in",
            options=[
                "pytorch_docs",
                "repo_devdocs_rag",
                "repo_auto_sentinel",
                "repo_devcontext_mcp",
                "repo_llmops_dashboard",
                "repo_csye6225",
            ],
            default=[],
        )
        top_k = st.slider("top_k", 1, 20, 5)

    question = st.text_input("Ask a question", value="")
    if st.button("Ask", type="primary") and question:
        st.write("---")
        st.write_stream(stream_tokens(question, namespaces, top_k))
        st.json(json.dumps({"namespaces": namespaces, "top_k": top_k}))


if __name__ == "__main__":
    main()
