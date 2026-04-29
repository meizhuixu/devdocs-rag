# devdocs-rag

> Code-aware retrieval-augmented generation over PyTorch docs and my own GitHub
> repos. Built as a personal knowledge base — dogfooded daily.

[![CI](https://github.com/meizhuixu/devdocs-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/meizhuixu/devdocs-rag/actions/workflows/ci.yml)
![status: phase 1 — scaffold](https://img.shields.io/badge/status-phase%201%20scaffold-yellow)
![python: 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)

---

## Why

I keep asking the same question across years of code: *"how did I do X
before?"* — AWS Cognito session refresh in CSYE6225, retry/backoff in
auto-sentinel, the right way to use `register_buffer` in PyTorch. Grep doesn't
cross repos. Embeddings alone miss exact symbol names. **devdocs-rag** is a
code-aware RAG system tuned for that question class:

- **Hybrid retrieval** — BM25 catches `register_buffer`; dense (`bge-large`)
  catches "how do I freeze layers in PyTorch?".
- **AST chunking** — code is split by function/class, so retrieved snippets
  are runnable units, not arbitrary windows.
- **Per-namespace index** — search "all my repos", or "only PyTorch docs", or
  "only the AWS course project".
- **Incremental** — a `commit_sha` per file means re-indexing only re-embeds
  what actually changed.

It's also the second project in my AI Native Portfolio matrix — the piece
that lets the other projects cite each other.

---

## Architecture

```
                                ┌──────────────────────┐
  Sources ─► Loader ─► Chunker ─► Embedder ─► Qdrant   │   (per namespace)
  (docs +   (AST     (semantic   (bge-large)│          │
   repos)    /md)     /heading)             │          │
                                            ▼          │
                                       commit_sha ─────┘
                                       metadata

  User ─► FastAPI ─► Hybrid Retrieval ─► Reranker ─► LLM ─► SSE tokens
                       │                  │           │
                       ├─ BM25            ├─ Cohere   └─ Anthropic
                       └─ Dense           └─ cross-encoder (fallback)
```

Detailed diagrams + decision log: **[ARCHITECTURE.md](ARCHITECTURE.md)**.
Project conventions: **[CLAUDE.md](CLAUDE.md)**.

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Vector DB | Qdrant | Fastest payload filters, native namespaces |
| Sparse retriever | BM25 (`rank-bm25`) | Recovers exact symbol names embeddings smooth away |
| Dense embedder | `bge-large-en-v1.5` (1024-d) | Strong on technical English |
| Fusion | Reciprocal Rank Fusion (k=60) | Cheap, well-studied |
| Reranker (prod) | Cohere `rerank-english-v3.0` | Quality > latency for top-5 |
| Reranker (fallback) | `ms-marco-MiniLM-L-6-v2` | Local, no API key |
| Code chunker | Python `ast` (others: tree-sitter — Phase 2) | Function/class boundaries |
| Doc chunker | Heading-aware → semantic windows | Keeps semantic units intact |
| LLM | Anthropic Claude (Phase 3) | Streaming + long context |
| Eval | Ragas + 50-item golden set | Regression gate in CI |
| API | FastAPI + SSE | Token streaming |
| Cache / queue | Redis | Embedding + result caching |
| UI | Streamlit | Single-page demo |
| Tooling | uv, ruff, mypy strict, pytest | One source of dep truth |

---

## Status

**Phase 1 — scaffold (current).** No real LLM, no real embeddings.

- [x] Repo + module skeleton
- [x] `docker-compose up` brings Qdrant + Redis online
- [x] CI green (ruff + mypy + pytest)
- [x] Mock pipeline runs end-to-end (mock embedder → empty retrieval → mock LLM → SSE)
- [x] FastAPI `/health` and `/query/stream` live with mock client

**Up next — Phase 2: real ingestion**
- [ ] PyTorch docs corpus → `pytorch_docs` namespace
- [ ] `bge-large-en-v1.5` via `sentence-transformers`
- [ ] Hybrid (BM25 + dense) wired against real Qdrant collections
- [ ] First end-to-end query against a real index (still mock LLM)

**Phase 3 — real LLM + reranker**
- [ ] Anthropic streaming client behind `LLMClient` interface
- [ ] Cohere Rerank API + cross-encoder fallback
- [ ] Streamlit UI proves the full streaming loop

**Phase 4 — multi-namespace**
- [ ] Self-index `repo_devdocs_rag`
- [ ] Add namespaces #1/#3/#4 as those projects come online
- [ ] Add legacy `repo_csye6225`

**Phase 5 — eval gate + fine-tune**
- [ ] 50-item hand-written golden set
- [ ] Ragas regression in CI
- [ ] Fine-tune `bge-small-en-v1.5` (contrastive); publish recall@10 vs base

---

## Roadmap (planned demo questions)

> "How did I handle AWS Cognito session refresh in the CSYE6225 project?"
> "What's the right way to use `register_buffer` in PyTorch?"
> "Show me every place I implemented exponential backoff across my repos."
> "Compare how I did rate limiting in auto-sentinel vs llmops-dashboard."

These are the queries the eval golden set must cover.

---

## Quickstart

```bash
# 1. Install deps (uv)
uv pip install -e ".[dev]"

# 2. Bring up infra (Qdrant + Redis)
docker compose up -d

# 3. Run the mock API
uvicorn devdocs_rag.api.main:app --reload

# 4. Hit it
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"hello","namespaces":[],"top_k":5}'

# 5. Or run the Streamlit UI
streamlit run src/devdocs_rag/ui/streamlit_app.py
```

---

## Development

```bash
ruff check .
ruff format --check .
mypy src/
pytest -q
```

CI runs all four on every push and PR.

---

## License

MIT.
