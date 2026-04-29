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

**Phase 2 — real ingestion (current).** Real `bge-base-en-v1.5` embeddings,
real Qdrant writes, incremental re-index by `commit_sha`. Still mock LLM.

- [x] Phase 1 scaffold (mock pipeline, FastAPI + SSE, CI green)
- [x] PyTorch docs corpus → `pytorch_docs` namespace via sparse-checkout
- [x] `bge-base-en-v1.5` via `sentence-transformers` (MPS-accelerated, Redis-cached)
- [x] Qdrant collection bootstrap with payload indexes
- [x] Incremental re-index: per-file `commit_sha` change detection
- [ ] Hybrid (BM25 + dense) wired against real Qdrant collections (next)
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

### Ingest PyTorch docs (Phase 2)

```bash
# Sparse-checkout docs/source/ from the PyTorch repo (~30-60 MB)
./scripts/fetch_pytorch_docs.sh

# Smoke run on a single file (validates the whole pipeline)
USE_MOCK_EMBEDDINGS=false uv run python -m devdocs_rag.ingestion \
  --namespace pytorch_docs \
  --source data/raw/pytorch/docs/source \
  --repo-root data/raw/pytorch \
  --smoke docs/source/notes/extending.rst

# Full run
USE_MOCK_EMBEDDINGS=false uv run python -m devdocs_rag.ingestion \
  --namespace pytorch_docs \
  --source data/raw/pytorch/docs/source \
  --repo-root data/raw/pytorch
```

---

## Known limitations / gaps

**Phase 2 ingests static .rst content only — autodoc-rendered API reference is not yet covered.**

PyTorch's documentation is built with Sphinx. Many `.rst` files in the API
reference (e.g. `torch.rst`, `nn.rst`, `optim.rst`) are mostly directives like
`.. autoclass:: torch.nn.Linear`; the actual prose comes from Python
docstrings rendered at Sphinx build time. Static `.rst` parsing therefore
captures the prose-heavy `notes/`, tutorials, and quantization guides, but
**misses inline API documentation**.

**What Phase 2 does cover:**
- All of `docs/source/notes/*.rst` (~50 substantial tutorials)
- `quantization.rst`, `distributed.rst`, `cuda.rst`, etc. (prose pages)
- File-level prose surrounding autodoc directives

**What Phase 2 does not cover:**
- API reference pulled from Python docstrings via Sphinx autodoc
- Examples embedded in autodoc'd docstrings

**Planned remediation — Phase 2.5:** run `make html` against the PyTorch docs
source and parse rendered HTML, then extract per-API-symbol chunks. Adds ~5
min build cost and `sphinx + torch + sphinx_rtd_theme` to the ingestion env.
Tracked as a Phase 2.5 milestone — not blocking Phase 3.

A query like *"how does `nn.Linear` work?"* will currently return the
notes/extending.rst chunk on subclassing `nn.Module` rather than the
`nn.Linear` docstring. Notes-style questions ("how do I extend autograd?",
"how do I write custom CUDA ops?") work as designed.

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
