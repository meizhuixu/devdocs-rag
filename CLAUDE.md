# READ THIS FIRST

This file is the **constitution** for `devdocs-rag`. Every contributor — human or
agent — reads it before touching code. If you're an LLM agent: do not skip the
DO/DON'T list below, do not invent file layouts, do not "improve" the design
decisions silently. If a rule conflicts with a user instruction, surface the
conflict; do not silently override.

The first 50 lines are the load-bearing rules. The rest is context.

---

## DO / DON'T (the only 50 lines you're required to memorize)

**DO**

1. Use Python 3.11+ everywhere. Type hints are mandatory on every public function.
2. Use Pydantic v2 for every config object, every retrieval state, every API
   request/response model. No bare dicts at module boundaries.
3. Keep the five module boundaries clean: `ingestion / retrieval / generation /
   eval / api`. Cross-module calls go through the public interface in each
   module's top-level `__init__.py` or a typed function — never reach into a
   sibling module's internals.
4. Treat `MockLLMClient` and the real client as interchangeable: same method
   signatures, same return types, same streaming contract. Tests must run
   without any real API key.
5. Use `logging` (`logger = logging.getLogger(__name__)`). Configure once, in
   `config.py` or `api/main.py`.
6. Each module ships at least one smoke test under `tests/`.
7. Every Qdrant chunk carries metadata: `source` (namespace), `commit_sha`,
   `indexed_at`, `path`, `chunk_type` (code|doc), `symbol` (function/class name
   for code, heading for docs). This is what makes incremental indexing work.
8. Treat the five source repos as **namespaces**, not collections. Code must
   not assume any specific namespace exists.
9. Run `ruff check . && mypy src/ && pytest` before committing. CI enforces
   the same three.
10. Update `ARCHITECTURE.md` when a design decision changes. Stale architecture
    docs are worse than none.

**DON'T**

1. Don't `print()` for debugging in `src/`. Use `logger.debug` /
   `logger.info`. (Tests and one-off scripts in `eval/` may print.)
2. Don't import `langchain` outside of `ingestion/loaders/` chunking utilities.
   It's a chunking dependency only — not a framework.
3. Don't hard-code API keys, URLs, model names. Everything goes through
   `config.py` (`pydantic-settings`).
4. Don't write a real LLM call in this scaffold. Phase 1 is mock-only. The real
   client lands in Phase 3 behind the same interface.
5. Don't add a new top-level module without updating this file and
   `ARCHITECTURE.md`.
6. Don't commit data. `data/raw/`, `data/processed/`, `qdrant_storage/` are
   gitignored. The `data/` directory ships with a README only.
7. Don't bypass `ruff` / `mypy` with `# type: ignore` or `# noqa` unless you
   leave a one-line comment explaining why.
8. Don't write a "fix" by mocking what should be tested. If a test fails because
   the contract changed, update the contract on purpose.
9. Don't introduce a new vector DB, embedder, or reranker without an entry in
   ARCHITECTURE.md's "Decision log" section.
10. Don't break the streaming contract on the API: SSE endpoints must yield
    incremental tokens, never buffer-then-flush.

---

## Project goal

A code-aware RAG system that indexes:

1. **PyTorch official docs** — public reference, broad surface area.
2. **My own GitHub repos** as separate namespaces — dogfooding:
   - `auto-sentinel` (matrix project #1)
   - `devdocs-rag` (this repo, self-indexing)
   - `devcontext-mcp` (matrix project #3)
   - `llmops-dashboard` (matrix project #4)
   - `csye6225-cloud-computing` (legacy course project — AWS deep)

The user (me) asks questions like:

- "How did I handle AWS Cognito session refresh in the CSYE6225 project?"
- "What's the right way to use `register_buffer` in PyTorch?"
- "Show me every place I implemented exponential backoff across my repos."

The system is **incremental**: namespaces #1/#3/#4 don't exist yet. Code paths
must tolerate a missing namespace and skip cleanly. Re-indexing checks
`commit_sha` per file and only re-embeds files whose `commit_sha` changed.

---

## Architecture at a glance

```
Sources ──► Loader ──► Chunker ──► Embedder ──► Qdrant (per namespace)
                                                   │
User ──► API (FastAPI/SSE) ──► Hybrid Retrieval ──┤
                                  │                │
                                  ├─► BM25 ────────┤
                                  └─► Dense ───────┘
                                       │
                                       ▼
                                  Reranker (Cohere → cross-encoder fallback)
                                       │
                                       ▼
                                  LLM (real | mock) ──► SSE tokens
```

Full diagram and decision log live in `ARCHITECTURE.md`.

---

## Module boundaries

| Module | Owns | Public interface |
|---|---|---|
| `ingestion` | Loading source files, chunking, embedding, writing to Qdrant. Knows about `commit_sha` and incremental updates. | `pipeline.run(namespace, source_path)` |
| `retrieval` | Hybrid search (BM25 + dense), reranking. Stateless w.r.t. user. | `hybrid.search(query, namespaces, top_k)` returning `list[RetrievedChunk]` |
| `generation` | Prompt templating, LLM client (real or mock), token streaming. | `llm_client.stream(messages)` yielding tokens |
| `eval` | Ragas + golden-set runner. Imports from the other modules read-only. | `ragas_runner.run(dataset_path) -> EvalReport` |
| `api` | FastAPI app, SSE streaming, request validation. Composes the four above. | HTTP endpoints |

**A module never imports another module's internal submodules.** Only the
top-level interface.

---

## Coding conventions

### Python style

- Python 3.11+. We use `match` statements where they help.
- Type hints on every public function. `from __future__ import annotations` at
  the top of every `src/` file.
- Pydantic v2 (`BaseModel`, `Field`, `model_validator`). No `@dataclass` for
  things that cross a module boundary.
- `pydantic-settings` for env loading.
- Snake_case for files, modules, functions. PascalCase for classes.
- Line length: 100. Configured in `pyproject.toml`.

### Linting & type checking

- `ruff` for linting + import sorting + formatting (we use ruff's formatter, not
  black).
- `mypy` in strict-ish mode on `src/`. Tests are checked with relaxed settings.
- CI runs both. Local: `ruff check . && mypy src/`.

### Logging

```python
import logging

logger = logging.getLogger(__name__)

def do_thing() -> None:
    logger.info("starting thing", extra={"namespace": ns})
```

- Never `print()` in `src/`.
- Use `extra={}` for structured fields.
- Set log level via `LOG_LEVEL` env var.

### Errors

- Raise typed exceptions from each module: `IngestionError`, `RetrievalError`,
  `GenerationError`. Catch at the API boundary, translate to HTTP responses.
- Don't catch `Exception` and continue silently. If you don't know what to do,
  let it bubble.

### Tests

- `pytest`, `pytest-asyncio` for async.
- Smoke test per module: import test + one call against a mock.
- Eventually: integration tests against ephemeral Qdrant via testcontainers.
- Goldens (Ragas dataset) live in `eval/datasets/`, loaded by `eval/`.

---

## Design decisions (locked unless ARCHITECTURE.md updated)

### Vector DB: Qdrant
- Self-hosted via docker-compose for dev; the same image scales to prod.
- Each chunk is a point with a payload containing `source`, `commit_sha`,
  `indexed_at`, `path`, `chunk_type`, `symbol`.
- Namespace isolation via Qdrant **collection names** (`pytorch_docs`,
  `repo_auto_sentinel`, `repo_devdocs_rag`, etc.).
- Why not pgvector / Weaviate / Chroma: Qdrant has the best filter performance
  for our metadata-heavy queries and supports payload indexes natively.

### Retrieval: hybrid (BM25 + dense)
- BM25 over raw tokens — catches exact symbol names (`register_buffer`, etc.)
  that semantic search smooths away.
- Dense via `bge-base-en-v1.5` (768-dim). See ARCHITECTURE.md D6 for the
  bge-large → bge-base downsize (M3 Air thermal/RAM budget; MTEB delta ~0.7).
- Fusion: reciprocal rank fusion (RRF) with `k=60`. Cheap and well-studied.

### Reranker
- **Production**: Cohere Rerank API (`rerank-english-v3.0`).
- **Fallback** (no API key / local-only): `cross-encoder/ms-marco-MiniLM-L-6-v2`
  via `sentence-transformers`.
- Choice driven by `COHERE_API_KEY` presence at runtime.

### Chunking
- **Code**: AST-based. Python via `ast`, others via `tree-sitter` (later). One
  chunk per top-level function or class. Class methods bundled with their
  class. Module-level docstring becomes its own chunk.
- **Docs**: semantic chunking — split on heading hierarchy first, then by
  ~512-token windows with 64-token overlap.
- Why AST for code: it preserves the unit of meaning (a function), so
  retrieval returns runnable, copy-pastable snippets.

### Eval: Ragas + 50-item golden set
- Ragas for context recall, faithfulness, answer relevancy.
- 50-item golden set hand-written by the user (me), covering all five
  namespaces.
- CI runs the eval gate on PRs that touch `retrieval/` or `generation/`.

### Incremental indexing
- For each file in a source repo: compute `commit_sha` (per-file via
  `git log -1 --format=%H -- <path>`).
- Compare against the most recent `commit_sha` stored in Qdrant for that file's
  chunks (filter by `path`).
- If different: delete old chunks for that path, re-chunk, re-embed, upsert.
- Files not present in the new tree: delete their chunks (deletion sweep).

### Embedding fine-tune
- Fine-tune `bge-small-en-v1.5` via contrastive learning on triples mined from
  user query logs (later phase).
- Compare base vs fine-tuned recall@10 on the golden set; show in README.

---

## Phases

- **Phase 1 (now, 2026-04-28)**: scaffold + mock pipeline + CI green +
  docker-compose up works. No real LLM, no real embeddings.
- **Phase 2**: real ingestion (PyTorch docs first), real embeddings, real
  hybrid retrieval, mock LLM still.
- **Phase 3**: real LLM via Anthropic + Cohere reranker, SSE streaming wired
  end-to-end, Streamlit UI live.
- **Phase 4**: namespace #2 (`devdocs_rag` self-index), then #1, #3, #4 as
  those projects come online.
- **Phase 5**: Ragas CI gate, embedding fine-tune comparison in README.
- **Phase 6 (done, 2026-07-03)**: real streaming LLM via `ArkLLMClient`
  (Volcano Ark gateway, doubao-seed-2.0-pro) behind the same `LLMClient`
  Protocol; LLMTracer → Langfuse tracing on every call; out-of-sample
  holdout eval for the fine-tune claims. `USE_MOCK_LLM=false` +
  `ARK_API_KEY` switches production on; tests stay hermetic on the mock.

---

## Working with this repo as an agent

If you (an LLM coding agent) are asked to make a change:

1. Read this file. Read `ARCHITECTURE.md`. Skim the module you're touching.
2. Run `ruff check . && mypy src/ && pytest` before you start, to confirm a
   green baseline.
3. Make the smallest change that satisfies the request. Do not refactor on the
   side. Do not "clean up" unrelated code.
4. If the change crosses a module boundary, surface that to the user — it
   probably needs a design discussion first.
5. Update tests. Add a smoke test if you added a new module.
6. Update `ARCHITECTURE.md` if you changed a design decision.
7. Update `docs/PROJECT.md`'s "当前状态（快照）" section and `DEBT.md` when the
   change lands as a PR: new debt gets an entry inline; resolved debt flips
   `[ ]` → `[X]` in the same commit as the fix (keep the entry). PROJECT.md is
   a snapshot / entry point — authoritative progress stays in `git log` + `DEBT.md`.
8. Run `ruff check . && mypy src/ && pytest` again.
9. Write a commit message that explains the **why**, not just the what.

---

## File layout (authoritative)

```
devdocs-rag/
├── CLAUDE.md                        ← this file
├── README.md                        ← public-facing
├── ARCHITECTURE.md                  ← diagrams + decision log
├── pyproject.toml                   ← uv-managed, single source of dep truth
├── Dockerfile
├── docker-compose.yml               ← Qdrant + Redis
├── .env.example                     ← every env var name, no values
├── .gitignore
├── .github/workflows/
│   ├── ci.yml                       ← ruff + mypy + pytest
│   └── eval.yml                     ← Ragas regression (Phase 5 gate)
├── src/devdocs_rag/
│   ├── __init__.py
│   ├── config.py                    ← pydantic-settings
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py                  ← FastAPI + SSE
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   └── loaders/
│   │       ├── __init__.py
│   │       ├── code_loader.py       ← AST chunking
│   │       └── doc_loader.py        ← markdown / semantic
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── bm25.py
│   │   ├── dense.py
│   │   ├── hybrid.py
│   │   └── reranker.py
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── prompts.py
│   │   └── llm_client.py            ← MockLLMClient + interface
│   └── ui/
│       └── streamlit_app.py
├── tests/
│   ├── test_smoke.py
│   └── test_eval_smoke.py           ← unit tests for eval/metrics.py
├── eval/
│   ├── README.md
│   ├── metrics.py                   ← recall@k, mrr@k, precision@k (no LLM)
│   ├── ragas_runner.py              ← golden-set runner → EvalReport
│   ├── datasets/
│   │   └── golden_qa.jsonl
│   └── finetune/
│       ├── README.md                ← setup + hyperparams + results table
│       ├── mine_triples.py          ← golden set → triples.jsonl (hard negs)
│       ├── train.py                 ← bge-small fine-tune (MNR loss)
│       ├── eval_comparison.py       ← 3-way dense recall@10 comparison
│       ├── triples.jsonl            (gitignored — generated)
│       └── bge-small-finetuned/     (gitignored — ~120 MB model weights)
└── data/
    └── README.md
```

If you need a file not on this list, add it here in the same PR.

---

## Glossary

- **Namespace**: a Qdrant collection. One per source repo, plus one for PyTorch
  docs. The boundary that lets us answer "search only my own code".
- **Chunk**: a single retrieval unit. Code: one function or class. Doc: one
  semantic window.
- **Golden set**: hand-written `(query, expected_chunks, expected_answer)`
  tuples used by Ragas for regression.
- **commit_sha**: per-file `git log -1 --format=%H -- <path>`, the canonical
  "this file's content hash" for incremental indexing.
- **Hybrid retrieval**: BM25 ∪ dense, fused by reciprocal rank fusion.

---

## Open questions (track here, resolve in ARCHITECTURE.md)

- How do we represent **cross-namespace** queries ("show me every place I did
  X")? Currently planned: parallel search per namespace, merge by RRF, dedupe
  by `(source, path, symbol)`.
- Reranker latency budget: Cohere is ~100ms p50, cross-encoder local is
  ~300ms for 50 docs. Decide top_k pre-rerank to keep p95 under 1s.
- Streamlit vs Next.js for the demo UI: Streamlit for Phase 3, possibly
  Next.js later if the portfolio matrix wants a unified frontend.

---

End of constitution. Re-read line 1 if you got lost.
