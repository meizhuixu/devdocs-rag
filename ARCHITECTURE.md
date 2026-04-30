# Architecture

This is the long-form companion to [README.md](README.md). It captures the
diagrams and the decision log. **Update this file when a design decision
changes** — see CLAUDE.md DON'T #5.

---

## 1. System overview

```mermaid
flowchart LR
    subgraph Sources
        D1[PyTorch docs]
        D2[repo_auto_sentinel]
        D3[repo_devdocs_rag]
        D4[repo_devcontext_mcp]
        D5[repo_llmops_dashboard]
        D6[repo_csye6225]
    end

    subgraph Ingestion
        L[Loader\nclassify code/doc]
        C[Chunker\nAST or semantic]
        E[Embedder\nbge-base-en-v1.5]
        S[commit_sha skip]
    end

    subgraph Storage
        Q[(Qdrant\nper-namespace collections)]
        R[(Redis\nembedding+result cache)]
    end

    subgraph Query
        API[FastAPI + SSE]
        H[Hybrid retrieval\nBM25Okapi + bge-base/Qdrant\n+ RRF k=60]
        RR[Reranker\ncross-encoder ms-marco-MiniLM-L-6-v2\nPhase 3 · Cohere in Phase 4]
        LLM[LLM\nMockLLMClient · Phase 4 swaps Anthropic]
    end

    U((User)) --> API
    API --> H
    H --> Q
    H --> RR
    RR --> LLM
    LLM --> API
    API --> U

    Sources --> L --> C --> E --> Q
    L -. skip if commit_sha unchanged .-> S
    E <-.cache.-> R
    H <-.cache.-> R
```

---

## 2. Ingestion flow

```mermaid
flowchart TD
    A[Source repo / docs tree] --> B{File type?}
    B -- .py / .ts / .go --> C[code_loader\nAST chunking]
    B -- .md / .rst / .txt --> D[doc_loader\nheading split]
    B -- other --> X[skip]

    C --> M[Chunk<br/>path, symbol, lines, text, type=code]
    D --> M

    M --> N[Compute commit_sha<br/>git log -1 -- path]
    N --> O{Existing chunks for path<br/>have same sha?}
    O -- yes --> P[Skip — file unchanged]
    O -- no --> E[Embed via bge-large]
    E --> F[Upsert to Qdrant<br/>collection=namespace<br/>payload=metadata]
    F --> G[Delete stale chunks<br/>where path matches but sha differs]
```

**Per-chunk payload** (Qdrant point):

```json
{
  "namespace": "repo_csye6225",
  "path": "src/auth/cognito.py",
  "symbol": "refresh_token",
  "chunk_type": "code",
  "language": "python",
  "start_line": 42,
  "end_line": 78,
  "commit_sha": "a1b2c3d4...",
  "indexed_at": "2026-04-28T18:00:00Z",
  "text": "def refresh_token(...): ..."
}
```

---

## 3. Query flow

```mermaid
flowchart LR
    U((User)) --> API[POST /query/stream]
    API --> V[QueryRequest validation<br/>pydantic]
    V --> H[Hybrid Retrieval<br/>BM25Okapi + bge-base/Qdrant<br/>+ RRF k=60]

    H --> B[BM25Okapi<br/>per-namespace<br/>scroll-rebuilt at startup]
    H --> D[Dense<br/>bge-base-en-v1.5<br/>via Qdrant query_points]
    B --> F[RRF fusion<br/>k=60 · rank-aware tie-break]
    D --> F

    F --> RR[Reranker<br/>cross-encoder ms-marco-MiniLM-L-6-v2<br/>MPS · top_50 → top_10]
    RR --> P[Build prompt<br/>system + context + question]

    P --> MK[MockLLMClient · Phase 3<br/>Phase 4 swaps Anthropic]
    MK --> S[SSE: retrieved → tokens → done]
    S --> U
```

---

## 4. Module dependency map

```mermaid
flowchart TB
    api --> retrieval
    api --> generation
    api --> ingestion

    generation --> retrieval

    eval --> retrieval
    eval --> generation
    eval --> ingestion

    ingestion --> config
    retrieval --> config
    generation --> config
    api --> config
```

A module **imports only from a module's top-level interface**, never from a
sibling's submodule. This is enforced by code review (and eventually by an
import linter).

---

## 5. Data model (cross-cutting)

| Object | Defined in | Why it's a Pydantic model |
|---|---|---|
| `Settings` | `config.py` | Single env-loading source of truth |
| `QueryRequest`, `HealthResponse` | `api/main.py` | Public API contract |
| `LLMMessage` | `generation/llm_client.py` | Crosses generation ↔ api boundary |
| `RetrievedChunk` | `retrieval/hybrid.py` | Crosses retrieval ↔ generation boundary |
| `CodeChunk`, `DocChunk` | `ingestion/loaders/*` | Cross loader → pipeline boundary |
| `IngestionReport` | `ingestion/pipeline.py` | Returned to caller (CLI / API) |

---

## 6. Decision log

Every entry: **decision · alternatives considered · why this · when revisit**.

### D1 — Qdrant for vector storage

- **Alternatives**: pgvector, Weaviate, Chroma, Milvus.
- **Why Qdrant**: best payload-filter performance, native multi-namespace via
  collections, `--storage-path` makes self-hosted dev trivial.
- **Revisit**: if we cross 100M chunks (we won't) or need SQL joins on payload.

### D2 — Hybrid (BM25 + dense), not dense-only

- **Alternatives**: dense-only with `bge-large`, or sparse-only with BM25.
- **Why hybrid**: code retrieval *requires* exact-symbol recall (e.g.
  `register_buffer`) that pure embeddings smooth away. Hybrid + RRF is a
  well-known answer.
- **Revisit**: if we add ColBERT/SPLADE — those would replace BM25.

### D3 — RRF (k=60) for fusion

- **Alternatives**: weighted score sum, learned reranker as fusion.
- **Why RRF**: parameter-free across retrievers with very different score
  distributions. k=60 is the literature default.
- **Revisit**: if a learned ranker beats it on the golden set.

### D4 — Cohere Rerank (prod) + cross-encoder fallback

- **Alternatives**: only cross-encoder, only LLM-as-reranker.
- **Why Cohere prod**: ~100ms p50 for top-50, quality is excellent for
  top-5 selection. Cross-encoder local removes the API dependency for offline
  use.
- **Revisit**: if Cohere pricing changes or a stronger open-weight reranker
  ships.

### D5 — AST chunking for code, heading-aware for docs

- **Alternatives**: fixed-window for everything, semantic-only.
- **Why AST**: a function/class is the natural unit of meaning in code; a
  retrieved chunk should be runnable. Headings carry structure in markdown.
- **Revisit**: if we add notebook (.ipynb) support — needs a different
  splitter.

### D6 — `bge-base-en-v1.5` (768-d) as base embedder

- **Alternatives**: `text-embedding-3-large` (OpenAI), `bge-large` (1024-d),
  `bge-small`, `e5-large`.
- **Why bge-base** (updated 2026-04-28, was bge-large): open weights, strong
  on technical English, license-clean for self-host. **MTEB delta vs bge-large
  is ~0.7**; the 3× model-size win matters on M3 Air with 16 GB (thermal +
  RAM headroom for Qdrant + IDE). Phase 5 fine-tunes `bge-small` and shows
  the comparison.
- **Revisit**: when a clear successor ships with measurable recall gains.

### D7 — Phase 1 ships a mock LLM behind a Protocol

- **Alternatives**: skip the API surface until Phase 3.
- **Why mock**: lets us prove the SSE plumbing, FastAPI wiring, and CI
  pipeline before paying for tokens. The Protocol guarantees the real client
  drops in without a refactor.
- **Revisit**: never — this is a Phase-1-only artifact, deleted in Phase 3
  (the `MockLLMClient` stays for tests; the *default* swaps).

### D8 — Per-file `commit_sha` for incremental indexing

- **Alternatives**: hash file content, hash mtime, full re-index.
- **Why commit_sha**: stable across machines, deterministic for unchanged
  files even if mtime changes (e.g. fresh clone), no need for a separate
  manifest.
- **Revisit**: if we ever index files that aren't in a git repo (we won't).

### D9 — Streamlit for the demo UI

- **Alternatives**: Next.js, plain HTML.
- **Why Streamlit**: zero-frontend-cost SSE consumer, fits the dogfood persona
  (one user — me).
- **Revisit**: when the portfolio matrix wants a unified Next.js frontend
  across all five projects.

### D10 — `uv` for dependency management

- **Alternatives**: poetry, pip-tools, plain pip.
- **Why uv**: fast, lockfile-by-default, single source of dep truth in
  `pyproject.toml`. Matches CI install path.
- **Revisit**: if uv stops being the consensus.

### D11 — Qdrant payload indexes on namespace / file_path / commit_sha / chunk_type

- **Alternatives**: no payload indexes (linear filter scan), index everything.
- **Why these four**: the incremental-indexing algorithm filters on
  `(namespace, file_path)` every run to fetch existing `commit_sha`. Query-time
  filters on `(namespace, chunk_type)` for "search only my code in repo X".
  Indexing other payload fields (`heading_path`, `symbol`, `level`) costs RAM
  for queries we don't run.
- **Revisit**: when we add full-text payload search ("show me chunks where
  heading_path contains 'autograd'") — that field gets a `text` index then.

### D12 — Qdrant is the checkpoint, not SQLite

- **Alternatives**: SQLite sidecar, JSONL manifest in `data/processed/`.
- **Why Qdrant**: it already holds `(file_path, commit_sha)` per chunk. A
  scroll over `with_payload=["file_path","commit_sha"]` reconstructs "what's
  done" in O(n_files) and is by definition consistent with what's actually
  indexed. A separate checkpoint store creates a second source of truth that
  can drift from Qdrant on crash.
- **Revisit**: if `scroll_file_shas()` becomes a bottleneck at >1M chunks (it
  won't for the user's corpus). Then add a dedicated metadata collection.

### D13 — Phase 1 mock pipeline ships before Phase 2 real ingestion

- **Alternatives**: skip Phase 1, build retrieval against real data.
- **Why**: lets us prove FastAPI + SSE + CI + module boundaries with zero LLM
  cost and zero data dependencies. The `Embedder` / `LLMClient` / `Reranker`
  Protocols guarantee Phase 2/3 swap without callsite churn.
- **Outcome**: Phase 2 turned out to be a pure swap of two `get_*()`
  factories. No callsite changed. The Protocol design paid off.

### D14 — Qdrant server pinned to v1.16.3

- **Alternatives**: stay on v1.12.4 (Phase 1 default), match the
  locally-installed client exactly (v1.17.1).
- **History**: bumped v1.12.4 → v1.13.6 first as a conservative one-step
  hop. Policy review caught that v1.13.6 vs client v1.17.1 is still 4
  minors apart (Qdrant policy: server ≤1 minor behind client), so a
  second bump moved the server to v1.16.3 — within policy, while still
  one minor below the client to avoid running on the very newest tag.
  The two-step iteration is preserved in git history (commits `31d5942`
  → this one) for the audit trail.
- **Why v1.16.3 specifically**: latest patch in the v1.16 line; one
  minor below the client v1.17.1, satisfying the policy without sitting
  on bleeding-edge tags. No storage-format break across v1.12 → v1.16.
- **Revisit**: when client crosses a major (v2.x), or when a future
  bump lands a feature we want (e.g. native sparse vectors for hybrid
  search, available in v1.10+ — already covered).

### D15 — Per-namespace ignore-globs (gitignore-style)

- **Alternatives**: hard-coded skip-paths in the loader, a global ignore list
  shared across namespaces, full `.gitignore` parsing via `pathspec`.
- **Why per-namespace dict**: PyTorch docs need to skip Sphinx build
  scripts (`docs/source/scripts/**`); user-repo namespaces will need
  different patterns (`node_modules/**`, `__pycache__/**`, `dist/**`,
  `*.lock`). A `dict[namespace, list[glob]]` in `Settings` keeps that
  per-namespace tailoring without code changes.
- **Why our own glob matcher** (not `pathspec`): minimal surface area —
  `**` for recursive segments + `fnmatch` for single-segment wildcards
  covers our needs in ~25 lines and one stdlib import. Adding `pathspec`
  would buy full gitignore semantics (negation, `!`, anchored slashes)
  we don't have a use case for. Tested in `tests/test_state.py` against
  11 representative cases.
- **Outcome**: 10 Sphinx-helper-script chunks removed from
  `pytorch_docs` (2143 → 2133). The matcher is the same one user-repo
  namespaces will use in Phase 4.
- **Revisit**: if we hit a pattern we can't express (e.g. `!keep.py`
  inside an ignored dir), swap to `pathspec` then.

### D16 — BM25 in-memory, rebuilt from Qdrant scroll at API startup

- **Alternatives**: persist a serialized BM25 index alongside Qdrant; build
  on first query (lazy, but visible latency on the first request); rebuild
  from a separate corpus file.
- **Why in-memory + scroll-rebuild**: 2,133 chunks → ~10 MB resident, ~1 s
  rebuild from a single Qdrant scroll (verified empirically). FastAPI
  lifespan primes the registry before serving traffic, so first-query
  latency is the same as warm. Persistence would require a serialization
  format choice + version-compat handling + drift-detection vs Qdrant; not
  worth it at this scale.
- **Concurrency**: `_bm25_registry.get_bm25_index()` uses double-checked
  locking so concurrent first-callers don't double-build.
- **Refresh after re-ingestion**: out of Phase 3 scope. Restart the API
  process to pick up new chunks. Phase 4 will add an `/admin/reload`
  endpoint when multi-namespace incremental updates need it.
- **Revisit**: if rebuild crosses ~5 s (i.e. corpus reaches ~10 k chunks
  with high token counts) or if multi-namespace bring-up makes startup
  unacceptable. Then switch to a persisted artifact + checksum check.

### D17 — RRF k=60, rank-aware deterministic tie-break

- **Alternatives**: weighted-sum fusion (BM25_score + α × cosine), learned
  reranker as fusion, raw-score tiebreaker.
- **Why RRF k=60**: Cormack et al. (2009) — parameter-free across retrievers
  with vastly different score distributions. k=60 is the literature default,
  matched by Elasticsearch / Vespa / Weaviate. Weighted-sum would require
  hand-tuning weights per namespace; learned ranker is a Phase 5 question.
- **Why rank-aware tie-break (not raw-score)**: BM25 (0..30, unbounded)
  and cosine similarity (-1..1, real range ~0.4..0.9) live on incompatible
  scales — a raw-score tiebreaker mixes apples and oranges. Tie key is
  `(-fused_score, min(bm25_rank, dense_rank), doc_id)`: same axis as RRF
  itself (rank), and `doc_id` is the deterministic anchor.
- **Empty-list guards**: BM25 or dense returning zero hits is treated as
  "skip in fusion" so the surviving list's rank order passes through
  monotonically. BM25 with all-zero scores (query had no in-vocab tokens)
  filters out at the BM25Index level via `score > 0`.
- **Revisit**: if a learned ranker (e.g. ColBERT-as-retriever, SPLADE)
  measurably beats RRF on the Phase 5 golden set.

### D18 — Local cross-encoder for Phase 3, Cohere deferred to Phase 4

- **Alternatives**: ship Cohere Rerank API on day one (production-grade,
  ~100 ms p50); skip rerank entirely in Phase 3.
- **Why local first**: keeps Phase 3 hermetic — no API key, no network
  dependency, no billing — while still proving the reranker delivers
  measurable quality lift over raw RRF (verified by M2 probe: Stream
  Sanitizer #5→#2 on the CUDA query, register_buffer noise pruned). The
  cross-encoder model is `cross-encoder/ms-marco-MiniLM-L-6-v2` — ~90 MB
  on disk, ~150 MB on MPS, ~700 ms per top-50 rerank pass on M3 MPS.
- **Why Cohere in Phase 4**: production deployment hits public network and
  scales beyond a single MPS device. Cohere `rerank-english-v3.0` p50 is
  ~100 ms for top-50 (~7× faster than local cross-encoder), better
  quality on out-of-domain queries.
- **Switch cost**: 5 lines + a new `cross_encoder_reranker.py`-shaped file
  for Cohere. Both implement the `Reranker` Protocol; the dispatcher in
  `reranker.py` already branches on `settings.reranker_type`; tests use
  `identity` reranker via `tests/conftest.py` so neither path runs in CI.
- **Revisit**: when Phase 4 lands. The local cross-encoder stays as a
  no-API-key fallback for offline / self-host scenarios.

---

## 7. Open questions

These are tracked here, not in CLAUDE.md, because they will resolve into
decisions and land back in section 6.

- **Cross-namespace ranking**: per-namespace BM25/dense → RRF works for one
  namespace. For *N* namespaces, do we (a) RRF the per-namespace top-k
  separately then re-RRF, or (b) merge everything into one BM25 index? Lean
  (a) — preserves namespace independence and is incrementally updatable.
- **Reranker latency budget**: Cohere ~100ms; cross-encoder ~300ms for 50
  docs. Pre-rerank `top_k` is currently 20 — may need to drop to 15 to keep
  end-to-end p95 under 1s once the LLM call lands.
- **Notebook (.ipynb) ingestion**: outside Phase 1–3 scope. Probably handled
  by extracting code cells through `nbformat` and feeding them to the AST
  chunker.
- **Eval against multiple LLMs**: Ragas is LLM-judge based. Do we judge with
  Claude (consistency with prod) or a separate model (avoid grading our own
  homework)? Lean GPT-4 as judge for independence.

---

## 8. Operational notes

- **Qdrant storage** lives in `./qdrant_storage/` (gitignored). Wipe to
  re-index from scratch.
- **Redis** is unauthenticated in dev; production deployment must change
  this.
- **API `/health`** returns `mock_llm: true` while we're in Phase 1 — useful
  signal that production isn't accidentally live with mocks.
- **Logs** are configured at app startup via `configure_logging()`. Set
  `LOG_LEVEL=DEBUG` to see retrieval scores.
