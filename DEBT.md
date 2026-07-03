# Technical Debt Register

Technical debt items for devdocs-rag (Project 2). `[ ]` = open, `[X]` = resolved.

When code changes surface a new debt item, Claude Code adds an entry inline. When an item is
resolved, mark it `[X]` in the same commit that lands the fix — keep the entry (with its commit
ref), do not delete it. Format is kept consistent with `auto-sentinel/DEBT.md`.

Debt here is anchored to the phase roadmap (Phase 1-5 complete → Phase 6 real LLM + eval hardening).

---

## Phase 6 Anchors (real LLM integration)

- [X] **MockLLMClient → real provider**: the generation layer ran on `MockLLMClient` behind the
  `LLMClient` Protocol. **Resolved (2026-07-03, Phase 6)**: `ArkLLMClient` (async streaming,
  openai SDK → Volcano Ark gateway, doubao-seed-2.0-pro, shared portfolio ARK_API_KEY) landed
  behind the unchanged Protocol; `USE_MOCK_LLM=false` dispatch raises typed `GenerationError`
  on missing key instead of silently falling back to mock. Verified end-to-end over the real
  API (`/query/stream` SSE to `done`, grounded answer, trace in Langfuse).

- [X] **Streaming token counts with LLMTracer (cross-repo, mirrors llmops-dashboard DEBT)**:
  `completion_tokens` is only known after the SSE stream closes. **Resolved (2026-07-03)**:
  `stream_options={"include_usage": True}` delivers usage in a final chunk; post-stream
  `set_tokens()`/`set_cost_breakdown()` before context exit shipped exact numbers on a real
  run (621+383 tokens, ¥0.0081152 CNY — matches the ¥3.2/¥16 price table). Pattern + the
  trace-ownership gotcha (injected trace_id without a parent-trace owner ⇒ orphan generation)
  documented in llmops-dashboard `docs/onboarding.md`; llmops DEBT entry flipped in the same
  change.

- [X] **Out-of-sample eval set**: fine-tune training data and golden set were same-source, so
  recall@10 = 1.00 was in-sample memorization. **Resolved (2026-07-03)**: deterministic 40/10
  stratified holdout (`eval/finetune/split_holdout.py`), triples re-mined from the 40 only,
  bge-small re-trained, scored on the 10 held-out queries: prod-base 0.80 / small-base 0.85 /
  **fine-tuned 0.85 — zero held-out lift**. In-sample 1.00 confirmed as memorisation; both
  standing decisions (prod = bge-base, no more fine-tuning) now have honest evidence. Full
  write-up in `eval/finetune/README.md`.

- [ ] **Client disconnect mid-stream ships a token-less span**: if the SSE consumer drops the
  connection before the stream finishes (observed with `curl | head`), the async generator is
  cancelled before the usage chunk arrives and the LLMTracer span ships with 0 tokens / no
  cost — while the provider still bills for whatever was generated. The span's error metadata
  records the cancellation, so observability is honest, but cost accounting undercounts.
  **何时修 (revisit with M4/MCP integration or llmops Phase 3 alerting)**: either estimate
  tokens from the streamed fragments, or accept and document the gap as a known limit of
  streaming cost capture.
  *Update (2026-07-03, `feat/m4-mcp-enabler`)*: materially mitigated for the MCP consumer —
  `QueryRequest.retrieval_only=true` skips LLM generation entirely (no client call, no tracer
  span), so devcontext-mcp's retrieval traffic can never hit this path. The disconnect gap
  still exists for normal (generating) queries; entry stays open.

---

## Cross-Project Coordination Anchors

- [ ] **repo_auto_sentinel corpus stale + golden items reference v1 files (blocked on
  auto-sentinel Sprint 6)**: the `repo_auto_sentinel` namespace was indexed in May (418 chunks
  across 16 pre-Sprint-5 commit shas) — none of the Sprint 5 `llm/` / agents code is searchable.
  Worse, golden items **q011–q015** all point at v1 single-agent files
  (`autosentinel/nodes/parse_log.py`, `analyze_error.py`, `execute_fix.py`, `format_report.py`,
  `autosentinel/graph.py`) which Sprint 6's v1 retirement deletes; the incremental deletion sweep
  will then remove their chunks and the eval gate breaks on dead ground truth. **何时修 (right
  after Sprint 6 merges)**: re-ingest `repo_auto_sentinel`, audit `eval/datasets/golden_qa.jsonl`
  (and the committed 40/10 split files) for paths that no longer exist, re-author the affected
  golden items against the 6-agent codebase, re-run the eval gate and re-baseline recall numbers
  if they move.

## Process Debt

- [X] **Phase 5 merge not pushed**: local `main` was ahead of `origin/main` by 7 commits (the
  whole `feat/phase-5-eval-finetune` line + its merge); remote still showed Phase 4 as the latest
  state. **Resolved 2026-07-03**: owner confirmed the push; main pushed together with the docs
  commit that introduces this register.
