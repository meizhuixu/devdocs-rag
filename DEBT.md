# Technical Debt Register

Technical debt items for devdocs-rag (Project 2). `[ ]` = open, `[X]` = resolved.

When code changes surface a new debt item, Claude Code adds an entry inline. When an item is
resolved, mark it `[X]` in the same commit that lands the fix — keep the entry (with its commit
ref), do not delete it. Format is kept consistent with `auto-sentinel/DEBT.md`.

Debt here is anchored to the phase roadmap (Phase 1-5 complete → Phase 6 real LLM + eval hardening).

---

## Phase 6 Anchors (real LLM integration)

- [ ] **MockLLMClient → real provider**: the generation layer runs on `MockLLMClient` behind the
  `LLMClient` Protocol. Phase 6 adds a real implementation class (Anthropic / OpenAI); no other
  code changes by design. Blocked on: no real API key configured yet (only `.env.example`), and
  the portfolio's serial ordering (auto-sentinel Sprint 5 ✅ → llmops-dashboard Phase 2 →
  **this**). Do NOT wire a real client before Phase 6 formally starts.

- [ ] **Streaming token counts with LLMTracer (cross-repo, mirrors llmops-dashboard DEBT)**:
  DevDocs serves over SSE, so `completion_tokens` is only known after the stream closes, but
  `LLMTracer` requires `set_tokens()` before the `with` block exits. The "set tokens post-stream,
  pre-exit" pattern is unproven against a real streaming call. Verify during Phase 6 wiring and
  document the streaming usage pattern if it differs from non-streaming.

- [ ] **Out-of-sample eval set**: the bge-small fine-tune training data and the 50-query golden
  set are same-source, so the fine-tuned model's recall@10 = 1.00 is in-sample memorization
  (real-world estimate ~0.78). Before publishing any fine-tune claim, build a held-out eval set.
  Note: do NOT redo the fine-tuning itself — the bge-base vs bge-small gap is a settled 1%
  decision; only the eval methodology needs the out-of-sample fix.

---

## Process Debt

- [X] **Phase 5 merge not pushed**: local `main` was ahead of `origin/main` by 7 commits (the
  whole `feat/phase-5-eval-finetune` line + its merge); remote still showed Phase 4 as the latest
  state. **Resolved 2026-07-03**: owner confirmed the push; main pushed together with the docs
  commit that introduces this register.
