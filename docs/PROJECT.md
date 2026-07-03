# DevDocs RAG — Project Context & Status

> 项目上下文 + 进度快照。**维护规则：每次代码 PR 落地，顺手更新「当前状态」段和涉及的决策段。**
> 权威进度以 `git log` / `DEBT.md` 为准，本文件是快照 + 上下文入口。
> 矩阵全局上下文见 `~/Repo/PORTFOLIO.md`（本地文件，不入库）。

---

## 当前状态（快照 2026-07-03）

- ✅ **Phase 1-5 全部完成**（2026-05-05），main 干净，130 tests passing。
- ⏳ **Phase 6 待启动**：MockLLMClient → 真实 LLM + out-of-sample eval set。
  排在项目 1（auto-sentinel）Sprint 5 之后串行启动——项目 1 已完成，项目 4 Phase 2 接入
  auto-sentinel 之后就轮到本项目。真实 LLM API key 尚未配置（仅 `.env.example`）。
- Phase 6 待办细项见 `DEBT.md`（本次新建）。

---

## 项目是什么

工程师私人代码知识库 RAG：对自己写过的所有代码 + PyTorch 文档做语义搜索。
"how did I do X before?" 这一类问题的专用检索系统。

## Corpus（2,905 chunks，4 namespace）

| Namespace | 内容 | Chunks |
|---|---|---|
| pytorch_docs | PyTorch 官方文档（253 RST） | 2,133 |
| repo_devdocs_rag | 本项目自身代码 | 264 |
| repo_auto_sentinel | 项目 1 代码 | 415 |
| repo_devcontext_mcp | 项目 3 代码 | 93 |

增量索引：每 chunk 存 `commit_sha`，只重 ingest 变更文件；`POST /admin/reload` 不重启刷新。

## 关键技术

- **Retrieval**：BM25 + dense（bge-base-en-v1.5, 768d）两路并行 → RRF (k=60) 融合 →
  cross-encoder（ms-marco-MiniLM-L-6-v2）精排
- **Chunking**：代码走 AST（按函数/类切），文档走 semantic（RST heading-split）
- **Performance**：warm 无 rerank ~20ms / 带 rerank ~720ms / cold start 4.4s
- **Generation**：`LLMClient` Protocol 抽象，当前 MockLLMClient；换真实 provider 只新增实现类，
  其余代码零修改（Phase 6 入口）

## Eval（核心差异化）

50 条手写 golden set（query → 期望命中 chunk），覆盖 4 namespace + 3 跨 namespace 查询。

| Model | recall@10 |
|---|---|
| **bge-base-en-v1.5（生产）** | **0.79** |
| bge-small-en-v1.5 base | 0.78 |
| bge-small fine-tuned | 1.00（in-sample，真实估 ~0.78） |

fine-tune 用对比学习 + hard negative mining，结论：bge-base 多花的维度只换 1%，
**不要重做 fine-tuning**。CI gate：recall@10 ≥ 0.70，PR 触碰 retrieval/eval 代码即跑，不达标 fail。

## 技术栈

Qdrant / bge-base-en-v1.5 / BM25Okapi / RRF / ms-marco-MiniLM-L-6-v2 /
sentence-transformers + MultipleNegativesRankingLoss / FastAPI + SSE / Streamlit /
pytest（130 tests）/ GitHub Actions（ruff + mypy + pytest + eval gate）/ uv

## Phase 6 范围（待启动）

1. 接真实 LLM（Anthropic / OpenAI，经 `LLMClient` Protocol 新增实现类）
2. LLM 调用接入项目 4 的 LLMTracer；注意 SSE 流式下 `completion_tokens` 只能在 stream
   结束后拿到（对应 llmops-dashboard DEBT 的 "Streaming token counts unverified"）
3. Out-of-sample eval set：fine-tune 训练数据与 golden set 同源（in-sample memorization），需重做

## 跨项目依赖

- **项目 4 LLMOps Dashboard**：Phase 6 接 LLMTracer 上报（push 模式，trace 带 `project` + `component` tag）
- **项目 3 DevContext MCP**：Phase 2 将通过 HTTP 调用本项目（`search_codebase` 等 3 个 tool）
