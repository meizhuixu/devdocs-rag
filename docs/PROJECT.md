# DevDocs RAG — Project Context & Status

> 项目上下文 + 进度快照。**维护规则：每次代码 PR 落地，顺手更新「当前状态」段和涉及的决策段。**
> 权威进度以 `git log` / `DEBT.md` 为准，本文件是快照 + 上下文入口。
> 矩阵全局上下文见 `~/Repo/PORTFOLIO.md`（本地文件，不入库）。

---

## 当前状态（快照 2026-07-03）

- ✅ **Phase 1-6 全部完成**。Phase 6（2026-07-03）：`ArkLLMClient` 真实流式 LLM
  （openai SDK → 火山方舟，doubao-seed-2.0-pro，复用组合级 ARK_API_KEY）进 `LLMClient`
  Protocol 后面，API 层零改动；139 tests passing（新增 9 个，全 hermetic）。
- ✅ **LLMTracer → Langfuse 实测贯通**：真实 SSE 全链路（检索→精排→真 LLM→`done`）trace 落库,
  tokens 621+383、cost ¥0.0081152 CNY 与价格表精确对账。顺带修掉一个真 bug：注入 trace_id
  会让 LLMTracer 只发孤儿 generation（详见 llmops onboarding 的 Trace Ownership 节）。
- ✅ **Out-of-sample eval 完成**：40/10 分层 holdout 重训重测——held-out recall@10：
  prod-base 0.80 / small-base 0.85 / **fine-tuned 0.85（零增益）**，in-sample 1.00 确证为
  记忆效应；生产维持 bge-base，不再做 fine-tune。
- 范围决策（2026-07-03）：Cohere Rerank 移出范围（本地 cross-encoder 即生产方案）；
  Anthropic/DeepSeek 备选由"单 Ark 网关复用"决策取代，需要时加一个 Protocol 实现类即可。
- 新增技术债 1 条：SSE 客户端中途断连 → span 记 0 token（供应商仍计费），见 `DEBT.md`。
- 下一步：**项目 3 DevContext MCP Phase 2**（M4）——本项目作为其真实后端之一。

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

| Model | 全集 recall@10 | held-out recall@10（n=10，2026-07-03） |
|---|---|---|
| **bge-base-en-v1.5（生产）** | **0.79** | 0.80 |
| bge-small-en-v1.5 base | 0.78 | 0.85 |
| bge-small fine-tuned | 1.00（in-sample） | **0.85（零增益）** |

fine-tune 用对比学习 + hard negative mining。out-of-sample holdout（40 训练 / 10 held-out）
证实 in-sample 1.00 是记忆效应、fine-tune 无真实增益——**不要重做 fine-tuning**，生产维持
bge-base。CI gate：recall@10 ≥ 0.70，PR 触碰 retrieval/eval 代码即跑，不达标 fail。

## 技术栈

Qdrant / bge-base-en-v1.5 / BM25Okapi / RRF / ms-marco-MiniLM-L-6-v2 /
sentence-transformers + MultipleNegativesRankingLoss / FastAPI + SSE / Streamlit /
pytest（130 tests）/ GitHub Actions（ruff + mypy + pytest + eval gate）/ uv

## Phase 6 落地内容（2026-07-03 完成）

1. `ArkLLMClient`（`generation/ark_client.py`）：异步流式，`stream_options={"include_usage": True}`
   取最终 usage chunk；`GenerationError` 类型化错误；SDK 注入缝保持测试 hermetic
2. LLMTracer 可选接入（tracing extra → 本地 editable llmops-dashboard）；流式 token 计数
   与 CNY 成本实测精确；单调用客户端不注入 trace_id（tracer 自持父 trace）
3. Out-of-sample holdout eval：`eval/finetune/split_holdout.py` + 复现命令与数字见
   `eval/finetune/README.md`

## 跨项目依赖

- **项目 4 LLMOps Dashboard**：Phase 6 接 LLMTracer 上报（push 模式，trace 带 `project` + `component` tag）
- **项目 3 DevContext MCP**：Phase 2 将通过 HTTP 调用本项目（`search_codebase` 等 3 个 tool）
