# Embedding Fine-tune — bge-small-en-v1.5

**Goal**: show whether domain-specific contrastive fine-tuning on the 50-item
golden set improves dense recall, using a same-size (384d) comparison so the
lift is attributable to the fine-tune, not the model size.

The production embedder (`bge-base-en-v1.5`, 768d) is unchanged. Fine-tuning
targets `bge-small-en-v1.5` (384d, 4× smaller) because training on M3 Air
fits comfortably in RAM and completes in minutes.

---

## Prerequisites

- Qdrant running with the full indexed corpus (all 4 namespaces, 2 905 chunks).
- `USE_MOCK_EMBEDDINGS=false` set in the shell.
- `uv pip install -e ".[dev]"` completed (sentence-transformers, datasets, torch).

---

## Step-by-step

### 1. Mine hard-negative triples

```bash
USE_MOCK_EMBEDDINGS=false python eval/finetune/mine_triples.py
```

Reads `eval/datasets/golden_qa.jsonl`. For each of the 50 golden items:
- Fetches the positive chunk text(s) from Qdrant by `(namespace, file_path, symbol)`.
- Runs `hybrid.search(top_k=20)` to find semantically-close non-relevant chunks
  as hard negatives.
- Emits `{"query", "positive", "hard_negative"}` to `triples.jsonl`.

Expected output: ~150–225 triples (50 items × 1–2 positives × 3 hard negs).

Flags:

| Flag | Default | Description |
|---|---|---|
| `--golden` | `../datasets/golden_qa.jsonl` | Golden-set path |
| `--output` | `triples.jsonl` | Output path |
| `--top-k` | 20 | Retrieval pool for hard negatives |
| `--hard-negs` | 3 | Hard negatives per golden item |

### 2. Fine-tune

```bash
python eval/finetune/train.py
```

Reads `triples.jsonl`. Trains `bge-small-en-v1.5` with
`MultipleNegativesRankingLoss` and saves the model to `bge-small-finetuned/`
(gitignored).

Flags:

| Flag | Default | Description |
|---|---|---|
| `--triples` | `triples.jsonl` | Triples file |
| `--output` | `bge-small-finetuned/` | Save path |
| `--base-model` | `BAAI/bge-small-en-v1.5` | Starting checkpoint |
| `--epochs` | 5 | Training epochs |
| `--batch` | 16 | Per-device batch size |
| `--device` | auto (cuda→cpu) | Force device; use `--device mps` only if ≥ 4 GB free |

#### Hyperparameters

| Param | Value | Rationale |
|---|---|---|
| Base model | `BAAI/bge-small-en-v1.5` | 384d, 33M params — trains in ~20 min on CPU (M3 Air) |
| Loss | `MultipleNegativesRankingLoss` | In-batch negatives + explicit hard negatives |
| Epochs | 5 | 258 triples / 16 batch → 16 steps/epoch → 85 total |
| Batch size | 16 | More in-batch negatives = stronger signal |
| LR | 2e-5 | Standard for sentence-transformer fine-tuning |
| Warmup | 10% of total steps | Prevents early instability |
| Precision | bf16 (CUDA) / fp32 (CPU/MPS) | MPS bf16 training support is unreliable |

> **Note on MPS OOM**: training on an M3 Air (20 GB unified memory) defaults to CPU because
> near-full memory allocation causes MPS training to crash after step 1. CPU training takes
> ~20 minutes for 85 steps and is perfectly reproducible.

### 3. Evaluate

```bash
USE_MOCK_EMBEDDINGS=false python eval/finetune/eval_comparison.py
```

Scrolls all 2 905 corpus chunks from Qdrant, embeds them with each of the
three models, and computes dense-only recall@10 (no BM25, no reranker) so
the comparison isolates the embedding quality.

---

## Results

Measured on 2026-05-05. Corpus: 2 919 chunks across 4 namespaces. 50 golden
queries. In-memory cosine similarity (no BM25, no reranker).

| Model | Dim | recall@10 (dense-only) |
|---|---|---|
| bge-base-en-v1.5 (prod) | 768 | 0.7900 |
| bge-small-en-v1.5 (base) | 384 | 0.7800 |
| bge-small-en-v1.5 (fine-tuned) | 384 | 1.0000 |

**Interpretation**:

- **Fine-tuned 1.0000 is in-sample**: triples were mined from the same 50 golden
  queries used for evaluation, so the model has effectively memorised the
  query→chunk pairings. **Real-world estimate for fine-tuned bge-small ≈ 0.78**
  (same as the base model, since no held-out set was used).

- **The meaningful comparison is base vs prod**: bge-small (384d, 0.78) closes to
  within 1% of bge-base (768d, 0.79) out-of-the-box, with 4× fewer parameters.
  This confirms the D6 decision to use bge-base in prod is conservative — the
  smaller model is already nearly as good.

- **Decision point**: fine-tuned bge-small does not surpass bge-base by the >5 pp
  threshold set in D23. Production embedder stays bge-base. A fairer fine-tune
  comparison would hold out 10 golden items from mining and evaluate only on those;
  that is a separate, future experiment.

---

## Gitignored artifacts

| Path | Why gitignored |
|---|---|
| `eval/finetune/triples.jsonl` | Generated from indexed corpus; reproducible |
| `eval/finetune/bge-small-finetuned/` | ~120 MB model weights |
