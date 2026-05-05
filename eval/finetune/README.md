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

#### Hyperparameters

| Param | Value | Rationale |
|---|---|---|
| Base model | `BAAI/bge-small-en-v1.5` | 384d, 33M params — trains in minutes on M3 Air |
| Loss | `MultipleNegativesRankingLoss` | In-batch negatives + explicit hard negatives |
| Epochs | 5 | ~225 triples / 16 batch → ~70 steps/epoch → 350 total |
| Batch size | 16 | Fits 16 GB unified memory; more in-batch negatives = stronger signal |
| LR | 2e-5 | Standard for sentence-transformer fine-tuning |
| Warmup | 10% of total steps | Prevents early instability |
| Precision | bf16 (CUDA) / fp32 (MPS/CPU) | MPS bf16 training support is unreliable |

### 3. Evaluate

```bash
USE_MOCK_EMBEDDINGS=false python eval/finetune/eval_comparison.py
```

Scrolls all 2 905 corpus chunks from Qdrant, embeds them with each of the
three models, and computes dense-only recall@10 (no BM25, no reranker) so
the comparison isolates the embedding quality.

---

## Results

> Run `eval_comparison.py` to fill in the table. Results are intentionally not
> committed as they depend on local training and will be added after the run.

| Model | Dim | recall@10 (dense-only) |
|---|---|---|
| bge-base-en-v1.5 (prod) | 768 | TBD |
| bge-small-en-v1.5 (base) | 384 | TBD |
| bge-small-en-v1.5 (fine-tuned) | 384 | TBD |

**Interpretation**: the recall delta between bge-small base and fine-tuned shows
the value of domain adaptation. If fine-tuned bge-small approaches bge-base
recall while being 4× smaller, it's a strong signal to consider re-indexing with
the fine-tuned model.

---

## Gitignored artifacts

| Path | Why gitignored |
|---|---|
| `eval/finetune/triples.jsonl` | Generated from indexed corpus; reproducible |
| `eval/finetune/bge-small-finetuned/` | ~120 MB model weights |
