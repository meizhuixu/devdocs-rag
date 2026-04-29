# Data

This directory holds source corpora for ingestion. **Nothing here is committed
to git** (`data/raw/` and `data/processed/` are gitignored).

## Sources

devdocs-rag indexes one collection per source, mapped to a Qdrant namespace.

| Namespace | Source | Phase |
|---|---|---|
| `pytorch_docs` | PyTorch official docs (`pytorch.org/docs`) | 2 |
| `repo_devdocs_rag` | This repo, self-indexing | 4 |
| `repo_auto_sentinel` | Matrix project #1 — when it exists | 4 |
| `repo_devcontext_mcp` | Matrix project #3 — when it exists | 4 |
| `repo_llmops_dashboard` | Matrix project #4 — when it exists | 4 |
| `repo_csye6225` | Legacy course project (AWS-heavy) | 4 |

## Layout (when populated)

```
data/
├── raw/            ← upstream snapshots (git submodules / clones / mirrors)
│   ├── pytorch-docs/
│   └── repos/
│       ├── auto-sentinel/
│       ├── devdocs-rag/
│       ├── devcontext-mcp/
│       ├── llmops-dashboard/
│       └── csye6225-cloud-computing/
└── processed/      ← post-chunking artifacts (chunk JSONL, embeddings cache)
```

## How to populate (Phase 2)

PyTorch docs:

```bash
git clone --depth 1 https://github.com/pytorch/pytorch.git data/raw/pytorch-src
# Phase 2: run `python -m devdocs_rag.ingestion ... --namespace pytorch_docs`
```

User repos:

```bash
mkdir -p data/raw/repos
cd data/raw/repos
for r in auto-sentinel devdocs-rag devcontext-mcp llmops-dashboard csye6225-cloud-computing; do
  git clone "git@github.com:meizhuixu/$r.git" "$r" 2>/dev/null || echo "skip: $r not yet"
done
```

## Incremental indexing

The pipeline keys on per-file `commit_sha`. To force a re-index, delete the
namespace in Qdrant — the next run will re-embed every file.
