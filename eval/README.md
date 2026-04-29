# Eval

This directory holds the regression suite for retrieval and generation quality.

## What's here

- `ragas_runner.py` — entry point that loads a golden dataset, runs the live
  retrieval + generation stack on each query, and scores it with
  [Ragas](https://docs.ragas.io/) metrics.
- `datasets/golden_qa.jsonl` — the hand-written golden set. Each line:
  ```json
  {
    "question": "...",
    "namespaces": ["repo_csye6225"],
    "expected_contexts": ["repo_csye6225:src/auth/cognito.py:refresh_token"],
    "reference_answer": "..."
  }
  ```

## Phase 1

`ragas_runner.py` is a stub that imports cleanly and prints a TODO. The golden
JSONL is empty — Phase 5 fills it with 50 hand-written items covering every
namespace.

## Running

```bash
uv run python -m eval.ragas_runner --dataset eval/datasets/golden_qa.jsonl
```

## CI gate

`.github/workflows/eval.yml` runs this on PRs touching `retrieval/`,
`generation/`, or `eval/datasets/`. The gate fires when context_recall or
faithfulness regresses against `main` by more than the configured threshold.

## Metrics tracked

- `context_recall` — did we retrieve the chunks the answer needed?
- `context_precision` — were the retrieved chunks relevant?
- `faithfulness` — did the answer stick to the retrieved context?
- `answer_relevancy` — did the answer actually address the question?
