"""Deterministic stratified train/holdout split of the golden-QA set.

Phase 6 out-of-sample eval: the original fine-tune mined triples from ALL 50
golden items and evaluated on the same 50, so its recall@10 = 1.0 was in-sample
memorisation. This script holds out 10 single-namespace items (stratified,
evenly spaced within each namespace bucket, no randomness) so a retrained
model can be scored on queries it never saw during mining.

Cross-namespace items (the hardest, only 4) all stay in the training split —
holding them out would leave too few to train that signal.

Usage:
    python eval/finetune/split_holdout.py
    # writes eval/datasets/golden_train_40.jsonl + golden_holdout_10.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

_GOLDEN = Path(__file__).parent.parent / "datasets" / "golden_qa.jsonl"
_TRAIN_OUT = Path(__file__).parent.parent / "datasets" / "golden_train_40.jsonl"
_HOLDOUT_OUT = Path(__file__).parent.parent / "datasets" / "golden_holdout_10.jsonl"

# Holdout quota per single-namespace bucket (total = 10).
_HOLDOUT_QUOTA = {
    "pytorch_docs": 3,
    "repo_devdocs_rag": 3,
    "repo_auto_sentinel": 2,
    "repo_devcontext_mcp": 2,
}


def main() -> int:
    items = [json.loads(line) for line in _GOLDEN.read_text().splitlines() if line.strip()]

    buckets: dict[str, list[dict[str, object]]] = {}
    for item in items:
        namespaces = item["namespaces"]
        key = namespaces[0] if len(namespaces) == 1 else "_cross"
        buckets.setdefault(key, []).append(item)

    holdout_ids: set[str] = set()
    for ns, quota in _HOLDOUT_QUOTA.items():
        bucket = sorted(buckets[ns], key=lambda it: str(it["id"]))
        # Evenly spaced picks across the bucket — deterministic, no seed.
        step = len(bucket) / quota
        for i in range(quota):
            holdout_ids.add(str(bucket[int(i * step + step / 2)]["id"]))

    train = [it for it in items if it["id"] not in holdout_ids]
    holdout = [it for it in items if it["id"] in holdout_ids]
    assert len(holdout) == 10, f"expected 10 holdout items, got {len(holdout)}"

    _TRAIN_OUT.write_text("".join(json.dumps(it) + "\n" for it in train))
    _HOLDOUT_OUT.write_text("".join(json.dumps(it) + "\n" for it in holdout))
    print(f"train: {len(train)} -> {_TRAIN_OUT.name}")
    print(f"holdout: {len(holdout)} -> {_HOLDOUT_OUT.name}: {sorted(holdout_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
