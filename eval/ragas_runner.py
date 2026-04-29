"""Ragas regression runner.

Phase 1: a callable stub that loads the golden JSONL and reports counts. Phase
5 wires Ragas metrics on top of the live retrieval + generation stack.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalReport:
    items: int
    todo: str = "Phase 5: wire Ragas metrics (context_recall, faithfulness, ...)"


def load_dataset(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def run(dataset_path: Path) -> EvalReport:
    rows = load_dataset(dataset_path)
    return EvalReport(items=len(rows))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Ragas regression on a golden set.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("eval/datasets/golden_qa.jsonl"),
    )
    args = parser.parse_args(argv)
    report = run(args.dataset)
    print(f"items={report.items} todo={report.todo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
