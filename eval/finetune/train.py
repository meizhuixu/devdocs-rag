"""Fine-tune BAAI/bge-small-en-v1.5 on mined hard-negative triples.

Reads eval/finetune/triples.jsonl (from mine_triples.py), trains with
MultipleNegativesRankingLoss, and saves the fine-tuned model to
eval/finetune/bge-small-finetuned/ (gitignored).

Precision strategy (D25):
  - CUDA available  → bf16 training (fast, stable)
  - MPS (Apple Silicon) → fp32 (MPS bf16 training support is unreliable)
  - CPU only → fp32

Usage:
    python eval/finetune/train.py
    python eval/finetune/train.py --triples eval/finetune/triples.jsonl --epochs 5 --batch 16
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_TRIPLES = Path(__file__).parent / "triples.jsonl"
_OUTPUT = Path(__file__).parent / "bge-small-finetuned"
_BASE_MODEL = "BAAI/bge-small-en-v1.5"


def _device_and_precision() -> tuple[str, bool, bool]:
    """Return (device, use_bf16, use_fp16) based on hardware availability."""
    import torch

    if torch.cuda.is_available():
        return "cuda", True, False  # bf16 on CUDA
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", False, False  # fp32 on MPS (bf16 unreliable in training)
    return "cpu", False, False  # fp32 on CPU


def train(
    triples_path: Path,
    output_path: Path,
    base_model: str,
    epochs: int,
    batch_size: int,
) -> None:
    """Fine-tune `base_model` on triples and save to `output_path`."""
    import torch  # noqa: I001 (lazy imports; torch before datasets alphabetically)
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.trainer import SentenceTransformerTrainer
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments

    device, use_bf16, use_fp16 = _device_and_precision()
    logger.info(
        "device=%s  bf16=%s  fp16=%s  epochs=%d  batch=%d",
        device,
        use_bf16,
        use_fp16,
        epochs,
        batch_size,
    )

    # Load triples
    triples = [
        json.loads(line)
        for line in triples_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not triples:
        raise ValueError(f"no triples found in {triples_path}")
    logger.info("loaded %d triples", len(triples))
    if len(triples) < batch_size:
        logger.warning(
            "only %d triples, fewer than batch_size=%d — training may be unstable",
            len(triples),
            batch_size,
        )

    train_dataset = Dataset.from_dict(
        {
            "anchor": [t["query"] for t in triples],
            "positive": [t["positive"] for t in triples],
            "negative": [t["hard_negative"] for t in triples],
        }
    )

    model = SentenceTransformer(base_model, device=device)
    loss = MultipleNegativesRankingLoss(model)

    steps_per_epoch = max(1, len(triples) // batch_size)
    logging_steps = max(1, steps_per_epoch // 4)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        warmup_ratio=0.1,
        bf16=use_bf16,
        fp16=use_fp16,
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=logging_steps,
        report_to="none",
        learning_rate=2e-5,
        weight_decay=0.01,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )

    logger.info("starting training …")
    trainer.train()

    # Save the final fine-tuned model (separate from checkpoint state)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    logger.info("saved fine-tuned model → %s", output_path)

    # Quick sanity check: verify the saved model loads and embeds
    check = SentenceTransformer(str(output_path), device=device)
    vec = check.encode(["sanity check"], normalize_embeddings=True)
    logger.info(
        "sanity embed: shape=%s  norm=%.4f", vec.shape, float(torch.norm(torch.tensor(vec)))
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune bge-small-en-v1.5 on mined triples.")
    parser.add_argument("--triples", type=Path, default=_TRIPLES)
    parser.add_argument("--output", type=Path, default=_OUTPUT)
    parser.add_argument("--base-model", default=_BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args(argv)

    if not args.triples.exists():
        print(
            f"ERROR: {args.triples} not found. Run mine_triples.py first.",
            file=sys.stderr,
        )
        return 1

    train(
        triples_path=args.triples,
        output_path=args.output,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
