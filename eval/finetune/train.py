"""Fine-tune BAAI/bge-small-en-v1.5 on mined hard-negative triples.

Reads eval/finetune/triples.jsonl (from mine_triples.py), trains with
MultipleNegativesRankingLoss, and saves the fine-tuned model to
eval/finetune/bge-small-finetuned/ (gitignored).

Precision strategy (D25):
  - CUDA available  → bf16 training (fast, stable)
  - MPS (Apple Silicon) → fp32, but only if --device mps is passed explicitly.
    Default is CPU because 20 GB unified memory on M3 Air is typically exhausted
    by other processes, making MPS training crash at step 1.
  - CPU (default on non-CUDA) → fp32

Usage:
    python eval/finetune/train.py
    python eval/finetune/train.py --triples eval/finetune/triples.jsonl --epochs 5 --batch 16
    python eval/finetune/train.py --device mps   # only if memory headroom is ≥ 4 GB
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


def _device_and_precision(requested: str | None = None) -> tuple[str, bool, bool]:
    """Return (device, use_bf16, use_fp16) based on hardware and optional override.

    MPS is not used by default: on a 20 GB M3 Air the unified memory is typically
    almost exhausted by the OS + other apps, so the MPS training allocator crashes
    at the first backward pass. Pass requested="mps" only when you have ≥ 4 GB free.
    """
    import torch

    if requested is not None:
        device = requested
        use_bf16 = device == "cuda"
        return device, use_bf16, False

    if torch.cuda.is_available():
        return "cuda", True, False  # bf16 on CUDA
    return "cpu", False, False  # fp32 on CPU (MPS requires explicit --device mps)


def train(
    triples_path: Path,
    output_path: Path,
    base_model: str,
    epochs: int,
    batch_size: int,
    device_override: str | None = None,
) -> None:
    """Fine-tune `base_model` on triples and save to `output_path`."""
    device, use_bf16, use_fp16 = _device_and_precision(device_override)

    from datasets import Dataset  # noqa: I001 (torch follows sentence_transformers for the MPS patch)
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.trainer import SentenceTransformerTrainer
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments
    import torch

    # Patch out MPS so TransformersTrainingArguments.device (and Accelerate's PartialState)
    # return CPU instead of MPS.  Both call torch.backends.mps.is_available() at the point
    # they're constructed — not at import time — so patching here (after imports, before
    # args construction) is sufficient.  Without this the Trainer calls model.to("mps")
    # even when we passed device="cpu" to SentenceTransformer, and training OOMs at step 2
    # on a near-full M3 Air.  The original function is restored after trainer.train().
    _orig_mps_available = torch.backends.mps.is_available
    if device == "cpu" and hasattr(torch.backends, "mps"):
        torch.backends.mps.is_available = lambda: False  # type: ignore[method-assign]
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
        warmup_steps=0.1,  # float = ratio in Transformers v5+ (replaces warmup_ratio)
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
    torch.backends.mps.is_available = _orig_mps_available  # type: ignore[method-assign]

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
    parser.add_argument(
        "--device",
        default=None,
        help="Force a specific device (cpu/cuda/mps). Default: cuda if available, else cpu.",
    )
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
        device_override=args.device,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
