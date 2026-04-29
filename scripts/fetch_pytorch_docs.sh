#!/usr/bin/env bash
# Sparse-checkout PyTorch's docs/source/ into data/raw/pytorch.
# Idempotent: if already cloned, runs git pull instead.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data/raw/pytorch"

mkdir -p "$ROOT/data/raw"

if [ -d "$DEST/.git" ]; then
    echo "[fetch] $DEST exists — pulling latest"
    git -C "$DEST" pull --depth=1 origin main
else
    echo "[fetch] cloning pytorch (sparse: docs/source) → $DEST"
    git clone --filter=blob:none --no-checkout --depth=1 \
        https://github.com/pytorch/pytorch.git "$DEST"
    git -C "$DEST" sparse-checkout init --cone
    git -C "$DEST" sparse-checkout set docs/source
    git -C "$DEST" checkout
fi

echo "[fetch] docs size:"
du -sh "$DEST/docs/source/" || true
echo "[fetch] .rst file count:"
find "$DEST/docs/source/" -name "*.rst" | wc -l
echo "[fetch] done"
