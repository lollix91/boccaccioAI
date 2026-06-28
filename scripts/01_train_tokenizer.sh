#!/bin/bash
# BoccaccioAI - Fase 1: Train BPE Tokenizer
# De Lauretis Tech
set -euo pipefail

echo "=== BoccaccioAI - Training Tokenizer BPE 32K ==="

python -m src.tokenizer.train_tokenizer \
    --config configs/tokenizer.yaml \
    --output-dir tokenizer/ \
    --vocab-size 32000 \
    --corpus-size-gb 5

echo "=== Tokenizer training complete ==="
