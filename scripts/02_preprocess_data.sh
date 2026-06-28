#!/bin/bash
# BoccaccioAI - Fase 2: Data Pipeline (Download + Filter + Tokenize)
# De Lauretis Tech
set -euo pipefail

echo "=== BoccaccioAI - Data Pipeline ==="

# Step 1: Download CulturaX Italian
echo "[1/3] Downloading CulturaX IT..."
python -m src.data.download \
    --output-dir data/raw \
    --max-size-gb 30

# Step 2: Filter and deduplicate
echo "[2/3] Filtering and deduplicating..."
python -m src.data.filter \
    --input-dir data/raw \
    --output-dir data/filtered \
    --jaccard-threshold 0.85 \
    --min-doc-length 200 \
    --num-workers 3

# Step 3: Pre-tokenize into binary format
echo "[3/3] Pre-tokenizing corpus..."
python -m src.data.tokenize_corpus \
    --input-dir data/filtered \
    --output-dir data/tokenized/pretrain \
    --tokenizer-path tokenizer/boccaccio-32k.json \
    --sequence-length 2048 \
    --val-split 0.005

echo "=== Data pipeline complete ==="
