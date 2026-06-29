#!/bin/bash
# BoccaccioAI - Fase 3: Pre-training on H100
# De Lauretis Tech
set -euo pipefail

echo "=== BoccaccioAI - Pre-training 700M Model ==="

python -m src.training.train \
    --mode pretrain \
    --model-config configs/model.yaml \
    --model-variant model \
    --training-config configs/training.yaml \
    --data-path data/tokenized/pretrain \
    --tokenizer-path tokenizer/boccaccio-32k.json

echo "=== Pre-training complete ==="
