#!/bin/bash
# BoccaccioAI - Fase 4: Instruction Fine-Tuning
# De Lauretis Tech
set -euo pipefail

echo "=== BoccaccioAI - Instruction Fine-Tuning ==="

python -m src.training.train \
    --mode finetune \
    --model-config configs/model.yaml \
    --model-variant model \
    --training-config configs/training.yaml \
    --data-path data/tokenized/finetune \
    --tokenizer-path tokenizer/boccaccio-32k.json \
    --resume-from checkpoints/pretrain/last.ckpt

echo "=== Fine-tuning complete ==="
