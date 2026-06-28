#!/bin/bash
# BoccaccioAI - Fase 5: Evaluation and Export
# De Lauretis Tech
set -euo pipefail

echo "=== BoccaccioAI - Evaluation ==="

# Q&A mode test
python -m src.inference.generate \
    --model-dir checkpoints/finetune \
    --tokenizer-path tokenizer/boccaccio-32k.json \
    --config-path configs/model.yaml \
    --mode qa \
    --context "Roma e' la capitale della Repubblica Italiana. Si trova nella regione Lazio, nell'Italia centrale." \
    --question "Qual e' la capitale dell'Italia?" \
    --max-new-tokens 128 \
    --temperature 0.3

echo "---"

# Free generation test
python -m src.inference.generate \
    --model-dir checkpoints/finetune \
    --tokenizer-path tokenizer/boccaccio-32k.json \
    --config-path configs/model.yaml \
    --mode generate \
    --prompt "L'intelligenza artificiale in Italia" \
    --max-new-tokens 256 \
    --temperature 0.7

echo "=== Evaluation complete ==="
