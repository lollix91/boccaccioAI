#!/bin/bash
# BoccaccioAI - Fasi 1-2 su VM Hetzner
# De Lauretis Tech
#
# Lancia tokenizer training + data pipeline completa.
# Da eseguire sulla VM dopo aver fatto il setup con vm_setup.sh.
#
# Uso:
#   source /opt/boccaccio-venv/bin/activate
#   cd /opt/boccaccioAI
#   bash scripts/run_fases_1_2.sh

set -euo pipefail

# Attiva venv se non gia' attivo
if [ -z "${VIRTUAL_ENV:-}" ]; then
    source /opt/boccaccio-venv/bin/activate
fi

cd /opt/boccaccioAI

echo "============================================"
echo "  BoccaccioAI - Fasi 1-2 su VM"
echo "  Inizio: $(date)"
echo "============================================"
echo ""

# ─── Fase 1: Tokenizer ─────────────────────────────────
echo "=== FASE 1: Training Tokenizer BPE 32K ==="
echo "Inizio: $(date)"
python -m src.tokenizer.train_tokenizer \
    --config configs/tokenizer.yaml \
    --output-dir tokenizer/ \
    --vocab-size 32000 \
    --corpus-size-gb 5
echo "Fine: $(date)"
echo ""

# ─── Fase 2a: Download CulturaX ────────────────────────
echo "=== FASE 2a: Download CulturaX IT ==="
echo "Inizio: $(date)"
python -m src.data.download \
    --output-dir data/raw \
    --max-size-gb 30
echo "Fine: $(date)"
echo ""

# ─── Fase 2b: Filtering ────────────────────────────────
echo "=== FASE 2b: Filtering e Deduplication ==="
echo "Inizio: $(date)"
python -m src.data.filter \
    --input-dir data/raw \
    --output-dir data/filtered \
    --jaccard-threshold 0.85 \
    --min-doc-length 200 \
    --num-workers 6
echo "Fine: $(date)"
echo ""

# ─── Fase 2c: Pre-tokenizzazione ───────────────────────
echo "=== FASE 2c: Pre-tokenizzazione ==="
echo "Inizio: $(date)"
python -m src.data.tokenize_corpus \
    --input-dir data/filtered \
    --output-dir data/tokenized/pretrain \
    --tokenizer-path tokenizer/boccaccio-32k.json \
    --sequence-length 2048 \
    --val-split 0.005
echo "Fine: $(date)"
echo ""

# ─── Riepilogo ─────────────────────────────────────────
echo "============================================"
echo "  Fasi 1-2 completate!"
echo "  Fine: $(date)"
echo "============================================"
echo ""
echo "Output generati:"
echo "  Tokenizer:     tokenizer/boccaccio-32k.json"
echo "  Dati train:    data/tokenized/pretrain/train.bin"
echo "  Dati val:      data/tokenized/pretrain/val.bin"
echo "  Metadata:      data/tokenized/pretrain/meta.json"
echo ""
echo "Dimensioni file:"
du -sh tokenizer/boccaccio-32k.json
du -sh data/tokenized/pretrain/train.bin
du -sh data/tokenized/pretrain/val.bin
echo ""
echo "Per scaricare i risultati in locale:"
echo "  scp root@<VM_IP>:/opt/boccaccioAI/tokenizer/boccaccio-32k.json ./tokenizer/"
echo "  scp root@<VM_IP>:/opt/boccaccioAI/data/tokenized/pretrain/*.bin ./data/tokenized/pretrain/"
echo "  scp root@<VM_IP>:/opt/boccaccioAI/data/tokenized/pretrain/meta.json ./data/tokenized/pretrain/"
