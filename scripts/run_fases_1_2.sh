#!/bin/bash
# BoccaccioAI - Fasi 1-2 su VM Hetzner
# De Lauretis Tech
#
# Lancia tokenizer training + data pipeline completa.
# Scrive progresso in /opt/boccaccioAI/progress.json per monitoraggio remoto.
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

PROGRESS_FILE="/opt/boccaccioAI/progress.json"
START_TIME=$(date +%s)

write_progress() {
    local stage="$1"
    local stage_name="$2"
    local percent="$3"
    local status="$4"
    local elapsed=$(( $(date +%s) - START_TIME ))
    cat > "$PROGRESS_FILE" << EOF
{
  "stage": "$stage",
  "stage_name": "$stage_name",
  "percent": $percent,
  "status": "$status",
  "elapsed_seconds": $elapsed,
  "started_at": "$(date -d @$START_TIME -u +%Y-%m-%dT%H:%M:%SZ)",
  "updated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
}

echo "============================================"
echo "  BoccaccioAI - Fasi 1-2 su VM"
echo "  Inizio: $(date)"
echo "============================================"
echo ""

# ─── Fase 1: Tokenizer ─────────────────────────────────
write_progress "fase_1_tokenizer" "Training Tokenizer BPE 32K" 0 "running"
echo "=== FASE 1: Training Tokenizer BPE 32K ==="
echo "Inizio: $(date)"
python -m src.tokenizer.train_tokenizer \
    --config configs/tokenizer.yaml \
    --output-dir tokenizer/ \
    --vocab-size 32000 \
    --corpus-size-gb 5 \
    2>&1 | tee logs/fase_1_tokenizer.log
write_progress "fase_1_tokenizer" "Training Tokenizer BPE 32K" 100 "completed"
echo "Fine: $(date)"
echo ""

# ─── Fase 2a: Download CulturaX ────────────────────────
write_progress "fase_2a_download" "Download CulturaX IT" 0 "running"
echo "=== FASE 2a: Download CulturaX IT ==="
echo "Inizio: $(date)"
python -m src.data.download \
    --output-dir data/raw \
    --max-size-gb 30 \
    2>&1 | tee logs/fase_2a_download.log
write_progress "fase_2a_download" "Download CulturaX IT" 100 "completed"
echo "Fine: $(date)"
echo ""

# ─── Fase 2b: Filtering ────────────────────────────────
write_progress "fase_2b_filtering" "Filtering e Deduplication" 0 "running"
echo "=== FASE 2b: Filtering e Deduplication ==="
echo "Inizio: $(date)"
python -m src.data.filter \
    --input-dir data/raw \
    --output-dir data/filtered \
    --jaccard-threshold 0.85 \
    --min-doc-length 200 \
    --num-workers 6 \
    2>&1 | tee logs/fase_2b_filtering.log
write_progress "fase_2b_filtering" "Filtering e Deduplication" 100 "completed"
echo "Fine: $(date)"
echo ""

# ─── Fase 2c: Pre-tokenizzazione ───────────────────────
write_progress "fase_2c_tokenize" "Pre-tokenizzazione" 0 "running"
echo "=== FASE 2c: Pre-tokenizzazione ==="
echo "Inizio: $(date)"
python -m src.data.tokenize_corpus \
    --input-dir data/filtered \
    --output-dir data/tokenized/pretrain \
    --tokenizer-path tokenizer/boccaccio-32k.json \
    --sequence-length 2048 \
    --val-split 0.005 \
    2>&1 | tee logs/fase_2c_tokenize.log
write_progress "fase_2c_tokenize" "Pre-tokenizzazione" 100 "completed"
echo "Fine: $(date)"
echo ""

# ─── Completato ─────────────────────────────────────────
write_progress "completed" "Fasi 1-2 completate" 100 "completed"

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
