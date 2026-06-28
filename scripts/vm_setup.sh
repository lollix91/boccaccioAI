#!/bin/bash
# BoccaccioAI - VM Setup Script (run on Hetzner VM after SSH)
# De Lauretis Tech
#
# Provisioning automatico di una VM Hetzner Cloud per le Fasi 1-2.
# Da lanciare come root subito dopo aver creato la VM.
#
# Uso:
#   ssh root@<VM_IP> 'bash -s' < scripts/vm_setup.sh
#
# Oppure copiare il file sulla VM e lanciarlo:
#   scp scripts/vm_setup.sh root@<VM_IP>:/root/
#   ssh root@<VM_IP> 'bash /root/vm_setup.sh'

set -euo pipefail

echo "============================================"
echo "  BoccaccioAI - VM Hetzner Setup"
echo "============================================"
echo ""

# ─── 1. Aggiornamento sistema ──────────────────────────
echo "[1/6] Aggiornamento sistema..."
apt-get update -qq
apt-get upgrade -y -qq
apt-get install -y -qq \
    python3 python3-pip python3-venv \
    git curl wget build-essential \
    cmake pkg-config \
    libffi-dev libssl-dev

echo "  Python3: $(python3 --version)"
echo ""

# ─── 2. Creazione ambiente virtuale ────────────────────
echo "[2/6] Creazione ambiente virtuale Python..."
python3 -m venv /opt/boccaccio-venv
source /opt/boccaccio-venv/bin/activate
pip install --upgrade pip -q
echo "  venv attivo: $(which python)"
echo ""

# ─── 3. Clone del repository ───────────────────────────
echo "[3/6] Clone repository..."
cd /opt
if [ -d "boccaccioAI" ]; then
    cd boccaccioAI
    git pull
else
    git clone https://github.com/lollix91/boccaccioAI.git
    cd boccaccioAI
fi
echo "  Repo in: $(pwd)"
echo "  Commit: $(git rev-parse --short HEAD)"
echo ""

# ─── 4. Installazione dipendenze ───────────────────────
echo "[4/6] Installazione dipendenze Python..."
pip install -r requirements.txt -q 2>&1 | tail -5
echo "  Dipendenze installate"
echo ""

# ─── 5. Verifica installazione ─────────────────────────
echo "[5/6] Verifica installazione..."
python -c "
import torch
import yaml
import numpy
import tokenizers
import datasets
from datasketch import MinHash
print(f'  torch: {torch.__version__}')
print(f'  tokenizers: {tokenizers.__version__}')
print(f'  datasets: {datasets.__version__}')
print('  Tutti i moduli importati correttamente')
"
echo ""

# ─── 6. Configurazione disk space check ────────────────
echo "[6/6] Verifica spazio disco..."
df -h / | awk 'NR==2 {print "  Disco root: " $4 " liberi su " $2}'
echo ""

# ─── Creazione directory dati ──────────────────────────
mkdir -p data/raw data/filtered data/tokenized/pretrain
mkdir -p tokenizer checkpoints

echo "============================================"
echo "  Setup completato!"
echo "============================================"
echo ""
echo "Prossimi passi:"
echo "  1. Fase 1:  python -m src.tokenizer.train_tokenizer"
echo "  2. Fase 2:  python -m src.data.download && python -m src.data.filter && python -m src.data.tokenize_corpus"
echo ""
echo "Oppure lancia lo script completo:"
echo "  bash scripts/run_fases_1_2.sh"
echo ""
echo "Per attivare il venv in futuro:"
echo "  source /opt/boccaccio-venv/bin/activate"
