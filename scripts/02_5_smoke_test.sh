#!/bin/bash
# BoccaccioAI - Fase 2.5: Smoke Test su GPU Locale
# De Lauretis Tech
#
# Addestra il modello nano (11M parametri) per pochi step sui dati
# reali pre-tokenizzati per verificare che l'intera pipeline funzioni
# prima di pagare l'H100.
#
# Requisiti: 1x GPU con >=4GB VRAM (RTX 3060 12GB va benissimo)
# Tempo stimato: ~10-15 minuti
# Costo: 0 EUR

set -euo pipefail

echo "=== BoccaccioAI - Smoke Test su GPU Locale ==="
echo ""
echo "Modello: nano (11M parametri)"
echo "Obiettivo: verificare pipeline end-to-end su dati reali"
echo ""

# Configurazione smoke test
SMOKE_STEPS=200
SMOKE_OUTPUT_DIR="checkpoints/smoke_test"

# Verifica che i dati pre-tokenizzati esistano
if [ ! -f "data/tokenized/pretrain/train.bin" ]; then
    echo "ERRORE: data/tokenized/pretrain/train.bin non trovato."
    echo "Esegui prima: bash scripts/02_preprocess_data.sh"
    exit 1
fi

# Verifica che il tokenizer esista
if [ ! -f "tokenizer/boccaccio-32k.json" ]; then
    echo "ERRORE: tokenizer/boccaccio-32k.json non trovato."
    echo "Esegui prima: bash scripts/01_train_tokenizer.sh"
    exit 1
fi

# Verifica GPU
echo "Verifica GPU..."
python -c "
import torch
if not torch.cuda.is_available():
    print('ERRORE: CUDA non disponibile. Serve una GPU per questo test.')
    exit(1)
gpu_name = torch.cuda.get_device_name(0)
gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
print(f'  GPU rilevata: {gpu_name} ({gpu_mem:.1f} GB VRAM)')
print(f'  CUDA version: {torch.version.cuda}')
"
echo ""

# Crea config di training smoke test on-the-fly
echo "Creazione config smoke test..."
python -c "
import yaml

config = {
    'pretrain': {
        'dataset_path': 'data/tokenized/pretrain',
        'sequence_length': 512,           # Ridotto per nano
        'num_tokens': 200 * 512 * 4,      # 200 steps * 512 seq * batch 4

        'micro_batch_size': 4,
        'gradient_accumulation_steps': 1,
        # Effective batch: 4 * 1 * 512 = 2048 tokens/step

        'optimizer': 'adamw',
        'learning_rate': 3.0e-4,
        'min_learning_rate': 3.0e-5,
        'weight_decay': 0.1,
        'beta1': 0.9,
        'beta2': 0.95,
        'eps': 1.0e-8,

        'scheduler': 'cosine',
        'warmup_steps': 20,

        'gradient_clip_val': 1.0,
        'gradient_clip_algorithm': 'norm',

        'precision': 'bf16-mixed',
        'compile_model': False,           # Disattivato per compatibilita'
        'use_flash_attention': True,

        'checkpoint_every_n_steps': 100,
        'checkpoint_dir': 'checkpoints/smoke_test',
        'save_top_k': 1,

        'log_every_n_steps': 10,
        'val_check_interval': 100,
        'val_split_ratio': 0.005,

        'accelerator': 'gpu',
        'devices': 1,
        'strategy': 'auto',
        'num_workers': 2,
    }
}

with open('configs/training_smoke.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
print('  Config salvato: configs/training_smoke.yaml')
"
echo ""

# Lancia il training smoke test
echo "Avvio smoke test (${SMOKE_STEPS} steps)..."
echo ""
python -m src.training.train \
    --mode pretrain \
    --model-config configs/model.yaml \
    --model-variant nano \
    --training-config configs/training_smoke.yaml \
    --data-path data/tokenized/pretrain \
    --tokenizer-path tokenizer/boccaccio-32k.json \
    --wandb-offline

echo ""
echo "=== Smoke Test Complete ==="
echo ""
echo "Verifica i risultati:"
echo "  1. La loss dovrebbe scendere (controlla i log o wandb)"
echo "  2. I checkpoint dovrebbero essere in ${SMOKE_OUTPUT_DIR}/"
echo "  3. Nessun errore di OOM o NaN"
echo ""
echo "Se tutto e' andato bene, puoi procedere con la Fase 3 su H100."
echo "Se ci sono problemi, correggi qui gratis invece che sull'H100 a 1.80 EUR/ora."
