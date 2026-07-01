#!/bin/bash
# Auto-upload checkpoint daemon per Vast.ai (v2 - no version bloat)
# - Carica ogni checkpoint nominato (epoch=0-step=N.ckpt) UNA sola volta
#   con il suo nome originale (non sovrascrive last.ckpt ad ogni ciclo)
# - Quando il training termina, carica last.ckpt (finale) una sola volta
# - Mantiene un registro locale dei checkpoint gia' caricati
# - Pulisce i backup -v*.ckpt per risparmiare spazio

set -e

PROJECT_DIR="/workspace/boccaccioAI"
CKPT_DIR="$PROJECT_DIR/checkpoints/pretrain"
HF_TOKEN="${HF_TOKEN:-$(cat /workspace/boccaccioAI/.hf_token 2>/dev/null)}"
HF_REPO="lollix91/boccaccio-data"
UPLOAD_INTERVAL=300
UPLOADED_LOG="$CKPT_DIR/.uploaded_registry"

mkdir -p "$CKPT_DIR"
touch "$UPLOADED_LOG"

echo "[daemon] Avvio auto-upload checkpoint daemon v2..."
echo "[daemon] Monitor: $CKPT_DIR/epoch=0-step=*.ckpt"
echo "[daemon] Repo HF: $HF_REPO"
echo "[daemon] Registro: $UPLOADED_LOG"

upload_ckpt() {
    LOCAL="$1"
    REPO_PATH="$2"
    TMP_CKPT="/tmp/ckpt_upload_$(date +%s).ckpt"
    cp "$LOCAL" "$TMP_CKPT" 2>/dev/null
    if [ ! -f "$TMP_CKPT" ]; then
        echo "[daemon] ERRORE copia temporanea di $(basename $LOCAL), riprovo."
        return 1
    fi
    echo "[daemon] Upload $REPO_PATH ($(du -h "$TMP_CKPT" | cut -f1))..."
    python3 -c "
from huggingface_hub import HfApi
api = HfApi(token='$HF_TOKEN')
api.upload_file(
    path_or_fileobj='$TMP_CKPT',
    path_in_repo='$REPO_PATH',
    repo_id='$HF_REPO',
    repo_type='dataset',
    token='$HF_TOKEN',
)
print('[daemon] Upload completato: $REPO_PATH')
" 2>&1
    RESULT=$?
    rm -f "$TMP_CKPT" 2>/dev/null
    return $RESULT
}

is_uploaded() {
    grep -qxF "$1" "$UPLOADED_LOG" 2>/dev/null
}

mark_uploaded() {
    echo "$1" >> "$UPLOADED_LOG"
}

while true; do
    rm -f $CKPT_DIR/last-v*.ckpt $CKPT_DIR/epoch=0-step=*-v*.ckpt 2>/dev/null

    # 1. Carica ogni checkpoint nominato NON ancora caricato
    for CKPT in $(find $CKPT_DIR -maxdepth 1 -name 'epoch=0-step=*.ckpt' ! -name '*-v*.ckpt' -printf '%f\n' 2>/dev/null | sort); do
        if is_uploaded "$CKPT"; then
            continue
        fi
        CKPT_PATH="$CKPT_DIR/$CKPT"
        # Verifica che il file sia completo (non in scrittura)
        SIZE1=$(stat -c '%s' "$CKPT_PATH" 2>/dev/null || echo 0)
        sleep 2
        SIZE2=$(stat -c '%s' "$CKPT_PATH" 2>/dev/null || echo 0)
        if [ "$SIZE1" != "$SIZE2" ] || [ "$SIZE1" -lt 1000000 ]; then
            echo "[daemon] $CKPT ancora in scrittura, salto."
            continue
        fi
        echo "[daemon] Nuovo checkpoint: $CKPT"
        if upload_ckpt "$CKPT_PATH" "checkpoints/pretrain/$CKPT"; then
            mark_uploaded "$CKPT"
            echo "[daemon] Registrato: $CKPT"
            # Pulisci checkpoint locali vecchi: tieni solo gli ultimi 2 + last.ckpt
            UPLOADED_CKPTS=$(grep '^epoch=0-step=' "$UPLOADED_LOG" 2>/dev/null | sort)
            CKPT_COUNT=$(echo "$UPLOADED_CKPTS" | grep -c . 2>/dev/null || echo 0)
            if [ "$CKPT_COUNT" -gt 2 ]; then
                TO_DELETE=$(echo "$UPLOADED_CKPTS" | head -n -2)
                for OLD in $TO_DELETE; do
                    if [ -f "$CKPT_DIR/$OLD" ]; then
                        rm -f "$CKPT_DIR/$OLD"
                        echo "[daemon] Pulito checkpoint locale vecchio: $OLD"
                    fi
                done
            fi
        else
            echo "[daemon] ERRORE upload $CKPT, riprovo al prossimo ciclo."
        fi
    done

    # 2. Controlla se il training e' terminato (processo non attivo)
    TRAINING_RUNNING=$(pgrep -f 'src.training.train' | head -1)
    if [ -z "$TRAINING_RUNNING" ]; then
        echo "[daemon] Training non piu' attivo. Upload finale last.ckpt..."
        if [ -f "$CKPT_DIR/last.ckpt" ]; then
            if ! is_uploaded "last.ckpt"; then
                if upload_ckpt "$CKPT_DIR/last.ckpt" "checkpoints/pretrain/last.ckpt"; then
                    mark_uploaded "last.ckpt"
                    echo "[daemon] Upload finale completato. last.ckpt su HF Hub."
                else
                    echo "[daemon] ERRORE upload finale last.ckpt, riprovo."
                fi
            fi
        fi
        # Carica anche l'ultimo checkpoint nominato come last.ckpt
        LATEST_NAMED=$(find $CKPT_DIR -maxdepth 1 -name 'epoch=0-step=*.ckpt' ! -name '*-v*.ckpt' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2)
        if [ -n "$LATEST_NAMED" ] && ! is_uploaded "last.ckpt_final"; then
            echo "[daemon] Upload $(basename $LATEST_NAMED) come last.ckpt finale..."
            if upload_ckpt "$LATEST_NAMED" "checkpoints/pretrain/last.ckpt"; then
                mark_uploaded "last.ckpt_final"
                echo "[daemon] Checkpoint finale sincronizzato come last.ckpt."
            fi
        fi
        echo "[daemon] Training terminato. Daemon in attesa (non esce per permettere retry)."
    fi

    sleep $UPLOAD_INTERVAL
done
