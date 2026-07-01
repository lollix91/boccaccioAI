"""BoccaccioAI - Vast.ai Setup & Training with auto-upload to HF Hub.

Si collega a un'istanza Vast.ai via SSH, scarica dati e checkpoint da HF Hub,
avvia il training in tmux con resume da ultimo checkpoint, e lancia un daemon
che carica automaticamente i checkpoint su HF Hub (fail-safe con doppia copia).

Uso:
    python scripts/vast_setup.py --host <ip> --port <porta> --key <path_ssh_key>
    python scripts/vast_setup.py --host 1.2.3.4 --port 12345 --key ~/.ssh/id_rsa

De Lauretis Tech
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import paramiko


# ─── Configurazione ───────────────────────────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = "lollix91/boccaccio-data"

PROJECT_DIR = "/workspace/boccaccioAI"
PYTHON = "python3"  # Vast.ai template usa python3, non python
PIP = "pip3"


# ─── Helper ───────────────────────────────────────────────────

def run(ssh: paramiko.SSHClient, cmd: str, timeout: int = 300) -> tuple[int, str]:
    """Esegue un comando SSH e ritorna (exit_code, output)."""
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return exit_code, (out + err).strip()


# ─── Step 1: Verifica GPU ─────────────────────────────────────

def check_gpu(ssh: paramiko.SSHClient) -> bool:
    print()
    print("=== Step 1: Verifica GPU ===")
    code, out = run(ssh, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null")
    if code != 0 or not out:
        print("  ERRORE: GPU non disponibile.")
        return False
    print(f"  GPU: {out}")
    return "H100" in out or "A100" in out or "RTX" in out


# ─── Step 2: Setup progetto e dipendenze ──────────────────────

def setup_project(ssh: paramiko.SSHClient) -> bool:
    print()
    print("=== Step 2: Setup progetto ===")

    # Vast.ai ha /workspace come storage persistente
    run(ssh, "mkdir -p /workspace")

    code, out = run(ssh, f"test -d {PROJECT_DIR}/.git && echo exists")
    if "exists" in out:
        print("  Repo presente, faccio pull...")
        run(ssh, f"cd {PROJECT_DIR} && git pull origin main 2>&1")
    else:
        print("  Clono il repo...")
        code, out = run(ssh, "cd /workspace && git clone https://github.com/lollix91/boccaccioAI.git 2>&1")
        if code != 0:
            print(f"  ERRORE: {out}")
            return False

    print("  Verifico compatibilita' PyTorch + CUDA driver...")
    # Il template Vast.ai puo' avere PyTorch compilato per CUDA 13.0
    # ma il driver GPU supporta solo 12.6. In tal caso, reinstalliamo PyTorch.
    code, out = run(ssh, f"{PYTHON} -c 'import torch; print(torch.__version__)' 2>/dev/null")
    code, driver_out = run(ssh, "nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null")
    code, cuda_out = run(ssh, "nvidia-smi | grep 'CUDA Version' | head -1 2>/dev/null")
    print(f"    PyTorch: {out.strip()}")
    print(f"    {cuda_out.strip()}")

    # Se PyTorch ha cu130 ma driver e' 12.6, reinstalliamo con cu126
    if "cu130" in out or "cu128" in out:
        print("  PyTorch incompatibile con driver CUDA 12.6. Reinstallo con CUDA 12.6...")
        run(ssh, f"{PIP} install torch==2.9.0 torchvision --index-url https://download.pytorch.org/whl/cu126 --force-reinstall 2>&1 | tail -5", timeout=600)
        code, out = run(ssh, f"{PYTHON} -c 'import torch; print(torch.__version__)' 2>/dev/null")
        print(f"    Nuovo PyTorch: {out.strip()}")

    print("  Verifico dipendenze...")
    deps = ["lightning", "tokenizers", "tqdm", "pyyaml", "numpy", "xxhash", "huggingface_hub", "flash_attn"]
    missing = []
    for dep in deps:
        code, out = run(ssh, f"{PYTHON} -c 'import {dep}' 2>/dev/null && echo OK || echo MISSING")
        if "MISSING" in out:
            missing.append(dep)
        else:
            print(f"    {dep}: OK")

    if missing:
        # flash_attn richiede build, lo installiamo separatamente
        non_flash = [d for d in missing if d != "flash_attn"]
        if non_flash:
            print(f"  Installo: {', '.join(non_flash)}")
            run(ssh, f"{PIP} install {' '.join(non_flash)} 2>&1 | tail -3", timeout=180)
        if "flash_attn" in missing:
            print("  Installo flash-attn (puo' richiedere 5-10 min)...")
            run(ssh, f"{PIP} install flash-attn --no-build-isolation 2>&1 | tail -5", timeout=600)

    print("  Verifico PyTorch + CUDA...")
    code, out = run(ssh, f'{PYTHON} -c "import torch; print(f\'torch={{torch.__version__}} cuda={{torch.cuda.is_available()}} gpu={{torch.cuda.get_device_name(0)}}\')"')
    print(f"    {out}")
    return "cuda=True" in out or "cuda=1" in out


# ─── Step 3: Download dati e checkpoint da HF Hub ─────────────

def download_data(ssh: paramiko.SSHClient) -> bool:
    print()
    print("=== Step 3: Download dati + checkpoint da HuggingFace Hub ===")
    print(f"  Repo: {HF_REPO}")
    print("  ~21GB totali (13GB train.bin + 8GB checkpoint + piccoli)")
    print()

    download_script = f"""
import os
from huggingface_hub import snapshot_download

os.environ["HF_TOKEN"] = "{HF_TOKEN}"
repo = "{HF_REPO}"
project = "{PROJECT_DIR}"

print(f"Downloading snapshot to {{project}}...")
snapshot_download(
    repo_id=repo,
    repo_type="dataset",
    local_dir=project,
    token=os.environ["HF_TOKEN"],
)
print("=== Verifica file ===")
expected = [
    "data/tokenized/pretrain/train.bin",
    "data/tokenized/pretrain/val.bin",
    "data/tokenized/pretrain/meta.json",
    "tokenizer/boccaccio-32k.json",
    "checkpoints/pretrain/last.ckpt",
    "configs/model.yaml",
    "configs/training.yaml",
    "configs/tokenizer.yaml",
]
all_ok = True
for f in expected:
    path = os.path.join(project, f)
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1e6
        print(f"OK: {{f}} ({{size_mb:.1f}} MB)")
    else:
        print(f"MISSING: {{f}}")
        all_ok = False
print("=== Download completo ===" if all_ok else "=== Download incompleto ===")
"""

    run(ssh, f"mkdir -p {PROJECT_DIR}/scripts")
    run(ssh, f"cat > {PROJECT_DIR}/scripts/_hf_download.py << 'PYEOF'\n{download_script}\nPYEOF")
    code, out = run(ssh, f"{PYTHON} {PROJECT_DIR}/scripts/_hf_download.py 2>&1", timeout=1800)
    safe = out.encode("ascii", errors="replace").decode("ascii")
    lines = [l for l in safe.split("\n") if l.strip()]
    for line in lines[-25:]:
        print(f"  {line}")

    run(ssh, f"rm {PROJECT_DIR}/scripts/_hf_download.py 2>/dev/null")

    return "Download completo" in out


# ─── Step 4: Crea script di auto-upload checkpoint ────────────

def create_upload_daemon(ssh: paramiko.SSHClient) -> bool:
    """Crea uno script bash che monitora last.ckpt e lo carica su HF Hub."""
    print()
    print("=== Step 4: Creazione daemon auto-upload checkpoint ===")

    daemon_script = f"""#!/bin/bash
# Auto-upload checkpoint daemon per Vast.ai (v2 - no version bloat)
# - Carica ogni checkpoint nominato (epoch=0-step=N.ckpt) UNA sola volta
#   con il suo nome originale (non sovrascrive last.ckpt ad ogni ciclo)
# - Quando il training termina, carica last.ckpt (finale) una sola volta
# - Mantiene un registro locale dei checkpoint gia' caricati
# - Pulisce i backup -v*.ckpt per risparmiare spazio

set -e

PROJECT_DIR="{PROJECT_DIR}"
CKPT_DIR="$PROJECT_DIR/checkpoints/pretrain"
HF_TOKEN="{HF_TOKEN}"
HF_REPO="{HF_REPO}"
UPLOAD_INTERVAL=300  # controlla ogni 5 min
UPLOADED_LOG="$CKPT_DIR/.uploaded_registry"
LAST_UPLOADED=""

# Carica registro checkpoint gia' caricati
mkdir -p "$CKPT_DIR"
touch "$UPLOADED_LOG"

echo "[daemon] Avvio auto-upload checkpoint daemon v2..."
echo "[daemon] Monitor: $CKPT_DIR/epoch=0-step=*.ckpt"
echo "[daemon] Repo HF: $HF_REPO"
echo "[daemon] Registro: $UPLOADED_LOG"

upload_ckpt() {{
    # $1 = path locale, $2 = path in repo
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
}}

is_uploaded() {{
    grep -qxF "$1" "$UPLOADED_LOG" 2>/dev/null
}}

mark_uploaded() {{
    echo "$1" >> "$UPLOADED_LOG"
}}

while true; do
    # Pulisci backup di Lightning (-v*.ckpt) per risparmiare spazio
    rm -f $CKPT_DIR/last-v*.ckpt $CKPT_DIR/epoch=0-step=*-v*.ckpt 2>/dev/null

    # 1. Carica ogni checkpoint nominato NON ancora caricato
    for CKPT in $(find $CKPT_DIR -maxdepth 1 -name 'epoch=0-step=*.ckpt' ! -name '*-v*.ckpt' -printf '%f\\n' 2>/dev/null | sort); do
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
        # Carica anche l'ultimo checkpoint nominato come last.ckpt se diverso
        LATEST_NAMED=$(find $CKPT_DIR -maxdepth 1 -name 'epoch=0-step=*.ckpt' ! -name '*-v*.ckpt' -printf '%T@ %p\\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2)
        if [ -n "$LATEST_NAMED" ] && ! is_uploaded "last.ckpt_final"; then
            echo "[daemon] Upload $LATEST_NAMED come last.ckpt finale..."
            if upload_ckpt "$LATEST_NAMED" "checkpoints/pretrain/last.ckpt"; then
                mark_uploaded "last.ckpt_final"
                echo "[daemon] Checkpoint finale sincronizzato come last.ckpt."
            fi
        fi
        echo "[daemon] Training terminato. Daemon in attesa (non esce per permettere retry)."
    fi

    sleep $UPLOAD_INTERVAL
done
"""

    run(ssh, f"cat > {PROJECT_DIR}/scripts/auto_upload_ckpt.sh << 'BASHEOF'\n{daemon_script}\nBASHEOF")
    run(ssh, f"chmod +x {PROJECT_DIR}/scripts/auto_upload_ckpt.sh")
    print("  Script creato: scripts/auto_upload_ckpt.sh")
    return True


# ─── Step 5: Avvia training + daemon in tmux ──────────────────

def start_training(ssh: paramiko.SSHClient) -> bool:
    print()
    print("=== Step 5: Avvio training + daemon ===")

    run(ssh, "tmux kill-session -t boccaccio 2>/dev/null")
    run(ssh, "tmux kill-session -t uploader 2>/dev/null")
    run(ssh, f"mkdir -p {PROJECT_DIR}/checkpoints/pretrain {PROJECT_DIR}/logs")

    # Training con resume da last.ckpt
    train_cmd = (
        f"cd {PROJECT_DIR} && "
        f"{PYTHON} -m src.training.train "
        "--mode pretrain "
        "--model-config configs/model.yaml "
        "--model-variant model "
        "--training-config configs/training.yaml "
        "--data-path data/tokenized/pretrain "
        "--tokenizer-path tokenizer/boccaccio-32k.json "
        "--resume-from checkpoints/pretrain/last.ckpt "
        "--wandb-offline "
        "2>&1 | tee logs/pretrain_training.log"
    )

    # Daemon di auto-upload
    upload_cmd = f"cd {PROJECT_DIR} && bash scripts/auto_upload_ckpt.sh 2>&1 | tee logs/auto_upload.log"

    print("  Avvio daemon auto-upload in tmux 'uploader'...")
    code, out = run(ssh, f"tmux new-session -d -s uploader '{upload_cmd}'")
    if code != 0:
        print(f"  ERRORE tmux uploader: {out}")
        return False

    print("  Avvio training in tmux 'boccaccio' (con resume da step 4000)...")
    code, out = run(ssh, f"tmux new-session -d -s boccaccio '{train_cmd}'")
    if code != 0:
        print(f"  ERRORE tmux: {out}")
        return False

    time.sleep(10)
    code, out = run(ssh, "tmux list-sessions 2>/dev/null")
    print(f"  Sessioni tmux: {out}")

    if "boccaccio" not in out:
        print("  ERRORE: tmux training non avviato.")
        code, log = run(ssh, f"tail -30 {PROJECT_DIR}/logs/pretrain_training.log 2>/dev/null")
        if log:
            print(f"  Log: {log}")
        return False

    print()
    print("  Training avviato in tmux 'boccaccio'")
    print("  Daemon upload avviato in tmux 'uploader'")
    return True


# ─── Main ─────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BoccaccioAI - Vast.ai setup & training")
    parser.add_argument("--host", type=str, required=True, help="IP dell'istanza Vast.ai")
    parser.add_argument("--port", type=int, default=22, help="Porta SSH")
    parser.add_argument("--user", type=str, default="root", help="User SSH (default: root)")
    parser.add_argument("--key", type=str, default=os.path.expanduser("~/.ssh/id_rsa"), help="Chiave SSH privata")
    parser.add_argument("--skip-download", action="store_true", help="Salta download dati (se gia' presenti)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not HF_TOKEN:
        print("ERRORE: Imposta HF_TOKEN environment variable.")
        print("  export HF_TOKEN=hf_xxxxxxxx")
        sys.exit(1)

    print("=" * 60)
    print("  BoccaccioAI - Vast.ai Setup & Training")
    print("=" * 60)
    print(f"  Host: {args.host}:{args.port} (user: {args.user})")
    print(f"  HF Repo: {HF_REPO}")
    print(f"  Project: {PROJECT_DIR}")

    # Connessione SSH
    print()
    print("Connessione SSH...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(args.host, port=args.port, username=args.user, key_filename=args.key, timeout=30)
    except Exception as e:
        print(f"ERRORE connessione SSH: {e}")
        sys.exit(1)
    print("  Connesso.")

    # Step 1: GPU
    if not check_gpu(ssh):
        print("GPU non valida. Uscita.")
        ssh.close()
        sys.exit(1)

    # Step 2: Setup progetto
    if not setup_project(ssh):
        print("Setup progetto fallito. Uscita.")
        ssh.close()
        sys.exit(1)

    # Step 3: Download dati
    if not args.skip_download:
        if not download_data(ssh):
            print("Download dati fallito. Uscita.")
            ssh.close()
            sys.exit(1)
    else:
        print()
        print("=== Step 3: Download saltato (--skip-download) ===")

    # Step 4: Crea daemon auto-upload
    create_upload_daemon(ssh)

    # Step 5: Avvia training
    if not start_training(ssh):
        print("Avvio training fallito. Uscita.")
        ssh.close()
        sys.exit(1)

    print()
    print("=" * 60)
    print("  SETUP COMPLETATO!")
    print("=" * 60)
    print()
    print("  Monitora con:")
    print(f"    python scripts/vast_monitor.py --host {args.host} --port {args.port} --key {args.key}")
    print()
    print("  SSH interattivo:")
    print(f"    ssh -p {args.port} {args.user}@{args.host}")
    print(f"    tmux attach -t boccaccio    # training")
    print(f"    tmux attach -t uploader     # daemon upload")
    print()
    print("  I checkpoint vengono caricati automaticamente su HF Hub.")
    print("=" * 60)

    ssh.close()


if __name__ == "__main__":
    main()
