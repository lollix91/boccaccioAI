"""BoccaccioAI - Setup Vast.ai per fine-tuning (Fase 4).

Scarica dati + checkpoint da HF Hub, configura il daemon di upload,
e avvia il fine-tuning in tmux.

Uso:
    python scripts/vast_finetune.py --host 115.124.123.240 --port 23014 --key ~/.ssh/vast_rsa

De Lauretis Tech
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import paramiko

# ─── Config ───────────────────────────────────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = "lollix91/boccaccio-data"

PROJECT_DIR = "/workspace/boccaccioAI"
PYTHON = "python3"
PIP = "pip3"


# ─── Helper ───────────────────────────────────────────────────

def run(ssh: paramiko.SSHClient, cmd: str, timeout: int = 300) -> tuple[int, str]:
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


# ─── Step 2: Setup progetto ───────────────────────────────────

def setup_project(ssh: paramiko.SSHClient) -> bool:
    print()
    print("=== Step 2: Setup progetto ===")

    run(ssh, "mkdir -p /workspace")

    code, out = run(ssh, f"test -d {PROJECT_DIR}/.git && echo exists")
    if "exists" in out:
        print("  Repo presente, git pull...")
        run(ssh, f"cd {PROJECT_DIR} && git pull origin main 2>&1")
    else:
        print("  Clone repo...")
        code, out = run(ssh, "cd /workspace && git clone https://github.com/lollix91/boccaccioAI.git 2>&1")
        if code != 0:
            print(f"  ERRORE clone: {out}")
            return False

    # Verifica dipendenze
    deps = ["torch", "lightning", "tokenizers", "tqdm", "yaml", "huggingface_hub"]
    for dep in deps:
        code, out = run(ssh, f"{PYTHON} -c 'import {dep}' 2>/dev/null && echo OK || echo MISSING")
        if "MISSING" in out:
            print(f"  Installando {dep}...")
            run(ssh, f"{PIP} install {dep} 2>&1 | tail -3", timeout=180)

    code, out = run(ssh, f'{PYTHON} -c "import torch; print(f\'torch={{torch.__version__}} cuda={{torch.cuda.is_available()}} gpu={{torch.cuda.get_device_name(0)}}\')"')
    print(f"  {out}")
    return True


# ─── Step 3: Download dati finetune + checkpoint da HF ────────

def download_data(ssh: paramiko.SSHClient) -> bool:
    print()
    print("=== Step 3: Download dati finetune + checkpoint da HF Hub ===")
    print(f"  Repo: {HF_REPO}")
    print("  ~22GB totali (13GB pretrain train.bin + 8GB checkpoint + 200MB finetune)")
    print()

    download_script = f'''
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
    "data/tokenized/finetune/train.bin",
    "data/tokenized/finetune/val.bin",
    "data/tokenized/finetune/meta.json",
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
'''

    run(ssh, f"mkdir -p {PROJECT_DIR}/scripts")
    run(ssh, f"cat > {PROJECT_DIR}/scripts/_hf_download.py << 'PYEOF'\n{download_script}\nPYEOF")
    code, out = run(ssh, f"{PYTHON} {PROJECT_DIR}/scripts/_hf_download.py 2>&1", timeout=1800)
    print(out)
    run(ssh, f"rm {PROJECT_DIR}/scripts/_hf_download.py 2>/dev/null")

    return "Download completo" in out


# ─── Step 4: Crea daemon auto-upload ──────────────────────────

def create_upload_daemon(ssh: paramiko.SSHClient) -> bool:
    print()
    print("=== Step 4: Configurazione daemon auto-upload ===")

    # Scrivi il token HF su file (per il daemon)
    run(ssh, f"echo '{HF_TOKEN}' > {PROJECT_DIR}/.hf_token && chmod 600 {PROJECT_DIR}/.hf_token")

    # Il daemon script e' gia nel repo (auto_upload_ckpt.sh), ma va adattato per finetune
    daemon_script = """#!/bin/bash
# Auto-upload daemon per checkpoint di fine-tuning BoccaccioAI
# Monitora checkpoints/finetune/ e uploada su HF Hub

PROJECT_DIR="/workspace/boccaccioAI"
CKPT_DIR="${PROJECT_DIR}/checkpoints/finetune"
HF_REPO="lollix91/boccaccio-data"
HF_TOKEN="${HF_TOKEN:-$(cat ${PROJECT_DIR}/.hf_token 2>/dev/null)}"
UPLOADED_LOG="${PROJECT_DIR}/.uploaded_finetune_registry"
RECHECK_INTERVAL=60

mkdir -p "$CKPT_DIR"
touch "$UPLOADED_LOG"

echo "[daemon] Avvio daemon upload finetune (v1)"
echo "[daemon] Monitor: $CKPT_DIR"
echo "[daemon] Intervallo: ${RECHECK_INTERVAL}s"

while true; do
    if [ ! -d "$CKPT_DIR" ]; then
        sleep "$RECHECK_INTERVAL"
        continue
    fi

    # Trova tutti i file .ckpt
    for ckpt in "$CKPT_DIR"/*.ckpt; do
        [ -e "$ckpt" ] || continue

        filename=$(basename "$ckpt")

        # Salta file temporanei di Lightning
        if echo "$filename" | grep -q "\\.tmp"; then
            continue
        fi

        # Verifica che il file sia completo (non in scrittura)
        size1=$(stat -c%s "$ckpt" 2>/dev/null || echo 0)
        sleep 2
        size2=$(stat -c%s "$ckpt" 2>/dev/null || echo 0)
        if [ "$size1" != "$size2" ]; then
            continue
        fi
        if [ "$size2" -lt 100000 ]; then
            continue
        fi

        # Già caricato?
        if grep -q "^${filename}$" "$UPLOADED_LOG" 2>/dev/null; then
            continue
        fi

        echo "[daemon] Trovato nuovo checkpoint: $filename ($(( size2 / 1000000 ))MB)"

        # Copia temporanea per evitare problemi di lettura
        tmp_ckpt="/tmp/_upload_${filename}"
        cp "$ckpt" "$tmp_ckpt" 2>/dev/null

        # Upload su HF
        python3 -c "
from huggingface_hub import HfApi
import os, sys
api = HfApi(token='${HF_TOKEN}')
try:
    api.upload_file(
        path_or_fileobj='${tmp_ckpt}',
        path_in_repo='checkpoints/finetune/${filename}',
        repo_id='${HF_REPO}',
        repo_type='dataset',
        token='${HF_TOKEN}',
    )
    print('OK')
except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(1)
" 2>&1

        result=$?
        rm -f "$tmp_ckpt" 2>/dev/null

        if [ $result -eq 0 ]; then
            echo "[daemon] Upload OK: $filename"
            echo "$filename" >> "$UPLOADED_LOG"

            # Mantieni solo gli ultimi 2 checkpoint locali + last.ckpt
            ls -t "$CKPT_DIR"/epoch=*.ckpt 2>/dev/null | tail -n +3 | while read old_ckpt; do
                echo "[daemon] Rimuovo vecchio checkpoint locale: $(basename "$old_ckpt")"
                rm -f "$old_ckpt"
            done
        else
            echo "[daemon] Upload FALLITO: $filename, riprovo al prossimo ciclo"
        fi
    done

    sleep "$RECHECK_INTERVAL"
done
"""

    run(ssh, f"cat > {PROJECT_DIR}/scripts/auto_upload_finetune.sh << 'BASHEOF'\n{daemon_script}\nBASHEOF")
    run(ssh, f"chmod +x {PROJECT_DIR}/scripts/auto_upload_finetune.sh")
    print("  Daemon script creato: scripts/auto_upload_finetune.sh")
    return True


# ─── Step 5: Avvia fine-tuning ────────────────────────────────

def start_training(ssh: paramiko.SSHClient) -> bool:
    print()
    print("=== Step 5: Avvio fine-tuning + daemon ===")

    run(ssh, "tmux kill-session -t boccaccio 2>/dev/null")
    run(ssh, "tmux kill-session -t uploader 2>/dev/null")
    run(ssh, f"mkdir -p {PROJECT_DIR}/checkpoints/finetune {PROJECT_DIR}/logs")

    # Training: fine-tuning da checkpoint pretrain
    train_cmd = (
        f"cd {PROJECT_DIR} && "
        f"{PYTHON} -m src.training.train "
        "--mode finetune "
        "--model-config configs/model.yaml "
        "--model-variant model "
        "--training-config configs/training.yaml "
        "--data-path data/tokenized/finetune "
        "--tokenizer-path tokenizer/boccaccio-32k.json "
        "--resume-from checkpoints/pretrain/last.ckpt "
        "--wandb-offline "
        "2>&1 | tee logs/finetune_training.log"
    )

    # Daemon di auto-upload per finetune
    upload_cmd = f"cd {PROJECT_DIR} && bash scripts/auto_upload_finetune.sh 2>&1 | tee logs/auto_upload_finetune.log"

    print("  Avvio daemon auto-upload in tmux 'uploader'...")
    code, out = run(ssh, f"tmux new-session -d -s uploader '{upload_cmd}'")
    if code != 0:
        print(f"  ERRORE tmux uploader: {out}")
        return False

    print("  Avvio fine-tuning in tmux 'boccaccio'...")
    code, out = run(ssh, f"tmux new-session -d -s boccaccio '{train_cmd}'")
    if code != 0:
        print(f"  ERRORE tmux: {out}")
        return False

    time.sleep(10)
    code, out = run(ssh, "tmux list-sessions 2>/dev/null")
    print(f"  Sessioni tmux: {out}")

    if "boccaccio" not in out:
        print("  ERRORE: tmux training non avviato.")
        code, log = run(ssh, f"tail -30 {PROJECT_DIR}/logs/finetune_training.log 2>/dev/null")
        if log:
            print(f"  Log: {log}")
        return False

    print()
    print("  Fine-tuning avviato in tmux 'boccaccio'")
    print("  Daemon upload avviato in tmux 'uploader'")
    return True


# ─── Main ─────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BoccaccioAI - Vast.ai fine-tuning setup")
    parser.add_argument("--host", type=str, required=True, help="IP dell'istanza Vast.ai")
    parser.add_argument("--port", type=int, default=22, help="Porta SSH")
    parser.add_argument("--user", type=str, default="root", help="User SSH")
    parser.add_argument("--key", type=str, default=os.path.expanduser("~/.ssh/id_rsa"), help="Chiave SSH privata")
    parser.add_argument("--skip-download", action="store_true", help="Salta download dati (se gia presenti)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not HF_TOKEN:
        print("ERRORE: HF_TOKEN non impostato.")
        print("  export HF_TOKEN=hf_xxxxxxxx")
        sys.exit(1)

    print("=" * 60)
    print("  BoccaccioAI - Vast.ai Fine-Tuning Setup")
    print("=" * 60)
    print(f"  Host: {args.host}:{args.port} (user: {args.user})")
    print(f"  Key: {args.key}")
    print(f"  HF Repo: {HF_REPO}")
    print()

    # Connessione SSH
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
        ssh.close()
        sys.exit(1)

    # Step 2: Setup
    if not setup_project(ssh):
        ssh.close()
        sys.exit(1)

    # Step 3: Download
    if not args.skip_download:
        if not download_data(ssh):
            print("ERRORE: Download dati fallito.")
            ssh.close()
            sys.exit(1)
    else:
        print("\n=== Step 3: Skip download (flag --skip-download) ===")

    # Step 4: Daemon
    create_upload_daemon(ssh)

    # Step 5: Training
    if not start_training(ssh):
        ssh.close()
        sys.exit(1)

    print()
    print("=" * 60)
    print("  Setup completato!")
    print()
    print("  Monitora:")
    print(f"    python scripts/vast_monitor.py --host {args.host} --port {args.port} --key {args.key}")
    print()
    print("  SSH:")
    print(f"    ssh -p {args.port} -i {args.key} {args.user}@{args.host}")
    print(f"    tmux attach -t boccaccio")
    print("=" * 60)

    ssh.close()


if __name__ == "__main__":
    main()
