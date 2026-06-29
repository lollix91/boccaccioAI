"""BoccaccioAI - Lightning.ai Studio Setup & Training Launcher.

Si collega allo Studio via SSH, carica i dati pre-tokenizzati dal PC locale,
verifica le dipendenze, e lancia il training in una sessione tmux persistente.
Chiudendo PowerShell il training continua sul server.

Uso:
    python scripts/lightning_setup.py
    python scripts/lightning_setup.py --skip-upload  # se dati gia' caricati

De Lauretis Tech
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import paramiko


# ─── Configurazione Lightning.ai ──────────────────────────────

LIGHTNING_HOST = "ssh.lightning.ai"
LIGHTNING_PORT = 22
LIGHTNING_USER = "s_01kw9jgs29f9znwd4cwpcctbpa"
LIGHTNING_KEY = os.path.expanduser("~/.ssh/lightning_rsa")

# Ambiente conda sullo Studio
CONDA_PYTHON = "/home/zeus/miniconda3/envs/cloudspace/bin/python"
CONDA_PIP = "/home/zeus/miniconda3/envs/cloudspace/bin/pip"
PROJECT_DIR = "/home/zeus/content/boccaccioAI"


# ─── Helper ───────────────────────────────────────────────────

def run(ssh: paramiko.SSHClient, cmd: str, timeout: int = 120) -> tuple[int, str]:
    """Esegue un comando SSH e ritorna (exit_code, output)."""
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return exit_code, (out + err).strip()


def upload_file(sftp: paramiko.SFTPClient, local_path: str, remote_path: str) -> bool:
    """Carica un file locale sul server remoto con progress bar."""
    if not os.path.exists(local_path):
        print(f"  SKIP: {local_path} (non trovato localmente)")
        return False

    local_size = os.path.getsize(local_path)
    print(f"  Upload: {os.path.basename(local_path)} ({local_size / 1e6:.1f} MB)")

    last_print = [0]

    def callback(transferred: int, total: int) -> None:
        if total > 0:
            pct = transferred / total * 100
            if pct - last_print[0] >= 10 or transferred == total:
                last_print[0] = pct
                print(f"    {pct:.0f}%", end="\r", flush=True)

    sftp.put(local_path, remote_path, callback=callback)
    print(f"    100% - completato")
    return True


# ─── Step 1: Verifica GPU ─────────────────────────────────────

def check_gpu(ssh: paramiko.SSHClient) -> bool:
    """Verifica che la GPU H100 sia disponibile."""
    print()
    print("=== Step 1: Verifica GPU ===")
    code, out = run(ssh, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null")
    if code != 0 or not out:
        print("  ERRORE: nvidia-smi non trovato. GPU non disponibile?")
        return False
    print(f"  GPU rilevata: {out}")
    if "H100" not in out:
        print("  ATTENZIONE: GPU non e' H100.")
    return True


# ─── Step 2: Setup progetto e dipendenze ──────────────────────

def setup_project(ssh: paramiko.SSHClient) -> bool:
    """Clona il repo e installa le dipendenze mancanti."""
    print()
    print("=== Step 2: Setup progetto ===")

    # Verifica se il repo esiste gia'
    code, out = run(ssh, f"test -d {PROJECT_DIR}/.git && echo exists")
    if "exists" in out:
        print("  Repo gia' presente, faccio pull...")
        run(ssh, f"cd {PROJECT_DIR} && git pull origin main 2>&1")
    else:
        print("  Clonando il repo...")
        code, out = run(ssh, f"cd /home/zeus/content && git clone https://github.com/lollix91/boccaccioAI.git 2>&1")
        if code != 0:
            print(f"  ERRORE: git clone fallito: {out}")
            return False

    # Verifica dipendenze
    print("  Verifico dipendenze...")
    deps_to_check = ["lightning", "tokenizers", "tqdm", "pyyaml", "numpy", "xxhash"]
    missing = []
    for dep in deps_to_check:
        code, out = run(ssh, f"{CONDA_PYTHON} -c 'import {dep}' 2>/dev/null && echo OK || echo MISSING")
        if "MISSING" in out:
            missing.append(dep)
        else:
            print(f"    {dep}: OK")

    if missing:
        print(f"  Installando: {', '.join(missing)}")
        run(ssh, f"{CONDA_PIP} install {' '.join(missing)} 2>&1 | tail -3", timeout=180)

    # Verifica PyTorch + CUDA
    print("  Verifico PyTorch + CUDA...")
    code, out = run(ssh, f'{CONDA_PYTHON} -c "import torch; print(f\'torch={{torch.__version__}} cuda={{torch.cuda.is_available()}} gpu={{torch.cuda.get_device_name(0)}}\')"')
    print(f"    {out}")
    if "cuda=True" not in out:
        print("  ERRORE: CUDA non disponibile.")
        return False

    return True


# ─── Step 3: Upload dati ──────────────────────────────────────

def upload_data(sftp: paramiko.SFTPClient, ssh: paramiko.SSHClient, local_root: str) -> bool:
    """Carica tokenizer e dati pre-tokenizzati dal PC locale."""
    print()
    print("=== Step 3: Upload dati (13GB) ===")
    print("  Questo richiedera' ~15-30 min a seconda della banda.")
    print()

    uploaded = 0

    # Tokenizer
    print("  [Tokenizer]")
    run(ssh, f"mkdir -p {PROJECT_DIR}/tokenizer")
    tok_path = os.path.join(local_root, "tokenizer", "boccaccio-32k.json")
    if upload_file(sftp, tok_path, f"{PROJECT_DIR}/tokenizer/boccaccio-32k.json"):
        uploaded += 1

    # Dati pre-tokenizzati
    print("  [Dati pre-tokenizzati]")
    data_dir = os.path.join(local_root, "data", "tokenized", "pretrain")
    if not os.path.exists(data_dir):
        print(f"  ERRORE: {data_dir} non trovato localmente.")
        return False

    run(ssh, f"mkdir -p {PROJECT_DIR}/data/tokenized/pretrain")
    for fname in ["train.bin", "val.bin", "meta.json"]:
        local_path = os.path.join(data_dir, fname)
        if upload_file(sftp, local_path, f"{PROJECT_DIR}/data/tokenized/pretrain/{fname}"):
            uploaded += 1

    print(f"  Upload completato: {uploaded} file")
    return True


# ─── Step 4: Verifica dati sul server ─────────────────────────

def verify_data(ssh: paramiko.SSHClient) -> bool:
    """Verifica che i file siano arrivati interi."""
    print()
    print("=== Step 4: Verifica dati ===")

    checks = [
        ("tokenizer/boccaccio-32k.json", 2_000_000),
        ("data/tokenized/pretrain/train.bin", 13_000_000_000),
        ("data/tokenized/pretrain/val.bin", 60_000_000),
        ("data/tokenized/pretrain/meta.json", 100),
    ]

    all_ok = True
    for rel_path, min_size in checks:
        full_path = f"{PROJECT_DIR}/{rel_path}"
        code, out = run(ssh, f"stat -c '%s' {full_path} 2>/dev/null")
        if code != 0 or not out:
            print(f"  MANCANTE: {rel_path}")
            all_ok = False
            continue
        size = int(out)
        size_mb = size / 1e6
        if size < min_size:
            print(f"  INCOMPLETO: {rel_path} ({size_mb:.1f} MB, atteso >= {min_size / 1e6:.1f} MB)")
            all_ok = False
        else:
            print(f"  OK: {rel_path} ({size_mb:.1f} MB)")

    return all_ok


# ─── Step 5: Avvia training in tmux ───────────────────────────

def start_training(ssh: paramiko.SSHClient) -> bool:
    """Lancia il training in una sessione tmux persistente."""
    print()
    print("=== Step 5: Avvio training ===")

    # Uccidi sessione tmux esistente
    run(ssh, "tmux kill-session -t boccaccio 2>/dev/null")

    # Crea directory
    run(ssh, f"mkdir -p {PROJECT_DIR}/checkpoints/pretrain {PROJECT_DIR}/logs")

    # Comando di training (usa il python del conda env cloudspace)
    train_cmd = (
        f"cd {PROJECT_DIR} && "
        f"{CONDA_PYTHON} -m src.training.train "
        "--mode pretrain "
        "--model-config configs/model.yaml "
        "--model-variant model "
        "--training-config configs/training.yaml "
        "--data-path data/tokenized/pretrain "
        "--tokenizer-path tokenizer/boccaccio-32k.json "
        "--wandb-offline "
        "2>&1 | tee logs/pretrain_training.log"
    )

    # Avvia in tmux
    tmux_cmd = f"tmux new-session -d -s boccaccio '{train_cmd}'"
    code, out = run(ssh, tmux_cmd)
    if code != 0:
        print(f"  ERRORE: impossibile avviare tmux: {out}")
        return False

    # Attendi e verifica
    time.sleep(8)
    code, out = run(ssh, "tmux list-sessions 2>/dev/null")
    if "boccaccio" not in out:
        print("  ERRORE: sessione tmux non trovata dopo l'avvio.")
        # Mostra log per debug
        code, log = run(ssh, f"tail -20 {PROJECT_DIR}/logs/pretrain_training.log 2>/dev/null")
        if log:
            print(f"  Log: {log}")
        return False

    print("  Training avviato in sessione tmux 'boccaccio'")
    print("  Il training continuera' anche se chiudi PowerShell.")
    return True


# ─── Main ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="BoccaccioAI - Lightning.ai Setup & Training")
    parser.add_argument("--host", default=LIGHTNING_HOST, help="Studio SSH host")
    parser.add_argument("--port", type=int, default=LIGHTNING_PORT, help="SSH port")
    parser.add_argument("--user", default=LIGHTNING_USER, help="SSH user")
    parser.add_argument("--password", default=None, help="SSH password")
    parser.add_argument("--key", default=LIGHTNING_KEY, help="SSH private key path")
    parser.add_argument("--local-root", default=".", help="Directory locale del progetto")
    parser.add_argument("--skip-upload", action="store_true", help="Salta upload dati (gia' caricati)")
    args = parser.parse_args()

    print("=" * 60)
    print("  BoccaccioAI - Lightning.ai Setup & Training")
    print("=" * 60)

    # ─── Connessione ──────────────────────────────────────────
    print(f"\nConnessione a {args.host}...", end=" ")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(
            args.host,
            port=args.port,
            username=args.user,
            password=args.password,
            key_filename=args.key,
            timeout=15,
        )
    except Exception as e:
        print(f"FALLITO: {e}")
        sys.exit(1)
    print("OK")

    sftp = ssh.open_sftp()

    # ─── Step 1: GPU ──────────────────────────────────────────
    if not check_gpu(ssh):
        print("\nERRORE: GPU non disponibile.")
        ssh.close()
        sys.exit(1)

    # ─── Step 2: Progetto e dipendenze ────────────────────────
    if not setup_project(ssh):
        print("\nERRORE: setup progetto fallito.")
        ssh.close()
        sys.exit(1)

    # ─── Step 3: Upload dati ──────────────────────────────────
    if not args.skip_upload:
        if not upload_data(sftp, ssh, args.local_root):
            print("\nERRORE: upload dati fallito.")
            ssh.close()
            sys.exit(1)
    else:
        print("\n=== Step 3: Upload dati (saltato) ===")

    # ─── Step 4: Verifica ─────────────────────────────────────
    if not verify_data(ssh):
        print("\nERRORE: verifica dati fallita.")
        ssh.close()
        sys.exit(1)

    # ─── Step 5: Training ─────────────────────────────────────
    if not start_training(ssh):
        print("\nERRORE: avvio training fallito.")
        ssh.close()
        sys.exit(1)

    sftp.close()
    ssh.close()

    print()
    print("=" * 60)
    print("  SETUP COMPLETATO - Training in corso!")
    print("=" * 60)
    print()
    print("  Il training gira in tmux sul server.")
    print("  Puoi chiudere PowerShell: il training continua.")
    print()
    print("  Per monitorare:")
    print("    python scripts/lightning_monitor.py")
    print()
    print("  Per scaricare i risultati:")
    print("    python scripts/lightning_download.py")
    print()


if __name__ == "__main__":
    main()
