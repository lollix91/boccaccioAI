"""BoccaccioAI - Lightning.ai Studio Setup & Training Launcher.

Si collega allo Studio via SSH, carica i dati pre-tokenizzati dal PC locale,
verifica le dipendenze, e lancia il training in una sessione tmux persistente.
Chiudendo PowerShell il training continua sul server.

Uso:
    python scripts/lightning_setup.py --host <IP> --port <PORT> --password <PASSWORD>

De Lauretis Tech
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import paramiko


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
    print(f"  Upload: {local_path} ({local_size / 1e6:.1f} MB) -> {remote_path}")

    downloaded = [0]
    last_print = [0]

    def callback(transferred: int, total: int) -> None:
        downloaded[0] = transferred
        if total > 0:
            pct = transferred / total * 100
            if pct - last_print[0] >= 10 or transferred == total:
                last_print[0] = pct
                print(f"    {pct:.0f}%", end="\r", flush=True)

    sftp.put(local_path, remote_path, callback=callback)
    print(f"    100% - completato")
    return True


def upload_dir(sftp: paramiko.SFTPClient, ssh: paramiko.SSHClient,
               local_dir: str, remote_dir: str) -> int:
    """Carica ricorsivamente una directory. Ritorna il numero di file caricati."""
    run(ssh, f"mkdir -p {remote_dir}")
    count = 0
    for item in os.listdir(local_dir):
        local_path = os.path.join(local_dir, item)
        remote_path = f"{remote_dir}/{item}"
        if os.path.isfile(local_path):
            if upload_file(sftp, local_path, remote_path):
                count += 1
        elif os.path.isdir(local_path):
            count += upload_dir(sftp, ssh, local_path, remote_path)
    return count


# ─── Step 1: Verifica GPU e ambiente ──────────────────────────

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
        print(f"  ATTENZIONE: GPU non e' H100. Potrebbe essere piu' lenta del previsto.")
    return True


# ─── Step 2: Setup progetto e dipendenze ──────────────────────

def setup_project(ssh: paramiko.SSHClient) -> bool:
    """Clona il repo e installa le dipendenze."""
    print()
    print("=== Step 2: Setup progetto ===")

    # Verifica se il repo esiste gia'
    code, _ = run(ssh, "test -d /root/boccaccioAI/.git && echo exists")
    if "exists" in _:
        print("  Repo gia' presente, faccio pull...")
        run(ssh, "cd /root/boccaccioAI && git pull origin main 2>&1")
    else:
        print("  Clonando il repo...")
        code, out = run(ssh, "cd /root && git clone https://github.com/lollix91/boccaccioAI.git 2>&1")
        if code != 0:
            print(f"  ERRORE: git clone fallito: {out}")
            return False

    # Installa dipendenze
    print("  Verifico dipendenze...")
    deps = ["lightning", "tokenizers", "tqdm", "pyyaml", "numpy", "xxhash"]
    for dep in deps:
        code, out = run(ssh, f"python -c 'import {dep}' 2>/dev/null && echo OK || echo MISSING")
        if "MISSING" in out:
            print(f"    Installando {dep}...")
            run(ssh, f"pip install {dep} 2>&1 | tail -1", timeout=120)
        else:
            print(f"    {dep}: OK")

    # Verifica PyTorch con CUDA
    print("  Verifico PyTorch + CUDA...")
    code, out = run(ssh, "python -c \"import torch; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')\"")
    print(f"    {out}")
    if "cuda=True" not in out:
        print("  ERRORE: CUDA non disponibile su PyTorch.")
        return False

    return True


# ─── Step 3: Upload dati ──────────────────────────────────────

def upload_data(sftp: paramiko.SFTPClient, ssh: paramiko.SSHClient, local_root: str) -> bool:
    """Carica tokenizer e dati pre-tokenizzati dal PC locale."""
    print()
    print("=== Step 3: Upload dati ===")

    remote_root = "/root/boccaccioAI"
    uploaded = 0

    # Tokenizer
    print("  [Tokenizer]")
    run(ssh, f"mkdir -p {remote_root}/tokenizer")
    tok_path = os.path.join(local_root, "tokenizer", "boccaccio-32k.json")
    if upload_file(sftp, tok_path, f"{remote_root}/tokenizer/boccaccio-32k.json"):
        uploaded += 1

    # Dati pre-tokenizzati
    print("  [Dati pre-tokenizzati]")
    data_dir = os.path.join(local_root, "data", "tokenized", "pretrain")
    if not os.path.exists(data_dir):
        print(f"  ERRORE: {data_dir} non trovato localmente.")
        return False

    run(ssh, f"mkdir -p {remote_root}/data/tokenized/pretrain")
    for fname in ["train.bin", "val.bin", "meta.json"]:
        local_path = os.path.join(data_dir, fname)
        if upload_file(sftp, local_path, f"{remote_root}/data/tokenized/pretrain/{fname}"):
            uploaded += 1

    print(f"  Upload completato: {uploaded} file")
    return True


# ─── Step 4: Verifica dati sul server ─────────────────────────

def verify_data(ssh: paramiko.SSHClient) -> bool:
    """Verifica che i file siano arrivati interi."""
    print()
    print("=== Step 4: Verifica dati ===")

    checks = [
        ("tokenizer/boccaccio-32k.json", 2_000_000),  # ~2.2MB
        ("data/tokenized/pretrain/train.bin", 13_000_000_000),  # ~13GB
        ("data/tokenized/pretrain/val.bin", 60_000_000),  # ~64MB
        ("data/tokenized/pretrain/meta.json", 100),  # ~138B
    ]

    all_ok = True
    for rel_path, min_size in checks:
        full_path = f"/root/boccaccioAI/{rel_path}"
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

    # Verifica che tmux sia installato
    code, out = run(ssh, "which tmux 2>/dev/null")
    if code != 0 or not out:
        print("  Installo tmux...")
        run(ssh, "apt-get update -qq && apt-get install -y -qq tmux 2>&1 | tail -1", timeout=120)

    # Uccidi sessione tmux esistente se presente
    run(ssh, "tmux kill-session -t boccaccio 2>/dev/null")

    # Crea directory per checkpoint
    run(ssh, "mkdir -p /root/boccaccioAI/checkpoints/pretrain")

    # Comando di training
    train_cmd = (
        "cd /root/boccaccioAI && "
        "python -m src.training.train "
        "--mode pretrain "
        "--model-config configs/model.yaml "
        "--model-variant model "
        "--training-config configs/training.yaml "
        "--data-path data/tokenized/pretrain "
        "--tokenizer-path tokenizer/boccaccio-32k.json "
        "--wandb-offline "
        "2>&1 | tee logs/pretrain_training.log"
    )

    # Crea directory logs
    run(ssh, "mkdir -p /root/boccaccioAI/logs")

    # Avvia in tmux
    tmux_cmd = f"tmux new-session -d -s boccaccio '{train_cmd}'"
    code, out = run(ssh, tmux_cmd)
    if code != 0:
        print(f"  ERRORE: impossibile avviare tmux: {out}")
        return False

    # Attendi qualche secondo e verifica che il processo sia attivo
    time.sleep(5)
    code, out = run(ssh, "tmux list-sessions 2>/dev/null")
    if "boccaccio" not in out:
        print("  ERRORE: sessione tmux non trovata dopo l'avvio.")
        return False

    print("  Training avviato in sessione tmux 'boccaccio'")
    print("  Il training continuera' anche se chiudi PowerShell.")
    return True


# ─── Main ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="BoccaccioAI - Lightning.ai Setup & Training")
    parser.add_argument("--host", required=True, help="Studio SSH host")
    parser.add_argument("--port", type=int, default=22, help="SSH port")
    parser.add_argument("--user", default="root", help="SSH user")
    parser.add_argument("--password", default=None, help="SSH password")
    parser.add_argument("--key", default=None, help="SSH private key path (alternativa a password)")
    parser.add_argument("--local-root", default=".", help="Directory locale del progetto")
    parser.add_argument("--skip-upload", action="store_true", help="Salta upload dati (gia' caricati)")
    args = parser.parse_args()

    print("=" * 60)
    print("  BoccaccioAI - Lightning.ai Setup & Training")
    print("=" * 60)

    # ─── Connessione ──────────────────────────────────────────
    print(f"\nConnessione a {args.host}:{args.port}...", end=" ")
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
        print("\nERRORE: GPU non disponibile. Annullamento.")
        ssh.close()
        sys.exit(1)

    # ─── Step 2: Progetto e dipendenze ────────────────────────
    if not setup_project(ssh):
        print("\nERRORE: setup progetto fallito. Annullamento.")
        ssh.close()
        sys.exit(1)

    # ─── Step 3: Upload dati ──────────────────────────────────
    if not args.skip_upload:
        if not upload_data(sftp, ssh, args.local_root):
            print("\nERRORE: upload dati fallito. Annullamento.")
            ssh.close()
            sys.exit(1)
    else:
        print("\n=== Step 3: Upload dati (saltato) ===")

    # ─── Step 4: Verifica ─────────────────────────────────────
    if not verify_data(ssh):
        print("\nERRORE: verifica dati fallita. Annullamento.")
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
    print("    python scripts/lightning_monitor.py \\")
    print(f"      --host {args.host} --port {args.port} --password <PASSWORD>")
    print()
    print("  Per scaricare i risultati:")
    print("    python scripts/lightning_download.py \\")
    print(f"      --host {args.host} --port {args.port} --password <PASSWORD>")
    print()


if __name__ == "__main__":
    main()
